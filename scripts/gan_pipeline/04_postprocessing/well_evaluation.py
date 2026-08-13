import os
import sys
import glob
import math
import random
import pathlib
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.ndimage import label
from scipy.spatial.distance import jensenshannon

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import torch
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import MDS
from voxgan.models.metrics import MSSWD

import pickle
from tqdm import tqdm
from skimage.util import view_as_windows

scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from gan_pipeline.core.custom_plots import apply_custom_plotting_flavor, FaciesColorMap

apply_custom_plotting_flavor()


def get_facies_config(num_classes=3):
    """Centralized configuration for facies properties to prevent hardcoding.

    Args:
        num_classes (int): Number of geological classes (3 or 9).

    Returns:
        dict: Configuration dictionary containing codes, names, colors, and mapping.

    Raises:
        ValueError: If num_classes is not 3 or 9.
    """
    if num_classes == 3:
        return {
            'codes': [1, 4, 8],
            'names': {
                1: 'Sand body deposits',
                4: 'Crevasse splay & Levee deposits',
                8: 'Clay deposits'
            },
            'colors': {
                1: '#f1970f',
                4: '#fffc65',
                8: '#33ff00'
            },
            'mapping': {0: 1, 1: 4, 2: 8} 
        }
    elif num_classes == 9:
        props = FaciesColorMap.FACIES_PROPERTIES
        codes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        names_dict = {}
        colors_dict = {}
        
        for key, info in props.items():
            val = info['val']
            if val in codes:
                names_dict[val] = key.replace('_', ' ').title()
                colors_dict[val] = info['color']
                
        return {
            'codes': codes,
            'names': names_dict,
            'colors': colors_dict,
            'mapping': {i-1: i for i in range(1, 10)}
        }
    else:
        raise ValueError(f"num_classes={num_classes} not supported.")


class WellMismatch:
    """Quantifies the degree of mismatch between 3D realizations and well data.

    Calculates the Macro F1 score to evaluate how accurately the generated grid
    captures the well data. Macro F1 accounts for class imbalance, weighting 
    minority facies equally.
    """

    def __init__(self, flumy_name, gan_name, data_dir, well_data_path, num_classes=3):
        """Initializes the WellMismatch class with directory and configuration settings.

        Args:
            flumy_name (str): Name of the Flumy simulation setting.
            gan_name (str): Name of the GAN configuration.
            data_dir (str or Path): Glob pattern or directory path for realization files.
            well_data_path (str or Path): Path to the Excel spreadsheet containing well data.
            num_classes (int, optional): Number of facies classes. Defaults to 3.
        """
        self.flumy_name = flumy_name
        self.gan_name = gan_name
        self.num_classes = num_classes
        self.cfg = get_facies_config(num_classes)

        self.data_files = sorted(glob.glob(str(data_dir)))
        if not self.data_files:
            print(f"Warning: No files found matching {data_dir}")

        self.well_data_path = well_data_path
        self.well_coords = []
        self.well_true_facies = []

        self._load_well_data()

    def _load_well_data(self):
        """Loads and maps well data coordinates and true facies from Excel sheets."""
        try:
            ORIGIN_E = 84337.0
            ORIGIN_N = 445750.0
            SPACING = 20.0

            tabs = ["DEL-GT-01", "DEL-GT-02-S2"]

            codes = self.cfg.get("codes", [1, 2, 3])
            if len(codes) < 3 and self.num_classes == 3:
                codes = [1, 4, 8]

            # Initialize the tracking list for well names
            self.well_names = []

            for tab in tabs:
                df = pd.read_excel(self.well_data_path, sheet_name=tab)
                df_32 = df.head(32)

                facies_col = "Facies " if "Facies " in df.columns else "Facies"

                for i, row in df_32.iterrows():
                    z = i
                    y = int(np.round((row["GRID N"] - ORIGIN_N) / SPACING))
                    x = int(np.round((row["GRID E"] - ORIGIN_E) / SPACING))

                    if 0 <= x < 128 and 0 <= y < 128:
                        raw_facies = row[facies_col]

                        if self.num_classes == 3:
                            if 1 <= raw_facies <= 3:
                                class_idx = int(raw_facies - 1)
                                mapped_facies = codes[class_idx]
                            else:
                                mapped_facies = codes[0]

                        elif self.num_classes == 9:
                            mapped_facies = (
                                raw_facies if 1 <= raw_facies <= 9 else 1
                            )
                        else:
                            mapped_facies = 1

                        # Keep this as 3 elements to protect existing code compatibility
                        self.well_coords.append((z, y, x))
                        self.well_true_facies.append(mapped_facies)
                        # Store the name in the parallel list
                        self.well_names.append(tab)

            print(f"Loaded {len(self.well_coords)} well conditioning points.")

        except Exception as e:
            print(f"Error loading well data: {e}")

    def _load_and_map_realization(self, file_path):
        """Loads a single 3D realization file and maps it to physical facies codes.

        Args:
            file_path (str or Path): Path to the realization file (.npy or .npz).

        Returns:
            np.ndarray: Evaluated grid mapped to physical facies codes.

        Raises:
            ValueError: If the grid dimension or structure is unsupported.
        """
        if file_path.endswith(".npz"):
            with np.load(file_path) as data:
                raw_arr = (
                    data["facies"] if "facies" in data else data[data.files[0]]
                )
        else:
            raw_arr = np.load(file_path)

        if raw_arr.ndim == 4:
            class_indices = np.argmax(raw_arr, axis=0)
        elif raw_arr.ndim == 3:
            class_indices = np.round(raw_arr).astype(int)
        else:
            raise ValueError(
                f"Unexpected array shape {raw_arr.shape} in {file_path}"
            )

        codes = self.cfg.get("codes", [1, 2, 3])
        if len(codes) < 3 and self.num_classes == 3:
            codes = [1, 4, 8]

        mapping = np.array(codes)
        class_indices = np.clip(class_indices, 0, len(mapping) - 1)
        return mapping[class_indices]

    def compute_mismatch(self):
        """Computes Macro and Per-Class F1 scores for all loaded realizations.

        Returns:
            pd.DataFrame: DataFrame containing F1 metrics for each evaluated realization.
        """
        if not self.well_coords:
            print("No well coordinates loaded. Cannot compute mismatch.")
            return None

        results = []
        expected_labels = self.cfg.get("codes", [1, 2, 3])
        if len(expected_labels) < 3 and self.num_classes == 3:
            expected_labels = [1, 4, 8]

        for file_path in self.data_files:
            try:
                grid = self._load_and_map_realization(file_path)

                y_pred = []
                y_true_valid = []

                for (z, y, x), true_facies in zip(
                    self.well_coords, self.well_true_facies
                ):
                    if (
                        0 <= z < grid.shape[0]
                        and 0 <= y < grid.shape[1]
                        and 0 <= x < grid.shape[2]
                    ):
                        y_pred.append(grid[z, y, x])
                        y_true_valid.append(true_facies)

                macro_f1 = f1_score(y_true_valid, y_pred, average="macro", zero_division=0)
                per_class_f1 = f1_score(y_true_valid, y_pred, average=None, labels=expected_labels, zero_division=0)

                results.append(
                    {
                        "Realization": os.path.basename(file_path),
                        "Macro_F1_Score": macro_f1,
                        "F1_Class_1_Channel": per_class_f1[0],
                        "F1_Class_2_Levee": per_class_f1[1],
                        "F1_Class_3_Overbank": per_class_f1[2],
                    }
                )

            except Exception as e:
                print(f"Error processing {file_path} for mismatch: {e}")

        df_results = pd.DataFrame(results)

        if not df_results.empty:
            mean_f1 = df_results["Macro_F1_Score"].mean()
            print(f"\n--- Well Data Mismatch ---")
            print(f"Evaluated {len(results)} realizations.")
            print(f"Average Macro F1 Score: {mean_f1:.4f}")
            print(f"Average F1 Class 1 (Channel): {df_results['F1_Class_1_Channel'].mean():.4f}")
            print(f"Average F1 Class 2 (Levee): {df_results['F1_Class_2_Levee'].mean():.4f}")
            print(f"Average F1 Class 3 (Overbank): {df_results['F1_Class_3_Overbank'].mean():.4f}")

        return df_results

    def plot_split_entropy(
        self,
        df_mismatch,
        axis="Z",
        num_slices=3,
        figsize=None,
        show_plot=True,
        save_plot=False,
        output_dir="outputs",
    ):
        """Plots split entropy maps comparing perfect and imperfect alignment subgroups.

        Args:
            df_mismatch (pd.DataFrame): DataFrame containing mismatch scores for realizations.
            axis (str, optional): Axis along which to slice the grid ('X', 'Y', or 'Z'). Defaults to "Z".
            num_slices (int, optional): Number of evenly spaced slices to plot. Defaults to 3.
            figsize (tuple, optional): Explicit figure dimensions. Defaults to None.
            show_plot (bool, optional): Whether to display the plot. Defaults to True.
            save_plot (bool, optional): Whether to save the plot image to disk. Defaults to False.
            output_dir (str, optional): Output directory path for saving plots. Defaults to "outputs".

        Returns:
            dict: Summary metrics including subgroup counts and mean entropy values.

        Raises:
            ValueError: If the specified axis is invalid.
        """
        if df_mismatch is None or df_mismatch.empty:
            print("Error: Provide a valid mismatch DataFrame.")
            return {}

        if not self.data_files:
            print("No realization files available.")
            return {}

        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            raise ValueError("axis must be 'X', 'Y', or 'Z'.")

        sample_grid = self._load_and_map_realization(self.data_files[0])
        nz, ny, nx = sample_grid.shape
        dims = {"Z": nz, "Y": ny, "X": nx}
        max_slices = dims[axis]

        all_slice_indices = list(range(max_slices))

        num_slices = min(num_slices, max_slices)
        if num_slices == 1:
            plot_slice_indices = [max_slices // 2]
        else:
            plot_slice_indices = np.linspace(0, max_slices - 1, num_slices, dtype=int).tolist()

        basename_to_path = {os.path.basename(fp): fp for fp in self.data_files}

        perfect_names = df_mismatch[df_mismatch["Macro_F1_Score"] == 1.0]["Realization"]
        imperfect_names = df_mismatch[df_mismatch["Macro_F1_Score"] < 1.0]["Realization"]

        perfect_paths = [basename_to_path[n] for n in perfect_names if n in basename_to_path]
        imperfect_paths = [basename_to_path[n] for n in imperfect_names if n in basename_to_path]

        print(f"\n--- Running Split Entropy Optimization Loop ({axis}-Axis) ---")
        print(f"  Perfect Alignment Subgroup: {len(perfect_paths)} files")
        print(f"  Imperfect Alignment Subgroup: {len(imperfect_paths)} files")

        perf_maps_all, perf_mean = self._extract_group_entropy(perfect_paths, axis, all_slice_indices, dims)
        imperf_maps_all, imperf_mean = self._extract_group_entropy(imperfect_paths, axis, all_slice_indices, dims)

        perf_maps = [perf_maps_all[i] for i in plot_slice_indices] if perf_maps_all is not None else None
        imperf_maps = [imperf_maps_all[i] for i in plot_slice_indices] if imperf_maps_all is not None else None

        if show_plot:
            figsize = figsize or (5 * num_slices, 8)
            fig, axes = plt.subplots(2, num_slices, figsize=figsize, sharex=True, sharey=True)
            axes = np.atleast_2d(axes)

            norm = mcolors.Normalize(vmin=0, vmax=1.0)
            xlabel, ylabel = {"Z": ("X", "Y"), "Y": ("X", "Z"), "X": ("Y", "Z")}[axis]

            for idx, slice_val in enumerate(plot_slice_indices):
                if perf_maps is not None:
                    im = axes[0, idx].imshow(perf_maps[idx], cmap="magma", origin="lower", norm=norm)
                    axes[0, idx].set_title(f"Perfect | {axis}-Slice {slice_val}", fontsize=10, c='#595959')
                else:
                    axes[0, idx].text(0.5, 0.5, "No Data", ha="center", va="center")

                if imperf_maps is not None:
                    im = axes[1, idx].imshow(imperf_maps[idx], cmap="magma", origin="lower", norm=norm)
                    axes[1, idx].set_title(f"Imperfect | {axis}-Slice {slice_val}", fontsize=10, c='#595959')
                else:
                    axes[1, idx].text(0.5, 0.5, "No Data", ha="center", va="center")

                axes[0, idx].set_ylabel(ylabel)
                axes[1, idx].set_ylabel(ylabel)
                axes[1, idx].set_xlabel(xlabel)

            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.89, 0.25, 0.015, 0.5])
            fig.colorbar(im, cax=cbar_ax).set_label("Normalized Entropy (0 to 1)", rotation=270, labelpad=15)

            plt.suptitle(f"Perfect $H_n$: {perf_mean:.4f} | Imperfect $H_n$: {imperf_mean:.4f}", fontsize=12, y=0.93, c='#595959')

            if save_plot:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, f"split_entropy_{axis}_{self.gan_name}.png")
                plt.savefig(path, bbox_inches="tight", dpi=400)
                print(f"Saved split entropy plot to: {path}")

            plt.show()

        return {"perfect_count": len(perfect_paths), "imperfect_count": len(imperfect_paths),
                "perfect_mean": perf_mean, "imperfect_mean": imperf_mean}

    def _extract_group_entropy(self, file_paths, axis, slice_indices, dims):
        """Extracts group cross-sections and calculates normalized spatial entropy.

        Args:
            file_paths (list): List of file paths to the realization files.
            axis (str): Cross-sectional axis ('X', 'Y', or 'Z').
            slice_indices (list): List of selected indices along the axis.
            dims (dict): Dictionary defining grid resolution dimensions.

        Returns:
            tuple: A list of 2D entropy maps and the average spatial entropy value.
        """
        num_files = len(file_paths)
        if num_files <= 1:
            return None, 0.0

        nz, ny, nx = dims["Z"], dims["Y"], dims["X"]
        h_max = np.log2(self.num_classes)

        if axis == "Z":
            shape = (num_files, len(slice_indices), ny, nx)
        elif axis == "Y":
            shape = (num_files, len(slice_indices), nz, nx)
        elif axis == "X":
            shape = (num_files, len(slice_indices), nz, ny)

        slices_stack = np.zeros(shape, dtype=np.uint8)

        for i, fp in enumerate(file_paths):
            grid = self._load_and_map_realization(fp)
            if axis == "Z":
                slices_stack[i] = grid[slice_indices, :, :]
            elif axis == "Y":
                slices_stack[i] = grid[:, slice_indices, :].swapaxes(0, 1)
            elif axis == "X":
                slices_stack[i] = grid[:, :, slice_indices].transpose(2, 0, 1)

        dim_y, dim_x = shape[2], shape[3]
        entropy_maps = []
        slice_means = []

        codes = self.cfg.get("codes", [1, 2, 3])
        if len(codes) < 3 and self.num_classes == 3:
            codes = [1, 4, 8]

        for idx in range(len(slice_indices)):
            probs = np.zeros((len(codes), dim_y, dim_x))
            for i, f_val in enumerate(codes):
                probs[i] = (
                    np.sum(slices_stack[:, idx, :, :] == f_val, axis=0)
                    / num_files
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entropy_map = entropy(probs, base=2, axis=0)
                if h_max > 0:
                    entropy_map = entropy_map / h_max

            entropy_maps.append(entropy_map)
            slice_means.append(np.mean(entropy_map))

        return entropy_maps, np.mean(slice_means)

    def plot_comparative_f1_distributions(self, f1_scores_results, validators_dict, save_path=None):
        """Plots histograms with uniform styling and a unified, bottom-centered legend."""
        metrics = {
            "Macro_F1_Score": "Macro F1 Score",
            "F1_Class_1_Channel": "F1 Score: Channel",
            "F1_Class_2_Levee": "F1 Score: Levee",
            "F1_Class_3_Overbank": "F1 Score: Overbank"
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 6), sharey=True)
        axes = axes.flatten()
        sns.set_theme(style="whitegrid")

        # Setup legend tracking
        legend_handles = []
        # Use a dictionary to keep track of colors per run for consistent patching
        run_colors = {}
        palette = sns.color_palette("deep", len(f1_scores_results))

        for idx, (col_name, display_name) in enumerate(metrics.items()):
            ax = axes[idx]
            
            for i, (run_name, df_f1) in enumerate(f1_scores_results.items()):
                if col_name in df_f1.columns:
                    data = df_f1[col_name]
                    color = palette[i]
                    run_colors[run_name] = color
                    
                    sns.histplot(
                        data=data,
                        element="step",
                        stat="count",
                        bins=20,
                        kde=False,
                        alpha=0.3,           # Match the 0.3 alpha from your other plot
                        color=color,
                        linewidth=0.5,
                        ax=ax,
                        label=run_name
                    )

            ax.set_xlim(0.0, 1.05)
            ax.set_xlabel(display_name, fontsize=9)
            ax.set_ylabel("Count" if idx % 2 == 0 else "", fontsize=9)
            ax.set_title(display_name, fontsize=10, pad=10)

        # Construct unified legend
        for run_name, color in run_colors.items():
            patch = mpatches.Patch(color=color, label=run_name)
            legend_handles.append(patch)

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(f1_scores_results),
            fontsize=9,
            frameon=True,
            framealpha=1.0,
            facecolor='white',
            edgecolor='#e0e0e0',
            fancybox=True,
            shadow=True,
            borderpad=0.8
        )

        plt.suptitle("Comparative F1 Distributions", fontsize=12, y=0.96)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96]) # Adjust rect to make room for legend
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(fname=save_path, dpi=400, bbox_inches="tight")
        plt.show()

    def plot_hybrid_well_profile(self, validators_dict, save_path=None):
        """Plots ensemble facies prediction probabilities (left 3 panels) and 
        overall matching accuracy (rightmost panel) across the Z axis per well.
        Includes horizontal reference markers bounding contiguous true facies blocks
        and seamless shading for true facies intervals to eliminate transition gaps.
        """
        metrics = {
            "Class_1_Channel": "Probability: Channel",
            "Class_2_Levee": "Probability: Levee",
            "Class_3_Overbank": "Probability: Overbank",
            "Match_Accuracy": "Overall Match Accuracy"
        }
        
        sample_validator = next(iter(validators_dict.values()))
        unique_wells = sorted(list(set(sample_validator.well_names)))
        
        # Dynamically sample grid depth instead of hardcoding to prevent indexing mismatches
        sample_grid = sample_validator._load_and_map_realization(sample_validator.data_files[0])
        depth_size = sample_grid.shape[0]
        
        z_indices = np.arange(depth_size)
        depth_values = -z_indices

        for well_name in unique_wells:
            fig, axes = plt.subplots(1, 4, figsize=(8.27, 2.5), sharey=True)
            axes = axes.flatten()
            sns.set_theme(style="whitegrid")

            plt.rc('xtick', labelsize=7)    # Change x-axis tick font size
            plt.rc('ytick', labelsize=7)    # Change y-axis tick font size

            legend_handles = []
            legend_labels = []

            # Extract the true facies vertical profile for this specific well
            true_facies_profile = np.full(depth_size, -1)
            for (w_z, w_y, w_x), t_facies, w_name in zip(sample_validator.well_coords, sample_validator.well_true_facies, sample_validator.well_names):
                if w_name == well_name and 0 <= w_z < depth_size:
                    true_facies_profile[w_z] = t_facies

            expected_labels = sample_validator.cfg.get("codes", [1, 4, 8])
            if len(expected_labels) < 3 and sample_validator.num_classes == 3:
                expected_labels = [1, 4, 8]
                
            blocks = []
            current_facies = -1
            start_z = -1
            
            for z in z_indices:
                t_facies = true_facies_profile[z]
                if t_facies != current_facies:
                    if current_facies != -1:
                        blocks.append((current_facies, start_z, z - 1))
                    current_facies = t_facies
                    start_z = z
                    
            if current_facies != -1:
                blocks.append((current_facies, start_z, z_indices[-1]))

            for run_name, validator in validators_dict.items():
                c1_prob_per_z = []
                c2_prob_per_z = []
                c3_prob_per_z = []
                match_accuracy_per_z = []

                grids = [validator._load_and_map_realization(fp) for fp in validator.data_files]

                for z in z_indices:
                    y_true_z = None
                    y_pred_z = []

                    for (w_z, w_y, w_x), true_facies, w_name in zip(validator.well_coords, validator.well_true_facies, validator.well_names):
                        if w_z == z and w_name == well_name:
                            y_true_z = true_facies
                            for grid in grids:
                                if 0 <= w_y < grid.shape[1] and 0 <= w_x < grid.shape[2]:
                                    y_pred_z.append(grid[z, w_y, w_x])

                    if y_pred_z and y_true_z is not None:
                        preds = np.array(y_pred_z)
                        c1_prob_per_z.append(np.mean(preds == expected_labels[0]))
                        c2_prob_per_z.append(np.mean(preds == expected_labels[1]))
                        c3_prob_per_z.append(np.mean(preds == expected_labels[2]))
                        match_accuracy_per_z.append(np.mean(preds == y_true_z))
                    else:
                        c1_prob_per_z.append(0.0)
                        c2_prob_per_z.append(0.0)
                        c3_prob_per_z.append(0.0)
                        match_accuracy_per_z.append(0.0)

                # Plot probabilities in the left 3 panels
                for idx, data_stream in enumerate([c1_prob_per_z, c2_prob_per_z, c3_prob_per_z]):
                    data_stream_np = np.array(data_stream)
                    l_prob, = axes[idx].plot(data_stream_np, depth_values, lw=0.5)
                    axes[idx].fill_betweenx(depth_values, 0, data_stream_np, color=l_prob.get_color(), alpha=0.25)
                    
                    # Highlight gap to 1.05 with the facies color using continuous block spans to prevent gaps
                    for t_facies, z_top, z_bottom in blocks:
                        if t_facies == expected_labels[idx]:
                            facies_color = validator.cfg['colors'][t_facies]
                            # Extend the slice by 1 index to meet the adjacent block seamlessly (capped at depth_size)
                            end_idx = min(z_bottom + 2, depth_size)
                            z_slice = slice(z_top, end_idx)
                            
                            axes[idx].fill_betweenx(
                                depth_values[z_slice], 
                                data_stream_np[z_slice], 
                                1.05, 
                                color=facies_color, 
                                alpha=0.3
                            )
                
                # Plot accuracy in the rightmost highlighted panel
                l_acc, = axes[3].plot(match_accuracy_per_z, depth_values, lw=0.75)
                axes[3].fill_betweenx(depth_values, 0, match_accuracy_per_z, color=l_acc.get_color(), alpha=0.35)

                if run_name not in legend_labels:
                    patch = mpatches.Patch(color=l_acc.get_color(), label=run_name)
                    legend_handles.append(patch)
                    legend_labels.append(run_name)

            # 2. Draw dotted reference lines and label text targets reusing our clean blocks
            for t_facies, z_top, z_bottom in blocks:
                if t_facies in expected_labels:
                    f_idx = expected_labels.index(t_facies)
                    if f_idx < 3: # Keep bounded to the first 3 panels
                        # Draw top bounding line
                        axes[f_idx].axhline(y=-z_top, color='black', linestyle=':', linewidth=0.5, zorder=10)
                        
                        # --- Dynamic Boundary Check ---
                        if -z_top + 1.2 > 0:
                            y_pos = -z_top - 1.4
                            v_align = 'center'
                        else:
                            y_pos = -z_top + 1.4
                            v_align = 'center'
                            
                        axes[f_idx].text(0.98, y_pos, f"FA{f_idx+1}", fontsize=6, fontweight='bold', va=v_align, ha='right', zorder=11)
                        
                        # Draw bottom bounding line at the true base of the cell block
                        if z_bottom != z_top:
                            axes[f_idx].axhline(y=-(z_bottom + 1), linestyle=':', color='black', linewidth=0.5, zorder=10)

            for idx, display_name in enumerate(metrics.values()):
                ax = axes[idx]
                ax.set_title(display_name, fontsize=8, pad=10)
                ax.set_ylim(-depth_size, 0) # Explicitly anchors -32 at the bottom and 0 at the top
                ax.set_xlim(0.0, 1.05)
                ax.set_xlabel("Probability" if idx < 3 else "Match Rate", fontsize=8)
                
                if idx == 0:
                    ax.set_ylabel("Depth (m)", fontsize=8) # Uniform tracking nomenclature
                
                # Soft grey background accent for the performance metrics summary panel
                if idx == 3:
                    ax.set_facecolor('#f7f7f7')

            print(f"Mean accuracy score of {well_name}: {np.round(np.mean(match_accuracy_per_z),2)}")

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(validators_dict),
                fontsize=7
            )

            plt.suptitle(f"Vertical Well Hybrid Profiles - Well: {well_name}", fontsize=8, y=0.90)
            plt.tight_layout(rect=[0, 0, 1, 1])
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                base, ext = os.path.splitext(save_path)
                individual_save_path = f"{base}_{well_name}{ext}"
                plt.savefig(fname=individual_save_path, dpi=400, bbox_inches="tight")
            plt.show()

    def plot_well_trajectory_slices(self, well_name=None, num_samples=1, 
                                    plot_title=True, plot_suptitle=True,
                                    plot_axtitle=True, figsize=None,
                                     slice_range=None,
                                    show_plot=True, save_plot=False, 
                                    output_dir="outputs", show_legend=True):
        """Visualizes X and Y 2D cross-sections across the exact trajectory of a well.
        
        Uses the upper well cell (minimum Z) as the reference (X, Y) location. Pointers 
        are added at the top and bottom of the plot to indicate well location without 
        obscuring the facies data with a full line.
        """
        if not self.well_coords:
            print("No well coordinates loaded. Cannot plot trajectory.")
            return

        # Default to the first available well if none is specified
        if well_name is None:
            if not self.well_names:
                return
            well_name = self.well_names[0]
            
        # --- Determine Marker Color Based on Well Type ---
        if 'DEL-GT-01' in well_name:
            pointer_color = 'red'   # Producer (Hot Water)
        elif 'DEL-GT-02' in well_name:
            pointer_color = 'blue'  # Injector (Cold Water)
        else:
            pointer_color = 'gray'  # Default fallback
            
        # 1. Isolate well points and find the upper well cell (minimum Z)
        well_points = [(z, y, x) for (z, y, x), w in zip(self.well_coords, self.well_names) if w == well_name]
        if not well_points:
            available_wells = list(set(self.well_names))
            print(f"Well '{well_name}' not found. Available wells: {available_wells}")
            return

        top_cell = min(well_points, key=lambda item: item[0])
        ref_z, ref_y, ref_x = top_cell
        
        # 2. Prepare data and styling
        plot_limit = min(num_samples, len(self.data_files))
        target_files = self.data_files[:plot_limit]
        
        colors = [self.cfg['colors'][c] for c in self.cfg['codes']]
        cmap = ListedColormap(colors)
        labels = [self.cfg['names'][c] for c in self.cfg['codes']]
        legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

        # Physical spacing
        dx, dy, dz = 20, 20, 1
        
        effective_figsize = figsize or (8.27*2, 3 * plot_limit)
        fig, axes = plt.subplots(plot_limit, 2, figsize=effective_figsize, squeeze=False)
        
        for idx, file_path in enumerate(target_files):
            data_3d = self._load_and_map_realization(file_path)
            
            # Remap data for proper coloring according to ListedColormap
            display_data = np.zeros_like(data_3d)
            for map_idx, code in enumerate(self.cfg['codes']):
                display_data[data_3d == code] = map_idx
                
            depth, height, width = display_data.shape
            filename = os.path.basename(file_path)
            
            # -- Implement slice_range filtering --
            if slice_range is not None and isinstance(slice_range, int):
                x_min = max(0, ref_x - slice_range)
                x_max = min(width, ref_x + slice_range + 1)
                y_min = max(0, ref_y - slice_range)
                y_max = min(height, ref_y + slice_range + 1)
            else:
                x_min, x_max = 0, width
                y_min, y_max = 0, height
            
            # --- Y-Slice (XZ Plane): Vertical View (Fixed at ref_y, cropped in X) ---
            extent_y = [x_min * dx, x_max * dx, -depth * dz, 0]
            ax_y = axes[idx, 0]
            ax_y.imshow(display_data[:, ref_y, x_min:x_max], cmap=cmap, origin='lower', 
                        vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_y, aspect='20') 
            if plot_title:
                ax_y.set_title(f"Y-Slice (Y={ref_y * dy}m) | {filename}", y=1.05)
            if plot_axtitle:
                ax_y.set_xlabel("X Distance (m)")
                ax_y.set_ylabel("Z Depth (m)")
            
            # Pointers for well location in XZ plane (using ref_x)
            well_x_pos = (ref_x + 0.5) * dx
            ax_y.plot(well_x_pos, 0, marker='v', color='k', markerfacecolor=pointer_color, 
                      markersize=12, clip_on=False, zorder=10) # Downward triangle at top (0m)
            ax_y.plot(well_x_pos, -depth * dz, marker='^', color='k', markerfacecolor=pointer_color, 
                      markersize=12, clip_on=False, zorder=10) # Upward triangle at bottom (-32m)

            # --- X-Slice (YZ Plane): Vertical View (Fixed at ref_x, cropped in Y) ---
            extent_x = [y_min * dy, y_max * dy, -depth * dz, 0]
            ax_x = axes[idx, 1]
            ax_x.imshow(display_data[:, y_min:y_max, ref_x], cmap=cmap, origin='lower', 
                        vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_x, aspect='20')
            if plot_title:
                ax_x.set_title(f"X-Slice (X={ref_x * dx}m)", y=1.05)
            if plot_axtitle:
                ax_x.set_xlabel("Y Distance (m)")
                ax_x.set_ylabel("Z Depth (m)")
            
            # Pointers for well location in YZ plane (using ref_y)
            well_y_pos = (ref_y + 0.5) * dy
            ax_x.plot(well_y_pos, 0, marker='v', color='k', markerfacecolor=pointer_color, 
                      markersize=12, clip_on=False, zorder=10) 
            ax_x.plot(well_y_pos, -depth * dz, marker='^', color='k', markerfacecolor=pointer_color, 
                      markersize=12, clip_on=False, zorder=10) 

        if plot_suptitle:
            fig.suptitle(f"Well: {well_name}", fontsize=12)

        if show_legend:
            fig.legend(handles=legend_patches, loc='lower center', ncol=len(colors), 
                       bbox_to_anchor=(0.5, -0.05), fontsize=10)
            
        plt.tight_layout()
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"well_trajectory_{well_name}_{plot_limit}samples.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=400)
            print(f"Saved well trajectory slices to: {plot_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


    def plot_well_trajectory_entropy_slices(self, well_name=None, figsize=None, plot_title=True, plot_suptitle=True, plot_axtitle=True,
                                             slice_range=None,
                                            show_plot=True, save_plot=False, output_dir="outputs", show_legend=True):
        """Visualizes X and Y 2D cross-sections of cell-wise normalized spatial entropy 
        across the exact trajectory of a well.
        """
        if not getattr(self, 'well_coords', None):
            print("No well coordinates loaded. Cannot plot trajectory.")
            return

        # 1. Isolate well points and find the upper well cell (minimum Z)
        if well_name is None:
            if not self.well_names:
                return
            well_name = self.well_names[0]
            
        # --- Determine Marker Color Based on Well Type ---
        if 'DEL-GT-01' in well_name:
            pointer_color = 'red'   # Producer (Hot Water)
        elif 'DEL-GT-02' in well_name:
            pointer_color = 'blue'  # Injector (Cold Water)
        else:
            pointer_color = 'gray'  # Default fallback
            
        well_points = [(z, y, x) for (z, y, x), w in zip(self.well_coords, self.well_names) if w == well_name]
        if not well_points:
            available_wells = list(set(self.well_names))
            print(f"Well '{well_name}' not found. Available wells: {available_wells}")
            return

        top_cell = min(well_points, key=lambda item: item[0])
        ref_z, ref_y, ref_x = top_cell
        
        # 2. Extract Data and Compute Slice-Specific Entropy (Memory Optimized)
        print(f"\n--- Generating Normalized Entropy Slices for Well: {well_name} ({self.gan_name}) ---")
        
        num_files = len(self.data_files)
        if num_files == 0:
            print("No realization files found.")
            return
            
        # Get dimensions
        sample_grid = self._load_and_map_realization(self.data_files[0])
        depth, height, width = sample_grid.shape
        
        # -- Implement slice_range filtering --
        if slice_range is not None and isinstance(slice_range, int):
            x_min = max(0, ref_x - slice_range)
            x_max = min(width, ref_x + slice_range + 1)
            y_min = max(0, ref_y - slice_range)
            y_max = min(height, ref_y + slice_range + 1)
        else:
            x_min, x_max = 0, width
            y_min, y_max = 0, height
        
        # Pre-allocate arrays for JUST the two cropped slices we need
        y_slice_stack = np.zeros((num_files, depth, x_max - x_min), dtype=sample_grid.dtype)
        x_slice_stack = np.zeros((num_files, depth, y_max - y_min), dtype=sample_grid.dtype)
        
        for i, fp in enumerate(self.data_files):
            grid = self._load_and_map_realization(fp)
            y_slice_stack[i] = grid[:, ref_y, x_min:x_max]
            x_slice_stack[i] = grid[:, y_min:y_max, ref_x]
            
        # Entropy Calculation Core
        codes = self.cfg.get("codes", [1, 2, 3])
        if len(codes) < 3 and getattr(self, 'num_classes', 3) == 3:
            codes = [1, 4, 8]
        h_max = np.log2(getattr(self, 'num_classes', 3))
        
        def calc_2d_entropy(stack):
            probs = np.zeros((len(codes), stack.shape[1], stack.shape[2]))
            for idx, f_val in enumerate(codes):
                probs[idx] = np.sum(stack == f_val, axis=0) / num_files
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from scipy.stats import entropy
                e_map = entropy(probs, base=2, axis=0)
                if h_max > 0:
                    e_map = e_map / h_max
            return np.nan_to_num(e_map)

        entropy_y_slice = calc_2d_entropy(y_slice_stack)
        entropy_x_slice = calc_2d_entropy(x_slice_stack)
        
        # 3. Prepare styling and physical spacing
        dx, dy, dz = 20, 20, 1
        norm = mcolors.Normalize(vmin=0, vmax=1.0)
        cmap = 'magma'
        
        effective_figsize = figsize or (8.27 * 2, 3) 
        fig, axes = plt.subplots(1, 2, figsize=effective_figsize, squeeze=False)
        
        # --- Y-Slice (XZ Plane): Vertical View (Fixed at ref_y, cropped in X) ---
        extent_y = [x_min * dx, x_max * dx, -depth * dz, 0]
        ax_y = axes[0, 0]
        im_y = ax_y.imshow(entropy_y_slice, cmap=cmap, origin='lower', 
                           norm=norm, extent=extent_y, aspect='20') 
        if plot_title:
            ax_y.set_title(f"Y-Slice (Y={ref_y * dy}m) | Entropy", y=1.05)
        if plot_axtitle:
            ax_y.set_xlabel("X Distance (m)")
            ax_y.set_ylabel("Z Depth (m)")
        
        # Pointers for well location in XZ plane (using ref_x)
        well_x_pos = (ref_x + 0.5) * dx
        ax_y.plot(well_x_pos, 0, marker='v', color='k', markerfacecolor=pointer_color, 
                  markersize=12, clip_on=False, zorder=10)
        ax_y.plot(well_x_pos, -depth * dz, marker='^', color='k', markerfacecolor=pointer_color, 
                  markersize=12, clip_on=False, zorder=10)

        # --- X-Slice (YZ Plane): Vertical View (Fixed at ref_x, cropped in Y) ---
        extent_x = [y_min * dy, y_max * dy, -depth * dz, 0]
        ax_x = axes[0, 1]
        im_x = ax_x.imshow(entropy_x_slice, cmap=cmap, origin='lower', 
                           norm=norm, extent=extent_x, aspect='20')
        if plot_title:
            ax_x.set_title(f"X-Slice (X={ref_x * dx}m) | Entropy", y=1.05)
        if plot_axtitle:
            ax_x.set_xlabel("Y Distance (m)")
            ax_x.set_ylabel("Z Depth (m)")
        
        # Pointers for well location in YZ plane (using ref_y)
        well_y_pos = (ref_y + 0.5) * dy
        ax_x.plot(well_y_pos, 0, marker='v', color='k', markerfacecolor=pointer_color, 
                  markersize=12, clip_on=False, zorder=10) 
        ax_x.plot(well_y_pos, -depth * dz, marker='^', color='k', markerfacecolor=pointer_color, 
                  markersize=12, clip_on=False, zorder=10) 
        
        # 4. Global configurations and Colorbar
        if plot_suptitle:
            fig.suptitle(f"Well Trajectory Entropy Profiles: {well_name} ({self.gan_name})", fontsize=12, y=1.02)

        plt.tight_layout()
        
        fig.subplots_adjust(right=0.92)
        if show_legend:
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
            fig.colorbar(im_x, cax=cbar_ax).set_label('Absolute Mapped Bounded Entropy (0 to 1)', rotation=270, labelpad=15)
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"well_entropy_{well_name}_{self.gan_name.replace(' ', '_')}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=600)
            print(f"Saved well entropy slices to: {plot_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

def plot_dir_history(data_dir, ax=None, title_suffix=""):
    """Loads the first CSV file from a directory and plots its loss curves.

    Args:
        data_dir (str or Path): Path to the directory containing the training CSV logs.
        ax (matplotlib.axes.Axes, optional): Pre-existing matplotlib axis object. Defaults to None.
        title_suffix (str, optional): Text suffix appended to the plot title. Defaults to "".
    """
    path = Path(data_dir)

    try:
        history_files = list(path.glob("*.csv"))
        if not history_files:
            print(f"Warning: No CSV files found in {path}")
            return

        Loss = pd.read_csv(history_files[0])
        step = np.arange(len(Loss))

        if ax is None:
            ax = plt.gca()

        ax.grid(True)
        for col in Loss.columns:
            ax.plot(step, Loss[col], label=col)

        ax.set_title(f"Loss Curves {title_suffix}".strip())
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

        mean_min_loss = Loss.min().mean()
        print(f"[{path.name}] Mean of minimum losses across columns: {mean_min_loss:.4f}")

    except Exception as e:
        print(f"Error accessing history directory: {path}")
        print(e)

def compute_pixelwise_entropy(file_paths):
    """Computes pixel-wise Shannon entropy and mean normalized entropy.

    Args:
        file_paths (list): List of file paths pointing to categorical numpy arrays.

    Returns:
        tuple: A tuple containing the normalized entropy map (np.ndarray) and the 
            mean normalized entropy value (float).
    """
    if not file_paths:
        return None, 0.0

    arrays = [np.load(fp) for fp in file_paths]
    stacked = np.stack(arrays, axis=0)

    unique_classes = np.unique(stacked)
    num_classes = len(unique_classes)

    if num_classes <= 1 or len(file_paths) <= 1:
        return np.zeros(stacked.shape[1:]), 0.0

    entropy_map = np.zeros(stacked.shape[1:])

    for c in unique_classes:
        p_c = np.mean(stacked == c, axis=0)
        log_p_c = np.zeros_like(p_c)
        mask = p_c > 0
        log_p_c[mask] = np.log2(p_c[mask])
        entropy_map -= p_c * log_p_c

    norm_entropy_map = entropy_map / np.log2(num_classes)
    mean_norm_entropy = np.mean(norm_entropy_map)

    return norm_entropy_map, mean_norm_entropy