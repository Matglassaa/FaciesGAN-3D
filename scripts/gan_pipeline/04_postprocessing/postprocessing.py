import os
import sys
import glob
import math
import random
import pathlib
import warnings
from pathlib import Path

# Mathematical libraries
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.ndimage import label
from scipy.spatial.distance import jensenshannon

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ML libraries
import torch
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import MDS
from voxgan.models.metrics import MSSWD


# Miscellaneous
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from skimage.util import view_as_windows
from collections import Counter, defaultdict

# Adjust path based on your environment
scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from gan_pipeline.core.custom_plots import apply_custom_plotting_flavor, FaciesColorMap

# Apply custom plotting style globally
apply_custom_plotting_flavor()

# --- Centralized configuration ---
def get_facies_config(num_classes=3):
    """Centralized configuration for facies properties to prevent hardcoding."""
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
            # Maps 0-indexed categorical outputs to physical codes
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
                # E.g., 'crevasse_splay_core' -> 'Crevasse Splay Core'
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


import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


class PostProcessing:
    """Handles spatial and statistical validation metrics for GAN-generated geological facies.

    This class loads 3D arrays of geological facies (both Flumy samples and GAN-generated),
    computes spatial statistics (MPS, connectivity, normalized entropy, JSD), 
    and generates visual diagnostic plots including 3D renders.

    Attributes:
        flumy_name (str): Display name for the Flumy baseline dataset in plots.
        gan_name (str): Display name for the generated dataset in plots.
        output_dir (str): Directory where generated plots and CSVs will be saved.
        data_files (list): Sorted list of file paths for the Flumy dataset.
        gan_files (list): Sorted list of file paths for the GAN-generated dataset.
        flumy_samples (list): Loaded Flumy data arrays in memory.
        gan_data (list): Loaded GAN data arrays in memory.
        cfg (dict): Centralized facies configuration from get_facies_config().
        facies_mapping (dict): Mapping from categorical indices to specific geological codes.
    """
    
    def __init__(self, flumy_name, gan_name, gan_description, output_dir, data_path, gan_path, num_classes=3):
        """Initializes the class and locates files, but defers loading to save memory.

        Args:
            flumy_name (str): Display name for the Flumy baseline dataset in plots (e.g., 'Flumy Data').
            gan_name (str): Display name for the generated dataset in plots.
            output_dir (str): Directory to save outputs (e.g., 'outputs/metrics').
            data_path (str): Glob pattern matching Flumy data files (e.g., 'datasets/*.npy').
            gan_path (str): Glob pattern matching GAN data files (e.g., 'outputs/*.npy').
            num_classes (int, optional): Number of facies classes. Defaults to 3.
        """ 
        self.flumy_name = flumy_name
        self.gan_name = gan_name
        self.gan_desc = gan_description
        self.output_dir = output_dir
        self.num_classes = num_classes
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        self.data_files = sorted(glob.glob(str(data_path)))
        self.gan_files = sorted(glob.glob(str(gan_path)))
        
        self.flumy_samples = []
        self.gan_data = []
        
        self.cfg = get_facies_config(self.num_classes)
        self.facies_mapping = self.cfg['mapping']

    
    def load_flumy_samples(self, limit=100):
        """Loads Flumy data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading {self.flumy_name} samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.data_files[:limit] if limit else self.data_files
        self.flumy_samples = [self._load_data(f) for f in files_to_load]
        print(f"Loaded {len(self.flumy_samples)} {self.flumy_name} samples into memory.")

    def load_gan_samples(self, limit=100):
        """Loads GAN data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading {self.gan_name} samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.gan_files[:limit] if limit else self.gan_files
        self.gan_data = [self._load_gan(f) for f in files_to_load]
        print(f"Loaded {len(self.gan_data)} {self.gan_name} samples into memory.")

    def _load_data(self, file):
        """Loads a Flumy facies array and maps labels to geological codes.

        Args:
            file (str): File path to a saved .npy numpy array.

        Returns:
            numpy.ndarray: The 3D array of mapped geological facies.
        """
        data = np.load(file)
        if self.num_classes == 3:
            mapped_data = np.zeros_like(data)
            mapped_data[(data >= 0) & (data < 4)] = 1
            mapped_data[(data >= 4) & (data < 8)] = 4
            mapped_data[(data >= 8)] = 8
            return mapped_data
        elif self.num_classes == 9:
            # Map values correctly. Usually Flumy returns classes 1-9.
            return data
        return data
    
    def _load_gan(self, file):
        """Loads GAN output and standardizes it to a 3D categorical array.

        Args:
            file (str): File path to a saved .npz or .npy array.

        Returns:
            numpy.ndarray: The 3D array of mapped geological facies.
            
        Raises:
            ValueError: If the input array does not conform to 3D or 4D expectations.
        """
        if file.endswith('.npz'):
            with np.load(file) as data:
                raw_arr = data['facies'] if 'facies' in data else data[data.files[0]]
        else:
            raw_arr = np.load(file)

        if raw_arr.ndim == 4:
            class_indices = np.argmax(raw_arr, axis=0) 
        elif raw_arr.ndim == 3:
            if raw_arr.min() < 0 or raw_arr.dtype in [np.float32, np.float64]:
                if self.num_classes == 3:
                    class_indices = np.round(raw_arr + 1).astype(int)
                else:
                    class_indices = np.round(raw_arr).astype(int)
                class_indices = np.clip(class_indices, 0, self.num_classes - 1)
            else:
                class_indices = raw_arr.astype(int)
        else:
            raise ValueError(f"Unexpected array shape {raw_arr.shape} in file {file}")

        return self._map_indices_to_facies(class_indices)

    def _map_indices_to_facies(self, class_indices):
        """Maps 0-indexed categorical data to specific geological facies codes.

        Args:
            class_indices (numpy.ndarray): Array containing 0-indexed categorical values.

        Returns:
            numpy.ndarray: Mapped array containing physical facies codes.
        """
        mapping = np.array(self.cfg['codes'])
        class_indices = np.clip(class_indices, 0, len(mapping) - 1)
        return mapping[class_indices]

    def _get_global_proportions(self, data_list, target_facies=None):
        """Calculates the global probability of each facies across the entire dataset.

        Args:
            data_list (list): List of 3D numpy arrays containing geological facies.
            target_facies (list, optional): List of facies codes to calculate proportions for.
                Defaults to None, which uses self.cfg['codes'].

        Returns:
            list: Float probabilities of each target facies, summing to 1.0.
        """
        if target_facies is None:
            target_facies = self.cfg['codes']
        if not data_list: 
            return [0.0] * len(target_facies)
            
        counts = {f: 0 for f in target_facies}
        total_voxels = sum(arr.size for arr in data_list)
        
        for arr in data_list:
            for f_val in target_facies:
                counts[f_val] += np.sum(arr == f_val)
                
        return [counts[f] / total_voxels for f in target_facies]

    def _calculate_h_max(self, proportions=None):
        """Calculates the absolute theoretical maximum Shannon entropy for the categorical system.

        This baseline changes the reference from sample distribution uncertainty to the absolute
        upper bound defined by the number of classes, ensuring rigorous bounded normalization.

        Args:
            proportions (list, optional): Deprecated parameter kept for signature compatibility.
                Defaults to None.

        Returns:
            float: Theoretical maximum entropy in bits (log2 of the number of classes).
        """
        return float(np.log2(self.num_classes))

    def _create_colormap_and_legend(self):
        """Creates a ListedColormap and legend patches from the centralized facies config.

        Returns:
            tuple: (ListedColormap, list of matplotlib Patch objects for legends).
        """
        color_list = [self.cfg['colors'][c] for c in self.cfg['codes']]
        custom_cmap = ListedColormap(color_list)
        legend_patches = [
            mpatches.Patch(color=self.cfg['colors'][c], label=self.cfg['names'][c]) 
            for c in self.cfg['codes']
        ]
        return custom_cmap, legend_patches

    def get_pattern_counts(self, data_array, template_size=(3, 3, 3)):
        """Extracts and counts Multiple Point Statistics (MPS) patterns via sliding window.

        Args:
            data_array (numpy.ndarray): 3D geological grid sample.
            template_size (tuple, optional): Spatial dimensions of the window. Defaults to (3, 3, 3).

        Returns:
            tuple: (numpy.ndarray of unique patterns, numpy.ndarray of corresponding patch counts).
        """
        windows = view_as_windows(data_array, template_size)
        n_patches = np.prod(windows.shape[:3])
        patch_size = np.prod(template_size)
        flat_windows = windows.reshape(n_patches, patch_size)
        return np.unique(flat_windows, axis=0, return_counts=True)

    def analyze_connectivity(self, data_array, target_facies):
        """Calculates 3D structural connectivity sizes for a specific facies.

        Args:
            data_array (numpy.ndarray): 3D volume grid to label.
            target_facies (int): Voxel code specifying the depositional body to isolate.

        Returns:
            list: Integers representing the size in voxels of every isolated 3D cluster body.
        """
        binary_mask = (data_array == target_facies).astype(int)
        structure = np.ones((3, 3, 3)) 
        labeled_array, _ = label(binary_mask, structure=structure)
        unique_ids, blob_sizes = np.unique(labeled_array, return_counts=True)
        return blob_sizes[1:].tolist()

    def _plot_pattern_distributions(self, facies_list, flumy_patterns, gan_patterns):
        """Generates 2D scatter plots comparing pattern frequencies per facies.

        Args:
            facies_list (list): Active geological facies codes to process.
            flumy_patterns (Counter): Cached counts of reference baseline structural patterns.
            gan_patterns (Counter): Cached counts of generated model structural patterns.
        """
        print("\n--- Generating 2D Pattern Distribution Plots ---")
        
        for f_val in facies_list:
            f_flumy = {p: c for p, c in flumy_patterns.items() if p[len(p)//2] == f_val}
            f_gan = {p: c for p, c in gan_patterns.items() if p[len(p)//2] == f_val}
            
            total_flumy = sum(f_flumy.values()) if f_flumy else 1
            total_gan = sum(f_gan.values()) if f_gan else 1
            
            all_patterns = set(f_flumy.keys()).union(set(f_gan.keys()))
            
            flumy_freqs = []
            gan_freqs = []
            for p in all_patterns:
                flumy_freqs.append(f_flumy.get(p, 0) / total_flumy)
                gan_freqs.append(f_gan.get(p, 0) / total_gan)
                
            plt.figure(figsize=(6, 5))
            plt.scatter(flumy_freqs, gan_freqs, alpha=0.6, color='teal', edgecolor='k', s=25)
            
            max_val = max(max(flumy_freqs, default=0), max(gan_freqs, default=0))
            plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Alignment')
            
            plt.title(f"MPS Pattern Frequency Comparison - Facies {f_val}")
            plt.xlabel(f"{self.flumy_name} Relative Frequency")
            plt.ylabel(f"{self.gan_name} Relative Frequency")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _plot_connectivity_distributions(self, facies_list, all_flumy_blobs, all_gan_blobs):
        """Plots the connected component size distributions for each facies class in subplots."""
        blob_data = []
        for f_val in facies_list:
            facies_name = self.cfg['names'].get(f_val, f"Facies {f_val}")
            for size in all_flumy_blobs.get(f_val, []):
                blob_data.append({'Facies': facies_name, 'Blob_Size': size, 'Dataset': self.flumy_name})
            for size in all_gan_blobs.get(f_val, []):
                blob_data.append({'Facies': facies_name, 'Blob_Size': size, 'Dataset': self.gan_name})
        
        df_blobs = pd.DataFrame(blob_data)
        if df_blobs.empty:
            return

        n_facies = len(facies_list)
        ncols = min(n_facies, 3)
        nrows = math.ceil(n_facies / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_list = axes.flatten()
        
        for idx, f_val in enumerate(facies_list):
            ax = axes_list[idx]
            facies_name = self.cfg['names'].get(f_val, f"Facies {f_val}")
            sub_df = df_blobs[df_blobs['Facies'] == facies_name]
            
            if sub_df.empty:
                ax.set_title(facies_name)
                continue
                
            sns.histplot(
                data=sub_df,
                x='Blob_Size',
                hue='Dataset',
                kde=True,
                element='step',
                stat='density',
                common_norm=False,
                alpha=0.4,
                ax=ax,
                log_scale=True
            )
            ax.set_title(f"Connected Body Sizes: {facies_name}")
            ax.set_xlabel("Blob Size (voxels)")
            ax.set_ylabel("Density")
            
        for i in range(n_facies, len(axes_list)):
            axes_list[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()

    def connectivity_and_pattern_analysis(self, facies_list=None, sample_limit=10, recompute_flumy=False, plot=True):
        """Executes MPS and Connectivity comparisons for ALL facies using loaded data.

        Args:
            facies_list (list, optional): Mapped facies array filters. Defaults to None.
            sample_limit (int, optional): Evaluation count limits for RAM protection. Defaults to 10.
            recompute_flumy (bool, optional): Force cache bypass. Defaults to False.
            plot (bool, optional): Enable chart renderings. Defaults to True.

        Returns:
            tuple: (pandas.DataFrame of metrics summary, dict containing raw structural outputs).
        """
        if facies_list is None:
            facies_list = self.cfg['codes']

        limit_str = str(sample_limit) if sample_limit is not None else "ALL"
        
        baseline_csv_path = os.path.join(self.output_dir, f"baseline_connectivity_{limit_str}_samples.csv")
        baseline_pkl_path = os.path.join(self.output_dir, f"baseline_raw_data_{limit_str}_samples.pkl")
        gan_csv_path = os.path.join(self.output_dir, f"connectivity_comparison_{limit_str}_samples.csv")

        flumy_stats_dict = {}
        flumy_pattern_counter = Counter()
        all_flumy_blobs = defaultdict(list)

        if not recompute_flumy and os.path.exists(baseline_csv_path) and os.path.exists(baseline_pkl_path):
            print(f"\n--- Loading pre-computed {self.flumy_name} metrics from cache ---")
            df_baseline = pd.read_csv(baseline_csv_path)
            
            with open(baseline_pkl_path, 'rb') as f:
                cache_data = pickle.load(f)
                all_flumy_blobs = cache_data['blobs']
                flumy_pattern_counter = cache_data['patterns']
                
            for _, row in df_baseline.iterrows():
                flumy_stats_dict[row['Facies']] = {
                    'Flumy_Max_Blob': row['Flumy_Max_Blob'],
                    'Flumy_Median': row['Flumy_Median'],
                    'Flumy_Mean': row['Flumy_Mean']
                }
            total_flumy_patterns = len(flumy_pattern_counter)
        else:
            print(f"\n--- Computing {self.flumy_name} metrics for {limit_str} samples ---")
            if getattr(self, 'flumy_samples', None) is None or len(self.flumy_samples) == 0:
                self.load_flumy_samples(limit=sample_limit)
            
            flumy_to_process = self.flumy_samples[:sample_limit] if sample_limit else self.flumy_samples
            
            for data in flumy_to_process:
                patterns, counts = self.get_pattern_counts(data)
                for p, c in zip(patterns, counts):
                    flumy_pattern_counter[tuple(p)] += c 
                    
                for f_val in facies_list:
                    all_flumy_blobs[f_val].extend(self.analyze_connectivity(data, f_val))
                    
            total_flumy_patterns = len(flumy_pattern_counter)
            
            baseline_rows = []
            for f_val in facies_list:
                d_blobs = all_flumy_blobs[f_val]
                stats = {
                    'Flumy_Max_Blob': max(d_blobs) if d_blobs else 0,
                    'Flumy_Median': np.median(d_blobs) if d_blobs else 0,
                    'Flumy_Mean': np.mean(d_blobs) if d_blobs else 0
                }
                flumy_stats_dict[f_val] = stats
                
                row = {'Facies': f_val}
                row.update(stats)
                baseline_rows.append(row)
                
            pd.DataFrame(baseline_rows).to_csv(baseline_csv_path, index=False)
            with open(baseline_pkl_path, 'wb') as f:
                pickle.dump({'blobs': all_flumy_blobs, 'patterns': flumy_pattern_counter}, f)

        print(f"\n--- Computing {self.gan_name} metrics for {limit_str} samples ---")
        if getattr(self, 'gan_data', None) is None or len(self.gan_data) == 0:
            self.load_gan_samples(limit=sample_limit)
            
        gan_to_process = self.gan_data[:sample_limit] if sample_limit else self.gan_data
        gan_pattern_counter = Counter()
        all_gan_blobs = defaultdict(list)

        for data in gan_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                gan_pattern_counter[tuple(p)] += c
                
            for f_val in facies_list:
                all_gan_blobs[f_val].extend(self.analyze_connectivity(data, f_val))

        stats_data = []
        for f_val in facies_list:
            g_blobs = all_gan_blobs[f_val]
            d_stats = flumy_stats_dict.get(f_val, {'Flumy_Max_Blob': 0, 'Flumy_Median': 0, 'Flumy_Mean': 0})
            
            stats_data.append({
                'Facies': f_val,
                'Flumy_Max_Blob': d_stats['Flumy_Max_Blob'],
                'GAN_Max_Blob': max(g_blobs) if g_blobs else 0,
                'Flumy_Median': d_stats['Flumy_Median'],
                'GAN_Median': np.median(g_blobs) if g_blobs else 0,
                'Flumy_Mean': d_stats['Flumy_Mean'],
                'GAN_Mean': np.mean(g_blobs) if g_blobs else 0
            })
            
        df_stats = pd.DataFrame(stats_data)
        df_stats['Flumy_Mean'] = df_stats['Flumy_Mean'].round(2)
        df_stats['GAN_Mean'] = df_stats['GAN_Mean'].round(2)
        df_stats.to_csv(gan_csv_path, index=False)

        print("\n" + "="*50 + "\n MULTIPLE POINT STATISTICS (MPS)\n" + "="*50)
        print(f"Total unique patterns ({self.flumy_name}): {total_flumy_patterns}")
        print(f"Total unique patterns ({self.gan_name}): {len(gan_pattern_counter)}")

        if plot:
            #self._plot_pattern_distributions(facies_list, flumy_pattern_counter, gan_pattern_counter)
            self._plot_connectivity_distributions(facies_list, all_flumy_blobs, all_gan_blobs)

        raw_run_data = {
            'flumy_blobs': dict(all_flumy_blobs),
            'gan_blobs': dict(all_gan_blobs),
            'flumy_patterns': flumy_pattern_counter,
            'gan_patterns': gan_pattern_counter
        }

        return df_stats, raw_run_data

    def plot_facies_percentages(self, mode='gan', figsize=None, show_plot=True, save_plot=False):
        """Plots volume percentages of specific facies globally across the dataset.

        Args:
            mode (str, optional): Target to evaluate ('gan', 'flumy', or 'both'). Defaults to 'gan'.
            figsize (tuple, optional): Figure size dimensions. Defaults to (10, 6).
            show_plot (bool, optional): Render execution outputs. Defaults to True.
            save_plot (bool, optional): Save image files. Defaults to False.
        """
        valid_modes = ['gan', 'flumy', 'both']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        if mode in ['gan', 'both'] and not self.gan_data:
            self.load_gan_samples()
        if mode in ['flumy', 'both'] and not self.flumy_samples:
            self.load_flumy_samples()
            
        print(f"\n--- Generating Facies Distribution (Mode: {mode.upper()}) ---")

        figsize = figsize or (10, 6)
        target_facies = self.cfg['codes']
        labels = [self.cfg['names'][c] for c in self.cfg['codes']]
        colors = [self.cfg['colors'][c] for c in self.cfg['codes']]
        
        def calculate_percentages(data_list):
            probs = self._get_global_proportions(data_list, target_facies)
            return [p * 100 for p in probs]

        fig, ax = plt.subplots(figsize=figsize)

        if mode == 'both':
            flumy_percentages = calculate_percentages(self.flumy_samples)
            gan_percentages = calculate_percentages(self.gan_data)
            
            x = np.arange(len(labels))
            width = 0.35 
            
            bars_flumy = ax.bar(x - width/2, flumy_percentages, width, color=colors, edgecolor='black', alpha=0.5)
            bars_gan = ax.bar(x + width/2, gan_percentages, width, color=colors, edgecolor='black', hatch='//')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
            
            for bar in bars_flumy:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', rotation=45, va='bottom', fontsize=10, color='#555555')
            for bar in bars_gan:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', rotation=45, va='bottom', fontsize=10, fontweight='bold')
                
            legend_elements = [
                mpatches.Patch(facecolor='gray', alpha=0.5, edgecolor='black', label=f'{self.flumy_name} ({len(self.flumy_samples)} samples)'),
                mpatches.Patch(facecolor='gray', hatch='//', edgecolor='black', label=f'{self.gan_name} ({len(self.gan_data)} samples)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
            ax.set_title(f'Facies Volume Distribution: {self.flumy_name} vs. {self.gan_name}', fontsize=14, pad=15)
            
        else:
            target_data = self.gan_data if mode == 'gan' else self.flumy_samples
            percentages = calculate_percentages(target_data)
            
            bars = ax.bar(labels, percentages, color=colors, edgecolor='black', linewidth=1.2)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
            display_name = self.gan_name if mode == 'gan' else self.flumy_name
            ax.set_title(f'{display_name} Facies Volume Distribution ({len(target_data)} Samples)', fontsize=14, pad=15)

        ax.set_ylabel('Volume Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100) 
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plot:
            filename = f"facies_distribution_{mode}.png"
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, bbox_inches='tight', dpi=400)
            print(f"saved {mode} facies percentages plot to: \n{plot_path}")
            
        if show_plot: 
            plt.show()
        else: 
            plt.close(fig)

    def _compute_ensemble_entropy_map(self, data_list):
        """Computes voxel-wise Shannon entropy across an ensemble, normalized by absolute class limits.

        Ensures values are rigidly bounded to a [0.0, 1.0] scale by dividing by log2(K).

        Args:
            data_list (list): List of 3D numpy arrays containing geological data arrays.

        Returns:
            numpy.ndarray: 3D spatial array mapping coordinate uncertainty from 0.0 to 1.0.
        """
        num_realizations = len(data_list)
        if num_realizations == 0:
            return None
            
        nz, ny, nx = data_list[0].shape
        facies_values = self.cfg['codes']
        
        probs = np.zeros((len(facies_values), nz, ny, nx))
        ensemble_stack = np.stack(data_list, axis=0)
        
        for i, f_val in enumerate(facies_values):
            probs[i] = np.sum(ensemble_stack == f_val, axis=0) / num_realizations

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_entropy = entropy(probs, base=2, axis=0)
            
        h_max = np.log2(self.num_classes)
        return raw_entropy / h_max
    
    def get_baseline_entropy_metrics(self):
        """Extracts the spatial baseline entropy matrix and global mean directly from loaded reference arrays.

        Returns:
            tuple: (3D numpy.ndarray of normalized testing baseline entropy, float mean scalar value).
        """
        if not self.flumy_samples:
            self.load_flumy_samples(limit=None)
            
        entropy_map = self._compute_ensemble_entropy_map(self.flumy_samples)
        mean_entropy = np.mean(entropy_map) if entropy_map is not None else 0.0
        return entropy_map, mean_entropy
    
    def plot_entropy(self, data_source='gan', axis=None, num_slices=9, figsize=None, plot_title=True, show_plot=True, save_plot=False):
        """Calculates and plots cell-wise normalized spatial entropy profiles sliced across a targeted axis.

        Args:
            data_source (str, optional): Target to evaluate ('gan' or 'flumy'). Defaults to 'gan'.
            axis (str, optional): Cross-section slicing filter ('X', 'Y', 'Z'). Defaults to None.
            num_slices (int, optional): Number of visual layouts profiles. Defaults to 9.
            figsize (tuple, optional): Dimensions for subplots. Defaults to None.
            plot_title (bool, optional): Enable suptitles. Defaults to True.
            show_plot (bool, optional): Interactive plotting. Defaults to True.
            save_plot (bool, optional): Disk saving options. Defaults to False.
        """
        valid_sources = ['gan', 'flumy']
        if data_source not in valid_sources:
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
            
        if axis and axis.upper() not in ['X', 'Y', 'Z']:
            raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
        
        axis = axis.upper() if axis else None
        
        # Check if target data list is empty and load samples if necessary
        if data_source == 'gan':
            if not getattr(self, 'gan_data', None) or len(self.gan_data) == 0:
                self.load_gan_samples()
            target_data = self.gan_data
        else:
            if not getattr(self, 'flumy_samples', None) or len(self.flumy_samples) == 0:
                self.load_flumy_samples(limit=None)  # Load all available files for a robust baseline
            target_data = self.flumy_samples
            
        axes_to_plot = [axis] if axis else ['Z', 'Y', 'X']

        h_max = np.log2(self.num_classes)
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        print(f"\n--- Generating {display_name} Normalized Entropy Matrices ---")
        print(f"Theoretical Absolute Max Upper Bound (H_max): {h_max:.4f} bits")

        entropy_tensor = self._compute_ensemble_entropy_map(target_data)
        nz, ny, nx = entropy_tensor.shape

        dims = {'Z': nz, 'Y': ny, 'X': nx}
        slices_dict = {ax: sorted(random.sample(range(dims[ax]), num_slices)) for ax in axes_to_plot}

        if 'Z' in axes_to_plot: 
            self._plot_precomputed_entropy_helper(entropy_tensor, slices_dict['Z'], 'Z', 'X', 'Y', data_source, figsize, plot_title, show_plot, save_plot)
        if 'Y' in axes_to_plot: 
            self._plot_precomputed_entropy_helper(entropy_tensor, slices_dict['Y'], 'Y', 'X', 'Z', data_source, figsize, plot_title, show_plot, save_plot)
        if 'X' in axes_to_plot: 
            self._plot_precomputed_entropy_helper(entropy_tensor, slices_dict['X'], 'X', 'Y', 'Z', data_source, figsize, plot_title, show_plot, save_plot)

    def _plot_precomputed_entropy_helper(self, entropy_tensor, slice_indices, axis_name, xlabel, ylabel, data_source, figsize, plot_title, show_plot, save_plot):
        """Internal plotting module mapping array cross-sections cleanly to a fixed 0.0 - 1.0 limit.

        Args:
            entropy_tensor (numpy.ndarray): Fully pre-normalized continuous 3D spatial field.
            slice_indices (list): Index grid selections.
            axis_name (str): Slicing coordinate plane label.
            xlabel (str): Horizontal plot text.
            ylabel (str): Vertical plot text.
            data_source (str): Target group identifier.
            figsize (tuple): Dimensions configurations.
            plot_title (bool): Context header labels configurations.
            show_plot (bool): Visualization rendering options.
            save_plot (bool): Disk storage saving configurations.
        """
        n_slices = len(slice_indices)
        norm = mcolors.Normalize(vmin=0, vmax=1.0) 

        if figsize:
            ncols = min(n_slices, 4)
            nrows = math.ceil(n_slices / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        elif n_slices == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        elif n_slices <= 3:
            fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
        else:
            ncols = min(n_slices, 4)
            nrows = math.ceil(n_slices / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        axes_list = np.atleast_1d(axes).flatten()

        for idx, slice_val in enumerate(slice_indices):
            if axis_name == 'Z':
                slice_data = entropy_tensor[slice_val, :, :]
            elif axis_name == 'Y':
                slice_data = entropy_tensor[:, slice_val, :]
            else:
                slice_data = entropy_tensor[:, :, slice_val]

            im = axes_list[idx].imshow(slice_data, cmap='magma', origin='lower', norm=norm)
            axes_list[idx].set_title(f"{axis_name}-Slice = {slice_val}")
            
            ncols_actual = min(n_slices, 4) if n_slices > 3 else n_slices
            if idx % ncols_actual == 0:
                bytes_label = axes_list[idx].set_ylabel(ylabel)
            nrows_actual = math.ceil(n_slices / ncols_actual)
            if idx >= (nrows_actual - 1) * ncols_actual:
                axes_list[idx].set_xlabel(xlabel)

        for idx in range(n_slices, len(axes_list)):
            axes_list[idx].set_visible(False)

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label('Absolute Mapped Bounded Entropy (0 to 1)', rotation=270, labelpad=15)
        
        plane = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}[axis_name]
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        if plot_title:
            plt.suptitle(f"{display_name}: {plane} Plane Rigorous Normalized Cell-Wise Entropy", fontsize=16)

        if save_plot:
            path = os.path.join(self.output_dir, f"norm_entropy_{data_source}_{plane}_{n_slices}_slices.png")
            plt.savefig(path, bbox_inches='tight', dpi=600)
            
        if show_plot: 
            plt.show()
        else: 
            plt.close(fig)
    
    def generate_2d_mds_plot(self, subset_size=100, figsize=(6, 8), save_plot=False):
        """Generates a 2D MDS plot comparing the spread of facies proportions between Flumy and GAN samples."""

        print("\n=== Generating Output: 2D MDS Flumy vs GAN Comparison Plot ===")
        
        if not self.gan_data:
            self.load_gan_samples(limit=subset_size)
        if not self.flumy_samples:
            self.load_flumy_samples(limit=subset_size)
            
        flumy_to_process = self.flumy_samples[:subset_size]
        gan_to_process = self.gan_data[:subset_size]
        
        all_features = []
        labels = []
        facies_codes = self.cfg['codes']
        
        for grid in flumy_to_process:
            props = [np.sum(grid == f) / grid.size for f in facies_codes]
            all_features.append(props)
            labels.append(self.flumy_name)
            
        for grid in gan_to_process:
            props = [np.sum(grid == f) / grid.size for f in facies_codes]
            all_features.append(props)
            labels.append(self.gan_name)
            
        features_matrix = np.array(all_features)
        
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, n_jobs=-1)
        embeddings = mds.fit_transform(features_matrix)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_labels = [self.flumy_name, self.gan_name]
        markers = ['o', 's']
        
        for i, l in enumerate(unique_labels):
            idx = [j for j, val in enumerate(labels) if val == l]
            ax.scatter(
                embeddings[idx, 0], embeddings[idx, 1],
                marker=markers[i], label=f"{l} (N={len(idx)})",
                s=35, alpha=0.8, edgecolors='white', linewidths=0.4
            )
            
        ax.set_title("MDS Spatial Clustering (Global Facies Proportions)", pad=10)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(color='#BFBFBF', alpha=0.4, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', frameon=True)
        
        if save_plot:
            out_mds_path = os.path.join(self.output_dir, f"mds_comparison_{subset_size}_samples.png")
            plt.savefig(out_mds_path, bbox_inches='tight', dpi=400)
            print(f"Saved 2D MDS Plot to: {out_mds_path}")
            
        plt.show()

    def plot_3d_entropy_pyvista(self, data_source='gan', figsize=None, show_plot=True, 
                                save_plot=True, black_background=True, show_legend=True):
        """Renders a 3D volumetric field of full continuous coordinates uncertainty bounded at [0.0, 1.0].

        This function always computes and displays the global mean voxel-wise entropy of the ensemble.

        Args:
            data_source (str, optional): Target sample ensemble evaluation group. Defaults to 'gan'.
            figsize (tuple, optional): Dimensions in pixels for the window layout. Defaults to None.
            show_plot (bool, optional): Interactive plotting execution. Defaults to True.
            save_plot (bool, optional): Screenshot file compression storage options. Defaults to True.
            black_background (bool, optional): Toggle dark or light scene rendering environment. Defaults to True.
            show_legend (bool, optional): Toggle color scale bar visibility. Defaults to True.
        """
        valid_sources = ['gan', 'flumy']
        if data_source not in valid_sources:
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
            
        try:
            import pyvista as pv
            pv.set_jupyter_backend('static') 
        except ImportError:
            print("Error: PyVista visualization module skipped.")
            return

        if data_source == 'gan':
            if not getattr(self, 'gan_data', None) or len(self.gan_data) == 0:
                self.load_gan_samples()
            target_data = self.gan_data
        else:
            if not getattr(self, 'flumy_samples', None) or len(self.flumy_samples) == 0:
                self.load_flumy_samples(limit=None)
            target_data = self.flumy_samples
            
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        print(f"\n--- Generating 3D PyVista Entropy Map ({display_name}) ---")

        entropy_map = self._compute_ensemble_entropy_map(target_data)
        
        # Compute and report the overall mean voxel-wise entropy of the 3D domain
        mean_voxel_entropy = float(np.mean(entropy_map))
        print(f"[{display_name}] Calculated Mean Voxel-Wise Entropy: {mean_voxel_entropy:.4f}")

        nz, ny, nx = entropy_map.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.cell_data['Entropy'] = entropy_map.transpose(2, 1, 0).flatten(order='F')

        window_size = figsize if figsize else (800, 500)
        plotter = pv.Plotter(off_screen=save_plot and not show_plot, window_size=window_size)
        
        # Apply background color toggle
        bg_color = 'black' if black_background else 'white'
        plotter.set_background(bg_color)
        
        # Adjust Scalar Bar color for readability based on background choice
        text_color = 'white' if black_background else 'black'

        plotter.enable_anti_aliasing('msaa') 

        plotter.add_mesh(
            grid, 
            scalars='Entropy',
            cmap='magma', 
            clim=[0.0, 1.0],  
            show_edges=False, 
            ambient=0.3,
            diffuse=0.7,
            show_scalar_bar=show_legend,               # Linked to your toggle argument
            scalar_bar_args={'color': text_color}      # Clean text configuration
        )

        plotter.view_isometric()
        plotter.camera.elevation -= 15  
        plotter.reset_camera()          
        plotter.camera.zoom(1.5)        

        plot_path = os.path.join(self.output_dir, f"3d_plot_{data_source}_ensemble_entropy.png")
        if save_plot:
            plotter.show(screenshot=plot_path)
        elif show_plot:
            plotter.show()
        else:
            plotter.close()

    def compute_spatial_delentropy(self, data_array):
        """Calculates the 3D spatial Delentropy of a single geological volume.
        
        This metric accounts for spatial structure by calculating the Shannon 
        entropy of the joint probability distribution of the 3D gradient vector field.
        
        Args:
            data_array (numpy.ndarray): 3D geological volume (Z, Y, X).
            
        Returns:
            float: The computed spatial delentropy in bits.
        """
        dz = np.pad(np.diff(data_array, axis=0), ((0, 1), (0, 0), (0, 0)), mode='edge')
        dy = np.pad(np.diff(data_array, axis=1), ((0, 0), (0, 1), (0, 0)), mode='edge')
        dx = np.pad(np.diff(data_array, axis=2), ((0, 0), (0, 0), (0, 1)), mode='edge')

        gradients = np.vstack([dz.flatten(), dy.flatten(), dx.flatten()]).T

        _, counts = np.unique(gradients, axis=0, return_counts=True)
        
        probabilities = counts / counts.sum()

        # The paper halves the 2D entropy to account for Papoulis' sampling redundancy, 
        # but for comparative metrics between datasets, the raw joint entropy is an excellent structural indicator.
        spatial_entropy = entropy(probabilities, base=2)
        
        # We divide by 2 following the parsimony principle outlined for Delentropy 
        # to halve the vector data rate using generalized sampling expansion.
        delentropy = spatial_entropy / 2.0 

        return delentropy

    def compare_structural_delentropy(self):
        """Evaluates and compares the spatial Delentropy of Flumy vs GAN datasets."""
        if not self.gan_data: self.load_gan_samples()
        if not self.flumy_samples: self.load_flumy_samples()

        print(f"\n--- Computing 3D Spatial Delentropy (Structural Complexity) ---")
        
        flumy_entropies = [self.compute_spatial_delentropy(arr) for arr in self.flumy_samples]
        gan_entropies = [self.compute_spatial_delentropy(arr) for arr in self.gan_data]

        flumy_mean = np.mean(flumy_entropies)
        gan_mean = np.mean(gan_entropies)
        
        print(f"{self.flumy_name} Mean Delentropy: {flumy_mean:.4f} bits")
        print(f"{self.gan_name} Mean Delentropy:  {gan_mean:.4f} bits")
        print(f"Difference ({self.gan_name} - {self.flumy_name}):   {gan_mean - flumy_mean:.4f} bits")
        
        if abs(gan_mean - flumy_mean) < 0.1:
            print(f"-> The {self.gan_name} exhibits excellent spatial structural fidelity compared to {self.flumy_name}.")
        else:
            print("-> A notable difference in spatial entropy suggests structural deviations (e.g., over-smoothing or excessive noise).")
            
        return {'flumy_delentropy_mean': flumy_mean, 'gan_delentropy_mean': gan_mean}

    def plot_local_delentropy_map(self, data_source='gan', slice_idx=None, window_size=5, figsize=None):
        """Generates and plots a 2D spatial map of local Delentropy using a sliding window.
        
        Args:
            data_source (str): 'gan' or 'flumy'.
            slice_idx (int, optional): The Z-slice index to plot. If None, picks the middle slice.
            window_size (int, optional): The size of the sliding window (must be odd, e.g., 5, 7, 9).
            figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (14, 6).
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number (e.g., 3, 5, 7).")

        if data_source == 'gan' and not self.gan_data: self.load_gan_samples()
        if data_source == 'flumy' and not self.flumy_samples: self.load_flumy_samples()
        
        target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
        volume = target_data[random.randint(0, len(target_data) - 1)]
        
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
            
        slice_2d = volume[slice_idx, :, :]
        ny, nx = slice_2d.shape
        
        pad_w = window_size // 2
        padded_slice = np.pad(slice_2d, pad_w, mode='edge')
        
        windows = view_as_windows(padded_slice, (window_size, window_size))
        
        local_entropy_map = np.zeros((ny, nx))
        
        print(f"Calculating local Delentropy map (Window: {window_size}x{window_size})...")
        
        for y in range(ny):
            for x in range(nx):
                patch = windows[y, x]
                
                dy = np.diff(patch, axis=0)[:, :-1].flatten()
                dx = np.diff(patch, axis=1)[:-1, :].flatten()
                
                gradients = np.vstack([dy, dx]).T
                
                _, counts = np.unique(gradients, axis=0, return_counts=True)
                probs = counts / counts.sum()
                
                # Halve the entropy for parsimony (as defined in the Delentropy literature)
                local_entropy_map[y, x] = entropy(probs, base=2) / 2.0

        figsize = figsize or (14, 6)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        custom_cmap, legend_patches = self._create_colormap_and_legend()
        display_slice = np.zeros_like(slice_2d)
        display_slice[slice_2d == 4] = 1
        display_slice[slice_2d == 8] = 2
        im0 = axes[0].imshow(display_slice, cmap=custom_cmap, origin='lower')
        axes[0].legend(handles=legend_patches, loc='upper right')
            
        axes[0].set_title(f"Original Facies (Z-Slice {slice_idx})")
        
        im1 = axes[1].imshow(local_entropy_map, cmap='magma', origin='lower')
        axes[1].set_title(f"Local Delentropy Map ({window_size}x{window_size} Window)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Local Spatial Entropy (bits)")
        
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        plt.suptitle(f"{display_name} - Spatial Structural Mapping", fontsize=16)
        plt.tight_layout()
        plt.show()


    def compute_slice_metrics(self, axis='Z', plot=False, save_plot=False):
        """Computes aggregate scalar profiles along an axis comparing the GAN directly to the true baseline.

        Provides exact comparison matrices mapping spatial Divergence alongside model vs reference
        variability ratios.

        Args:
            axis (str, optional): Cross-section slicing filter ('X', 'Y', 'Z'). Defaults to 'Z'.
            plot (bool, optional): Visual chart renderings configurations. Defaults to False.

        Returns:
            tuple: (pandas.DataFrame matching all individual slices data, dict of global summary metrics).
        """
        if not self.gan_data: 
            self.load_gan_samples()
        if not self.flumy_samples: 
            self.load_flumy_samples()

        print(f"\n--- Computing Scalar Metrics comparing {self.gan_name} to Baseline across {axis}-Axis ---")
        
        facies_values = self.cfg['codes']
        num_gan_samples = len(self.gan_data)
        num_flumy_samples = len(self.flumy_samples)
        
        entropy_flumy_3d = self._compute_ensemble_entropy_map(self.flumy_samples)
        entropy_gan_3d = self._compute_ensemble_entropy_map(self.gan_data)
        nz, ny, nx = entropy_flumy_3d.shape

        p_flumy = np.zeros((len(facies_values), nz, ny, nx))
        p_gan = np.zeros((len(facies_values), nz, ny, nx))
        
        flumy_vol = np.array(self.flumy_samples)
        gan_vol = np.array(self.gan_data)
        
        for i, f_val in enumerate(facies_values):
            p_flumy[i] = np.sum(flumy_vol == f_val, axis=0) / num_flumy_samples
            p_gan[i] = np.sum(gan_vol == f_val, axis=0) / num_gan_samples
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jsd_3d = jensenshannon(p_flumy, p_gan, base=2, axis=0) ** 2

        if axis == 'Z':
            agg_axes = (1, 2)
        elif axis == 'Y':
            agg_axes = (0, 2)
        else:
            agg_axes = (0, 1)
            
        slice_entropy_flumy = np.nanmean(entropy_flumy_3d, axis=agg_axes)
        slice_entropy_gan = np.nanmean(entropy_gan_3d, axis=agg_axes)
        slice_jsd = np.nanmean(jsd_3d, axis=agg_axes)
        
        num_slices = {'Z': nz, 'Y': ny, 'X': nx}[axis]
        results = [{
            'Axis': axis,
            'Slice_Index': idx,
            'Flumy_Norm_Entropy': slice_entropy_flumy[idx],
            'GAN_Norm_Entropy': slice_entropy_gan[idx],
            'JSD_Flumy_vs_GAN': slice_jsd[idx]
        } for idx in range(num_slices)]
        
        df_results = pd.DataFrame(results)
        
        summary_stats = {
            'Flumy_Entropy_Mean': df_results['Flumy_Norm_Entropy'].mean(),
            'Flumy_Entropy_Std': df_results['Flumy_Norm_Entropy'].std(),
            'GAN_Entropy_Mean': df_results['GAN_Norm_Entropy'].mean(),
            'GAN_Entropy_Std': df_results['GAN_Norm_Entropy'].std(),
            'JSD_Mean': df_results['JSD_Flumy_vs_GAN'].mean(),
            'JSD_Std': df_results['JSD_Flumy_vs_GAN'].std(),
        }

        print(f"\n--- Summary Statistics ({axis}-Axis) ---")
        print(f"Test Baseline Entropy: {summary_stats['Flumy_Entropy_Mean']:.4f} ± {summary_stats['Flumy_Entropy_Std']:.4f}")
        print(f"{self.gan_name} Model Entropy:  {summary_stats['GAN_Entropy_Mean']:.4f} ± {summary_stats['GAN_Entropy_Std']:.4f}")
        print(f"Model-to-Baseline Spatial Entropy Ratio: {summary_stats['GAN_Entropy_Mean'] / summary_stats['Flumy_Entropy_Mean']:.4f}")
        print(f"Model-to-baseline JSD: {summary_stats['JSD_Mean']:.4f} ± {summary_stats['JSD_Std']:.4f}")
        
        csv_path = os.path.join(self.output_dir, f"slice_metrics_normalized_{axis}.csv")
        df_results.to_csv(csv_path, index=False)

        if plot:
            self._plot_metric_distributions(df_results, axis, save_plot)
            
        return df_results, summary_stats

    def _plot_metric_distributions(self, df_results, axis, save_plot):
        """Plots the metric distributions across slices.
        
        Args:
            df_results (pd.DataFrame): DataFrame containing slice metrics.
            axis (str): The axis along which slices were taken.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        slices = df_results['Slice_Index']
        flumy_entropy = df_results['Flumy_Norm_Entropy']
        gan_entropy = df_results['GAN_Norm_Entropy']
        jsd = df_results['JSD_Flumy_vs_GAN']
        
        axes[0].plot(slices, flumy_entropy, label=f'{self.flumy_name}', color='tab:blue', linewidth=2)
        axes[0].plot(slices, gan_entropy, label=f'{self.gan_name}', color='tab:orange', linewidth=2, linestyle='--')
        axes[0].set_title(f'Normalized Entropy across {axis}-Axis', fontsize=14)
        axes[0].set_xlabel('Slice Index', fontsize=12)
        axes[0].set_ylabel('Normalized Entropy', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        axes[1].plot(slices, jsd, color='tab:purple', linewidth=2)
        axes[1].set_title(f'Jensen-Shannon Divergence across {axis}-Axis', fontsize=14)
        axes[1].set_xlabel('Slice Index', fontsize=12)
        axes[1].set_ylabel('JSD', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_plot:
                    plot_path = os.path.join(self.output_dir, f"JSD_{axis}_axis.png")
                    plt.savefig(plot_path, bbox_inches='tight', dpi=400)
                    print(f"Saved JSD plot to: {plot_path}")
        plt.show()
    
    def plot_2d_slices(self, data_source='gan', num_samples=1, num_slices=1, axis=None, 
                       slice_range=None, slice_indices=None, figsize=None, 
                       plot_title=False, plot_filename=True, show_plot=True, save_plot=False, 
                       combine_samples=True, show_legend=True):
        """Visualizes 2D cross-sections (Horizontal and Vertical) of 3D realizations scaled to physical dimensions.
        
        Args:
            data_source (str): Target dataset to evaluate ('gan' or 'flumy').
            num_samples (int): Number of realizations to plot.
            num_slices (int): Number of slices to extract per realization. Ignored when
                slice_indices is provided.
            axis (str, optional): Specific axis to slice ('X', 'Y', 'Z'). If None, plots all 3.
            slice_range (tuple, optional): Restricts slicing to a sub-range of the axis as
                (start, end). When provided, num_slices evenly-spaced positions are sampled
                within this range. E.g., slice_range=(80, 120) with num_slices=5.
            slice_indices (list, optional): Explicit list of integer slice positions. Overrides
                both num_slices and slice_range when provided.
            figsize (tuple, optional): Figure size as (width, height) in inches. If None,
                auto-computed based on layout.
            show_plot (bool): If True, displays the plot interactively.
            save_plot (bool): If True, saves the plot to the output directory.
            combine_samples (bool): If True, combines all samples into a single subplot grid.
            show_legend (bool): If True, displays the facies classification color legend.
        """
        if axis and axis.upper() not in ['X', 'Y', 'Z']:
            raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
            
        axis = axis.upper() if axis else None

        if data_source == 'gan' and not self.gan_data: 
            self.load_gan_samples(limit=num_samples)
        if data_source == 'flumy' and not self.flumy_samples: 
            self.load_flumy_samples(limit=num_samples)
            
        target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
        files_list = self.gan_files if data_source == 'gan' else self.data_files
        
        plot_limit = min(num_samples, len(target_data))
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name

        colors = [self.cfg['colors'][c] for c in self.cfg['codes']]
        cmap = ListedColormap(colors)
        labels = [self.cfg['names'][c] for c in self.cfg['codes']]
        legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

        # Physical spacing parameters (Z, Y, X order matching array format: depth, height, width)
        dx, dy, dz = 20, 20, 1

        def _resolve_slice_indices(dim_size):
            if slice_indices is not None:
                return np.array(slice_indices, dtype=int)
            elif slice_range is not None:
                return np.linspace(slice_range[0], slice_range[1], num_slices, dtype=int)
            else:
                return np.linspace(0, dim_size - 1, num_slices, dtype=int)

        sample_data_0 = target_data[0]
        depth, height, width = sample_data_0.shape
        z_idx_template = _resolve_slice_indices(depth)
        effective_num_slices = len(z_idx_template)

        if combine_samples:
            if axis is None:
                total_rows = plot_limit * effective_num_slices
                effective_figsize = figsize or (18, 5 * total_rows)
                fig, axes = plt.subplots(total_rows, 3, figsize=effective_figsize, squeeze=False)
                
                plot_row = 0
                for idx in range(plot_limit):
                    data_3d = target_data[idx]
                    display_data = np.zeros_like(data_3d)
                    for map_idx, code in enumerate(self.cfg['codes']):
                        display_data[data_3d == code] = map_idx
                    if plot_filename:    
                        filename = os.path.basename(files_list[idx]) if idx < len(files_list) else f"Sample_{idx}"
                    else:
                        filename=""
                    z_idx = _resolve_slice_indices(data_3d.shape[0])
                    y_idx = _resolve_slice_indices(data_3d.shape[1])
                    x_idx = _resolve_slice_indices(data_3d.shape[2])
                    
                    for i in range(effective_num_slices):
                        # Z-Slice (XY Plane): Horizontal View
                        extent_z = [0, width * dx, 0, height * dy]
                        axes[plot_row, 0].imshow(display_data[z_idx[i], :, :], cmap=cmap, origin='lower', 
                                                 vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_z, aspect='equal')
                        axes[plot_row, 0].set_title(f"{filename} | Horiz (Z={z_idx[i] * dz}m)")
                        axes[plot_row, 0].set_xlabel("X Distance (m)")
                        axes[plot_row, 0].set_ylabel("Y Distance (m)")

                        # Y-Slice (XZ Plane): Vertical View
                        extent_y = [0, width * dx, 0, depth * dz]
                        axes[plot_row, 1].imshow(display_data[:, y_idx[i], :], cmap=cmap, origin='lower', 
                                                 vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_y, aspect='20') 
                        axes[plot_row, 1].set_title(f"{filename} | Vert (Y={y_idx[i] * dy}m)")
                        axes[plot_row, 1].set_xlabel("X Distance (m)")
                        axes[plot_row, 1].set_ylabel("Z Distance (m)")

                        # X-Slice (YZ Plane): Vertical View
                        extent_x = [0, height * dy, 0, depth * dz]
                        axes[plot_row, 2].imshow(display_data[:, :, x_idx[i]], cmap=cmap, origin='lower', 
                                                 vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_x, aspect='20')
                        axes[plot_row, 2].set_title(f"{filename} | Vert (X={x_idx[i] * dx}m)")
                        axes[plot_row, 2].set_xlabel("Y Distance (m)")
                        axes[plot_row, 2].set_ylabel("Z Distance (m)")
                        plot_row += 1
            else:
                total_plots = plot_limit * effective_num_slices
                ncols = min(total_plots, 3)
                nrows = math.ceil(total_plots / ncols)
                effective_figsize = figsize or (5 * ncols, 5 * nrows)
                
                fig, axes = plt.subplots(nrows, ncols, figsize=effective_figsize)
                axes_list = np.atleast_1d(axes).flatten()
                
                plot_idx = 0
                for idx in range(plot_limit):
                    data_3d = target_data[idx]
                    display_data = np.zeros_like(data_3d)
                    for map_idx, code in enumerate(self.cfg['codes']):
                        display_data[data_3d == code] = map_idx
                    
                    if plot_filename:
                        filename = os.path.basename(files_list[idx]) if idx < len(files_list) else f"Sample_{idx}"
                    else:
                        filename=""
                    dim_size = data_3d.shape[{'Z': 0, 'Y': 1, 'X': 2}[axis]]
                    current_indices = _resolve_slice_indices(dim_size)
                    
                    for i in range(effective_num_slices):
                        ax = axes_list[plot_idx]
                        if axis == 'Z':
                            extent_z = [0, width * dx, 0, height * dy]
                            ax.imshow(display_data[current_indices[i], :, :], cmap=cmap, origin='lower', 
                                      vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_z, aspect='equal')
                            ax.set_title(f"{filename} (depth={current_indices[i] * dz}m)")
                            ax.set_xlabel("X Distance (m)")
                            ax.set_ylabel("Y Distance (m)")
                        elif axis == 'Y':
                            extent_y = [0, width * dx, 0, depth * dz]
                            ax.imshow(display_data[:, current_indices[i], :], cmap=cmap, origin='lower', 
                                      vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_y, aspect=20)
                            ax.set_title(f"{filename} (Y= {current_indices[i] * dy}m)")
                            ax.set_xlabel("X Distance (m)")
                            ax.set_ylabel("Z Depth (m)")
                        elif axis == 'X':
                            extent_x = [0, height * dy, 0, depth * dz]
                            ax.imshow(display_data[:, :, current_indices[i]], cmap=cmap, origin='lower', 
                                      vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_x, aspect=20)
                            ax.set_title(f"{filename} (X= {current_indices[i] * dx}m)")
                            ax.set_xlabel("Y Distance (m)")
                            ax.set_ylabel("Z Depth (m)")
                        plot_idx += 1
                        
                for i in range(plot_idx, len(axes_list)):
                    axes_list[i].set_visible(False)

            if show_legend:
                fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8)
                
            if plot_title:
                ax_label = axis if axis else "All_Axes"
                plt.suptitle(f"{display_name}", fontsize=12, fontweight='bold', x= 0.15, y=0.95)
                
            if save_plot:
                ax_label = axis if axis else "All_Axes"
                plot_path = os.path.join(self.output_dir, f"combined_grid_{data_source}_{ax_label}_{plot_limit}samples.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"Saved combined 2D grid matrix to: {plot_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        else:
            for idx in range(plot_limit):
                data_3d = target_data[idx]
                display_data = np.zeros_like(data_3d)
                for map_idx, code in enumerate(self.cfg['codes']):
                    display_data[data_3d == code] = map_idx

                depth, height, width = display_data.shape
                z_idx = _resolve_slice_indices(depth)
                y_idx = _resolve_slice_indices(height)
                x_idx = _resolve_slice_indices(width)
                
                filename = os.path.basename(files_list[idx]) if idx < len(files_list) else f"Sample_{idx}"

                if axis is None:
                    effective_figsize = figsize or (18, 5 * effective_num_slices)
                    fig, axes = plt.subplots(effective_num_slices, 3, figsize=effective_figsize)
                    if effective_num_slices == 1:
                        axes = np.expand_dims(axes, axis=0) 
                    
                    for i in range(effective_num_slices):
                        # XY Plane
                        extent_z = [0, width * dx, 0, height * dy]
                        axes[i, 0].imshow(display_data[z_idx[i], :, :], cmap=cmap, origin='lower', 
                                          vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_z, aspect='equal')
                        axes[i, 0].set_title(f"Horizontal Slice (depth= {z_idx[i] * dz}m)")
                        axes[i, 0].set_xlabel("X Distance (m)")
                        axes[i, 0].set_ylabel("Y Distance (m)")

                        # XZ Plane
                        extent_y = [0, width * dx, 0, depth * dz]
                        axes[i, 1].imshow(display_data[:, y_idx[i], :], cmap=cmap, origin='lower', 
                                          vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_y, aspect='auto') 
                        axes[i, 1].set_title(f"Vertical Section (Y={y_idx[i] * dy}m)")
                        axes[i, 1].set_xlabel("X Distance (m)")
                        axes[i, 1].set_ylabel("Z Distance (m)")

                        # YZ Plane
                        extent_x = [0, height * dy, 0, depth * dz]
                        axes[i, 2].imshow(display_data[:, :, x_idx[i]], cmap=cmap, origin='lower', 
                                          vmin=0, vmax=len(self.cfg['codes'])-1, extent=extent_x, aspect='auto')
                        axes[i, 2].set_title(f"Vertical Section (X={x_idx[i] * dx}m)")
                        axes[i, 2].set_xlabel("Y Distance (m)")
                        axes[i, 2].set_ylabel("Z Distance (m)")
                else:
                    ncols = min(effective_num_slices, 4)
                    nrows = math.ceil(effective_num_slices / ncols)
                    effective_figsize = figsize or (5 * ncols, 5 * nrows)
                    
                    fig, axes = plt.subplots(nrows, ncols, figsize=effective_figsize)
                    axes_list = np.atleast_1d(axes).flatten()

                    idx_map = {'Z': z_idx, 'Y': y_idx, 'X': x_idx}
                    current_indices = idx_map[axis]

                    for i in range(effective_num_slices):
                        ax = axes_list[i]
                        if axis == 'Z':
                            extent_z = [0, width * dx, 0, height * dy]
                            ax.imshow(display_data[current_indices[i], :, :], cmap=cmap, origin='lower',
                                      vmin=0, vmax=len(self.cfg['codes'])-1, interpolation='antialiased', interpolation_stage='rgba', extent=extent_z, aspect='equal')
                            ax.set_title(f"Horizontal Slice (Z={current_indices[i] * dz}m)")
                            ax.set_xlabel("X Distance (m)")
                            ax.set_ylabel("Y Distance (m)")
                        elif axis == 'Y':
                            extent_y = [0, width * dx, 0, depth * dz]
                            ax.imshow(display_data[:, current_indices[i], :], cmap=cmap, origin='lower',
                                      vmin=0, vmax=len(self.cfg['codes'])-1, interpolation='antialiased', interpolation_stage='rgba', extent=extent_y, aspect='auto')
                            ax.set_title(f"Vertical Section (Y={current_indices[i] * dy}m)")
                            ax.set_xlabel("X Distance (m)")
                            ax.set_ylabel("Z Distance (m)")
                        elif axis == 'X':
                            extent_x = [0, height * dy, 0, depth * dz]
                            ax.imshow(display_data[:, :, current_indices[i]], cmap=cmap, origin='lower',
                                      vmin=0, vmax=len(self.cfg['codes'])-1, interpolation='antialiased', interpolation_stage='rgba', extent=extent_x, aspect='auto')
                            ax.set_title(f"Vertical Section (X={current_indices[i] * dx}m)")
                            ax.set_xlabel("Y Distance (m)")
                            ax.set_ylabel("Z Distance (m)")

                    for i in range(effective_num_slices, len(axes_list)):
                        axes_list[i].set_visible(False)

                if show_legend:
                    fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=12)
                    
                if plot_title:
                    plt.suptitle(f"{display_name} | Realization: {filename} | Shape: {data_3d.shape}", fontsize=16, fontweight='bold', y=1.02)
                
                if save_plot:
                    ax_label = axis if axis else "All_Axes"
                    plot_path = os.path.join(self.output_dir, f"2d_slices_{data_source}_{ax_label}_{effective_num_slices}slices_{filename.replace('.npy', '.png')}")
                    plt.savefig(plot_path, bbox_inches='tight', dpi=400)
                    print(f"Saved 2D slices to: {plot_path}")
                    
                if show_plot:
                    plt.show()
                else:
                    plt.close(fig)

    def plot_3d_pyvista(self, data_source='gan', mode='separate', target_filename=None, target_facies=None, 
                            figsize=None, show_legend=True, show_plot=True, save_plot=True, black_background=False):
            """Renders an interactive or static 3D volumetric plot of a geological sample."""
            valid_sources = ['gan', 'flumy']
            valid_modes = ['separate', 'combined']
            
            if data_source not in valid_sources:
                raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")
                
            try:
                import pyvista as pv
                pv.set_jupyter_backend('static') 
            except ImportError:
                print("Error: 'pyvista' is not installed. Skipping 3D plot.")
                return
            
            # 1. Define UI colors based on toggle
            bg_color = 'black' if black_background else 'white'
            text_color = 'white' if black_background else 'black'

            display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
            print(f"\n--- Generating 3D PyVista Plot ({display_name} | Mode: {mode.upper()}) ---")

            facies_colors = self.cfg['colors']
            facies_titles = self.cfg['names']

            # File loading logic
            plot_identifier = ""
            if target_filename:
                file_list = self.gan_files if data_source == 'gan' else self.data_files
                matched_file = next((f for f in file_list if target_filename in f), None)
                if matched_file:
                    data_3d = self._load_gan(matched_file) if data_source == 'gan' else self._load_data(matched_file)
                    plot_identifier = target_filename.split('.')[0]
                else:
                    print(f"Warning: '{target_filename}' not found. Falling back to random sample.")
                    target_filename = None

            if not target_filename:
                if data_source == 'gan' and not self.gan_data: self.load_gan_samples()
                elif data_source == 'flumy' and not self.flumy_samples: self.load_flumy_samples() 
                target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
                random_idx = random.randint(0, len(target_data) - 1)
                data_3d = target_data[random_idx]
                plot_identifier = f"sample_{random_idx}"

            nz, ny, nx = data_3d.shape
            grid = pv.ImageData()
            grid.dimensions = (nx + 1, ny + 1, nz + 1)
            grid.cell_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

            facies_to_plot = [target_facies] if isinstance(target_facies, int) else (target_facies or self.cfg['codes'])
            if len(facies_to_plot) == 1 and mode == 'separate': mode = 'combined'

            # Plotter Setup
            num_subplots = len(facies_to_plot) if mode == 'separate' else 1
            window_size = figsize if figsize else ((800 * num_subplots if mode == 'separate' else 800), 500)
            
            plotter = pv.Plotter(shape=(1, num_subplots), off_screen=save_plot and not show_plot, window_size=window_size)
            plotter.set_background(bg_color)
            plotter.enable_anti_aliasing('msaa') 
                
            # Single loop to add meshes
            for i, f_val in enumerate(facies_to_plot):
                if f_val not in facies_colors: continue
                    
                if mode == 'separate':
                    plotter.subplot(0, i)
                    plotter.add_text(facies_titles[f_val], font_size=20, color=text_color, shadow=True) 
                    
                threshed = grid.threshold([f_val - 0.5, f_val + 0.5], scalars='Facies')
                if threshed.n_points > 0:
                    plotter.add_mesh(threshed, color=facies_colors[f_val], show_edges=False, ambient=0.2, diffuse=0.8,
                                    label=facies_titles[f_val] if mode == 'combined' else None)
                    
                if mode == 'separate':
                    plotter.view_isometric()
                    plotter.camera.elevation -= 10

            # FIXED: Removed 'color=text_color' to resolve the PyVista Renderer crash
            if (mode == 'combined') and show_legend and len(facies_to_plot) > 1:
                plotter.add_legend(bcolor='grey', face=None, size=(0.2, 0.2))
            
            if mode == 'combined':
                plotter.view_isometric()
                plotter.camera.elevation -= 10
                plotter.camera.zoom(1.3)

            # Saving/Closing
            suffix = f"_{'-'.join(map(str, facies_to_plot))}" if target_facies else ""
            plot_path = os.path.join(self.output_dir, f"3d_plot_{data_source}_{plot_identifier}{mode}{suffix}.png")
            
            if save_plot:
                plotter.show(screenshot=plot_path)
                print(f"Saved 3D plot to: {plot_path}") 
            elif show_plot: plotter.show()
            else: plotter.close()

class DistributionEvaluator:
    """Evaluates geological distributions using Multi-Scale Sliced Wasserstein Distance (MS-SWD).
    
    Provides dataset-agnostic computation of pairwise MS-SWD distance matrices
    and multi-dataset MDS visualization. Each dataset's distance matrix and raw
    samples are cached to disk, enabling efficient multi-run comparisons without 
    reloading original data.

    Attributes:
        output_dir (str): Base directory where outputs are saved.
        device (torch.device): PyTorch computation device.
        ms_swd (MSSWD): The Multi-Scale Sliced Wasserstein Distance metric.
    """
    DISTANCE_FILENAME = "msswd_distances.npz"

    def __init__(self, output_dir, device_type=None):
        """Initializes the DistributionEvaluator.

        Args:
            output_dir (str): Base directory to save outputs. Individual dataset
                embeddings are saved to their own subdirectories.
            device_type (str, optional): PyTorch device string ('cuda:0', 'cpu'). 
                If None, auto-detects GPU availability.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if device_type:
            self.device = torch.device(device_type)
            print(f"using device {self.device}")
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"using device {self.device}")
        
        self.ms_swd = MSSWD(
            n_levels=3,
            n_descriptors=512,
            descriptor_size=(3, 7, 7),
            n_repeat=12,
            n_proj=128,
            padding_mode='circular',
            combine_levels=True
        )

    def _prepare_tensors(self, data_list):
        """Converts a list of 3D numpy arrays into PyTorch tensors with a channel dimension.

        Args:
            data_list (list): List of 3D numpy arrays of geological facies.

        Returns:
            torch.Tensor: 5D tensor of shape (N, 1, Z, Y, X) on the configured device.
        """
        arr = np.stack(data_list)
        
        if arr.ndim == 4:
            arr = np.expand_dims(arr, axis=1)
            
        tensor = torch.tensor(arr, dtype=torch.float32).to(self.device)
        return tensor

    def compute_embedding(self, samples, label, save_dir=None):
        """Computes pairwise MS-SWD distances for a single dataset and caches results.

        Calculates the full pairwise distance matrix between all samples using
        MS-SWD and saves it along with the raw sample arrays to a .npz file
        for later use by plot_multi_mds().

        Args:
            samples (list): List of 3D numpy arrays (geological volumes) for this dataset.
            label (str): Human-readable label for this dataset (e.g., 'Flumy', 'VoxGAN_run1').
                Used in progress bar and print statements.
            save_dir (str, optional): Directory to save the distance file. If None,
                uses self.output_dir. The file is always named 'msswd_distances.npz'.

        Returns:
            numpy.ndarray: Symmetric pairwise distance matrix of shape (N, N).
        """
        target_dir = save_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, self.DISTANCE_FILENAME)

        n_samples = len(samples)
        print(f"\nComputing MS-SWD distance matrix for '{label}' ({n_samples} samples)...")
        
        tensor = self._prepare_tensors(samples)
        distances = np.zeros((n_samples, n_samples))
        
        total_comparisons = (n_samples * (n_samples - 1)) // 2
        
        with torch.no_grad():
            with tqdm(total=total_comparisons, desc=f"MS-SWD [{label}]", unit="pair") as pbar:
                for j in range(n_samples):
                    for k in range(j + 1, n_samples):
                        dist = self.ms_swd(tensor[j:j+1], tensor[k:k+1]).item()
                        distances[j, k] = dist
                        distances[k, j] = dist
                        pbar.update(1)

        samples_array = np.stack(samples)
        np.savez(
            save_path,
            distances=distances,
            samples=samples_array
        )
        print(f"Saved distance matrix ({n_samples}x{n_samples}) and samples to: {save_path}")
        
        return distances

    def plot_multi_mds(self, datasets, random_state=42, figsize=None, 
                       save_plot=True, show_plot=True):
        """Generates a combined MDS scatter plot from multiple dataset distance caches.

        Loads cached distance matrices and raw samples from each dataset directory,
        computes cross-dataset pairwise MS-SWD distances, assembles a combined 
        distance matrix, and applies MDS to produce a unified 2D embedding.

        Args:
            datasets (dict): Mapping of dataset labels to their output directories.
                Each directory must contain 'msswd_distances.npz' (created by 
                compute_embedding). Example:
                    {
                        'Flumy': 'outputs/flumy/',
                        'VoxGAN_run1': 'outputs/run1/',
                        'VoxGAN_run2': 'outputs/run2/'
                    }
            random_state (int, optional): Seed for MDS reproducibility. Defaults to 42.
            figsize (tuple, optional): Figure size as (width, height) in inches. 
                Defaults to (8, 8).
            save_plot (bool, optional): If True, saves the plot to self.output_dir. 
                Defaults to True.
            show_plot (bool, optional): If True, displays the plot interactively. 
                Defaults to True.

        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes) for further customization.

        Raises:
            FileNotFoundError: If any dataset directory is missing its distance file.
                Run compute_embedding() for the missing dataset first.
        """
        print("\n--- Loading cached distance data ---")
        
        all_labels = list(datasets.keys())
        cached_data = {}
        
        for label, data_dir in datasets.items():
            dist_path = os.path.join(data_dir, self.DISTANCE_FILENAME)
            if not os.path.exists(dist_path):
                raise FileNotFoundError(
                    f"Distance file not found at '{dist_path}'. "
                    f"Run compute_embedding() for '{label}' first."
                )
            loaded = np.load(dist_path)
            cached_data[label] = {
                'distances': loaded['distances'],
                'samples': loaded['samples']
            }
            print(f"  Loaded '{label}': {loaded['distances'].shape[0]} samples")

        dataset_sizes = [cached_data[label]['distances'].shape[0] for label in all_labels]
        total_samples = sum(dataset_sizes)
        
        combined_distances = np.zeros((total_samples, total_samples))
        
        # Fill in within-dataset blocks from cached matrices
        offset = 0
        offsets = {}
        for label in all_labels:
            n = cached_data[label]['distances'].shape[0]
            offsets[label] = offset
            combined_distances[offset:offset+n, offset:offset+n] = cached_data[label]['distances']
            offset += n

        # Compute cross-dataset distances
        print("\nComputing cross-dataset MS-SWD distances...")
        for i, label_a in enumerate(all_labels):
            for label_b in all_labels[i+1:]:
                samples_a = cached_data[label_a]['samples']
                samples_b = cached_data[label_b]['samples']
                
                tensor_a = self._prepare_tensors(list(samples_a))
                tensor_b = self._prepare_tensors(list(samples_b))
                
                n_a = len(samples_a)
                n_b = len(samples_b)
                off_a = offsets[label_a]
                off_b = offsets[label_b]
                
                total_cross = n_a * n_b
                with torch.no_grad():
                    with tqdm(total=total_cross, desc=f"Cross [{label_a} × {label_b}]", unit="pair") as pbar:
                        for j in range(n_a):
                            for k in range(n_b):
                                dist = self.ms_swd(tensor_a[j:j+1], tensor_b[k:k+1]).item()
                                combined_distances[off_a + j, off_b + k] = dist
                                combined_distances[off_b + k, off_a + j] = dist
                                pbar.update(1)

        print("\nApplying Multidimensional Scaling (MDS) on combined distance matrix...")
        reducer = MDS(
            n_components=2,
            n_jobs=-1,
            random_state=random_state,
            dissimilarity='precomputed'
        )
        
        embeddings = reducer.fit_transform(combined_distances)
        print(f"MDS stress: {reducer.stress_:.4f}")

        # --- NEW CODE: Create a list of labels corresponding to each point ---
        expanded_labels = []
        for label in all_labels:
            n = cached_data[label]['distances'].shape[0]
            expanded_labels.extend([label] * n)
            
        # --- NEW CODE: Save coordinates to CSV ---
        df_coords = pd.DataFrame({
            'Dimension 1': embeddings[:, 0],
            'Dimension 2': embeddings[:, 1],
            'Dataset': expanded_labels
        })
        csv_path = os.path.join(self.output_dir, "msswd_mds_coordinates.csv")
        df_coords.to_csv(csv_path, index=False)
        print(f"Saved 2D coordinates to CSV: {csv_path}")
        
        # Plot
        figsize = figsize or (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        color_cycle = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']
        
        offset = 0
        for idx, label in enumerate(all_labels):
            n = cached_data[label]['distances'].shape[0]
            color = color_cycle[idx % len(color_cycle)]
            marker = markers[idx % len(markers)]
            
            ax.scatter(
                embeddings[offset:offset+n, 0],
                embeddings[offset:offset+n, 1],
                s=30, c=[color], marker=marker, lw=0.5, ec='white',
                label=f'{label} ({n} samples)'
            )
            offset += n
        
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('MS-SWD Multidimensional Scaling', fontsize=14, pad=15)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='#BFBFBF', alpha=0.5, linestyle='--', linewidth=0.5)
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=min(len(all_labels), 4), frameon=False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, "msswd_mds_multi_plot.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved multi-MDS plot to: {plot_path}")
            
        if show_plot:
            plt.show()
        
        return fig, ax