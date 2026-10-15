import os
import argparse
import numpy as np
import h5py
from glob import glob
import sys
from pathlib import Path


# Importing custom plots (kept in case you want to adapt them later, but disabled in main)
#from training_dataset_summary import plot_facies_distribution, plot_entropy
from custom_plots import apply_custom_plotting_flavor, FaciesColorMap
apply_custom_plotting_flavor()

def parse_args():
    parser = argparse.ArgumentParser(description="Convert multiple directories of .npz/.npy files to a single .h5 file")
    # CHANGED: Use nargs='+' to accept multiple directories
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Paths to the directories with .npz/.npy files')
    parser.add_argument('--output_file', type=str, required=True, help='Path and name for the output .h5 file (e.g., dataset.h5)')
    parser.add_argument('--samples_per_dir', type=int, default=10000, help='Number of samples to take from EACH dataset')
    parser.add_argument('--file_type', type=str, choices=['npz', 'npy'], default='npy', help='Type of input files (default: npy)')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save generated plots (default: same as output_file)')
    parser.add_argument('--num_plot_samples', type=int, default=10, help='Number of realizations to sample for the entropy plot')
    parser.add_argument('--target_dim', type=int, default=32, help='Target size for the dimension being cropped.')
    parser.add_argument('--disable_crop', type=bool, default=True, help='Disable the crop axis function')
    parser.add_argument('--crop_axis', type=int, default=0, help='Axis to crop. 0 for first, 1 for second (default), 2 for third.')
    
    return parser.parse_args()

def load_array(filepath, file_type):
    """Helper function to correctly load either .npz or .npy files."""
    if file_type == 'npz':
        return np.load(filepath)['facies']
    else:
        return np.load(filepath)

def center_crop(data, disable_crop, target_dim, axis):
    """Crops the specified axis to target_dim by taking the middle slices."""
    if not disable_crop:
        current_dim = data.shape[axis]
        if current_dim > target_dim:
            start = (current_dim - target_dim) // 2
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(start, start + target_dim)
            return data[tuple(slices)]
    return data

def main():
    args = parse_args()
    
    all_files = []
    search_pattern = f"*.{args.file_type}"

    # CHANGED: Iterate over all provided directories and grab the first N files from each
    for d in args.data_dirs:
        files_in_dir = sorted(glob(os.path.join(d, search_pattern)))
        
        # Check if the directory has enough files
        if len(files_in_dir) < args.samples_per_dir:
            print(f"Warning: Directory {d} only has {len(files_in_dir)} files. Taking all of them.")
            selected_files = files_in_dir
        else:
            selected_files = files_in_dir[:args.samples_per_dir]
            
        all_files.extend(selected_files)
        print(f"Collected {len(selected_files)} files from {d}")

    num_files = len(all_files)
    
    if num_files == 0:
        print("Error: No files found in the provided directories.")
        return

    print(f"\nTotal files to convert: {num_files}. Starting conversion...")

    # Load the first file and apply the center crop to establish standard dimensions
    first_file = load_array(all_files[0], args.file_type)
    first_file = center_crop(first_file, args.disable_crop, args.target_dim, args.crop_axis)
    
    grid_shape = first_file.shape
    dtype = first_file.dtype

    print(f"Final normalized grid shape for HDF5 dataset: {grid_shape}")

    # Create the HDF5 file
    with h5py.File(args.output_file, 'w') as h5f:
        dataset = h5f.create_dataset(
            'facies', 
            shape=(num_files, *grid_shape), 
            dtype=dtype,
            chunks=(1, *grid_shape),
            compression="lzf"
        )

        # Populate the dataset
        for i, file_path in enumerate(all_files):
            try:
                data = load_array(file_path, args.file_type)
                data = center_crop(data, args.disable_crop, args.target_dim, args.crop_axis)
                dataset[i] = data
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                print(f"Attempted to insert array of shape {data.shape} into dataset slot of shape {grid_shape}")
                sys.exit(1)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{num_files} files...")

    print(f"Success! HDF5 dataset saved to: {args.output_file}")

    # check dataset contents
    print("-" * 30)
    with h5py.File(args.output_file, 'r') as f:
        for key in f.keys():
            ds = f[key]
            print(f"Dataset Name : {key}")
            print(f"Shape        : {ds.shape}")
            print(f"Data Type    : {ds.dtype}")
            print(f"Chunk Size   : {ds.chunks}")
            print(f"Compression  : {ds.compression}")
            print("-" * 30)
            
    # CHANGED: Plotting disabled for combined datasets to avoid breaking custom scripts
    print("\nNote: Plotting is skipped. Custom plot scripts expect a single directory, but we combined multiple.")
    # apply_custom_plotting_flavor()
    # plot_dir = args.plot_dir if args.plot_dir else os.path.dirname(args.output_file)
    # if not plot_dir:
    #     plot_dir = "."
    # os.makedirs(plot_dir, exist_ok=True)
    # file_basenames = [os.path.basename(f) for f in all_files]
    # plot_facies_distribution(args.data_dirs[0], plot_dir, file_basenames)  # Would need rewrite
    # plot_entropy(args.data_dirs[0], plot_dir, file_basenames, args.num_plot_samples) # Would need rewrite

if __name__ == '__main__':
    main()