import os
import json
import argparse
import h5py
import numpy as np

def load_config(config_path):
    """
    Loads a JSON configuration file.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, save_path):
    """
    Saves a configuration dictionary to a JSON file.
    
    Args:
        config (dict): Configuration dictionary.
        save_path (str): Path to save the config file.
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

def parse_hybrid_args():
    """
    Parses arguments using a hybrid approach: CLI overrides JSON configs.
    
    Returns:
        dict: Final merged configuration.
    """
    parser = argparse.ArgumentParser(description="Train DCGAN Architecture 4 for Geomodelling")
    parser.add_argument('--config', type=str, default=argparse.SUPPRESS, help='Path to a JSON config file to load defaults from')
    parser.add_argument('--data_file', type=str, default=argparse.SUPPRESS, help='Path to the training data directory and name of the .h5 file')
    parser.add_argument('--output_dir', type=str, default=argparse.SUPPRESS, help='Base output directory')
    parser.add_argument('--run_name', type=str, default=argparse.SUPPRESS, help='Name of the run folder to be created inside output_dir')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=argparse.SUPPRESS, help='Number of samples to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=argparse.SUPPRESS, help='Batch size for validation')
    parser.add_argument('--num_gpus', type=int, default=argparse.SUPPRESS, help='Number of GPUs to use')
    parser.add_argument('--disable_one_hot', action='store_true', default=argparse.SUPPRESS, help='Disable one-hot encoding')
    parser.add_argument('--validation_size', type=float, default=argparse.SUPPRESS, help='Validation split ratio')
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Base Defaults
    config = {
        "data_file": None,
        "output_dir": "outputs",
        "run_name": "default_run",
        "epochs": 50,
        "num_samples":None,
        "batch_size": 8,
        "val_batch_size": 8,
        "num_gpus": 1,
        "disable_one_hot": False,
        "validation_size": 0.1
    }
    
    # Override with JSON config if provided
    if 'config' in args_dict:
        json_config = load_config(args_dict['config'])
        config.update(json_config)
    
    # Override with CLI arguments
    for key, value in args_dict.items():
        if key != 'config':
            config[key] = value

    if not config.get('data_file'):
        parser.error("--data_file is required (via CLI or config file).")

    # Save settings to settings.json file in output directory
    run_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    settings_path = os.path.join(run_dir, 'settings.json')
    settings_path = os.path.join(config['output_dir'], 'settings.json')
    try:
        with open(settings_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to: {settings_path}")
    except Exception as e:
        print(f"Warning: Could not save settings.json. Error: {e}")
        
    return config

def validate_dataset(h5_path):
    """
    Validates the HDF5 dataset before training to catch corrupted or malformed files.
    
    Args:
        h5_path (str): Path to the HDF5 dataset.
        
    Raises:
        ValueError: If the dataset is invalid or corrupted.
    """
    print(f"\n--- Validating dataset: {h5_path} ---")
    try:
        with h5py.File(h5_path, 'r') as h5f:
            if 'facies' not in h5f:
                raise ValueError(f"'facies' dataset not found in {h5_path}")
            
            ds = h5f['facies']
            shape = ds.shape
            print(f"Dataset shape: {shape}")
            
            if len(shape) != 4:
                raise ValueError(f"Expected 4D dataset (N, Z, Y, X), got {len(shape)}D")
            
            # Check a sample for actual values to ensure it's not corrupted
            sample = ds[0]
            unique_vals = np.unique(sample)
            print(f"Unique facies values in first sample: {unique_vals}")
            
            if len(unique_vals) <= 1 and unique_vals[0] == 0:
                print("WARNING: The first sample contains only zeros! Checking another sample...")
                if shape[0] > 1:
                    sample2 = ds[1]
                    unique_vals2 = np.unique(sample2)
                    if len(unique_vals2) <= 1 and unique_vals2[0] == 0:
                        raise ValueError("Dataset appears to be corrupted (all zeros).")
                        
            print("Dataset validation passed!")
            print("-" * 35,"\n")
            
    except Exception as e:
        raise ValueError(f"Failed to validate dataset: {str(e)}")
