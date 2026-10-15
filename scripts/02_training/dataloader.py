import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    """ 
    A PyTorch Dataset for loading and processing facies data from an HDF5 file.

    This dataset handles loading 3D or 2D facies arrays from an HDF5 file, mapping their raw integer values to defined classes, and applying preprocessing 
    such as one-hot encoding and scaling. It also supports preloading data into RAM to minimize disk I/O bottlenecks during training.

    Attributes:
        h5_path (str): Path to the HDF5 file containing the data.
        nz (int): Slicing parameter, typically determining the number of depth slices loaded.
        use_one_hot (bool): Indicates if the loaded data is converted to a one-hot representation.
        preload_ram (bool): Indicates if the dataset slices are preloaded into memory.
        length (int): Total number of samples along the primary axis of the dataset.
        data_cache (numpy.ndarray): In-memory cache of the data (if preload_ram is True).
        mapping (numpy.ndarray): Look-up array for fast forward mapping of facies values.
        reverse_mapping (dict): Dictionary mapping new values back to a representative raw value.
    """
    def __init__(self, h5_path, num_samples, nz=32, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, one_hot_all=False, preload_ram=True, **kwargs):
        self.h5_path = h5_path
        self.nz = nz
        self.use_one_hot = use_one_hot
        self.one_hot_all = one_hot_all
        self.num_classes = 9 if one_hot_all else 3  # Dynamically set the class count
        self.preload_ram = preload_ram
        
        with h5py.File(self.h5_path, 'r') as h5f:
            if 'facies' not in h5f:
                raise KeyError(f"Dataset 'facies' not found in {self.h5_path}")
            
            total_length = h5f['facies'].shape[0]
            self.length = min(num_samples, total_length) if num_samples is not None else total_length
            
            if self.preload_ram:
                print(f"Loading {self.length} samples into RAM. Please wait...")
                self.data_cache = h5f['facies'][:self.length, -nz:, :, :]
                num_loaded, depth, height, width = self.data_cache.shape
                print(f"Loading {num_loaded} samples into RAM complete!\n")
        
        # Define mappings   
        if facies_mapping is None:
            if self.one_hot_all:
                # --- NEW 9-CLASS MAPPING ---
                # Raw Flumy facies are 1-9, so we map them to 0-8 for one-hot encoding
                self.mapping = np.zeros(10, dtype=np.int64)
                self.mapping[1:10] = np.arange(9)
                self.reverse_mapping = {i: i+1 for i in range(9)}
            else:
                # --- ORIGINAL 3-CLASS MAPPING ---
                self.mapping = np.zeros(13, dtype=np.int64)
                self.mapping[1:4] = 0   
                self.mapping[4:8] = 1   
                self.mapping[8:13] = 2  
                self.reverse_mapping = {0: 1, 1: 4, 2: 8}
        else:
            max_val = max(facies_mapping.keys())
            self.mapping = np.zeros(max_val + 1, dtype=np.int64)
            self.reverse_mapping = {}
            for raw_val, new_val in facies_mapping.items():
                self.mapping[raw_val] = new_val
                if new_val not in self.reverse_mapping:
                    self.reverse_mapping[new_val] = raw_val

        if save_mapping_dir:
            self._save_mapping(save_mapping_dir)

    def _save_mapping(self, save_dir):
        """
        Saves the facies mapping configuration to a JSON file.

        Creates the directory if it does not exist and writes the forward mapping array,
        representative reverse mapping, and one-hot configuration to 'facies_mapping_config.json'.

        Args:
            save_dir (str): The directory where the configuration file will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "facies_mapping_config.json")
        config = {
            "forward_mapping_array": self.mapping.tolist(),
            "representative_reverse_mapping": self.reverse_mapping,
            "used_one_hot": self.use_one_hot
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves and processes a specific sample from the dataset.

        Fetches the raw data slice either from the RAM cache or directly from disk,
        applies the defined facies mapping, and formats the output tensor (scaling and/or 
        one-hot encoding based on initialization parameters).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed PyTorch tensor under the key 'data'.
        """
        if self.preload_ram:
            data = self.data_cache[idx]
        else:
            with h5py.File(self.h5_path, 'r') as h5f:
                data = h5f['facies'][idx]
        
        mapped_data = self.mapping[data]
        tensor_data = torch.from_numpy(mapped_data)
        
        if self.use_one_hot:
            processed_data = F.one_hot(tensor_data, num_classes=self.num_classes).permute(3, 0, 1, 2).float()
            processed_data = (processed_data * 2.0) - 1.0
        else:
            processed_data = tensor_data.unsqueeze(0).float()
            processed_data = processed_data - 1.0

        return {'data': processed_data}