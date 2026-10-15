import h5py
import pathlib
import numpy as np
import os

base = pathlib.Path("C:/Users/mathi/Downloads")

f = h5py.File(base / "training_dataset_nexus_20000_samples.h5", "r")

print(f"dataset keys: {f.keys()}")
print(f"dataset age shape: {f['age'].shape}")
print(f"dataset facies shape: {f['facies'].shape}")