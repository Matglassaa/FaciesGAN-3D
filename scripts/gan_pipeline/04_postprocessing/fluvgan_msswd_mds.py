"""
=============================================
Evaluating GAN models using MS-SWD and MDS
=============================================

Architecture aligned with training_cgan.py:
             + nl = (3, 5, 5)
             + Double convolutions = True
             + Double residual blocks = True
             + Input dimensions: 32x128x128 (Z, X, Y)
             + Single-pass processing preserving full shape dimensions
             + Capped sample constraints via --max_samples flag

             
Run 1= /home/nfs/mtorresruhe/data/outputs/UPD_20000_samples/25_epochs_3_classes_cgan_num_iter_1_no_penalty_lr_gen_1e3_disc_3e3_doubleconv_on_setting_2/architecture_4_dcgan_samples_one_hot_epochs_25_bs_64_run_1.pt
Run 2= /home/nfs/mtorresruhe/data/outputs/UPD_20000_samples/25_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_25_bs_64_run_1.pt
Run 3= /home/nfs/mtorresruhe/data/outputs/UPD_20000_samples/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_100_bs_64_run_1.pt

Adapted for HPC cluster execution.
nohup python fluvgan_msswd_mds.py \
    --test_data_dir /home/nfs/mtorresruhe/data/datasets/testing_dataset_nexus_1000_samples_ntg_0.67_chdepth_6_isbx_100/samples/facies/ \
    --model_paths ~/data/outputs/UPD_20000_samples/25_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_25_bs_64_run_1.pt,~/data/outputs/UPD_20000_samples/25_epochs_3_classes_cgan_num_iter_1_no_penalty_lr_gen_1e3_disc_3e3_doubleconv_on_setting_2/architecture_4_dcgan_samples_one_hot_epochs_25_bs_64_run_1.pt \
    --output_dir /home/nfs/mtorresruhe/data/outputs/post_training_plots/combined_mds_analysis/ \
    --nc 3 \
    --batch_size 32 > msswd_eval.out 2>&1 &
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
from sklearn.manifold import MDS

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from voxgan.networks import resnet
from voxgan.data.datasets import Compose, Crop, FillNaN, Scale, ToTensor
from voxgan.models.metrics import MSSWD


################################################################################
# Custom Dataset replicating your training script's Facies mapping & One-Hot logic

class NativeNpyDataset(Dataset):
    def __init__(self, directory_path, num_classes=3, max_samples=None, transform=None):
        self.directory_path = os.path.expanduser(directory_path)
        all_files = sorted(glob.glob(os.path.join(self.directory_path, "*.npy")))
        
        # Apply the sample limit constraint here
        if max_samples is not None:
            self.file_paths = all_files[:max_samples]
            print(f"Capping dataset at user-specified maximum: {len(self.file_paths)} samples (out of {len(all_files)} total files).")
        else:
            self.file_paths = all_files
            
        self.transform = transform
        self.num_classes = num_classes
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in the directory: {self.directory_path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        raw_grid = np.load(file_path).astype(np.int64) 
        
        raw_grid = np.clip(raw_grid, 0, 12)
        
        # 1. Map raw facies codes (1-12) to grouped classes (0-2)
        grouped_grid = self.mapping[raw_grid] if hasattr(self, 'mapping') else self._setup_and_map(raw_grid)
        
        # 2. Convert grouped classes to one-hot layout -> Shape: (32, 128, 128, 3)
        one_hot_grid = np.eye(self.num_classes)[grouped_grid]
        
        # 3. Transpose to expected PyTorch layout -> Shape: (3, 32, 128, 128)
        one_hot_grid = np.transpose(one_hot_grid, (3, 0, 1, 2)).astype(np.float32)
        
        sample = {'data': one_hot_grid}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def _setup_and_map(self, raw_grid):
        # Fallback initialization of lookup array mapping rules
        self.mapping = np.zeros(13, dtype=np.int64)
        self.mapping[1:4] = 0   
        self.mapping[4:8] = 1   
        self.mapping[8:13] = 2  
        return self.mapping[raw_grid]


################################################################################
# Main Execution

def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN models using MS-SWD and compute MDS projections")
    parser.add_argument('--test_data_dir', type=str, required=True, help="Path to the directory containing real .npy facies files")
    parser.add_argument('--model_paths', type=str, required=True, help="Comma-separated list of paths to generator .pt checkpoints")
    parser.add_argument('--output_dir', type=str, default='../outputs/msswd_eval', help="Directory to save the CSVs")
    parser.add_argument('--nz', type=int, default=100, help="Latent vector size")
    parser.add_argument('--nc', type=int, default=3, help="Number of facies channels (e.g., 3)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for generating samples")
    parser.add_argument('--max_samples', type=int, default=100, help="Maximum number of real samples to load and compute MDS for")
    parser.add_argument('--random_seed', type=int, default=43, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_path = Path(args.output_dir)

    ################################################################################
    # Load Real Test Samples (Capped by max_samples)
    
    print(f"Loading real test data from: {args.test_data_dir}")
    
    transform = Compose([
        Crop(((0, 3), (0, 32), (0, 128), None)), 
        FillNaN(((0., 'max+1'), (0., 'max+1'), (0., 'max+1'))), 
        Scale(((0, 1), None)), 
        ToTensor()
    ])
    
    dataset = NativeNpyDataset(args.test_data_dir, num_classes=args.nc, max_samples=args.max_samples, transform=transform)
    n_real_samples = len(dataset)
    print(f"Processing full 32x128x128 target dimensions across {n_real_samples} loaded samples.")
    
    test_samples = torch.empty((n_real_samples, args.nc, 32, 128, 128), device=device)
    labels = []
    latents = []

    for j in tqdm(range(n_real_samples), desc="Processing Real Data"):
        sample = dataset[j]
        test_samples[j] = sample['data'].to(device)
        labels.append("Flumy_Real")
        latents.append(np.full(args.nz, np.nan))

    ################################################################################
    # Load Models & Generate Fake Samples
    
    model_paths = [p.strip() for p in args.model_paths.split(',')]
    print(f"\nDetected {len(model_paths)} models to evaluate.")
    
    all_samples = [test_samples]
    
    for m_idx, m_path in enumerate(model_paths):
        print(f"\nProcessing Model {m_idx + 1}/{len(model_paths)}: {m_path}")
        resolved_path = os.path.expanduser(m_path)
        
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
        model_name = Path(resolved_path).stem
        
        generator = resnet.DeepGenerator3d(nz=args.nz,
                                           ngf=64,
                                           nc=args.nc,
                                           nl=(3, 5, 5),
                                           max_factor=16,
                                           residual_weight=1.,
                                           mode='nearest',
                                           kernel_size=3,
                                           layer_normalization=nn.BatchNorm3d,
                                           last_layer_normalization=nn.BatchNorm3d,
                                           weight_normalization=nn.utils.parametrizations.spectral_norm,
                                           activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                           last_activation=nn.Tanh,
                                           use_double_conv=True,
                                           use_double_resblocks=False,
                                           use_attention=False,
                                           skip_z=False,
                                           split_z=False)
        
        gen_state = {k.replace('module.', ''): v for k, v in checkpoint['generator'].items()}
        generator.load_state_dict(gen_state)
        
        generator.to(device).eval()
        for p in generator.parameters(): p.requires_grad = False
        
        fake_samples = torch.empty((n_real_samples, args.nc, 32, 128, 128), device=device)
        
        with torch.no_grad():
            for j in tqdm(range(0, n_real_samples, args.batch_size), desc=f"Generating Fake Samples"):
                curr_bs = min(args.batch_size, n_real_samples - j)
                z = torch.randn((curr_bs, args.nz), device=device)
                
                for z_vec in z.cpu().numpy():
                    latents.append(z_vec)
                    labels.append(f"Model_{m_idx+1}_{model_name}")
                
                generated = generator(z.view(curr_bs, args.nz, 1, 1, 1))
                fake_samples[j:j + curr_bs] = generated
                
        all_samples.append(fake_samples)

    combined_samples = torch.cat(all_samples, dim=0)
    total_n = combined_samples.shape[0]
    print(f"\nTotal combined samples for MS-SWD: {total_n}")

    ################################################################################
    # MS-SWD Pairwise Computation
    
    print("Initializing MS-SWD Metric...")
    ms_swd = MSSWD(n_levels=3,
                   n_descriptors=512,
                   descriptor_size=(4, 7, 7),
                   n_repeat=12,
                   n_proj=128,
                   padding_mode='circular',
                   combine_levels=True,
                   n_gpu=torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    distances = np.zeros((total_n, total_n))
    
    print("Computing MS-SWD Pairwise Distances...")
    with torch.no_grad():
        for j in tqdm(range(total_n), desc="Row-wise MS-SWD"):
            for k in range(j + 1, total_n):
                distance = ms_swd(combined_samples[j:j + 1], combined_samples[k:k + 1])
                dist_val = distance.item() if isinstance(distance, torch.Tensor) else distance
                distances[j, k] = dist_val
                distances[k, j] = dist_val

    ################################################################################
    # MDS Computation
    
    print("Computing 2D MDS Spatial Mapping...")
    reducer = MDS(n_components=2,
                  n_jobs=-1,
                  random_state=args.random_seed,
                  dissimilarity='precomputed')
                  
    coords_2d = reducer.fit_transform(distances)
    print(f"MDS Stress (lower is better): {reducer.stress_:.4f}")

    ################################################################################
    # Saving Results
    
    print(f"Saving outputs to {args.output_dir}...")
    
    df_coords = pd.DataFrame({
        'Label': labels,
        'Dimension_1': coords_2d[:, 0],
        'Dimension_2': coords_2d[:, 1]
    })
    coords_path = output_dir_path / 'mds_2d_coordinates.csv'
    df_coords.to_csv(coords_path, index=False)
    print(f"Saved 2D coordinates to: {coords_path}")
    
    latent_columns = [f'Latent_{i+1}' for i in range(args.nz)]
    df_embeddings = pd.DataFrame(latents, columns=latent_columns)
    df_embeddings.insert(0, 'Label', labels)
    df_embeddings['Sample_Index'] = range(total_n)
    
    embed_path = output_dir_path / 'latent_embeddings.csv'
    df_embeddings.to_csv(embed_path, index=False)
    print(f"Saved embeddings to: {embed_path}")
    
    dist_path = output_dir_path / 'msswd_distance_matrix.csv'
    pd.DataFrame(distances).to_csv(dist_path, index=False, header=False)
    
    print("\nEvaluation Complete!")

if __name__ == '__main__':
    main()