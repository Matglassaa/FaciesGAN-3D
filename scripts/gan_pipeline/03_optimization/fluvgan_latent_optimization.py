"""
==============================================
GAN inversion using latent-vector optimization
==============================================

Architecture DCGAN (4)
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator

Example Run:
nohup python fluvgan_2_inversion_3_prior-art_1.py \
    --model_path ~/data/outputs/UPD_20000_samples/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_100_bs_64_run_1.pt \
    --well_data_path ~/data/datasets/well_data/Well_data.xlsx \
    --output_dir ~/data/outputs/post_optimization_results/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2 \
    --n_samples 100 --steps 3000 > inversion.out 2>&1 &
"""

import os   
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from voxgan.networks import resnet

################################################################################
# Functions

class MultiChannelContextLoss(_Loss):
    def __init__(self, data_indices, data_values, threshold, shape, nc, spacing=None, p=1, device=None):
        super(MultiChannelContextLoss, self).__init__(None, None, 'mean')
        self.nc = nc
        
        if spacing is None:
            spacing = (5.0, 1.0, 1.0) 

        dist_input = torch.ones(shape, dtype=int)
        dist_input[data_indices[0], data_indices[1], data_indices[2]] = 0
        distance, indices = distance_transform_edt(dist_input.cpu(), sampling=spacing, return_indices=True)

        sigma = threshold
        self._mask = np.exp(-(distance**2) / (2 * sigma**2))
        self._mask[distance > (3 * sigma)] = 0.
        self._mask = torch.tensor(self._mask, device=device).float()

        self._target = torch.empty((nc, *shape), device=device)        
        num_points = data_values.size(0)

        one_hot_vals = torch.full((num_points, nc), -1.0, device=device)
        one_hot_vals[torch.arange(num_points), data_values.long()] = 1.0
            
        for c in range(nc):
            chan_target = torch.zeros(shape, device=device)
            chan_target[data_indices[0], data_indices[1], data_indices[2]] = one_hot_vals[:, c].float()
            self._target[c] = chan_target[indices[0], indices[1], indices[2]]

        self._p = torch.abs if p == 1 else partial(torch.pow, exponent=p)

    def forward(self, input):
        # input: (B, C, Z, Y, X), self._target: (C, Z, Y, X)
        diff = self._p(input - self._target)
        masked_diff = self._mask * diff
            
        return torch.sum(masked_diff)
    # def forward(self, input):
    #     # input: (B, C, Z, Y, X), self._target: (C, Z, Y, X)
    #     diff = self._p(input - self._target)
    #     masked_diff = self._mask * diff
        
    #     mask_sum = torch.sum(self._mask)
    #     if mask_sum > 0:
    #         return torch.sum(masked_diff) / mask_sum
    #     else:
    #         return torch.sum(masked_diff)

def map_facies(val):
    """Maps raw facies values to 3 classes (0=Channel, 1=Levee, 2=Overbank)."""
    if val == 1: return 0  # Maps to Channel
    if val == 2: return 1  # Maps to Levee
    if val == 3: return 2  # Maps to Overbank
    return 0 # Default fallback

def main():
    parser = argparse.ArgumentParser(description="GAN inversion for well conditioning")
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument('--well_data_path', type=str, required=True, help="Path to Well_data.xlsx")
    parser.add_argument('--output_dir', type=str, default='../outputs/inversion', help="Output directory")
    parser.add_argument('--nz', type=int, default=100, help="Latent vector size")
    parser.add_argument('--nc', type=int, default=3, help="Number of facies channels")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of realizations to optimize")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for latent optimization")
    parser.add_argument('--steps', type=int, default=3000, help="Number of optimization steps")
    parser.add_argument('--threshold', type=float, default=1.0, help="Distance threshold for context loss")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ################################################################################
    # Model Setup
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    nl = (3, 5, 5) # Matches training script
    
    generator = resnet.DeepGenerator3d(nz=args.nz,
                                       ngf=64,
                                       nc=args.nc,
                                       nl=nl,
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
                                       use_double_resblocks=True,
                                       use_attention=False,
                                       skip_z=False,
                                       split_z=False)
    
    discriminator = resnet.DeepDiscriminator3d(ndf=64,
                                               nc=args.nc,
                                               nl=nl,
                                               max_factor=16,
                                               residual_weight=1.,
                                               kernel_size=3,
                                               layer_normalization=None,
                                               weight_normalization=nn.utils.parametrizations.spectral_norm,
                                               activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                               use_double_conv=True,
                                               use_double_resblocks=True,
                                               use_attention=False)

    # Handle DataParallel if needed
    gen_state = {k.replace('module.', ''): v for k, v in checkpoint['generator'].items()}
    disc_state = {k.replace('module.', ''): v for k, v in checkpoint['discriminator'].items()}
    
    generator.load_state_dict(gen_state)
    discriminator.load_state_dict(disc_state)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator.to(device).eval()
    discriminator.to(device).eval()
    
    for p in generator.parameters(): p.requires_grad = False
    for p in discriminator.parameters(): p.requires_grad = False

    ################################################################################
    # Well Data Processing
    
    print("Loading well data...")
    # Map coordinates to grid indices
    # Constants derived from project notebooks
    ORIGIN_E = 84337.0  # (86237.0 - 1900)
    ORIGIN_N = 445750.0 # (446550.0 - 800)
    SPACING = 20.0
    
    tabs = ['DEL-GT-01', 'DEL-GT-02-S2']
    all_indices = []
    all_values = []
    debug_records = [] # Container to store details for validation printing
    
    for tab in tabs:
        df = pd.read_excel(args.well_data_path, sheet_name=tab)
        df_32 = df.head(32) # Take first 32 meters
        
        # Facies column might have a trailing space in some versions
        facies_col = 'Facies ' if 'Facies ' in df.columns else 'Facies'
        
        for i, row in df_32.iterrows():
            z = i # meter by meter
            y = int(np.round((row['GRID N'] - ORIGIN_N) / SPACING))
            x = int(np.round((row['GRID E'] - ORIGIN_E) / SPACING))
            
            # Ensure within grid bounds (128x128)
            if 0 <= x < 128 and 0 <= y < 128:
                raw_val = row[facies_col]
                mapped_val = map_facies(raw_val)
                
                all_indices.append([z, y, x])
                all_values.append(mapped_val)
                debug_records.append({
                    'well': tab,
                    'z': z, 'y': y, 'x': x,
                    'raw_value': raw_val,
                    'mapped_value': mapped_val
                })

    X = torch.tensor(all_indices).T # (3, N)
    y = torch.tensor(all_values)    # (N,)
    print(f"Conditioning on {len(all_values)} data points.")

    print("\n" + "="*85)
    print("                      WELL CONDITIONING VERIFICATION REPORT")
    print("="*85)
    print(f"{'Well Tab':<15} | {'Grid Coordinate (Z, Y, X)':<27} | {'Raw Facies':<12} | {'Mapped Class (ID)':<18}")
    print("-"*85)
    for rec in debug_records:
        class_name = {0: "Channel (0)", 1: "Levee (1)", 2: "Overbank (2)"}.get(rec['mapped_value'], "Unknown")
        coord_str = f"({rec['z']}, {rec['y']}, {rec['x']})"
        print(f"{rec['well']:<15} | {coord_str:<27} | {rec['raw_value']:<12.1f} | {class_name:<18}")
    
    print("-"*85)
    # Class distribution summary per well
    summary_df = pd.DataFrame(debug_records)
    print("Class distribution summary:")
    for well_name in tabs:
        well_subset = summary_df[summary_df['well'] == well_name]
        counts = well_subset['mapped_value'].value_counts()
        print(f"  * {well_name:<15} -> Channel (0): {counts.get(0, 0):<3} | Levee (1): {counts.get(1, 0):<3} | Overbank (2): {counts.get(2, 0):<3}")
    
    total_counts = summary_df['mapped_value'].value_counts()
    print(f"  * {'TOTAL':<15} -> Channel (0): {total_counts.get(0, 0):<3} | Levee (1): {total_counts.get(1, 0):<3} | Overbank (2): {total_counts.get(2, 0):<3}")
    print("="*85 + "\n")
    # ==============================================================================

    ################################################################################
    # Inference / Optimization
    
    z_optimized = torch.empty((args.n_samples, args.nz))
    history = dict()
    
    loss_fn_prior = nn.BCEWithLogitsLoss()
    label_real = torch.tensor([1.0], device=device)
    
    # Grid shape for context loss (Z, Y, X)
    grid_shape = (32, 128, 128)
    loss_fn_context = MultiChannelContextLoss(X, y, args.threshold, grid_shape, args.nc, device=device)

    for i in range(args.n_samples):
        print(f"Optimizing realization {i+1}/{args.n_samples}...")
        history[f'loss_{i+1}'] = []

        # Initialize latent vector
        _z = torch.randn(1, args.nz, 1, 1, 1, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([_z], lr=args.lr)

        pbar = tqdm(range(args.steps), leave=False)
        for step in pbar:
            optimizer.zero_grad()
            
            samples = generator(_z) # (1, nc, 32, 128, 128)
            proba = discriminator(samples)
            
            l_context = loss_fn_context(samples)
            l_prior = loss_fn_prior(proba['data'], label_real.expand_as(proba['data']))
            
            # Use the correct balanced weight strategy
            loss = 10.0 * l_context + 1.0 * l_prior
            
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            history[f'loss_{i+1}'].append(loss_val)
            if step % 100 == 0:
                pbar.set_description(f"Loss: {loss_val:.4f}")

        z_optimized[i] = _z.detach().cpu().view(-1)

    ################################################################################
    # Saving Results
    
    print(f"Saving results to {args.output_dir}...")
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, 'inversion_history.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'optimized_z.npy'), z_optimized.numpy())
    
    # Generate and save final realizations
    with torch.no_grad():
        for i in range(args.n_samples):
            _z = z_optimized[i].view(1, args.nz, 1, 1, 1).to(device)
            sample = generator(_z)
            # Rescale from [-1, 1] to [0, 1] and save as numpy
            sample_np = (0.5 * sample + 0.5).cpu().numpy()[0]
            np.save(os.path.join(args.output_dir, f'realization_{i+1}.npy'), sample_np)

    print("Inversion complete!")

if __name__ == '__main__':
    main()