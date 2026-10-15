"""
==================================
GAN inversion using pivotal tuning
==================================

Architecture DCGAN (4)
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator
             + R1 regularization

Adapted for cluster execution and Excel-based well data.

Example Run with Single Latents:
nohup python fluvgan_pivotal_tuning.py \
    --model_path ~/data/outputs/UPD_20000_samples/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_100_bs_64_run_1.pt \
    --optimized_z_path ~/data/outputs/post_optimization_results/Run_1_3_model_5_adjusted_loss_3000_steps_threshold_2/optimized_z.npy \
    --well_data_path ~/data/datasets/well_data/Well_data.xlsx \
    --output_dir ~/data/outputs/post_inversion_results/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/run_2/ \
    --batch_size 8 --steps 1000 > editing_th2.out 2>&1 &

Example Run with Combined Latents:
nohup python fluvgan_pivotal_tuning.py \
    --model_path ~/data/outputs/UPD_20000_samples/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/architecture_4_dcgan_samples_one_hot_epochs_100_bs_64_run_1.pt \
    --optimized_z_path ~/data/outputs/post_optimization_results/Run_1_1_model_5_adjusted_loss_3000_steps_threshold_1/optimized_z.npy, ~/data/outputs/post_optimization_results/Run_1_3_model_5_adjusted_loss_3000_steps_threshold_2/optimized_z.npy, ~/data/outputs/post_optimization_results/Run_1_2_model_5_adjusted_loss_3000_steps_threshold_5/optimized_z.npy\
    --well_data_path ~/data/datasets/well_data/Well_data.xlsx \
    --output_dir ~/data/outputs/post_inversion_results/100_epochs_3_classes_cgan_lr_gen_1e3_disc_3e3_doubleconv_doubleresblock_on_penalty_every_16_iter_setting_2/ \
    --batch_size 64 --steps 1000 > editing_th_combined.out 2>&1 &

"""

import os
import re
import json
import argparse
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from voxgan.networks import resnet

################################################################################
# Functions

class LPIPSLoss(_Loss):
    """
    LPIPS loss with a 3D discriminator.
    """
    def __init__(self, discriminator, reduction='mean', eps=1e-12):
        super(LPIPSLoss, self).__init__(None, None, None)
        self.discriminator = discriminator
        self.reduction = torch.mean if reduction == 'mean' else (torch.sum if reduction == 'sum' else lambda x: x)
        self.eps = eps

    def forward(self, input, target):
        loss = 0.
        disc = self.discriminator.module if isinstance(self.discriminator, nn.DataParallel) else self.discriminator
        
        curr_input = input
        curr_target = target
        
        for i in range(len(disc.main) - 1):
            curr_input = disc.main[i](curr_input)
            curr_target = disc.main[i](curr_target)
            
            # Normalize ONLY for the distance calculation
            norm_input = F.normalize(curr_input, eps=self.eps, dim=1)
            norm_target = F.normalize(curr_target, eps=self.eps, dim=1)
            
            loss += torch.mean(torch.sum((norm_input - norm_target)**2, 1), (1, 2, 3))

        return self.reduction(loss)

def map_facies(val):
    """Maps raw facies values to 3 classes (0=Channel, 1=Levee, 2=Overbank)."""
    if val == 1: return 0  # Maps to Channel
    if val == 2: return 1  # Maps to Levee
    if val == 3: return 2  # Maps to Overbank
    return 0 # Default fallback

def save_hyperparameters(args, final_output_dir, debug_records, loaded_paths=None):
    """Saves a json metadata record of the tuning setup and class distributions."""
    mapped_vals = [r['mapped_value'] for r in debug_records]
    counts = Counter(mapped_vals)
    class_dist = {
        "Channel (0)": counts.get(0, 0),
        "Levee (1)": counts.get(1, 0),
        "Overbank (2)": counts.get(2, 0)
    }
    
    metadata = {
        "model_path": args.model_path,
        "optimized_z_path": args.optimized_z_path,
        "loaded_latent_files": loaded_paths if loaded_paths else [args.optimized_z_path],
        "well_data_path": args.well_data_path,
        "output_dir": args.output_dir,
        "final_output_dir": final_output_dir,
        "nz": args.nz,
        "nc": args.nc,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "steps": args.steps,
        "total_conditioning_points": len(debug_records),
        "well_class_distribution": class_dist,
        "geological_mapping": {
            "0": "Channel",
            "1": "Levee",
            "2": "Overbank"
        }
    }
    
    json_path = os.path.join(final_output_dir, "tuning_hyperparameters.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved run configuration metadata to: {json_path}")

################################################################################
# Main Execution

def main():
    parser = argparse.ArgumentParser(description="GAN inversion using pivotal tuning (editing)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument('--optimized_z_path', type=str, required=True, 
                        help="Path to optimized_z.npy. Supports multiple comma-separated files (e.g. z1.npy,z2.npy,z3.npy)")
    parser.add_argument('--well_data_path', type=str, required=True, help="Path to Well_data.xlsx")
    parser.add_argument('--output_dir', type=str, default='../outputs/editing', help="Output directory")
    parser.add_argument('--nz', type=int, default=100, help="Latent vector size")
    parser.add_argument('--nc', type=int, default=3, help="Number of facies channels")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for tuning")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate for generator tuning")
    parser.add_argument('--steps', type=int, default=1000, help="Number of tuning steps")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Automate output directory structure
    model_parent = Path(args.model_path).parent.name
    final_output_dir = os.path.join(args.output_dir, model_parent)
    realizations_dir = os.path.join(final_output_dir, 'realizations')
    os.makedirs(realizations_dir, exist_ok=True)

    ################################################################################
    # Model Setup
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    nl = (3, 5, 5)
    
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
                                       activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=False),
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
                                               activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=False),
                                               use_double_conv=True, 
                                               use_double_resblocks=True,
                                               use_attention=False)

    # Clean state dicts
    gen_state = {k.replace('module.', ''): v for k, v in checkpoint['generator'].items()}
    disc_state = {k.replace('module.', ''): v for k, v in checkpoint['discriminator'].items()}
    
    generator.load_state_dict(gen_state)
    discriminator.load_state_dict(disc_state)
    
    generator.to(device).eval()
    discriminator.to(device).eval()
    
    for p in generator.parameters(): p.requires_grad = False
    for p in discriminator.parameters(): p.requires_grad = False

    ################################################################################
    # Data Processing
    
    print("Loading optimized latent vectors...")
    # Support comma-separated paths for combining multiple runs (e.g., 3 runs of 100 realizations)
    paths = [p.strip() for p in args.optimized_z_path.split(',')]
    latents_list = []
    
    for path in paths:
        # Resolve user home directory tildes cleanly (e.g. ~/data)
        resolved_path = os.path.expanduser(path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Optimized latent file not found at: {resolved_path}")
        print(f"  Loading: {resolved_path}")
        latents_list.append(np.load(resolved_path))
        
    # Concatenate along the sample axis (axis 0)
    p_z = np.concatenate(latents_list, axis=0)
    print(f"Combined {len(paths)} latent files. Total combined shape: {p_z.shape}")
    
    p_z = torch.tensor(p_z, device=device).float()
    n_samples = p_z.shape[0]

    print("Loading well data...")
    ORIGIN_E = 84337.0
    ORIGIN_N = 445750.0
    SPACING = 20.0
    
    tabs = ['DEL-GT-01', 'DEL-GT-02-S2']
    all_indices = []
    all_values = []
    debug_records = []
    
    for tab in tabs:
        df = pd.read_excel(args.well_data_path, sheet_name=tab)
        df_32 = df.head(32)
        facies_col = 'Facies ' if 'Facies ' in df.columns else 'Facies'
        
        for i, row in df_32.iterrows():
            z = i
            y = int(np.round((row['GRID N'] - ORIGIN_N) / SPACING))
            x = int(np.round((row['GRID E'] - ORIGIN_E) / SPACING))
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

    # --- FIX: Moved the spatial coordinates safely to the matching torch GPU device ---
    X = torch.tensor(all_indices, device=device).T # (3, N)
    print(f"well data shape: {X.shape}")
    print(f"Well data coordinates: \n{X}")
    
    # --- FIX: Bulletproof GPU-based target construction using advanced indexing ---
    num_points = len(all_values)
    y_vals_tensor = torch.tensor(all_values, dtype=torch.long, device=device)
    one_hot_targets = torch.full((num_points, args.nc), -1.0, device=device)
    one_hot_targets[torch.arange(num_points), y_vals_tensor] = 1.0
    one_hot_targets = one_hot_targets.T # (nc, N)

    # Save execution settings & logs
    save_hyperparameters(args, final_output_dir, debug_records, loaded_paths=paths)

    ################################################################################
    # Pivotal Tuning
    
    print(f"Starting pivotal tuning for {args.steps} steps...")
    tuned_generator = copy.deepcopy(generator)
    tuned_generator.requires_grad_(True)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for tuning!")
        tuned_generator = nn.DataParallel(tuned_generator)
        dist_discriminator = nn.DataParallel(discriminator)
    else:
        dist_discriminator = discriminator

    optimizer = torch.optim.Adam(tuned_generator.parameters(), lr=args.lr)
    
    loss_fn_tuning = nn.L1Loss()        
    loss_fn_reg_l2 = nn.MSELoss()       
    loss_fn_reg_lpips = LPIPSLoss(dist_discriminator)

    history = {'loss': []}
    pbar = tqdm(range(args.steps))
    
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.
        
        # Batch through the optimized z samples
        for i in range(0, n_samples, args.batch_size):
            s = slice(i, i + args.batch_size)
            curr_batch_size = p_z[s].shape[0]
            
            # 1. Tuning Loss: Match optimized z to well data
            samples = tuned_generator(p_z[s].view(curr_batch_size, args.nz, 1, 1, 1))
            # extracted: (B, nc, N)
            extracted = samples[:, :, X[0], X[1], X[2]]
            loss_tuning = loss_fn_tuning(extracted, one_hot_targets.expand(curr_batch_size, -1, -1))
            
            # 2. Regularization Loss: Keep the model near original for random z
            # Sample z near the pivot
            z_rand = torch.randn(curr_batch_size, args.nz, device=device)
            # Distance controlled interpolation (Roich et al. 2021 style)
            z_pivot = p_z[s]
            dist = torch.linalg.norm(z_rand - z_pivot, dim=1, keepdim=True)
            z_interp = z_pivot + 30. * (z_rand - z_pivot) / dist
            
            with torch.no_grad():
                samples_orig = generator(z_interp.view(curr_batch_size, args.nz, 1, 1, 1))
            
            samples_tuned = tuned_generator(z_interp.view(curr_batch_size, args.nz, 1, 1, 1))
            
            loss_reg_l2 = loss_fn_reg_l2(samples_tuned, samples_orig)
            loss_reg_lpips = loss_fn_reg_lpips(samples_tuned, samples_orig)
            
            # weighting from Roich et al. (2021) adapted for this setup
            loss = loss_tuning + 0.1 * (loss_reg_lpips + 1.0 * loss_reg_l2)
            
            loss = loss * (curr_batch_size / n_samples)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        history['loss'].append(total_loss)
        if step % 10 == 0:
            pbar.set_description(f"Loss: {total_loss:.4f}")

    ################################################################################
    # Saving Results
    
    print(f"Saving results to {final_output_dir}...")
    pd.DataFrame(history).to_csv(os.path.join(final_output_dir, 'tuning_history.csv'), index=False)
    
    # Save tuned model
    save_model = tuned_generator.module if isinstance(tuned_generator, nn.DataParallel) else tuned_generator
    torch.save(save_model.state_dict(), os.path.join(final_output_dir, 'tuned_generator.pt'))
    
    # Generate and save final realizations
    with torch.no_grad():
        save_model.eval()
        for i in range(n_samples):
            _z = p_z[i].view(1, args.nz, 1, 1, 1)
            sample = save_model(_z)
            sample_np = (0.5 * sample + 0.5).cpu().numpy()[0]
            np.save(os.path.join(realizations_dir, f'realization_{i+1}.npy'), sample_np)

    print("Editing complete!")

if __name__ == '__main__':
    main()