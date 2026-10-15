import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from voxgan.networks import resnet
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Loss and Generate Realizations")
    parser.add_argument('--csv_path', type=str, default='outputs/2000_training_samples/RUN_2000_samples_128xy_dataset_50_epochs/fluvgan_1_training_1_architecture_4_dcgan_no_one_hot_1_history.csv', help='Path to the history CSV for plotting losses')
    parser.add_argument('--ckpt_path', type=str, default=r'C:\Users\mathi\Desktop\TU Delft\TU Delft year 5\Master Thesis\Thesis-project-DGM\outputs\post_training\20000_training_samples\25_epochs_9_classes_cgan_lr_1e3_gen_3e3_disc_doubleconv_doubleresblock_on_penalty_every_4_iter_setting_2\architecture_4_dcgan_samples_one_hot_all_epochs_25_bs_64_run_1.pt', help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/post_training/20000_training_samples/25_epochs_9_classes_cgan_lr_1e3_gen_3e3_disc_doubleconv_doubleresblock_on_penalty_every_4_iter_setting_2', help='Output folder')
    parser.add_argument('--num_reals', type=int, default=100, help='Number of realizations to generate')

    return parser.parse_args()

def plot_losses(csv_path, output_dir):

    if not os.path.exists(csv_path):
        print(f"Loss CSV not found at '{csv_path}'. Skipping loss visualization.")
        return

    print(f"Plotting losses from {csv_path}...")
    loss = pd.read_csv(csv_path)
    x = loss["iteration"]

    plt.figure(figsize=(12, 10))

    # Subplot 1: Generator and Discriminator Loss
    plt.subplot(3, 1, 1)
    if "loss_discriminator" in loss.columns and "loss_generator" in loss.columns:
        plt.plot(x, loss["loss_discriminator"], label="Discriminator loss")
        plt.plot(x, loss["loss_generator"], label="Generator loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    # Subplot 2: D(x) and D(G(z)) Tracking
    plt.subplot(3, 1, 2)
    if "D(x)" in loss.columns and "D(G(z))" in loss.columns and "D_up(G(z))" in loss.columns:
        plt.plot(x, loss["D(x)"], label="D(x)")
        plt.plot(x, loss["D(G(z))"], label="D(G(z))")
        plt.plot(x, loss["D_up(G(z))"], label="D_up(G(z))")
    plt.legend()
    plt.ylabel("D(x) and D(G(z))")
    plt.xlabel("Iteration")

    plt.subplot(3, 1, 3)
    ms_swd_cols = [col for col in loss.columns if "MS-SWD" in col]
    for i, col in enumerate(ms_swd_cols):
        loss_valid = loss[loss[col].notna()]
        if not loss_valid.empty:
            plt.plot(loss_valid["iteration"], loss_valid[col], marker='.', label=f"MS-SWD (Channel {i})")
    
    if ms_swd_cols:
        plt.legend()
    plt.ylabel("MS-SWD")
    plt.xlabel("Iteration")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "loss_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to: {save_path}")

def generate_realizations(ckpt_path, output_dir, nc=9, nl = (3,5,5), num_realizations=100):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at '{ckpt_path}'. Skipping generation.")
        return
    output_dir = os.path.join(output_dir,'realizations')
    os.makedirs(output_dir, exist_ok=True)

    last_activation = nn.Tanh
    nz = 100
    ngf = 64
    max_factor = 16

    print("Building Generator...")
    gen_layer = resnet.DeepGenerator3d(
        nz=nz, ngf=ngf, nc=nc, nl=nl, max_factor=max_factor, residual_weight=1.,
        mode='nearest', kernel_size=3, layer_normalization=nn.BatchNorm3d,
        last_layer_normalization=nn.BatchNorm3d,
        weight_normalization=nn.utils.parametrizations.spectral_norm,
        activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        last_activation=last_activation, use_double_conv=True, use_double_resblocks=True,
        use_attention=False, skip_z=False, split_z=False
    )

    print(f"Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gen_state = checkpoint['generator'] if 'generator' in checkpoint else checkpoint
    if 'state_dict' in gen_state:
        gen_state = gen_state['state_dict']

    gen_layer = nn.DataParallel(gen_layer)
    gen_layer.load_state_dict(gen_state, strict=False)
    gen_layer = gen_layer.module
    gen_layer.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_layer = gen_layer.to(device)
    print("Success! Weights are perfectly aligned.")

    print(f"Generating {num_realizations} 3D River Block Realizations...")
    for i in range(num_realizations):
        with torch.no_grad():
            z = torch.randn(1, nz).to(device)
            z_input = z.view(z.shape + (1, 1, 1))
            output = gen_layer(z_input)
            
            sample = output['data'].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
            
            if nc > 1:
                volume = np.argmax(sample[0], axis=0).astype(np.uint8)
            else:
                volume = sample[0, 0, :, :, :]

            out_path = os.path.join(output_dir, f"realization_{i+1:02d}.npy")
            np.save(out_path, volume)
            print(f"Generated realization {i+1}/{num_realizations} -> {out_path}. Data shape= {volume.shape}")

def main():
    args = parse_args()
    #plot_losses(args.csv_path, args.output_dir)
    generate_realizations(ckpt_path=args.ckpt_path, output_dir=args.output_dir, num_realizations=args.num_reals)

if __name__ == '__main__':
    main()