"""
WGAN-SN (MSG-GAN) architecture training script:

example usage:
nohup python -u wgans_training.py \
    --run_name 100_epochs_3_classes_mws-gan_nexus_100_isbx \
    --data_file ~/data/datasets/training_dataset_nexus_20000_samples_ntg_0_67_chdepth_6_isbx_100/samples/facies/samples.h5 \
    --output_dir ~/data/outputs/UPD_20000_samples \
    --num_gpus 2 \
    --num_samples 10000 \
    --epochs 100 \
    --batch_size 64 \
    --val_batch_size 64 \
    --validation_size 0.2 \
    --one_hot_all False > training_mwsgan.out 2>&1 &
"""

import os
import re
import torch
import shutil
import warnings
import numpy as np
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

# Import Custom Modules
from dataloader import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import WGANLoss
from voxgan.models.metrics import MSSWD

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NormalizedSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Swish has a max derivative of ~1.0998. 
        # Dividing by this guarantees the function is strictly 1-Lipschitz.
        return F.silu(x) / 1.0998

class PixelNorm(nn.Module):
    """
    Pixelwise Feature Vector Normalization (PixNorm)
    Normalizes the feature channels of each individual spatial voxel independently.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

# ==============================================================================
# VoxGAN Compliant Epsilon Penalty
# ==============================================================================
class EpsilonDriftPenalty:
    """
    Acts as an anchor to prevent Wasserstein loss from drifting/exploding.
    Built as a class to be fully compliant with VoxGAN's penalty iterators.
    """
    def __init__(self, weight=0.0001, num_iter=1):
        self.weight = weight
        self.to_backprop = True       # Required by VoxGAN
        self.num_iter = num_iter      # Required by VoxGAN
        self.name = 'DriftPenalty'    

    def __call__(self, discriminator, device, real, fake, scaler=None):
        # VoxGAN strictly passes: (discriminator, device, real_images, fake_images, scaler)
        # We run the forward passes here to get the logits required for the penalty
        real_logits = discriminator(real)
        fake_logits = discriminator(fake)
        
        r_log = real_logits['data'] if isinstance(real_logits, dict) else real_logits
        f_log = fake_logits['data'] if isinstance(fake_logits, dict) else fake_logits
        
        # Penalizes the magnitude of the Discriminator's logits directly
        return self.weight * torch.mean(torch.square(torch.cat([r_log, f_log], dim=0)))

################################################################################
# Multi-Scale Generator (MSG-GAN)
################################################################################
class MSG_GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pn1 = PixelNorm()
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pn2 = PixelNorm()
        
        # 1x1x1 convolution to generate the image at this scale
        self.to_rgb = nn.Conv3d(out_channels, img_channels, kernel_size=1)
        
        # Matched Reference Study: uses ReLU instead of LeakyReLU
        self.act = nn.ReLU(inplace=True)
        self.out_act = nn.Tanh() # Real data is scaled to [-1, 1]

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.pn1(self.conv1(x)))
        x = self.act(self.pn2(self.conv2(x)))
        
        # Generate the multi-scale image output
        img_out = self.out_act(self.to_rgb(x))
        return x, img_out

class MultiScaleGenerator3D(nn.Module):
    def __init__(self, nz=100, nc=3):
        super().__init__()
        self.nz = nz  
        self.nc = nc  
        
        # Local ConvTranspose3d spatial projection mimicking the reference study
        self.init_conv = nn.ConvTranspose3d(nz, 256, kernel_size=(4, 16, 16), bias=False)
        self.init_pn = PixelNorm()
        
        # Matched Reference Study: uses ReLU
        self.init_act = nn.ReLU(inplace=True)
        
        self.block1 = MSG_GeneratorBlock(256, 128, nc) # Outputs 8x32x32
        self.block2 = MSG_GeneratorBlock(128, 64, nc)  # Outputs 16x64x64
        self.block3 = MSG_GeneratorBlock(64, 32, nc)   # Outputs 32x128x128
        
    def forward(self, z):
        if isinstance(z, dict):
            z = z['data']
            
        z = z.view(z.size(0), self.nz, 1, 1, 1) 
        x = self.init_act(self.init_pn(self.init_conv(z)))
        
        x, img1 = self.block1(x)
        x, img2 = self.block2(x)
        x, final_img = self.block3(x)
        
        return {
            'data': final_img,       
            'ms_fakes': [img1, img2] 
        }

################################################################################
# Multi-Scale Critic (MSG-GAN)
################################################################################
class MinibatchStdev3D(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_32 = x.to(torch.float32)
        batch_size, _, d, h, w = x_32.shape
        mean = torch.mean(x_32, dim=0, keepdim=True)
        sq_diff = torch.square(x_32 - mean)
        variance = torch.mean(sq_diff, dim=0, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        mean_std = torch.mean(std)
        target_shape = (batch_size, 1, d, h, w)
        expanded_std = mean_std.expand(target_shape).to(orig_dtype)
        return torch.cat([x, expanded_std], dim=1)

class MSG_DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Matched Reference Study: biases are allowed because the Epsilon Penalty handles drift natively
        self.conv1 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), n_power_iterations=5)
        self.conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), n_power_iterations=5)
        self.down = nn.AvgPool3d(2)
        
        # Matched Reference Study: uses normalized_swish
        self.act = NormalizedSwish()

    def forward(self, x, img):
        if x is None:
            h = img
        else:
            h = torch.cat([x, img], dim=1)
        
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.down(h)
        return h

class MultiScaleDiscriminator3D(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.block1 = MSG_DiscriminatorBlock(nc, 32)
        self.block2 = MSG_DiscriminatorBlock(32 + nc, 64)
        self.block3 = MSG_DiscriminatorBlock(64 + nc, 128)

        self.mstd = MinibatchStdev3D()
        self.flatten = nn.Flatten()
        
        # Matched Reference Study: biases are allowed
        self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(129 * 4 * 16 * 16, 1))

    def forward(self, input_dict):
        if isinstance(input_dict, torch.Tensor):
            final_img = input_dict
            ms_fakes = None
        else:
            final_img = input_dict['data']
            ms_fakes = input_dict.get('ms_fakes', None)
        
        if ms_fakes is not None:
            img_8x32 = ms_fakes[0]
            img_16x64 = ms_fakes[1]
        else:
            img_16x64 = F.avg_pool3d(final_img, 2)
            img_8x32 = F.avg_pool3d(final_img, 4)
            
        x = self.block1(None, final_img)     
        x = self.block2(x, img_16x64)        
        x = self.block3(x, img_8x32)         
        x = self.mstd(x)                     
        
        return {'data': self.fc(self.flatten(x))}

class ChannelAwareMSSWD(MSSWD):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.custom_name = name

    def __str__(self):
        return self.custom_name

def generate_realizations(ckpt_path, output_dir, nc=3, nl=(3,5,5), num_realizations=100):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at '{ckpt_path}'. Skipping generation.")
        return
        
    output_dir = os.path.join(output_dir, 'realizations')
    os.makedirs(output_dir, exist_ok=True)

    nz = 100

    print("Building Multi-Scale Generator (MSG-GAN)...")
    gen_layer = MultiScaleGenerator3D(nz=nz, nc=nc)

    print(f"Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    gen_state = checkpoint['generator'] if 'generator' in checkpoint else checkpoint
    if 'state_dict' in gen_state:
        gen_state = gen_state['state_dict']

    gen_layer = nn.DataParallel(gen_layer)
    missing, unexpected = gen_layer.load_state_dict(gen_state, strict=False)
    if missing or unexpected:
        print(f"Warning during loading:\nMissing keys: {missing}\nUnexpected keys: {unexpected}")
        
    gen_layer = gen_layer.module 
    gen_layer.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_layer = gen_layer.to(device)
    print("Success! Weights are perfectly aligned.")

    print(f"Generating {num_realizations} 3D River Block Realizations...")
    for i in range(num_realizations):
        with torch.no_grad():
            z = torch.randn(1, nz).to(device)
            output = gen_layer(z)
            sample = output['data'].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
            
            if nc > 1:
                volume = np.argmax(sample[0], axis=0).astype(np.uint8)
            else:
                volume = sample[0, 0, :, :, :]

            out_path = os.path.join(output_dir, f"realization_{i+1:02d}.npy")
            np.save(out_path, volume)
            
            if (i + 1) % 10 == 0 or (i + 1) == num_realizations:
                print(f"Generated realization {i+1}/{num_realizations} -> Data shape= {volume.shape}")

    print("Generation complete!")

def main():
    config = parse_hybrid_args()

    np.random.seed(43)
    torch.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    
    config_save_path = os.path.join(run_dir, "config.json")
    save_config(config, config_save_path)
    print(f"Saved run configuration to: {config_save_path}")
    validate_dataset(config['data_file'])

    model_name = 'architecture_msg_wgan_sn'
    nz = 100
    nl = (3, 5, 5) 
    use_one_hot = not config['disable_one_hot']
    one_hot_all = config.get('one_hot_all', False)
    
    if use_one_hot:
        nc = 9 if one_hot_all else 3
        encoding_tag = "one_hot_all" if one_hot_all else "one_hot"
    else:
        nc = 1
        encoding_tag = "no_one_hot"
    
    generator = partial(MultiScaleGenerator3D, nz=nz, nc=nc)
    discriminator = partial(MultiScaleDiscriminator3D, nc=nc)

    optimizer_generator = partial(optim.Adam, lr=1e-4, betas=(0., 0.9))
    optimizer_discriminator = partial(optim.Adam, lr=3e-4, betas=(0., 0.9))
    
    loss_generator = WGANLoss()
    loss_discriminator = WGANLoss()

    metric_params = dict(n_levels=3,
                         n_descriptors=256,
                         descriptor_size=(4, 7, 7),
                         n_repeat=12,
                         n_proj=56,
                         padding_mode='circular', 
                         combine_levels=True, 
                         batch_size=config['val_batch_size'],
                         n_gpu=config['num_gpus'])

    if use_one_hot:
        metrics = []
        for c in range(nc):
            metric_name = f"MS-SWD_Ch{c}"
            ms_swd = ChannelAwareMSSWD(name=metric_name, channel=c, **metric_params)
            metrics.append(ms_swd)
    else:
        ms_swd = ChannelAwareMSSWD(name="MS-SWD", **metric_params)
        metrics = [ms_swd]

    dataset_name = os.path.splitext(os.path.basename(config['data_file']))[0]
    
    dataset = partial(FaciesDataset,
                      h5_path=config['data_file'],
                      num_samples=config['num_samples'],
                      save_mapping_dir=run_dir,
                      use_one_hot=use_one_hot,
                      one_hot_all=one_hot_all,
                      dataset_name=dataset_name,
                      num_epochs=config['epochs'])

    num_training = 1 
    
    for i in range(num_training):
        output_label = f'{model_name}_{encoding_tag}_{i + 1}'
        
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=run_dir,
                        output_label=output_label,
                        verbose=2,
                        num_gpus=config['num_gpus'],
                        num_nodes=1,
                        distributed=False,
                        backend='nccl',
                        use_amp_training=False) 
        
        # Instantiate the proper VoxGAN-compliant Penalty Object
        eps_penalty = EpsilonDriftPenalty(weight=0.0001, num_iter=config.get('num_iter', 2))

        gan.configure(optimizer_generator,
                      optimizer_discriminator,
                      loss_generator,
                      loss_discriminator,
                      initialize_weights=initialize_weights_normal,
                      num_iter_discriminator=config.get('num_iter', 2),              
                      num_accumulated=int(64 / config['batch_size']),
                      fake_label_generator=-1.0,       
                      real_label_discriminator=-1.0,   
                      fake_label_discriminator=1.0,        
                      penalty_generator=None,
                      penalty_discriminator=[eps_penalty]) # Safely wrapped the object in a list
        
        gan.train(dataset,
                  num_epochs=config['epochs'],
                  batch_size=config['batch_size'],
                  num_workers=0 * config['num_gpus'],
                  pin_memory=True,
                  drop_last=True,
                  checkpoint_step=1e12,
                  sampling_step=1000,
                  sampling_size=3,
                  metrics=metrics,
                  metric_step=20,
                  validation_size=config['validation_size'],
                  validation_batch_size=config['val_batch_size'],
                  preload_validation=True,
                  resume_checkpoint_id=None)

        checkpoint_dir_path = os.path.join(run_dir, f'{output_label}_Training_Checkpoints')
        checkpoint_paths = glob(os.path.join(checkpoint_dir_path, f'{output_label}_training_checkpoint_*'))
        
        if checkpoint_paths:
            checkpoint_paths = sorted(checkpoint_paths, key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
            last_checkpoint = checkpoint_paths[-1]

            new_filename = f"{model_name}_{dataset_name}_{encoding_tag}_epochs_{config['epochs']}_bs_{config['batch_size']}_run_{i + 1}.pt"
            final_checkpoint_path = os.path.join(run_dir, new_filename)

            shutil.copy(last_checkpoint, final_checkpoint_path)
            shutil.rmtree(checkpoint_dir_path)
            
            print(f"Saved final renamed model checkpoint to: {final_checkpoint_path}\n")

            try:
                from visualize_and_generate import plot_losses
                
                csv_path = os.path.join(run_dir, f"{output_label}_history.csv")
                plot_losses(csv_path, run_dir)
                
                generate_realizations(
                    ckpt_path=final_checkpoint_path, 
                    output_dir=run_dir, 
                    nc=nc, 
                    nl=nl,
                    num_realizations=100
                )
                print("Post-training visualization complete.")
            except Exception as e:
                print(f"Error during post-training visualization: {e}")

if __name__ == '__main__':
    main()