"""
WGAN-SN (MSG-GAN) architecture training script (Pure PyTorch):

example usage:
nohup python -u wgans_training.py \
    --run_name 100_epochs_3_classes_mws-gan_nexus_100_isbx \
    --data_file ~/data/datasets/training_dataset_nexus_20000_samples_ntg_0_67_chdepth_6_isbx_100/samples/facies/samples.h5 \
    --output_dir ~/data/outputs/UPD_20000_samples \
    --num_gpus 2 \
    --num_samples 10000 \
    --epochs 100 \
    --batch_size 64 \
    --num_iter 2 \
    --one_hot_all False > training_mwsgan.out 2>&1 &
"""

import os
import csv
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Custom Modules
from dataloader import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# Weight Initialization
# ==============================================================================
def initialize_weights_normal(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ==============================================================================
# Custom Activations & Normalizations
# ==============================================================================
class NormalizedSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Swish has a max derivative of ~1.0998. 
        # Dividing by this guarantees the function is strictly 1-Lipschitz.
        return F.silu(x) / 1.0998

class GroupSort(nn.Module):
    def __init__(self, channels_per_group=2):
        super().__init__()
        self.c = channels_per_group

    def forward(self, x):
        # Sorts elements in groups of 2 along the channel dimension (dim=1)
        shape = x.shape
        x = x.view(shape[0], shape[1] // self.c, self.c, *shape[2:])
        x, _ = torch.sort(x, dim=2)
        return x.view(shape)

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
        
        self.act = nn.ReLU(inplace=True)
        self.out_act = nn.Tanh() # Real data is scaled to [-1, 1]

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.pn1(self.conv1(x)))
        x = self.act(self.pn2(self.conv2(x)))
        
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
        self.conv1 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), n_power_iterations=5)
        self.conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), n_power_iterations=5)
        self.down = nn.AvgPool3d(2)
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

################################################################################
# Generation Utility
################################################################################
def generate_realizations(ckpt_path, output_dir, nc=3, num_realizations=100):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at '{ckpt_path}'. Skipping generation.")
        return
        
    output_dir = os.path.join(output_dir, 'realizations')
    os.makedirs(output_dir, exist_ok=True)

    nz = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Building Multi-Scale Generator (MSG-GAN)...")
    gen_layer = MultiScaleGenerator3D(nz=nz, nc=nc).to(device)

    print(f"Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    gen_state = checkpoint['generator'] if 'generator' in checkpoint else checkpoint
    if 'state_dict' in gen_state:
        gen_state = gen_state['state_dict']

    missing, unexpected = gen_layer.load_state_dict(gen_state, strict=False)
    gen_layer.eval()
    
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

################################################################################
# Pure PyTorch Training Loop
################################################################################
def main():
    config = parse_hybrid_args()

    np.random.seed(43)
    torch.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    
    config_save_path = os.path.join(run_dir, "config.json")
    save_config(config, config_save_path)
    print(f"Saved run configuration to: {config_save_path}")
    validate_dataset(config['data_file'])

    model_name = 'architecture_msg_wgan_sn'
    nz = 100
    use_one_hot = not config.get('disable_one_hot', False)
    one_hot_all = config.get('one_hot_all', False)
    
    if use_one_hot:
        nc = 9 if one_hot_all else 3
        encoding_tag = "one_hot_all" if one_hot_all else "one_hot"
    else:
        nc = 1
        encoding_tag = "no_one_hot"
        
    dataset_name = os.path.splitext(os.path.basename(config['data_file']))[0]
    
    # 1. Dataset & DataLoader (Parameters match FaciesDataset definition exactly)
    dataset = FaciesDataset(h5_path=config['data_file'],
                            num_samples=config['num_samples'],
                            nz=32, # Ensure slices map to the Discriminator's expected depth
                            save_mapping_dir=run_dir,
                            use_one_hot=use_one_hot,
                            one_hot_all=one_hot_all,
                            preload_ram=True)

    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            drop_last=True)

    # 2. Models & Optimizers
    generator = MultiScaleGenerator3D(nz=nz, nc=nc).to(device)
    discriminator = MultiScaleDiscriminator3D(nc=nc).to(device)
    
    generator.apply(initialize_weights_normal)
    discriminator.apply(initialize_weights_normal)
    
    # DataParallel across GPUs
    if config['num_gpus'] > 1 and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # TTUR Learning Rates
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.0, 0.9))

    # 3. CSV Logger setup
    output_label = f'{model_name}_{encoding_tag}_1'
    csv_path = os.path.join(run_dir, f"{output_label}_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'iteration', 'loss_discriminator', 'D(x)', 'loss_generator', 'D(G(z))'])

    num_iter_d = config.get('num_iter', 2)
    epsilon_weight = 0.0001
    global_step = 0

    print("\nStarting Pure PyTorch Training Loop...")
    for epoch in range(1, config['epochs'] + 1):
        print(f"... Epoch {epoch}/{config['epochs']}...")
        
        for i, batch in enumerate(dataloader):
            global_step += 1
            
            # Extract real images (handles dict output correctly)
            if isinstance(batch, (list, tuple)):
                real_imgs = batch[0].to(device, dtype=torch.float32)
            elif isinstance(batch, dict):
                real_imgs = batch['data'].to(device, dtype=torch.float32)
            else:
                real_imgs = batch.to(device, dtype=torch.float32)
                
            batch_size = real_imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(num_iter_d):
                optimizer_D.zero_grad()
                
                # Generate Fake Images
                z = torch.randn(batch_size, nz).to(device)
                fake_dict = generator(z)
                fake_imgs = fake_dict['data']
                ms_fakes = fake_dict['ms_fakes']
                
                # Discriminator Predictions
                real_logits = discriminator({'data': real_imgs})['data']
                fake_logits = discriminator({'data': fake_imgs.detach(), 'ms_fakes': [f.detach() for f in ms_fakes]})['data']
                
                # WGAN Loss (Maximize: mean(real) - mean(fake) -> Minimize: mean(fake) - mean(real))
                d_loss_real = -torch.mean(real_logits)
                d_loss_fake = torch.mean(fake_logits)
                wgan_loss_D = d_loss_real + d_loss_fake
                
                # Epsilon Drift Penalty
                drift_penalty = epsilon_weight * torch.mean(torch.square(torch.cat([real_logits, fake_logits], dim=0)))
                
                loss_D = wgan_loss_D + drift_penalty
                loss_D.backward()
                optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, nz).to(device)
            fake_dict = generator(z)
            fake_logits = discriminator({'data': fake_dict['data'], 'ms_fakes': fake_dict['ms_fakes']})['data']
            
            # Generator wants Discriminator to think fakes are real (Maximize mean(fake) -> Minimize -mean(fake))
            loss_G = -torch.mean(fake_logits)
            
            loss_G.backward()
            optimizer_G.step()

            # -----------------
            #  Logging
            # -----------------
            if (i + 1) % 10 == 0 or i == 0:
                print(f"... ... Batch {i+1}/{len(dataloader)} "
                      f"| Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} "
                      f"| D(x): {-d_loss_real.item():.4f} | D(G(z)): {d_loss_fake.item():.4f}")
                
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, i+1, global_step, loss_D.item(), -d_loss_real.item(), loss_G.item(), d_loss_fake.item()])

    # 4. Save Final Checkpoint
    final_checkpoint_path = os.path.join(run_dir, f"{model_name}_{dataset_name}_{encoding_tag}_epochs_{config['epochs']}_bs_{config['batch_size']}_final.pt")
    
    # Save un-wrapped state dicts (removing DataParallel wrapper names for clean loading later)
    gen_state = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
    disc_state = discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict()
    
    torch.save({
        'generator': {'state_dict': gen_state},
        'discriminator': {'state_dict': disc_state}
    }, final_checkpoint_path)
    
    print(f"\nSaved final model checkpoint to: {final_checkpoint_path}")

    # 5. Post-Training Visualization
    try:
        from visualize_and_generate import plot_losses
        plot_losses(csv_path, run_dir)
        
        generate_realizations(ckpt_path=final_checkpoint_path, 
                              output_dir=run_dir, 
                              nc=nc, 
                              num_realizations=100)
        print("Post-training visualization complete.")
    except Exception as e:
        print(f"Error during post-training visualization: {e}")

if __name__ == '__main__':
    main()