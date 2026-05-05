"""
Text-to-SignLanguage Diffusion Model Training Script.

A simplified MDM-style (Motion Diffusion Model) implementation for
generating SMPL-X sign language motion from English text.

Architecture: Transformer Encoder + DDPM + CLIP text conditioning
Target: Train on A100 with expanded ASL dataset.

Usage:
    python train_diffusion.py \
        --pkl_dir data/cache/how2sign_pkls_cropTrue_shapeFalse \
        --mapping data/metadata/how2sign_mapping.json \
        --save_dir checkpoints/mdm_weights/sign_mdm_v1 \
        --batch_size 64 \
        --epochs 200 \
        --lr 1e-4
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import random
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
# A100 Optimization: Enable TensorFloat32 (TF32) for massive speedups on Ampere architecture
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.core.sign_language_dataset import SignLanguageDataset, SignLanguageDataModule


# ---------------------------------------------------------------------------
# CLIP Text Encoder Wrapper
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """Wraps a frozen CLIP model for text encoding."""

    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cuda"):
        super().__init__()
        self.device = device
        try:
            import clip as clip_module
            self.clip_model, _ = clip_module.load(clip_model, device=device)
            self.clip_model.eval()
            self.clip = clip_module
            self.text_dim = 512
            print(f"[CLIPTextEncoder] Loaded {clip_model}, dim={self.text_dim}")
        except ImportError:
            print("[CLIPTextEncoder] CLIP not available, using random embeddings")
            self.clip_model = None
            self.clip = None
            self.text_dim = 512

        # Freeze CLIP
        if self.clip_model:
            for p in self.clip_model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list) -> torch.Tensor:
        """Encode list of strings -> [B, 512] float tensor."""
        if self.clip_model and self.clip:
            tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
            features = self.clip_model.encode_text(tokens).float()
            return features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            return torch.randn(len(texts), self.text_dim, device=self.device)


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Anatomically Informed GNN & LSTM Modules
# ---------------------------------------------------------------------------

class AnatomicalGNN(nn.Module):
    """Message passing neural network for skeletal joints."""
    def __init__(self, num_joints=55, in_dim=3, hidden_dim=32, out_dim=256, layers=4):
        super().__init__()
        self.num_joints = num_joints
        self.joint_proj = nn.Linear(in_dim, hidden_dim)
        
        # Pose embedding to break permutation equivariance
        self.pose_emb = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        
        # Learnable adjacency matrices to implicitly learn the kinematic tree
        self.adjs = nn.ParameterList([
            nn.Parameter(torch.eye(num_joints) + torch.randn(num_joints, num_joints) * 0.01) 
            for _ in range(layers)
        ])
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(layers)
        ])
        
        self.out_proj = nn.Linear(num_joints * hidden_dim, out_dim)

    def forward(self, x):
        # x: [B*T, num_joints, in_dim]
        h = self.joint_proj(x)
        h = h + self.pose_emb
        
        for adj, mlp in zip(self.adjs, self.mlps):
            # Graph message passing
            h_msg = torch.matmul(adj, h)
            h = h + mlp(h_msg)
            
        h = h.view(h.shape[0], -1)  # Flatten joints
        return self.out_proj(h)


class ExpressionEncoder(nn.Module):
    """MLP encoder for facial expressions and non-joint parameters."""
    def __init__(self, in_dim=17, hidden_dim=128, out_dim=256):
        super().__init__()
        # Expression token to break permutation equivariance
        self.exp_emb = nn.Parameter(torch.randn(1, in_dim) * 0.02)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: [B*T, in_dim]
        return self.mlp(x + self.exp_emb)


class SignDiffusionModel(nn.Module):
    """
    GNN + LSTM based denoising network for SMPL-X motion generation.
    Matches the 'Neural Sign Actors' architecture.
    """
    def __init__(
        self,
        pose_dim: int = 182,
        latent_dim: int = 512,
        num_layers: int = 4,  # LSTM layers
        num_heads: int = 8,   # Unused now, kept for config compatibility
        max_frames: int = 300,
        text_dim: int = 512,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        
        # We assume first 165 dims are 55 joints (55 * 3), remaining 17 are expressions/shape/trans
        self.num_joints = 55
        self.joint_dim = 3
        self.exp_dim = pose_dim - (self.num_joints * self.joint_dim)

        # Encoders
        self.gnn_encoder = AnatomicalGNN(
            num_joints=self.num_joints, 
            in_dim=self.joint_dim, 
            hidden_dim=32, 
            out_dim=latent_dim // 2, 
            layers=4
        )
        self.exp_encoder = ExpressionEncoder(
            in_dim=self.exp_dim, 
            hidden_dim=128, 
            out_dim=latent_dim // 2
        )

        # Timestep & Text embeddings
        self.time_embed = TimestepEmbedding(latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Auto-regressive LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=latent_dim, 
            hidden_size=latent_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, pose_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_t: torch.Tensor,      # [B, T, D] noisy motion
        t: torch.Tensor,         # [B] timestep
        text_emb: torch.Tensor,  # [B, text_dim] CLIP embedding
        mask: torch.Tensor = None,  # [B, T] bool mask
    ) -> torch.Tensor:
        B, T, D = x_t.shape

        # Split features into joints and expressions
        joints = x_t[:, :, :self.num_joints*self.joint_dim].contiguous().view(B * T, self.num_joints, self.joint_dim)
        expr = x_t[:, :, self.num_joints*self.joint_dim:].contiguous().view(B * T, self.exp_dim)

        # Encode
        h_joints = self.gnn_encoder(joints)  # [B*T, latent/2]
        h_expr = self.exp_encoder(expr)      # [B*T, latent/2]

        # Combine
        h = torch.cat([h_joints, h_expr], dim=-1)  # [B*T, latent]
        h = h.view(B, T, -1)                       # [B, T, latent]

        # Add timestep and text conditioning (broadcast across time)
        t_emb = self.time_embed(t)      # [B, latent]
        c = self.text_proj(text_emb)    # [B, latent]
        h = h + t_emb.unsqueeze(1) + c.unsqueeze(1)

        # LSTM Auto-regressive decoding
        out, _ = self.lstm(h)  # [B, T, latent]

        # Project back to SMPL-X pose space
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# DDPM Noise Scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Standard DDPM with cosine noise schedule."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps

        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register = {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
            'sqrt_recip_alphas': torch.sqrt(1.0 / alphas),
            'posterior_variance': betas * (1.0 - torch.cat([torch.tensor([0.0]), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod),
        }

    def to(self, device):
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple:
        """q(x_t | x_0) - add noise at timestep t."""
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.register['sqrt_alphas_cumprod'][t][:, None, None]
        sqrt_one_minus = self.register['sqrt_one_minus_alphas_cumprod'][t][:, None, None]
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t_idx, text_emb, mask=None, cfg_scale=1.0, null_emb=None):
        """Single reverse diffusion step with CFG and x0 clipping."""
        B = x_t.shape[0]
        device = x_t.device
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # Classifier-Free Guidance
        if cfg_scale > 1.0 and null_emb is not None:
            pred_noise_text = model(x_t, t, text_emb, mask)
            pred_noise_null = model(x_t, t, null_emb, mask)
            pred_noise = pred_noise_null + cfg_scale * (pred_noise_text - pred_noise_null)
        else:
            pred_noise = model(x_t, t, text_emb, mask)

        alpha = self.register['alphas'][t_idx]
        alpha_cumprod = self.register['alphas_cumprod'][t_idx]
        beta = self.register['betas'][t_idx]

        # Predict x0 and clamp it to avoid mathematically exploding "alien" motions
        sqrt_alpha_cumprod = self.register['sqrt_alphas_cumprod'][t_idx]
        sqrt_one_minus = self.register['sqrt_one_minus_alphas_cumprod'][t_idx]
        pred_x0 = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha_cumprod
        pred_x0 = torch.clamp(pred_x0, -2.5, 2.5) # Bounds motion to realistic human range

        if t_idx > 0:
            alpha_cumprod_prev = self.register['alphas_cumprod'][t_idx - 1]
            mean = (sqrt_alpha_cumprod * beta / (1.0 - alpha_cumprod)) * pred_x0 + \
                   (alpha.sqrt() * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)) * x_t
            noise = torch.randn_like(x_t)
            var = self.register['posterior_variance'][t_idx]
            x_prev = mean + var.sqrt() * noise
        else:
            x_prev = pred_x0

        return x_prev

    @torch.no_grad()
    def sample(self, model, text_emb, max_frames, pose_dim, mask=None, device='cuda', cfg_scale=1.0, null_emb=None):
        """Full reverse diffusion: noise -> motion."""
        B = text_emb.shape[0]
        x = torch.randn(B, max_frames, pose_dim, device=device)

        for t in reversed(range(self.num_steps)):
            x = self.sample_step(model, x, t, text_emb, mask, cfg_scale, null_emb)
            if t % 100 == 0:
                print(f"  Sampling step {self.num_steps - t}/{self.num_steps}")

        return x


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # Data
    data_module = SignLanguageDataModule(
        pkl_dir=args.pkl_dir,
        mapping_path=args.mapping,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.05,
    )
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Text encoder
    text_encoder = CLIPTextEncoder(device=str(device))

    # Model
    model = SignDiffusionModel(
        pose_dim=args.pose_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_frames=args.max_frames,
        text_dim=text_encoder.text_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Scheduler
    scheduler = DDPMScheduler(num_steps=args.diffusion_steps).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, eps=1e-8,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config['num_params'] = num_params
    config['mean'] = data_module.mean.tolist()
    config['std'] = data_module.std.tolist()
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            motion = batch['motion'].to(device)   # [B, T, D]
            mask = batch['mask'].to(device)        # [B, T]
            texts = batch['text']                  # list of str

            # Encode text
            text_emb = text_encoder.encode(texts)  # [B, 512]

            # Sample random timesteps
            B = motion.shape[0]
            t = torch.randint(0, args.diffusion_steps, (B,), device=device)

            # Add noise
            x_t, noise = scheduler.add_noise(motion, t)

            # Predict noise
            pred_noise = model(x_t, t, text_emb, mask)

            # Loss (only on valid frames)
            loss = F.mse_loss(pred_noise * mask.unsqueeze(-1),
                              noise * mask.unsqueeze(-1))

            # Weighted loss for hands (dims 66-156) - 2x weight total
            hand_loss = F.mse_loss(
                pred_noise[:, :, 66:156] * mask.unsqueeze(-1),
                noise[:, :, 66:156] * mask.unsqueeze(-1),
            )
            total_loss = loss + 1.0 * hand_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={np.mean(train_losses[-50:]):.4f}")

        lr_scheduler.step()
        epoch_time = time.time() - t0
        avg_train = float(np.mean(train_losses)) if train_losses else float('inf')

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                mask = batch['mask'].to(device)
                texts = batch['text']
                text_emb = text_encoder.encode(texts)
                B = motion.shape[0]
                t = torch.randint(0, args.diffusion_steps, (B,), device=device)
                x_t, noise = scheduler.add_noise(motion, t)
                pred_noise = model(x_t, t, text_emb, mask)
                loss = F.mse_loss(pred_noise * mask.unsqueeze(-1),
                                  noise * mask.unsqueeze(-1))
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses)) if val_losses else float('inf')

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train={avg_train:.4f} val={avg_val:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6f} | "
              f"time={epoch_time:.1f}s")

        # Save checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'model_epoch{epoch+1:04d}.pt')
            torch.save({
                'epoch': int(epoch + 1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': float(avg_train),
                'val_loss': float(avg_val),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': int(epoch + 1),
                'model_state_dict': model.state_dict(),
                'val_loss': float(avg_val),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  New best model! val_loss={avg_val:.4f}")

    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(args):
    """Generate motion from text prompt using trained model."""
    device = torch.device('cpu')

    # Load config
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # Text encoder
    text_encoder = CLIPTextEncoder(device=str(device))

    # Model
    model = SignDiffusionModel(
        pose_dim=config['pose_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_frames=config['max_frames'],
        text_dim=text_encoder.text_dim,
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    # PyTorch 2.6+ defaults `weights_only=True`, which can reject older checkpoints
    # containing numpy scalar metadata. Our checkpoints are self-generated/trusted,
    # so we load with `weights_only=False` when supported.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scheduler = DDPMScheduler(num_steps=config.get('diffusion_steps', 1000)).to(device)

    # Encode text
    text_emb = text_encoder.encode([args.text]).to(device)

    # Null embedding for CFG (using empty string)
    null_emb = text_encoder.encode([""]).to(device)

    # Generate
    T = args.num_frames
    print(f"[Generate] Text: '{args.text}' | Frames: {T} | CFG Scale: 2.5")
    motion = scheduler.sample(
        model, 
        text_emb, 
        max_frames=T, 
        pose_dim=config['pose_dim'], 
        device=device,
        cfg_scale=2.5,
        null_emb=null_emb
    )
    motion = motion[0].cpu().numpy()  # [T, D]

    # Denormalize
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    motion = motion * std + mean

    # Trim to reasonable length
    motion = motion[:args.num_frames]

    # -----------------------------------------------------------------------
    # Post-processing: smooth jitter + clamp to realistic human ranges
    # (matches real How2Sign pkl statistics)
    # -----------------------------------------------------------------------
    from scipy.ndimage import gaussian_filter1d
    
    # 1. Per-region clamping (based on real data analysis)
    motion[:, 3:66] = np.clip(motion[:, 3:66], -1.8, 1.8)      # Body
    motion[:, 66:156] = np.clip(motion[:, 66:156], -1.5, 1.5)   # Hands
    
    # 2. Zero out shape params (real data always has constant neutral shape)
    motion[:, 159:169] = 0.0
    
    # 3. Clamp translation to realistic range
    motion[:, 179:182] = np.clip(motion[:, 179:182], -0.1, 20.0)
    
    # 4. Temporal smoothing (Gaussian filter to kill jitter)
    # Hands move faster, so they need less smoothing than the body
    for d in range(motion.shape[1]):
        if d < 3: # Global Orient (HEAVY smoothing to stop wobbling)
            s = 10.0
        elif d < 66: # Body
            s = 3.0
        elif d < 156: # Hands
            s = 1.0
        else: # Jaw/Transl
            s = 2.0
        motion[:, d] = gaussian_filter1d(motion[:, d], sigma=s, mode='nearest')

    # 5. Lock Translation (prevents floating/drifting)
    # The real dataset has a static camera, so translation shouldn't change much.
    # We freeze it to the dataset's average translation.
    motion[:, 179:182] = mean[179:182]

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'generated_motion.pkl')
    import pickle
    with open(out_path, 'wb') as f:
        motion_f32 = motion.astype(np.float32)
        pickle.dump({'smplx': motion_f32}, f)
    print(f"Saved: {out_path} shape={motion_f32.shape}")

    # Also save as npz for easy inspection
    np.savez(os.path.join(args.output_dir, 'generated_motion.npz'), smplx=motion_f32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sign Language Diffusion Model")
    sub = parser.add_subparsers(dest='command')

    # Train
    tp = sub.add_parser('train')
    tp.add_argument('--pkl_dir', required=True)
    tp.add_argument('--mapping', required=True)
    tp.add_argument('--save_dir', default='checkpoints/mdm_weights/sign_mdm_v1')
    tp.add_argument('--batch_size', type=int, default=64)
    tp.add_argument('--epochs', type=int, default=200)
    tp.add_argument('--lr', type=float, default=1e-4)
    tp.add_argument('--weight_decay', type=float, default=0.05)
    tp.add_argument('--max_frames', type=int, default=300)
    tp.add_argument('--pose_dim', type=int, default=182)
    tp.add_argument('--latent_dim', type=int, default=512)
    tp.add_argument('--num_layers', type=int, default=8)
    tp.add_argument('--num_heads', type=int, default=8)
    tp.add_argument('--diffusion_steps', type=int, default=1000)
    tp.add_argument('--num_workers', type=int, default=4)
    tp.add_argument('--save_every', type=int, default=10)

    # Generate
    gp = sub.add_parser('generate')
    gp.add_argument('--model_dir', required=True)
    gp.add_argument('--text', required=True)
    gp.add_argument('--num_frames', type=int, default=150)
    gp.add_argument('--output_dir', default='data/cache/generated/')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    else:
        parser.print_help()
