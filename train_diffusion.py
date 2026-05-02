"""
Text-to-SignLanguage Diffusion Model Training Script.

A simplified MDM-style (Motion Diffusion Model) implementation for
generating SMPL-X sign language motion from English text.

Architecture: Transformer Encoder + DDPM + CLIP text conditioning
Target: Train on A100 with expanded ASL dataset.

Usage:
    python train_diffusion.py \
        --pkl_dir unified_pkls/ \
        --mapping mapping.json \
        --save_dir checkpoints/sign_mdm_v1 \
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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sign_language_dataset import SignLanguageDataset, SignLanguageDataModule


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
# Motion Diffusion Transformer
# ---------------------------------------------------------------------------

class SignDiffusionModel(nn.Module):
    """
    Transformer-based denoising network for SMPL-X motion generation.
    
    Predicts noise epsilon given:
        - noisy motion x_t: [B, T, D]
        - timestep t: [B]
        - text embedding c: [B, text_dim]
    """

    def __init__(
        self,
        pose_dim: int = 182,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_frames: int = 300,
        text_dim: int = 512,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.max_frames = max_frames

        # Input projection
        self.input_proj = nn.Linear(pose_dim, latent_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, latent_dim) * 0.02)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(latent_dim)

        # Text conditioning projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
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

        # Project input
        h = self.input_proj(x_t)  # [B, T, latent]

        # Add positional encoding
        h = h + self.pos_embed[:, :T, :]

        # Add timestep embedding (broadcast across time)
        t_emb = self.time_embed(t)  # [B, latent]
        h = h + t_emb.unsqueeze(1)

        # Add text conditioning (broadcast across time)
        c = self.text_proj(text_emb)  # [B, latent]
        h = h + c.unsqueeze(1)

        # Create attention mask for padding
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # TransformerEncoder expects True=ignore

        # Transformer
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Output
        out = self.output_proj(h)  # [B, T, D]
        return out


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
    def sample_step(self, model, x_t, t_idx, text_emb, mask=None):
        """Single reverse diffusion step."""
        B = x_t.shape[0]
        device = x_t.device
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)

        pred_noise = model(x_t, t, text_emb, mask)

        alpha = self.register['alphas'][t_idx]
        alpha_cumprod = self.register['alphas_cumprod'][t_idx]
        beta = self.register['betas'][t_idx]

        coeff = beta / self.register['sqrt_one_minus_alphas_cumprod'][t_idx]
        mean = (1.0 / alpha.sqrt()) * (x_t - coeff * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            var = self.register['posterior_variance'][t_idx]
            x_prev = mean + var.sqrt() * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, model, text_emb, max_frames, pose_dim, mask=None, device='cuda'):
        """Full reverse diffusion: noise -> motion."""
        B = text_emb.shape[0]
        x = torch.randn(B, max_frames, pose_dim, device=device)

        for t in reversed(range(self.num_steps)):
            x = self.sample_step(model, x, t, text_emb, mask)
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

            # Weighted loss for hands (dims 66-156) - 2x weight
            hand_loss = F.mse_loss(
                pred_noise[:, :, 66:156] * mask.unsqueeze(-1),
                noise[:, :, 66:156] * mask.unsqueeze(-1),
            )
            total_loss = loss + 0.5 * hand_loss

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
        avg_train = np.mean(train_losses)

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

        avg_val = np.mean(val_losses) if val_losses else float('inf')

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train={avg_train:.4f} val={avg_val:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6f} | "
              f"time={epoch_time:.1f}s")

        # Save checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'model_epoch{epoch+1:04d}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  New best model! val_loss={avg_val:.4f}")

    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(args):
    """Generate motion from text prompt using trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    ckpt = torch.load(os.path.join(args.model_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scheduler = DDPMScheduler(num_steps=config.get('diffusion_steps', 1000)).to(device)

    # Encode text
    text_emb = text_encoder.encode([args.text])

    # Generate
    print(f"Generating motion for: '{args.text}'")
    motion = scheduler.sample(model, text_emb, config['max_frames'], config['pose_dim'], device=device)
    motion = motion[0].cpu().numpy()  # [T, D]

    # Denormalize
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    motion = motion * std + mean

    # Trim to reasonable length
    motion = motion[:args.num_frames]

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'generated_motion.pkl')
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump({'smplx': motion.astype(np.float32)}, f)
    print(f"Saved: {out_path} shape={motion.shape}")

    # Also save as npz for easy inspection
    np.savez(os.path.join(args.output_dir, 'generated_motion.npz'), smplx=motion)


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
    tp.add_argument('--save_dir', default='checkpoints/sign_mdm_v1')
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
    gp.add_argument('--output_dir', default='generated/')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    else:
        parser.print_help()
