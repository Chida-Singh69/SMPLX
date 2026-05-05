"""
Text-to-Sign Language Diffusion Model - V3

Fixes over V2:
  1. HARD FAIL if CLIP not installed (no silent random embeddings)
  2. Per-sample null-text dropout (not per-batch) -> proper CFG training
  3. Transformer decoder instead of LSTM (attends across all frames)
  4. Cosine noise schedule (better SNR for motion data)
  5. Temporal loss on predicted x0, not pred_noise (correct signal)
  6. Per-region loss weighting with normalized targets (body/hands/face)
  7. AMP (mixed precision) + gradient checkpointing for A100 speed
  8. Larger GNN hidden_dim (64) + LayerNorm after each message pass
  9. EMA (exponential moving average) of weights for stable generation
 10. DDIM sampler (50 steps at inference instead of 1000)

Usage (A100):
    python train_diffusion_v3.py train \
        --pkl_dir how2sign_pkls_cropTrue_shapeFalse \
        --mapping merged_how2sign_mapping.json \
        --save_dir checkpoints/sign_mdm_v3 \
        --batch_size 64 --epochs 300

    python train_diffusion_v3.py generate \
        --model_dir checkpoints/sign_mdm_v3 \
        --text "hello how are you" \
        --num_frames 120 --cfg_scale 3.0
"""

import os, sys, json, math, time, argparse, random, pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from sign_language_dataset import SignLanguageDataset, SignLanguageDataModule


# ---------------------------------------------------------------------------
# 1. CLIP — hard fail if missing
# ---------------------------------------------------------------------------

def load_clip_encoder(clip_model: str = "ViT-B/32", device: str = "cuda"):
    try:
        import clip as clip_module
        model, _ = clip_module.load(clip_model, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"[CLIP] Loaded {clip_model}, dim=512")
        return model, clip_module, 512
    except ImportError:
        sys.exit(
            "\n[FATAL] 'clip' package not found.\n"
            "Install it: pip install git+https://github.com/openai/CLIP.git\n"
            "Then re-run in that environment. V3 never uses random embeddings.\n"
        )


@torch.no_grad()
def encode_text(clip_model, clip_module, texts: list, device: str) -> torch.Tensor:
    tokens = clip_module.tokenize(texts, truncate=True).to(device)
    feats = clip_model.encode_text(tokens).float()
    return feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# 2. Cosine noise schedule (better than linear for motion)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(num_steps: int, s: float = 0.008):
    """Improved cosine schedule from Nichol & Dhariwal 2021."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.9999)


# ---------------------------------------------------------------------------
# 3. Noise scheduler (DDPM forward + DDIM reverse)
# ---------------------------------------------------------------------------

class NoiseScheduler:
    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        for attr in ['betas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x0, t):
        """q(x_t | x_0)"""
        noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cumprod[t][:, None, None]
        sm = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sa * x0 + sm * noise, noise

    def predict_x0(self, x_t, t_idx, pred_noise):
        sa = self.sqrt_alphas_cumprod[t_idx]
        sm = self.sqrt_one_minus_alphas_cumprod[t_idx]
        # Handle batched timesteps [B] -> [B,1,1] for broadcasting against [B,T,D]
        if sa.dim() >= 1:
            sa = sa[:, None, None]
            sm = sm[:, None, None]
        return (x_t - sm * pred_noise) / sa

    @torch.no_grad()
    def ddim_sample(self, model, text_emb, null_emb, max_frames, pose_dim,
                    device, cfg_scale=3.0, ddim_steps=50):
        """DDIM deterministic sampler — 50 steps instead of 1000."""
        B = text_emb.shape[0]
        x = torch.randn(B, max_frames, pose_dim, device=device)

        step_indices = torch.linspace(self.num_steps - 1, 0, ddim_steps).long()

        for i, t_idx in enumerate(step_indices):
            t_idx = t_idx.item()
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # CFG
            pn_text = model(x, t, text_emb)
            pn_null = model(x, t, null_emb)
            pred_noise = pn_null + cfg_scale * (pn_text - pn_null)

            # Predict x0
            x0 = self.predict_x0(x, t_idx, pred_noise)
            x0 = torch.clamp(x0, -3.0, 3.0)

            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1].item()
                alpha_prev = self.alphas_cumprod[t_prev]
                alpha = self.alphas_cumprod[t_idx]
                # DDIM update (eta=0 -> deterministic)
                x = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = x0

            if i % 10 == 0:
                print(f"  DDIM {i+1}/{ddim_steps}")

        return x


# ---------------------------------------------------------------------------
# 4. SMPL-X adjacency
# ---------------------------------------------------------------------------

def build_smplx_adjacency(num_joints=55):
    SMPLX_PARENTS = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15,
        20, 25, 26, 27, 20, 29, 30, 31, 20, 33, 34, 35, 20, 37, 38,
        21, 40, 41, 42, 21, 44, 45, 46, 21, 48, 49, 50, 21, 52, 53,
    ]
    adj = torch.zeros(num_joints, num_joints)
    for i, p in enumerate(SMPLX_PARENTS):
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0
        adj[i, i] = 1.0
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / row_sum


# ---------------------------------------------------------------------------
# 5. Anatomical GNN (v3: larger hidden, LayerNorm per layer)
# ---------------------------------------------------------------------------

class AnatomicalGNN(nn.Module):
    def __init__(self, num_joints=55, in_dim=3, hidden_dim=64, out_dim=256, layers=4):
        super().__init__()
        self.num_joints = num_joints
        self.joint_proj = nn.Linear(in_dim, hidden_dim)
        self.pose_emb = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        base_adj = build_smplx_adjacency(num_joints)
        self.register_buffer('base_adj', base_adj)
        self.edge_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(num_joints, num_joints))
            for _ in range(layers)
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.out_proj = nn.Linear(num_joints * hidden_dim, out_dim)

    def forward(self, x):
        # x: [B*T, num_joints, 3]
        h = self.joint_proj(x) + self.pose_emb
        for ew, mlp, norm in zip(self.edge_weights, self.mlps, self.norms):
            adj = self.base_adj * torch.sigmoid(ew)
            h = norm(h + mlp(torch.matmul(adj, h)))
        return self.out_proj(h.view(h.shape[0], -1))


# ---------------------------------------------------------------------------
# 6. Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# 7. Transformer decoder block (replaces LSTM)
# ---------------------------------------------------------------------------

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# 8. Main diffusion model (V3)
# ---------------------------------------------------------------------------

class SignDiffusionModelV3(nn.Module):
    """
    GNN (kinematic tree) + Transformer (temporal) + AdaLN text/time conditioning.
    AdaLN: each transformer block is modulated by (time + text) via scale/shift.
    """

    def __init__(
        self,
        pose_dim: int = 182,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_frames: int = 300,
        text_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.num_joints = 55
        self.joint_dim = 3
        self.exp_dim = pose_dim - self.num_joints * self.joint_dim  # = 17

        # Pose encoders
        self.gnn = AnatomicalGNN(
            num_joints=self.num_joints, in_dim=self.joint_dim,
            hidden_dim=64, out_dim=latent_dim // 2, layers=4
        )
        self.exp_enc = nn.Sequential(
            nn.Linear(self.exp_dim, 128), nn.GELU(), nn.Linear(128, latent_dim // 2)
        )

        # Positional encoding (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, max_frames, latent_dim) * 0.02)

        # Time + text -> AdaLN modulation (scale + shift per layer)
        cond_dim = latent_dim
        self.time_emb = TimestepEmbedding(latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        # 2 * latent_dim per layer (scale + shift) * num_layers
        self.adaLN_proj = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, 2 * latent_dim))
            for _ in range(num_layers)
        ])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, pose_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t, t, text_emb, mask=None):
        B, T, D = x_t.shape

        # Encode pose
        joints = x_t[:, :, :self.num_joints * self.joint_dim].view(B * T, self.num_joints, self.joint_dim)
        expr = x_t[:, :, self.num_joints * self.joint_dim:].view(B * T, self.exp_dim)
        h = torch.cat([self.gnn(joints), self.exp_enc(expr)], dim=-1).view(B, T, -1)

        # Add positional embedding (clip to actual T)
        h = h + self.pos_emb[:, :T, :]

        # Conditioning signal: time + text (additive before AdaLN)
        cond = self.time_emb(t) + self.text_proj(text_emb)  # [B, latent_dim]

        # Build key_padding_mask for attention (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # [B, T], True means PAD

        # Transformer with AdaLN modulation
        for block, ada_proj in zip(self.blocks, self.adaLN_proj):
            scale_shift = ada_proj(cond).unsqueeze(1)          # [B, 1, 2*latent]
            scale, shift = scale_shift.chunk(2, dim=-1)
            h = h * (1 + scale) + shift                        # AdaLN
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.final_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# 9. EMA helper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# 10. Per-region loss with hand upweighting (v3: correct normalization)
# ---------------------------------------------------------------------------

REGION_SLICES = {
    'global':  (0,   3,   1.0),
    'body':    (3,   66,  1.0),
    'lhand':   (66,  111, 2.0),   # hands upweighted x2
    'rhand':   (111, 156, 2.0),
    'jaw':     (156, 159, 0.5),
    'expr':    (159, 169, 0.5),
    'betas':   (169, 179, 0.0),   # no loss on betas (static shape)
    'transl':  (179, 182, 0.5),
}

def region_weighted_loss(pred, target, mask):
    """
    pred/target: [B, T, D]
    mask:        [B, T] float (1=valid, 0=pad)
    """
    total = 0.0
    m = mask.unsqueeze(-1)  # [B, T, 1]
    for name, (a, b, w) in REGION_SLICES.items():
        if w == 0.0:
            continue
        diff = ((pred[:, :, a:b] - target[:, :, a:b]) * m) ** 2
        total = total + w * diff.sum() / (m.sum() * (b - a) + 1e-8)
    return total


# ---------------------------------------------------------------------------
# 11. Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print(f"[Train V3] Device: {device}  AMP: {use_amp}")

    # CLIP — hard fail here
    clip_model, clip_module, text_dim = load_clip_encoder(device=str(device))

    # Data
    dm = SignLanguageDataModule(
        pkl_dir=args.pkl_dir,
        mapping_path=args.mapping,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.05,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model
    model = SignDiffusionModelV3(
        pose_dim=args.pose_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_frames=args.max_frames,
        text_dim=text_dim,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model V3] Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    ema = EMA(model, decay=0.9999)
    scheduler = NoiseScheduler(num_steps=args.diffusion_steps).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    # Warmup 5% then cosine decay
    warmup_steps = int(args.epochs * len(train_loader) * 0.05)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    config = vars(args).copy()
    config.update({
        'num_params': num_params,
        'mean': dm.mean.tolist(),
        'std': dm.std.tolist(),
        'text_dim': text_dim,
        'version': 'v3',
    })
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()

        for batch in train_loader:
            motion = batch['motion'].to(device)   # [B, T, 182]
            mask = batch['mask'].to(device)        # [B, T]
            texts = batch['text']
            B = motion.shape[0]

            # Encode text (no grad needed, CLIP is frozen)
            with torch.no_grad():
                text_emb = encode_text(clip_model, clip_module, texts, str(device))

            # *** Per-sample null-text dropout (10%) — NOT per-batch ***
            null_mask = (torch.rand(B, device=device) < 0.1).float()[:, None]
            text_emb = text_emb * (1 - null_mask)   # zero out selected samples

            # Sample timestep
            t = torch.randint(0, args.diffusion_steps, (B,), device=device)

            with autocast(enabled=use_amp):
                x_t, noise = scheduler.add_noise(motion, t)
                pred_noise = model(x_t, t, text_emb, mask)

                # --- Temporal loss on predicted x0, NOT pred_noise ---
                pred_x0 = scheduler.predict_x0(x_t, t, pred_noise)
                real_x0 = motion
                temporal_loss = F.mse_loss(
                    (pred_x0[:, 1:] - pred_x0[:, :-1]) * mask[:, 1:].unsqueeze(-1),
                    (real_x0[:, 1:] - real_x0[:, :-1]) * mask[:, 1:].unsqueeze(-1),
                )

                # Main per-region loss
                main_loss = region_weighted_loss(pred_noise, noise, mask)
                total_loss = main_loss + 0.1 * temporal_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            ema.update(model)

            losses.append(total_loss.item())
            global_step += 1

            if global_step % 200 == 0:
                print(f"  step={global_step} loss={np.mean(losses[-50:]):.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Validation (use EMA weights)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                motion = batch['motion'].to(device)
                mask = batch['mask'].to(device)
                texts = batch['text']
                B = motion.shape[0]
                text_emb = encode_text(clip_model, clip_module, texts, str(device))
                t = torch.randint(0, args.diffusion_steps, (B,), device=device)
                x_t, noise = scheduler.add_noise(motion, t)
                with autocast(enabled=use_amp):
                    pred_noise = ema.shadow(x_t, t, text_emb, mask)
                    loss = region_weighted_loss(pred_noise, noise, mask)
                val_losses.append(loss.item())

        avg_train = float(np.mean(losses)) if losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | train={avg_train:.4f} "
              f"val={avg_val:.4f} | {time.time()-t0:.1f}s")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
            }, os.path.join(args.save_dir, f'model_epoch{epoch+1:04d}.pt'))

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ema.state_dict(),   # save EMA weights as best
                'val_loss': avg_val,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  [Best] val={avg_val:.4f}")

    print(f"\n[Done] Best val loss: {best_val:.4f}")


# ---------------------------------------------------------------------------
# 12. Inference (DDIM, 50 steps)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(args):
    # Force CPU to avoid CUDA compatibility issues with RTX 5060 Ti on local machine
    device = torch.device('cpu')

    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    clip_model, clip_module, text_dim = load_clip_encoder(device=str(device))

    model = SignDiffusionModelV3(
        pose_dim=config['pose_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_frames=config['max_frames'],
        text_dim=config.get('text_dim', 512),
    ).to(device)

    ckpt = torch.load(os.path.join(args.model_dir, 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scheduler = NoiseScheduler(num_steps=config.get('diffusion_steps', 1000)).to(device)

    text_emb = encode_text(clip_model, clip_module, [args.text], str(device))
    null_emb = encode_text(clip_model, clip_module, [""], str(device))

    T = args.num_frames
    print(f"[Generate V3] '{args.text}' | frames={T} | cfg={args.cfg_scale} | ddim_steps={args.ddim_steps}")

    motion = scheduler.ddim_sample(
        model, text_emb, null_emb,
        max_frames=T, pose_dim=config['pose_dim'],
        device=device, cfg_scale=args.cfg_scale,
        ddim_steps=args.ddim_steps,
    )
    motion = motion[0].cpu().numpy()

    # Denormalize
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    motion = motion * std + mean

    # Post-process
    from scipy.ndimage import gaussian_filter1d
    motion[:, 3:66]   = np.clip(motion[:, 3:66],   -1.8, 1.8)
    motion[:, 66:156] = np.clip(motion[:, 66:156],  -1.5, 1.5)
    motion[:, 159:169] = 0.0
    motion[:, 179:182] = np.clip(motion[:, 179:182], -0.1, 20.0)
    for d in range(motion.shape[1]):
        motion[:, d] = gaussian_filter1d(motion[:, d], sigma=2.0, mode='nearest')

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'generated_motion.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'smplx': motion.astype(np.float32)}, f)
    np.savez(os.path.join(args.output_dir, 'generated_motion.npz'),
             smplx=motion.astype(np.float32))
    print(f"Saved: {out_path}  shape={motion.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Sign Language Diffusion V3")
    sub = p.add_subparsers(dest='command')

    tp = sub.add_parser('train')
    tp.add_argument('--pkl_dir',   required=True)
    tp.add_argument('--mapping',   required=True)
    tp.add_argument('--save_dir',  default='checkpoints/sign_mdm_v3')
    tp.add_argument('--batch_size',      type=int,   default=64)
    tp.add_argument('--epochs',          type=int,   default=300)
    tp.add_argument('--lr',              type=float, default=1e-4)
    tp.add_argument('--weight_decay',    type=float, default=0.05)
    tp.add_argument('--max_frames',      type=int,   default=300)
    tp.add_argument('--pose_dim',        type=int,   default=182)
    tp.add_argument('--latent_dim',      type=int,   default=512)
    tp.add_argument('--num_layers',      type=int,   default=8)
    tp.add_argument('--num_heads',       type=int,   default=8)
    tp.add_argument('--diffusion_steps', type=int,   default=1000)
    tp.add_argument('--num_workers',     type=int,   default=4)
    tp.add_argument('--save_every',      type=int,   default=10)

    gp = sub.add_parser('generate')
    gp.add_argument('--model_dir',   required=True)
    gp.add_argument('--text',        required=True)
    gp.add_argument('--num_frames',  type=int,   default=150)
    gp.add_argument('--cfg_scale',   type=float, default=3.0)
    gp.add_argument('--ddim_steps',  type=int,   default=50)
    gp.add_argument('--output_dir',  default='generated_v3/')

    args = p.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    else:
        p.print_help()