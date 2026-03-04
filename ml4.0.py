"""
Muon track reconstruction with a Transformer (Encoder-only).

Goal:
  Input  : per-event vector of photon counts on 48 SiPMs  (shape: [48])
           plus known geometry position of each SiPM       (shape: [48,3])
  Output : muon track params (x0, y0, z0=0, direction vx,vy,vz)

Notes / assumptions (based on your data snippet):
  - muonx,muony,muonz are one "reference point" on the muon track (e.g. SLab hit position).
  - (px,py,pz) are momentum components; direction is p normalized.
  - We compute the intersection with z=0 plane:
        x0 = x - (px/pz)*z
        y0 = y - (py/pz)*z
    (requires pz != 0)
  - SiPMHit_Det_ID gives list of detected SiPM IDs per event. We build count matrix.

If your SiPM geometry builder returns only 12 positions (as your function does),
we will repeat it 4 times to reach 48 by default. Please replace that part with
your true 48-position mapping if needed.
"""

import os
import math
import random
import numpy as np
import uproot
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# -------------------------
# 0) Repro / device
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 1) Geometry: SiPM positions (REAL 48 POS)
# -------------------------
Scintxlength = 20.0
Scintylength = 20.0
SiPMzlength  = 0.1
Scintzlength = 2.0
ESRthickness = 0.1
Tapethickness = 0.1
number_of_slabs = 4
slab_offsets = [-9.63, 4.2, 4.2, 4.2]

total_thickness = Scintzlength + ESRthickness + Tapethickness
slab_z = np.array([slab_offsets[j] + total_thickness * (j - 1) for j in range(4)], dtype=float)

x0 = Scintxlength / 2 + SiPMzlength / 2
y0 = Scintylength / 2 + SiPMzlength / 2

def build_local_sipms():
    local_pos = []
    for sign_x in (-1, 1):
        for i in range(3):
            y = Scintylength / 4 * (i - 1)
            local_pos.append((sign_x * x0, y, 0.0))
    for sign_y in (-1, 1):
        for i in range(3):
            x = Scintxlength / 4 * (i - 1)
            local_pos.append((x, sign_y * y0, 0.0))
    return np.array(local_pos, dtype=float)

LOCAL_POS_12 = build_local_sipms()

def build_global_sipms():
    sipm_pos = np.zeros((48, 3), dtype=float)
    for i in range(12):
        lx, ly, lz = LOCAL_POS_12[i]
        for j in range(number_of_slabs):
            sid = i * number_of_slabs + j   # sid in [0..47]
            sipm_pos[sid] = (lx, ly, slab_z[j] + lz)
    return sipm_pos

SIPM_POS = build_global_sipms().astype(np.float32)  # (48,3)

# -------------------------
# 2) Data loading + feature/label building
# -------------------------
def load_root_build_matrix(root_path):
    file = uproot.open(root_path)
    tree = file["Simu;1"]

    SiPMHit_Det_ID = tree["SiPMHit_Det_ID"].array()
    muonx = tree["SLabHit_pos_x"].array()
    muony = tree["SLabHit_pos_y"].array()
    muonz = tree["SLabHit_pos_z"].array()
    muonpx = tree["muon_px"].array()
    muonpy = tree["muon_py"].array()
    muonpz = tree["muon_pz"].array()

    particle_number = len(SiPMHit_Det_ID)
    num_detector = 48

    event_detID_matrix = np.zeros((particle_number, num_detector), dtype=np.float32)

    # Build count vector per event
    for i in range(particle_number):
        detectorID = SiPMHit_Det_ID[i]
        for ID in detectorID:
            # ensure int and in-range
            idx = int(ID)
            if 0 <= idx < num_detector:
                event_detID_matrix[i, idx] += 1.0

    # Build labels: (x0,y0,z0=0, vx,vy,vz)
    y = np.zeros((particle_number, 6), dtype=np.float32)

    bad = 0
    for i in range(particle_number):
        if len(muonx[i])>0:
            x = float(muonx[i][0])
            y0_ = float(muony[i][0])
            z = float(muonz[i][0])

            px = float(muonpx[i][0])
            py = float(muonpy[i][0])
            pz = float(muonpz[i][0])

            # direction
            pnorm = math.sqrt(px * px + py * py + pz * pz) + 1e-12
            vx, vy, vz = px / pnorm, py / pnorm, pz / pnorm

            # intersection with z=0 plane (line through (x,y,z) along p)
            # x(t) = x + (px)*t, z(t) = z + (pz)*t => t0 = -z/pz
            if abs(pz) < 1e-9:
                bad += 1
                # fallback: keep x,y as "x0,y0" if pz ~ 0 (rare; should inspect)
                x0 = x
                yy0 = y0_
            else:
                t0 = -z / pz
                x0 = x + px * t0
                yy0 = y0_ + py * t0

            y[i, 0] = x0
            y[i, 1] = yy0
            y[i, 2] = 0.0
            y[i, 3] = vx
            y[i, 4] = vy
            y[i, 5] = vz

    if bad > 0:
        print(f"[Warn] {bad}/{particle_number} events had |pz|~0; used fallback for x0,y0.")

    return event_detID_matrix, y


class MuonSipmDataset(Dataset):
    """
    Each sample:
      tokens: (48, feat_dim)
      target: (6,) -> [x0,y0,z0=0,vx,vy,vz]
    """
    def __init__(self, counts_48, targets_6, sipm_pos_48x3):
        super().__init__()
        assert counts_48.shape[1] == 48
        assert sipm_pos_48x3.shape == (48, 3)
        assert targets_6.shape[1] == 6

        self.counts = counts_48.astype(np.float32)
        self.targets = targets_6.astype(np.float32)
        self.pos = sipm_pos_48x3.astype(np.float32)

        # optional: normalize position inputs
        self.pos_mean = self.pos.mean(axis=0, keepdims=True)
        self.pos_std = self.pos.std(axis=0, keepdims=True) + 1e-6
        self.pos_norm = (self.pos - self.pos_mean) / self.pos_std  # (48,3)

        # normalize regression targets for x0,y0 (helps training); store stats
        xy = self.targets[:, :2]
        self.xy_mean = xy.mean(axis=0, keepdims=True).astype(np.float32)
        self.xy_std = (xy.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        c = self.counts[idx]  # (48,)
        t = self.targets[idx] # (6,)

        # features per token
        sumc = float(c.sum()) + 1e-6
        logc = np.log1p(c)
        relc = c / sumc
        mask = (c > 0).astype(np.float32)

        # token feature: [log1p(c), c/sum(c), mask, x,y,z(normed)]
        # -> shape (48, 6)
        feats = np.stack([logc, relc, mask], axis=1)  # (48,3)
        feats = np.concatenate([feats, self.pos_norm], axis=1)  # (48,6)

        # normalize targets (only x0,y0). direction stays unit.
        xy = (t[:2] - self.xy_mean[0]) / self.xy_std[0]
        out = np.concatenate([xy, t[2:]], axis=0).astype(np.float32)

        return torch.from_numpy(feats), torch.from_numpy(out)

    def denorm_xy(self, xy_norm):
        # xy_norm: (...,2)
        return xy_norm * torch.from_numpy(self.xy_std).to(xy_norm.device) + torch.from_numpy(self.xy_mean).to(xy_norm.device)


# -------------------------
# 3) Transformer model
# -------------------------
class TokenMLP(nn.Module):
    def __init__(self, in_dim, d_model, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MuonTransformer(nn.Module):
    def __init__(self, token_dim=6, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embed = TokenMLP(token_dim, d_model, hidden=2*d_model, dropout=dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # heads
        self.head_xy = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # normalized x0,y0
        )
        self.head_v = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # raw direction
        )

        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens):
        """
        tokens: (B, 48, token_dim)
        returns:
          xy_norm: (B,2)
          v_hat : (B,3) unit direction
        """
        B = tokens.size(0)
        x = self.token_embed(tokens)             # (B,48,d_model)
        cls = self.cls.expand(B, -1, -1)         # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)           # (B,49,d_model)
        h = self.encoder(x)                      # (B,49,d_model)
        h_cls = h[:, 0, :]                       # (B,d_model)

        xy_norm = self.head_xy(h_cls)
        v_raw = self.head_v(h_cls)
        v_hat = F.normalize(v_raw, dim=-1, eps=1e-8)
        return xy_norm, v_hat


def loss_fn(xy_pred_norm, v_pred, target):
    """
    target: (B,6) -> [x_norm,y_norm,z0,vx,vy,vz] where x_norm,y_norm are normalized already
    """
    xy_true = target[:, :2]
    v_true = target[:, 3:6]

    loss_xy = F.mse_loss(xy_pred_norm, xy_true)
    # cosine loss for direction
    cos = (v_pred * v_true).sum(dim=-1).clamp(-1.0, 1.0)
    loss_v = (1.0 - cos).mean()
    return loss_xy, loss_v


# -------------------------
# 4) Train / eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, dataset: MuonSipmDataset):
    model.eval()
    total = 0
    loss_xy_sum = 0.0
    loss_v_sum = 0.0

    # metrics in physical units
    pos_errs = []
    ang_errs = []

    for tokens, target in loader:
        tokens = tokens.to(DEVICE)
        target = target.to(DEVICE)

        xy_pred_norm, v_pred = model(tokens)
        lxy, lv = loss_fn(xy_pred_norm, v_pred, target)

        loss_xy_sum += float(lxy.item()) * tokens.size(0)
        loss_v_sum += float(lv.item()) * tokens.size(0)
        total += tokens.size(0)

        # denormalize xy
        xy_pred = dataset.denorm_xy(xy_pred_norm).cpu().numpy()
        xy_true = dataset.denorm_xy(target[:, :2]).cpu().numpy()

        # position error (Euclidean in xy)
        pe = np.sqrt(((xy_pred - xy_true) ** 2).sum(axis=1))
        pos_errs.append(pe)

        # angle error
        v_true = target[:, 3:6].cpu().numpy()
        v_pred_np = v_pred.cpu().numpy()
        dot = np.sum(v_true * v_pred_np, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        ang_errs.append(ang)

    pos_errs = np.concatenate(pos_errs) if pos_errs else np.array([])
    ang_errs = np.concatenate(ang_errs) if ang_errs else np.array([])

    return {
        "loss_xy": loss_xy_sum / max(total, 1),
        "loss_v": loss_v_sum / max(total, 1),
        "pos_err_mean": float(pos_errs.mean()) if pos_errs.size else float("nan"),
        "pos_err_median": float(np.median(pos_errs)) if pos_errs.size else float("nan"),
        "ang_err_mean_deg": float(ang_errs.mean()) if ang_errs.size else float("nan"),
        "ang_err_median_deg": float(np.median(ang_errs)) if ang_errs.size else float("nan"),
    }


def train(
    root_path,
    epochs=200,
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-3,
    lambda_xy=1.0,
    lambda_v=1.0,
):
    counts, targets = load_root_build_matrix(root_path)
    dataset = MuonSipmDataset(counts, targets, SIPM_POS)

    # split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MuonTransformer(token_dim=6, d_model=128, nhead=8, num_layers=4, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for tokens, target in train_loader:
            tokens = tokens.to(DEVICE)
            target = target.to(DEVICE)

            xy_pred_norm, v_pred = model(tokens)
            lxy, lv = loss_fn(xy_pred_norm, v_pred, target)
            loss = lambda_xy * lxy + lambda_v * lv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * tokens.size(0)
            n_seen += tokens.size(0)

        train_loss = loss_sum / max(n_seen, 1)

        val_metrics = evaluate(model, val_loader, dataset)
        val_loss = lambda_xy * val_metrics["loss_xy"] + lambda_v * val_metrics["loss_v"]

        history["train"].append({"epoch": ep, "loss": train_loss})
        history["val"].append({"epoch": ep, "loss": val_loss, **val_metrics})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:4d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"pos(mean/med) {val_metrics['pos_err_mean']:.3f}/{val_metrics['pos_err_median']:.3f} | "
                f"ang(mean/med) {val_metrics['ang_err_mean_deg']:.2f}/{val_metrics['ang_err_median_deg']:.2f} deg"
            )

    model.load_state_dict(best_state)
    return model, dataset, history, val_loader


# -------------------------
# 5) Visualization
# -------------------------
@torch.no_grad()
def visualize_predictions(model, dataset: MuonSipmDataset, loader, max_batches=3):
    model.eval()
    xs_true, ys_true, xs_pred, ys_pred = [], [], [], []
    ang_errs = []

    batches = 0
    for tokens, target in loader:
        tokens = tokens.to(DEVICE)
        target = target.to(DEVICE)
        xy_pred_norm, v_pred = model(tokens)

        xy_pred = dataset.denorm_xy(xy_pred_norm).cpu().numpy()
        xy_true = dataset.denorm_xy(target[:, :2]).cpu().numpy()

        xs_true.append(xy_true[:, 0]); ys_true.append(xy_true[:, 1])
        xs_pred.append(xy_pred[:, 0]); ys_pred.append(xy_pred[:, 1])

        v_true = target[:, 3:6].cpu().numpy()
        v_pred_np = v_pred.cpu().numpy()
        dot = np.sum(v_true * v_pred_np, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        ang_errs.append(ang)

        batches += 1
        if batches >= max_batches:
            break

    xs_true = np.concatenate(xs_true); ys_true = np.concatenate(ys_true)
    xs_pred = np.concatenate(xs_pred); ys_pred = np.concatenate(ys_pred)
    ang_errs = np.concatenate(ang_errs)

    plt.figure()
    plt.scatter(xs_true, ys_true, s=12, label="true")
    plt.scatter(xs_pred, ys_pred, s=12, label="pred")
    plt.xlabel("x0"); plt.ylabel("y0")
    plt.title("Intersection on z=0: true vs pred")
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(ang_errs, bins=30)
    plt.xlabel("direction angle error (deg)")
    plt.ylabel("count")
    plt.title("Direction error histogram")
    plt.show()

# -------------------------
# 5.5) Training curves & diagnostics
# -------------------------
def _extract_series(history):
    """
    history: dict with keys 'train' and 'val'
    Each is list of dicts (one per epoch).
    """
    train = history.get("train", [])
    val = history.get("val", [])

    epochs = [d["epoch"] for d in train] if train else [d["epoch"] for d in val]

    train_loss = [d.get("loss", np.nan) for d in train]
    val_loss   = [d.get("loss", np.nan) for d in val]

    # Optional metrics in val dict (from evaluate())
    val_loss_xy = [d.get("loss_xy", np.nan) for d in val]
    val_loss_v  = [d.get("loss_v", np.nan) for d in val]

    pos_mean = [d.get("pos_err_mean", np.nan) for d in val]
    pos_med  = [d.get("pos_err_median", np.nan) for d in val]
    ang_mean = [d.get("ang_err_mean_deg", np.nan) for d in val]
    ang_med  = [d.get("ang_err_median_deg", np.nan) for d in val]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_loss_xy": val_loss_xy,
        "val_loss_v": val_loss_v,
        "pos_mean": pos_mean,
        "pos_median": pos_med,
        "ang_mean": ang_mean,
        "ang_median": ang_med,
    }


def plot_training_curves(history):
    s = _extract_series(history)
    epochs = s["epochs"]

    # 1) total loss: train vs val
    plt.figure()
    plt.plot(epochs, s["train_loss"], label="train total loss")
    plt.plot(epochs, s["val_loss"], label="val total loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training curve: total loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2) val loss decomposition
    if not all(np.isnan(s["val_loss_xy"])) or not all(np.isnan(s["val_loss_v"])):
        plt.figure()
        plt.plot(epochs, s["val_loss_xy"], label="val loss_xy (MSE)")
        plt.plot(epochs, s["val_loss_v"], label="val loss_v (1-cos)")
        plt.xlabel("epoch")
        plt.ylabel("loss component")
        plt.title("Validation loss components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 3) position error curve
    if not all(np.isnan(s["pos_mean"])) or not all(np.isnan(s["pos_median"])):
        plt.figure()
        plt.plot(epochs, s["pos_mean"], label="val pos err mean")
        plt.plot(epochs, s["pos_median"], label="val pos err median")
        plt.xlabel("epoch")
        plt.ylabel("position error (same unit as x,y)")
        plt.title("Validation position error")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 4) angle error curve
    if not all(np.isnan(s["ang_mean"])) or not all(np.isnan(s["ang_median"])):
        plt.figure()
        plt.plot(epochs, s["ang_mean"], label="val angle err mean (deg)")
        plt.plot(epochs, s["ang_median"], label="val angle err median (deg)")
        plt.xlabel("epoch")
        plt.ylabel("angle error (deg)")
        plt.title("Validation direction error")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


@torch.no_grad()
def collect_predictions(model, dataset, loader, max_batches=None):
    """
    Collect predictions & targets in physical units (denormalized).
    Returns dict of numpy arrays.
    """
    model.eval()
    xy_true_all, xy_pred_all = [], []
    v_true_all, v_pred_all = [], []

    batches = 0
    for tokens, target in loader:
        tokens = tokens.to(DEVICE)
        target = target.to(DEVICE)

        xy_pred_norm, v_pred = model(tokens)

        xy_pred = dataset.denorm_xy(xy_pred_norm).detach().cpu().numpy()
        xy_true = dataset.denorm_xy(target[:, :2]).detach().cpu().numpy()

        v_true = target[:, 3:6].detach().cpu().numpy()
        v_pred_np = v_pred.detach().cpu().numpy()

        xy_true_all.append(xy_true)
        xy_pred_all.append(xy_pred)
        v_true_all.append(v_true)
        v_pred_all.append(v_pred_np)

        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    xy_true_all = np.concatenate(xy_true_all, axis=0)
    xy_pred_all = np.concatenate(xy_pred_all, axis=0)
    v_true_all = np.concatenate(v_true_all, axis=0)
    v_pred_all = np.concatenate(v_pred_all, axis=0)

    # errors
    pos_err = np.sqrt(((xy_pred_all - xy_true_all) ** 2).sum(axis=1))
    dot = np.sum(v_true_all * v_pred_all, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))

    return {
        "xy_true": xy_true_all,
        "xy_pred": xy_pred_all,
        "v_true": v_true_all,
        "v_pred": v_pred_all,
        "pos_err": pos_err,
        "ang_err_deg": ang_err_deg,
        "dot": dot,
    }


def plot_parity_and_residuals(pred_pack):
    xy_true = pred_pack["xy_true"]
    xy_pred = pred_pack["xy_pred"]
    pos_err = pred_pack["pos_err"]
    ang_err = pred_pack["ang_err_deg"]

    # Parity plot for x0
    plt.figure()
    plt.scatter(xy_true[:, 0], xy_pred[:, 0], s=10, alpha=0.7)
    mn = min(xy_true[:, 0].min(), xy_pred[:, 0].min())
    mx = max(xy_true[:, 0].max(), xy_pred[:, 0].max())
    plt.plot([mn, mx], [mn, mx])  # y=x
    plt.xlabel("true x0")
    plt.ylabel("pred x0")
    plt.title("Parity plot: x0")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Parity plot for y0
    plt.figure()
    plt.scatter(xy_true[:, 1], xy_pred[:, 1], s=10, alpha=0.7)
    mn = min(xy_true[:, 1].min(), xy_pred[:, 1].min())
    mx = max(xy_true[:, 1].max(), xy_pred[:, 1].max())
    plt.plot([mn, mx], [mn, mx])  # y=x
    plt.xlabel("true y0")
    plt.ylabel("pred y0")
    plt.title("Parity plot: y0")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Residual histograms
    res_x = xy_pred[:, 0] - xy_true[:, 0]
    res_y = xy_pred[:, 1] - xy_true[:, 1]

    plt.figure()
    plt.hist(res_x, bins=30)
    plt.xlabel("x0 residual (pred-true)")
    plt.ylabel("count")
    plt.title("Residual distribution: x0")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure()
    plt.hist(res_y, bins=30)
    plt.xlabel("y0 residual (pred-true)")
    plt.ylabel("count")
    plt.title("Residual distribution: y0")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Position error histogram
    plt.figure()
    plt.hist(pos_err, bins=30)
    plt.xlabel("position error |(x0,y0)|")
    plt.ylabel("count")
    plt.title("Position error histogram")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Angle error histogram
    plt.figure()
    plt.hist(ang_err, bins=30)
    plt.xlabel("direction angle error (deg)")
    plt.ylabel("count")
    plt.title("Angle error histogram")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Angle error CDF
    ang_sorted = np.sort(ang_err)
    cdf = np.arange(1, len(ang_sorted) + 1) / len(ang_sorted)
    plt.figure()
    plt.plot(ang_sorted, cdf)
    plt.xlabel("direction angle error (deg)")
    plt.ylabel("CDF")
    plt.title("Angle error CDF")
    plt.grid(True, alpha=0.3)
    plt.show()

# -------------------------
# 6) Run
# -------------------------
if __name__ == "__main__":
    ROOT_PATH = r"C:/Users/22205/Documents/muon_detector/output/output_500range.root"

    model, dataset, history, val_loader = train(
        ROOT_PATH,
        epochs=200,
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-3,
        lambda_xy=1.0,
        lambda_v=1.0,
    )
    plot_training_curves(history)

    pred_pack = collect_predictions(model, dataset, val_loader, max_batches=None)
    plot_parity_and_residuals(pred_pack)

    visualize_predictions(model, dataset, val_loader, max_batches=10)

    # Example: predict one event
    model.eval()
    feats, target = dataset[0]
    with torch.no_grad():
        xy_norm, v_hat = model(feats.unsqueeze(0).to(DEVICE))
        xy = dataset.denorm_xy(xy_norm).cpu().numpy()[0]
        v = v_hat.cpu().numpy()[0]

    print("Pred:", "x0,y0,z0=", xy[0], xy[1], 0.0, "dir=", v)
    # denorm true
    t_xy = dataset.denorm_xy(target[:2].unsqueeze(0)).cpu().numpy()[0]
    print("True:", "x0,y0,z0=", t_xy[0], t_xy[1], 0.0, "dir=", target[3:].numpy())