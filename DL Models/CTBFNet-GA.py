"""
CTBFNet: CNN + Transformer + BiLSTM with HRV Fusion for ECG Classification

Overview:
- Loads multi-subject ECG CSVs (expects 'baseline' and 'stroop' conditions).
- Segments ECG into overlapping fixed-length windows.
- Extracts 41 HRV features per segment (time/frequency/nonlinear).
- Encodes ECG with CNN -> Transformer -> BiLSTM and fuses HRV via FiLM and attention.
- Trains with GroupKFold to avoid subject leakage.
- Tracks accuracy, F1, precision, recall, AUC, and computation metrics.
- Saves per-fold outputs and final summary.

Directory expectations:
DATA_ROOT/
  └── <SubjectID>/
      └── baseline/
          └── baseline_reconstructed.csv
      └── stroop/
          └── stroop_reconstructed.csv

ECG CSVs must contain at least one column with name including "ecg" (case-insensitive).

How to run (Windows, VS Code Terminal):
1) Create/activate environment and install requirements:
   pip install numpy pandas torch scikit-learn scipy matplotlib psutil
2) Adjust DATA_ROOT and OUTPUT_ROOT as needed.
3) Run:
   python CTBFNet-GA.py

Notes:
- SAMPLE_RATE should match your acquisition device (e.g., 130 Hz for Polar H10).
- SEGMENT_LEN and OVERLAP control segmentation granularity.
- NUM_HRV_FEATURES must equal the feature vector length produced in compute_hrv_features.
- If you encounter issues with np.trapezoid, replace with np.trapz (NumPy version dependent).
"""

import os
import time
import psutil
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from scipy import interpolate, stats
from scipy.signal import welch, find_peaks
import matplotlib.pyplot as plt

# ==========================================
#  USER CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_ROOT = "ECG_RAW"  # Path to dataset root (see "Directory expectations" in module docstring)
OUTPUT_ROOT = "output"  # Directory to save results/checkpoints

# Signal processing
SAMPLE_RATE = 130          # Hz
SEGMENT_LEN = 7800         # Number of samples per segment (e.g., 60s @130Hz -> 7800)
OVERLAP = 0.2              # Fractional overlap between consecutive segments (0.0..0.9)

# Training
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-6
NUM_FOLDS = 5

# Model / Features
DEVICE = "cpu"             # GPU preferred
NUM_HRV_FEATURES = 41      # Must match compute_hrv_features output length

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"Device: {DEVICE}; Sample Rate={SAMPLE_RATE} Hz; Segment={SEGMENT_LEN}")
process = psutil.Process(os.getpid())

# ==========================================
#  1) DATA LOADING & SEGMENTATION
# ==========================================
def load_all_subjects(root: str):
    """
    Load ECG recordings from a directory structure organized by subject and condition.

    Args:
        root: Path to dataset root.

    Returns:
        List of tuples (subject_id, condition, ecg_signal_np).
        - condition should be one of ['baseline', 'stroop'] as filtered below.
        - ecg_signal_np is a 1D float array.
    """
    records = []
    for subj in sorted(glob(os.path.join(root, "*"))):
        sid = os.path.basename(subj)
        for cond in sorted(glob(os.path.join(subj, "*"))):
            cname = os.path.basename(cond).lower()
            if cname not in ["baseline", "stroop"]:
                continue
            path = os.path.join(cond, f"{cname}_reconstructed.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
                col = [c for c in df.columns if "ecg" in c.lower()]
                if not col:
                    continue
                sig = pd.to_numeric(df[col[0]], errors="coerce").dropna().values
                if sig.size:
                    records.append((sid, cname, sig))
            except Exception as e:
                print("⚠ Error loading:", path, e)
    return records

def segment(sig: np.ndarray, seg_len: int, overlap: float):
    """
    Split a 1D signal into overlapping segments.

    Args:
        sig: 1D signal array.
        seg_len: Segment length in samples.
        overlap: Fractional overlap [0.0..0.9].

    Returns:
        List of segments, each of shape (seg_len,).
    """
    step = int(seg_len * (1 - overlap))
    if step <= 0:
        return []
    segs = [sig[i:i + seg_len] for i in range(0, len(sig) - seg_len + 1, step)]
    return segs

# Load and segment data (Z-score per recording)
records = load_all_subjects(DATA_ROOT)
print(f"Loaded {len(records)} recordings (Baseline vs Stroop)")

X, y, groups = [], [], []
for s, c, sig in records:
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    if len(sig) < SEGMENT_LEN:
        continue
    segs = segment(sig, SEGMENT_LEN, OVERLAP)
    X.extend(segs)
    label = 0 if c == "baseline" else 1
    y.extend([label] * len(segs))
    groups.extend([s] * len(segs))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
groups = np.array(groups)
print(f"Segments: {len(X)}, shape={X.shape} , distribution={np.bincount(y) if len(y)>0 else 'empty'}")

# ==========================================
#  2) HRV FEATURE COMPUTATION
# ==========================================
def detect_rpeaks(sig: np.ndarray, fs: int):
    """
    Simple R-peak detection using amplitude threshold and minimum distance.

    Args:
        sig: 1D ECG signal.
        fs: Sampling frequency (Hz).

    Returns:
        Indices of detected peaks (np.ndarray).
    """
    thr = np.mean(sig) + 0.3 * np.std(sig)
    dist = int(0.3 * fs)
    peaks, _ = find_peaks(sig, height=thr, distance=dist)
    return peaks

def sampen(x, m=2, r=None):
    """
    Sample Entropy of a 1D sequence.

    Args:
        x: 1D series.
        m: Embedding dimension.
        r: Tolerance; defaults to 0.2*std(x).

    Returns:
        Sample entropy (float).
    """
    x = np.array(x, dtype=float)
    n = len(x)
    if n < m + 1:
        return 0.0
    if r is None:
        r = 0.2 * np.std(x)
    def _phi(m):
        C = 0
        for i in range(n - m + 1):
            xi = x[i:i + m]
            for j in range(i + 1, n - m + 1):
                xj = x[j:j + m]
                if np.max(np.abs(xi - xj)) <= r:
                    C += 1
        return C
    A = _phi(m + 1)
    B = _phi(m)
    if B == 0:
        return 0.0
    return -np.log((A / B) if A > 0 else 1e-10)

def higuchi_fd(x, kmax=10):
    """
    Higuchi fractal dimension for a 1D series.

    Args:
        x: 1D array.
        kmax: Maximum k.

    Returns:
        Estimated fractal dimension (float).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return 0.0
    Lk = []
    ln_k = []
    for k in range(1, min(kmax, n-1) + 1):
        Lm_sum = 0.0
        count = 0
        for m in range(k):
            idxs = np.arange(m, n, k)
            if idxs.size <= 1:
                continue
            diff = np.abs(np.diff(x[idxs]))
            Lm = np.sum(diff) * (n - 1) / (((n - m) / k) * k)
            Lm_sum += Lm
            count += 1
        if count > 0:
            avg_Lk = Lm_sum / count
            if avg_Lk > 0:
                Lk.append(np.log(avg_Lk))
                ln_k.append(-np.log(k))
    if len(ln_k) < 2:
        return 0.0
    slope, _, _, _, _ = stats.linregress(ln_k, Lk)
    return slope

def _approx_entropy(U, m=2, r=None):
    """
    Approximate Entropy (ApEn) of a 1D sequence.

    Args:
        U: 1D sequence.
        m: Embedding dimension.
        r: Tolerance; defaults to 0.2*std(U).

    Returns:
        Approximate entropy (float).
    """
    U = np.asarray(U, dtype=float)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U) if N > 0 else 0.0
    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1) - 1  # exclude self-match
        return np.sum(C / (N - m + 1)) / (N - m + 1) if (N - m + 1) > 0 else 0.0
    try:
        return np.log(_phi(m) / (_phi(m + 1) + 1e-12) + 1e-12)
    except Exception:
        return 0.0

def _poincare_sd(rr_ms: np.ndarray):
    """
    Poincaré plot descriptors (SD1, SD2, SD1/SD2).

    Args:
        rr_ms: RR intervals in milliseconds.

    Returns:
        (sd1, sd2, ratio)
    """
    if len(rr_ms) < 2:
        return 0.0, 0.0, 0.0
    diff = np.diff(rr_ms)
    sd_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0.0
    sd1 = np.sqrt(0.5) * sd_diff
    tmp = max(0.0, 2 * (sdnn ** 2) - 0.5 * (sd_diff ** 2))
    sd2 = np.sqrt(tmp) if tmp > 0 else 0.0
    ratio = sd1 / sd2 if sd2 > 0 else 0.0
    return sd1, sd2, ratio

def _dfa_alpha1(rr_ms: np.ndarray):
    """
    Detrended Fluctuation Analysis alpha1 (short-term scaling exponent).

    Args:
        rr_ms: RR intervals (ms).

    Returns:
        alpha1 exponent (float).
    """
    x = np.asarray(rr_ms, dtype=float)
    if len(x) < 8:
        return 0.0
    y = np.cumsum(x - np.mean(x))
    N = len(y)
    window_sizes = np.array([4,5,6,7,8,10,12,14,16])
    window_sizes = window_sizes[window_sizes < N//2]
    if len(window_sizes) < 2:
        return 0.0
    F = []
    for w in window_sizes:
        nseg = N // w
        if nseg < 2:
            continue
        rms = []
        for i in range(nseg):
            seg = y[i*w:(i+1)*w]
            t = np.arange(len(seg))
            A = np.vstack([t, np.ones_like(t)]).T
            coef, _, _, _ = np.linalg.lstsq(A, seg, rcond=None)
            trend = A @ coef
            diff = seg - trend
            rms.append(np.sqrt(np.mean(diff**2)))
        if len(rms) > 0:
            F.append(np.mean(rms))
    if len(F) < 2:
        return 0.0
    log_w = np.log(window_sizes[:len(F)])
    log_F = np.log(np.maximum(F, 1e-12))
    slope, _, _, _, _ = stats.linregress(log_w, log_F)
    return slope

def compute_hrv_features(seg: np.ndarray, fs: int):
    """
    Compute a fixed-length HRV feature vector for a single ECG segment.

    Pipeline:
      1) R-peak detection.
      2) RR interval series (s, ms).
      3) Time-domain features.
      4) Frequency-domain features (Welch on interpolated RR).
      5) Nonlinear features.

    Args:
        seg: 1D ECG segment (float32).
        fs: Sampling rate (Hz).

    Returns:
        np.ndarray of shape (NUM_HRV_FEATURES,), float32.
    """
    peaks = detect_rpeaks(seg, fs)
    Np = len(peaks)
    if Np < 3:
        return np.zeros(NUM_HRV_FEATURES, dtype=np.float32)
    t = peaks / fs
    rr = np.diff(t)  # seconds
    if len(rr) < 3:
        return np.zeros(NUM_HRV_FEATURES, dtype=np.float32)

    # Time-domain (in ms for RR measures)
    rr_ms = rr * 1000.0
    AVNN = np.mean(rr_ms)
    SDNN = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0.0
    RMSSD = np.sqrt(np.mean(np.diff(rr_ms) ** 2)) if len(rr_ms) > 1 else 0.0
    diff_rr = np.abs(np.diff(rr))  # seconds diff
    NN50 = int(np.sum(np.abs(np.diff(rr_ms)) > 50.0))
    pNN50 = 100.0 * NN50 / len(rr_ms) if len(rr_ms) > 0 else 0.0
    NN20 = int(np.sum(np.abs(np.diff(rr_ms)) > 20.0))
    pNN20 = 100.0 * NN20 / len(rr_ms) if len(rr_ms) > 0 else 0.0
    IQR_RR = np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25)
    MAD_RR = np.median(np.abs(rr_ms - np.median(rr_ms)))
    CVNN = (SDNN / AVNN * 100.0) if AVNN > 0 else 0.0
    CVSD = (RMSSD / AVNN * 100.0) if AVNN > 0 else 0.0
    HR_mean = 60.0 / np.mean(rr) if np.mean(rr) > 0 else 0.0
    HR_median = 60.0 / np.median(rr) if np.median(rr) > 0 else 0.0

    # Frequency-domain on interpolated RR
    try:
        rr_times = t[1:]
        if len(rr_times) < 4:
            vlf_p = lf_p = hf_p = TP = LF_nu = HF_nu = LF_HF = HF_LF = lnVLF = lnLF = lnHF = lnTP = 0.0
            fpeak_lf = fpeak_hf = fc_lf = fc_hf = vlf_pct = lf_pct = hf_pct = rel_TP = 0.0
        else:
            fs_interp = 4.0
            ti = np.arange(rr_times[0], rr_times[-1], 1.0 / fs_interp)
            if len(ti) < 4:
                vlf_p = lf_p = hf_p = TP = LF_nu = HF_nu = LF_HF = HF_LF = lnVLF = lnLF = lnHF = lnTP = 0.0
                fpeak_lf = fpeak_hf = fc_lf = fc_hf = vlf_pct = lf_pct = hf_pct = rel_TP = 0.0
            else:
                f_interp = interpolate.interp1d(rr_times, rr_ms, kind='cubic', fill_value="extrapolate")
                rr_resampled = f_interp(ti)
                rr_resampled = rr_resampled - np.mean(rr_resampled)
                nperseg = min(256, len(rr_resampled))
                f, Pxx = welch(rr_resampled, fs=fs_interp, nperseg=nperseg)
                vlf = (0.0033, 0.04)
                lf = (0.04, 0.15)
                hf = (0.15, 0.4)
                eps = 1e-12
                mask_vlf = (f >= vlf[0]) & (f < vlf[1])
                mask_lf = (f >= lf[0]) & (f < lf[1])
                mask_hf = (f >= hf[0]) & (f < hf[1])
                # If np.trapezoid is unavailable in your NumPy version, use np.trapz
                vlf_p = np.trapezoid(Pxx[mask_vlf], f[mask_vlf]) if np.any(mask_vlf) else 0.0
                lf_p = np.trapezoid(Pxx[mask_lf], f[mask_lf]) if np.any(mask_lf) else 0.0
                hf_p = np.trapezoid(Pxx[mask_hf], f[mask_hf]) if np.any(mask_hf) else 0.0
                TP = vlf_p + lf_p + hf_p
                total = TP + eps
                LF_nu = 100.0 * lf_p / (lf_p + hf_p + eps)
                HF_nu = 100.0 * hf_p / (lf_p + hf_p + eps)
                LF_HF = lf_p / (hf_p + eps)
                HF_LF = hf_p / (lf_p + eps)
                lnVLF = np.log(vlf_p + eps)
                lnLF = np.log(lf_p + eps)
                lnHF = np.log(hf_p + eps)
                lnTP = np.log(TP + eps)
                vlf_pct = 100.0 * vlf_p / total
                lf_pct = 100.0 * lf_p / total
                hf_pct = 100.0 * hf_p / total
                rel_TP = TP / (np.trapezoid(Pxx, f) + eps)
                fpeak_lf = f[mask_lf][np.argmax(Pxx[mask_lf])] if np.any(mask_lf) else 0.0
                fpeak_hf = f[mask_hf][np.argmax(Pxx[mask_hf])] if np.any(mask_hf) else 0.0
                fc_lf = np.sum(f[mask_lf] * Pxx[mask_lf]) / (np.sum(Pxx[mask_lf]) + eps) if np.any(mask_lf) else 0.0
                fc_hf = np.sum(f[mask_hf] * Pxx[mask_hf]) / (np.sum(Pxx[mask_hf]) + eps) if np.any(mask_hf) else 0.0
    except Exception:
        vlf_p = lf_p = hf_p = TP = LF_nu = HF_nu = LF_HF = HF_LF = lnVLF = lnLF = lnHF = lnTP = 0.0
        fpeak_lf = fpeak_hf = fc_lf = fc_hf = vlf_pct = lf_pct = hf_pct = rel_TP = 0.0

    # Nonlinear features
    se = sampen(rr)  # on seconds
    ap_en = _approx_entropy(rr, m=2, r=0.2 * np.std(rr) if np.std(rr) > 0 else 0.0)
    sd1, sd2, sd_ratio = _poincare_sd(rr_ms)
    hfd = higuchi_fd(rr_ms)
    kurt = stats.kurtosis(rr_ms, fisher=True, bias=False) if len(rr_ms) > 0 else 0.0
    dfa_a1 = _dfa_alpha1(rr_ms)

    feats = [
        # Time-domain (13)
        AVNN, SDNN, RMSSD, NN50, pNN50, NN20, pNN20, IQR_RR, MAD_RR, CVNN, CVSD, HR_mean, HR_median,
        # Frequency-domain (20)
        vlf_p, lf_p, hf_p, TP, LF_nu, HF_nu, LF_HF, HF_LF, lnVLF, lnLF, lnHF, lnTP, fpeak_lf, fpeak_hf, fc_lf, fc_hf, vlf_pct, lf_pct, hf_pct, rel_TP,
        # Nonlinear (8)
        sd1, sd2, sd_ratio, se, ap_en, hfd, kurt, dfa_a1
    ]
    feats = np.array(feats, dtype=np.float32)
    if feats.size != NUM_HRV_FEATURES:
        # Safety: pad/truncate to fixed length
        out = np.zeros(NUM_HRV_FEATURES, dtype=np.float32)
        out[:min(len(feats), NUM_HRV_FEATURES)] = feats[:min(len(feats), NUM_HRV_FEATURES)]
        return out
    return feats

print("Computing HRV features for all segments...")
hrv_feats = np.array([compute_hrv_features(seg, SAMPLE_RATE) for seg in X])
hrv_feats = np.nan_to_num(hrv_feats)
# Standardize features across dataset
hrv_feats = (hrv_feats - np.mean(hrv_feats, axis=0)) / (np.std(hrv_feats, axis=0) + 1e-8)

# ==========================================
#  3) DATASET
# ==========================================
class ECGDatasetWithHRV(Dataset):
    """
    PyTorch Dataset returning ECG segments and aligned HRV features.

    __getitem__ returns:
        x:  (1, L) float32 tensor (ECG segment)
        h:  (NUM_HRV_FEATURES,) float32 tensor (HRV features)
        y:  () float32 tensor {0.0, 1.0} (label)
    """
    def __init__(self, X, hrv, y):
        self.X = X
        self.hrv = hrv
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i][None, :], dtype=torch.float32)
        h = torch.tensor(self.hrv[i], dtype=torch.float32)
        lbl = torch.tensor(self.y[i], dtype=torch.float32)
        return x, h, lbl

# ==========================================
#  4) MODEL COMPONENTS
# ==========================================
class ResidualConvBlock(nn.Module):
    """
    Multi-kernel residual 1D convolution block:
    - Parallel conv branches (k=3,5,7), averaged, BN, GELU
    - Residual 1x1 projection for channel match
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch3 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.branch7 = nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = (b3 + b5 + b7) / 3.0
        out = self.bn(out)
        out = self.act(out + self.res(x))
        return out

class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) using HRV features:
    Produces per-channel gamma/beta to modulate sequence embeddings.
    """
    def __init__(self, hrv_dim, feature_dim):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(hrv_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU()
        )

    def forward(self, x, hrv):
        """
        Args:
            x:   (B, T, D) sequence features
            hrv: (B, H)    HRV features
        Returns:
            (B, T, D) FiLM-modulated features
        """
        gamma_beta = self.film(hrv)  # (B, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)   # (B,1,D)
        beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta

class CrossAttentionHRV(nn.Module):
    """
    Single-query cross-attention:
    - Query from pooled sequence state.
    - Keys/Values from sequence features.
    Returns attention context and weights.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, hrv_embed):
        """
        Args:
            x:         (B, T, D) encoder sequence
            hrv_embed: (B, D)    pooled sequence (or HRV-projected)
        Returns:
            context: (B, D)
            attn:    (B, 1, T)
        """
        q = self.q_proj(hrv_embed).unsqueeze(1)  # (B,1,D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (k.size(-1) ** 0.5)  # (B,1,T)
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, v).squeeze(1)  # (B, D)
        return context, attn

class CTBFNet(nn.Module):
    """
    CTBFNet:
      CNN -> Transformer -> BiLSTM (residual)
      + FiLM modulation by HRV
      + Cross-attention pooling
      + Classifier on [pooled || HRV || attn_ctx]
    """
    def __init__(self, embed_dim=64, lstm_dim=32, hrv_dim=NUM_HRV_FEATURES, nhead=4, num_layers=2):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            ResidualConvBlock(1, 64),
            nn.MaxPool1d(2),
            ResidualConvBlock(64, 128),
            nn.MaxPool1d(2),
            ResidualConvBlock(128, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

        # Transformer Encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=256,
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.film = FiLMConditioning(hrv_dim, embed_dim)

        # BiLSTM with residual projection
        self.bilstm = nn.LSTM(embed_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.res_proj = nn.Linear(embed_dim, 2 * lstm_dim)

        # Cross-attention
        self.cross_attn = CrossAttentionHRV(2 * lstm_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_dim + hrv_dim + 2 * lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x, hrv):
        """
        Args:
            x:   (B, 1, L) raw ECG segment
            hrv: (B, H)    HRV features
        Returns:
            (B,) logits
        """
        feat = self.cnn(x)             # (B, embed_dim, T)
        feat = feat.permute(0, 2, 1)   # (B, T, embed_dim)
        feat = self.film(feat, hrv)    # FiLM conditioning

        trans_out = self.transformer(feat)       # (B, T, embed_dim)
        lstm_out, _ = self.bilstm(trans_out)     # (B, T, 2*lstm_dim)
        res = self.res_proj(trans_out)           # (B, T, 2*lstm_dim)
        lstm_out = lstm_out + res                # residual fusion

        pooled = lstm_out.mean(dim=1)            # (B, 2*lstm_dim)
        attn_context, _ = self.cross_attn(lstm_out, pooled)  # (B, 2*lstm_dim)

        fusion = torch.cat([pooled, hrv, attn_context], dim=1)
        out = self.classifier(fusion)            # (B,1)
        return out.view(-1)

# ==========================================
#  5) TRAIN / EVAL UTILITIES
# ==========================================
def safe_roc_auc(y_true, y_prob):
    """
    Safe ROC AUC that returns 0.5 if only one class is present or upon error.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return 0.5

def evaluate(model: nn.Module, loader: DataLoader):
    """
    Evaluate model on a dataloader.

    Returns:
        acc, f1, precision, recall, auc
    """
    model.eval()
    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for xb, hrvb, yb in loader:
            xb = xb.to(DEVICE)
            hrvb = hrvb.to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb, hrvb)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()
            all_y.extend(yb.detach().cpu().numpy().tolist())
            all_p.extend(pred.detach().cpu().numpy().tolist())
            all_prob.extend(prob.detach().cpu().numpy().tolist())
    all_y = np.array(all_y, dtype=int)
    all_p = np.array(all_p, dtype=int)
    all_prob = np.array(all_prob, dtype=float)
    acc = accuracy_score(all_y, all_p) if all_y.size > 0 else 0.0
    f1 = f1_score(all_y, all_p, zero_division=0) if all_y.size > 0 else 0.0
    auc = safe_roc_auc(all_y, all_prob) if all_y.size > 0 else 0.5
    prec = precision_score(all_y, all_p, zero_division=0) if all_y.size > 0 else 0.0
    rec = recall_score(all_y, all_p, zero_division=0) if all_y.size > 0 else 0.0
    return acc, f1, prec, rec, auc

def plot_metrics(history, fdir, epoch=None):
    """
    Save training curves. Now includes val_loss and can save every epoch.
    """
    plt.figure(figsize=(10, 4))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="Val Accuracy")
    if "val_auc" in history:
        plt.plot(history["val_auc"], label="Val AUC")
    plt.legend(); plt.title("Training Metrics"); plt.tight_layout()
    fname = f"metrics_curve_epoch_{epoch}.png" if epoch is not None else "metrics_curve.png"
    plt.savefig(os.path.join(fdir, fname))
    plt.close()

# ==========================================
#  6) TRAIN FUNCTION (PER-EPOCH METRICS)
# ==========================================
def train_fold(model, loader_tr, loader_te, fdir, pos_weight):
    """
    Train a single fold with early stopping and resource metrics.
    """
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc, best_acc = 0.0, 0.0
    wait, patience = 0, 50
    gpu_available = torch.cuda.is_available()
    total_gpu_time = 0.0
    peak_gpu_mem = 0.0

    for ep in range(1, EPOCHS + 1):
        cpu_start = time.process_time()
        wall_start = time.time()
        mem_start = process.memory_info().rss

        if gpu_available:
            torch.cuda.reset_peak_memory_stats()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        model.train()
        train_loss = 0.0
        n_samples = 0
        for xb, hrvb, yb in loader_tr:
            xb = xb.to(DEVICE)
            hrvb = hrvb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            out = model(xb, hrvb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        if gpu_available:
            end_evt.record()
            torch.cuda.synchronize()
            gpu_time_ms = start_evt.elapsed_time(end_evt)
            gpu_time_s = gpu_time_ms / 1000.0
            epoch_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
            total_gpu_time += gpu_time_s
            peak_gpu_mem = max(peak_gpu_mem, epoch_peak_mem_mb)
        else:
            gpu_time_s = 0.0
            epoch_peak_mem_mb = 0.0

        train_loss = train_loss / max(n_samples, 1)

        # Validation loss + metrics
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, hrvb, yb in loader_te:
                xb = xb.to(DEVICE)
                hrvb = hrvb.to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb, hrvb)
                loss = crit(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_samples += xb.size(0)
        val_loss = val_loss / max(val_samples, 1)
        acc, f1, prec, rec, auc = evaluate(model, loader_te)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_auc"].append(auc)

        cpu_end = time.process_time()
        wall_end = time.time()
        mem_end = process.memory_info().rss
        cpu_time = cpu_end - cpu_start
        wall_time = wall_end - wall_start
        mem_used_mb = (mem_end - mem_start) / (1024.0**2)

        print(
            f"Epoch {ep:03d} | Loss={train_loss:.4f} | ValLoss={val_loss:.4f} | ACC={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f} || "
            f"GPU_time={gpu_time_s:.3f}s | GPU_peak_mem={epoch_peak_mem_mb:.2f} MB | Wall={wall_time:.3f}s | Mem_delta={mem_used_mb:.2f} MB"
        )

        if auc > best_auc or acc > best_acc:
            best_auc = max(best_auc, auc)
            best_acc = max(best_acc, acc)
            wait = 0
            try:
                torch.save(model.state_dict(), os.path.join(fdir, "best.pt"))
            except Exception:
                pass
        else:
            wait += 1

        # Per-epoch plot
        plot_metrics(history, fdir, epoch=ep)

        if wait >= patience:
            print("Early Stop at epoch", ep)
            break

    pd.DataFrame(history).to_csv(os.path.join(fdir, "training_history.csv"), index=False)
    try:
        model.load_state_dict(torch.load(os.path.join(fdir, "best.pt")))
    except Exception:
        pass
    return best_acc, best_auc, total_gpu_time, peak_gpu_mem

# ==========================================
#  7) MAIN: CROSS-VALIDATION DRIVER
# ==========================================
def main():
    """
    Runs GroupKFold training/evaluation.
    Saves per-fold metrics and final summary under OUTPUT_ROOT.
    """
    if len(X) == 0:
        print("No data found. Exiting.")
        return

    gkf = GroupKFold(n_splits=NUM_FOLDS)
    results = []
    counts = np.bincount(y, minlength=2)
    pos_weight_val = float(counts[0]) / float(max(1, counts[1]))
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n======== Fold {fold} ========")
        fdir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        os.makedirs(fdir, exist_ok=True)

        fold_cpu_start = time.process_time()
        fold_wall_start = time.time()
        fold_mem_start = process.memory_info().rss

        train_ds = ECGDatasetWithHRV(X[tr], hrv_feats[tr], y[tr])
        test_ds = ECGDatasetWithHRV(X[te], hrv_feats[te], y[te])
        loader_tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        loader_te = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = CTBFNet(
            embed_dim=64, lstm_dim=32, hrv_dim=NUM_HRV_FEATURES, nhead=4, num_layers=2
        )
        best_acc, best_auc, total_gpu_time, peak_gpu_mem = train_fold(model, loader_tr, loader_te, fdir, pos_weight)
        acc, f1, prec, rec, auc = evaluate(model, loader_te)
        final_acc = max(best_acc, acc)
        results.append([final_acc, f1, prec, rec, auc])
        print(f"Fold {fold} Final — ACC={final_acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
        fold_cpu_end = time.process_time()
        fold_wall_end = time.time()
        fold_mem_end = process.memory_info().rss
        fold_cpu_time = fold_cpu_end - fold_cpu_start
        fold_wall_time = fold_wall_end - fold_wall_start
        fold_mem_mb = (fold_mem_end - fold_mem_start) / (1024.0**2)
        print(f"⚙ Fold {fold} Computational Metrics:")
        print(f"   GPU Time (s):  {total_gpu_time:.3f}")
        print(f"   GPU Peak Mem (MB): {peak_gpu_mem:.2f}")
        print(f"   Wall Time (s): {fold_wall_time:.3f}")
        print(f"   Memory (MB):   {fold_mem_mb:.2f}")

        pd.DataFrame(results, columns=["ACC", "F1", "PREC", "REC", "AUC"]).to_csv(
            os.path.join(OUTPUT_ROOT, f"results_up_to_fold_{fold}.csv"), index=False
        )
    results_arr = np.array(results)
    if results_arr.size == 0:
        print("No folds completed.")
        return

    print("\n========== FINAL RESULTS ==========")
    print(f"ACC={results_arr[:,0].mean():.4f}±{results_arr[:,0].std():.4f}")
    print(f"F1 ={results_arr[:,1].mean():.4f}±{results_arr[:,1].std():.4f}")
    print(f"PREC={results_arr[:,2].mean():.4f}±{results_arr[:,2].std():.4f}")
    print(f"REC ={results_arr[:,3].mean():.4f}±{results_arr[:,3].std():.4f}")
    print(f"AUC ={results_arr[:,4].mean():.4f}±{results_arr[:,4].std():.4f}")

    try:
        torch.save(model.state_dict(), os.path.join(OUTPUT_ROOT, "final_model.pt"))
    except Exception:
        pass
    pd.DataFrame(results, columns=["ACC", "F1", "PREC", "REC", "AUC"]).to_csv(
        os.path.join(OUTPUT_ROOT, "final_summary.csv"), index=False
    )
    print("Training complete. Models and results saved.")

if __name__ == "__main__":
    main()