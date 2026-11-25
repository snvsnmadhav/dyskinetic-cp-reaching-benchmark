#!/usr/bin/env python3
"""
==============================================================================
COMPLETE PIPELINE: FFRNN + DONN + Multi-Model Comparison
Upper Limb Reaching Classification — TD vs DCP
==============================================================================

Dataset : Vanmechelen et al. (2024) J Biomech 173:112233
          doi:10.48804/BZM9SJ  —  reaching-sideways task
          2D markerless (DeepLabCut) stick-figure kinematics
          30 DCP + 21 TD subjects × 8 reps = 408 trials

Models:
  ── From-scratch RNN cells (numpy) ──
  1. 1D FlipFlop RNN   (JK flip-flop, Kumari et al. 2023)
  2. Simple RNN          (Elman)
  3. LSTM-lite
  4. GRU-lite

  ── Sklearn classifiers & regressors ──
  5. Random Forest
  6. SVM (RBF)
  7. MLP Neural Network
  8. Logistic / Ridge Regression
  9. Gradient Boosting

  ── Deep oscillatory (PyTorch, if available) ──
  10. DONN (ResHopf oscillatory classifier)

Tasks:
  A — Classification: TD vs DCP (binary)
  B — Regression: Joint angle prediction from landmarks

Ref: https://github.com/vigneswaran-chandrasekaran/ffrnn
     Kumari, Chandrasekaran & Chakravarthy (2023) Neural Comput & Applic 35
==============================================================================
"""

import os, sys, glob, re, math, copy, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # Uncomment if running headless (no display)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from sklearn.model_selection import (StratifiedKFold, KFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error, ConfusionMatrixDisplay
)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor

from scipy.signal import savgol_filter, resample
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# numpy 2.x compat
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ── PyTorch: optional ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import init
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_TORCH = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INIT] PyTorch {torch.__version__} on {device}')
except ImportError:
    HAS_TORCH = False
    print('[INIT] PyTorch not available — DONN will be skipped.')

SEED = 42
np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN      = 50
MAX_SEQ_LEN  = 50       # for numpy-based RNNs
N_FOLDS      = 5
FPS          = 30.0
N_BOOTSTRAP  = 500

# DONN config
HIDDEN_DIM   = 64
OSC_UNITS    = 64
DT           = 0.01
MIN_OMEGA    = 0.5
MAX_OMEGA    = 5.0
INPUT_SCALER = 5.0
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
EPOCHS       = 1000
PATIENCE     = 10

# =============================================================================
# 1. FILENAME PARSING
# =============================================================================
_FNAME_RE = re.compile(
    r'^(?P<subject>\d+)_(?P<side>left|right)_(?P<cls>TD|DCP)_re[po](?P<rep>\d+)$',
    re.IGNORECASE)

def parse_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _FNAME_RE.match(base)
    if m is None:
        raise ValueError(f"Bad filename: '{base}'")
    cls = m.group('cls').upper()
    return {
        'subject_id': m.group('subject'),
        'side':       m.group('side').lower(),
        'class_str':  cls,
        'label':      0 if cls == 'TD' else 1,
        'repetition': int(m.group('rep')),
        'filename':   os.path.basename(fname),
    }

# =============================================================================
# 2. COLUMN DETECTION + HELPERS
# =============================================================================
JOINT_NAMES = {
    'shoulder_r': 'shoulder_right', 'shoulder_l': 'shoulder_left',
    'elbow_r': 'elbow_right',       'elbow_l': 'elbow_left',
    'wrist_r': 'wrist_right',       'wrist_l': 'wrist_left',
    'hand_r': 'MCP3_right',         'hand_l': 'MCP3_left',
    'forehead': 'forehead',          'chin': 'chin',
    'object': 'object',
}

LANDMARK_COLS_XY = [
    "('forehead', 'x')", "('forehead', 'y')",
    "('chin', 'x')", "('chin', 'y')",
    "('ankle1', 'x')", "('ankle1', 'y')",
    "('knee_right', 'x')", "('knee_right', 'y')",
    "('shoulder_right', 'x')", "('shoulder_right', 'y')",
    "('elbow_right', 'x')", "('elbow_right', 'y')",
    "('wrist_right', 'x')", "('wrist_right', 'y')",
    "('MCP3_right', 'x')", "('MCP3_right', 'y')",
    "('ankle2', 'x')", "('ankle2', 'y')",
    "('knee_left', 'x')", "('knee_left', 'y')",
    "('shoulder_left', 'x')", "('shoulder_left', 'y')",
    "('elbow_left', 'x')", "('elbow_left', 'y')",
    "('wrist_left', 'x')", "('wrist_left', 'y')",
    "('MCP3_left', 'x')", "('MCP3_left', 'y')",
    "('object', 'x')", "('object', 'y')",
]
ANGLE_COLS = ['R-Elbow', 'L-Elbow', 'R-Wrist', 'L-Wrist']

def _detect_xy_col(df_columns, joint_name):
    patterns = [
        (f"('{joint_name}', 'x')", f"('{joint_name}', 'y')"),
        (f'{joint_name}_x', f'{joint_name}_y'),
    ]
    for cx, cy in patterns:
        if cx in df_columns and cy in df_columns:
            return cx, cy
    cols = list(df_columns)
    xm = [c for c in cols if joint_name.lower() in str(c).lower() and "'x'" in str(c).lower()]
    ym = [c for c in cols if joint_name.lower() in str(c).lower() and "'y'" in str(c).lower()]
    if xm and ym:
        return xm[0], ym[0]
    return None, None

def get_xy(df, canonical):
    jname = JOINT_NAMES.get(canonical, canonical)
    cx, cy = _detect_xy_col(df.columns, jname)
    T = len(df)
    if cx is None:
        return np.full((T, 2), np.nan, dtype=np.float64), 1.0
    x = df[cx].interpolate(method='linear', limit_direction='both').ffill().bfill()
    y = df[cy].interpolate(method='linear', limit_direction='both').ffill().bfill()
    miss = max(df[cx].isnull().mean(), df[cy].isnull().mean())
    return np.stack([x.values, y.values], 1).astype(np.float64), miss

def angle_3pt(a, b, c):
    ba, bc = a - b, c - b
    cos_a = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def smooth(s, window=5, polyorder=2):
    if len(s) < max(window, polyorder + 1):
        return s.copy()
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(s, w, polyorder)

def deriv(s, dt=1.0 / 30.0):
    return np.gradient(s, dt)

def jerk_rms(s, dt=1.0 / 30.0):
    return np.sqrt(np.mean(deriv(deriv(deriv(s, dt), dt), dt) ** 2))

def norm_jerk(s, dt=1.0 / 30.0):
    j = deriv(deriv(deriv(s, dt), dt), dt)
    dur = len(s) * dt
    rng = max(np.ptp(s), 1e-8)
    return np.sqrt(0.5 * np.trapezoid(j ** 2, dx=dt) * dur ** 5 / rng ** 2)

def convex_hull_area(points_2d):
    try:
        pts = points_2d[~np.isnan(points_2d).any(axis=1)]
        if len(pts) < 3:
            return 0.0
        if np.linalg.matrix_rank(pts - pts.mean(0)) < 2:
            return 0.0
        return ConvexHull(pts).volume
    except:
        return 0.0

def resample_temporal(seq, target_len):
    T, D = seq.shape
    if T == target_len:
        return seq
    return resample(seq, target_len, axis=0).astype(np.float32)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

# =============================================================================
# 3. UPPER-LIMB FEATURE EXTRACTION (from DONN v3)
# =============================================================================
def extract_upper_limb_features(df, meta):
    T = len(df)
    dt = 1.0 / FPS
    active_side = meta['side']
    act, pas = ('l', 'r') if active_side == 'left' else ('r', 'l')

    shoulder_act, m1 = get_xy(df, f'shoulder_{act}')
    shoulder_pas, m2 = get_xy(df, f'shoulder_{pas}')
    elbow_act, _     = get_xy(df, f'elbow_{act}')
    elbow_pas, _     = get_xy(df, f'elbow_{pas}')
    wrist_act, _     = get_xy(df, f'wrist_{act}')
    wrist_pas, _     = get_xy(df, f'wrist_{pas}')
    hand_act, _      = get_xy(df, f'hand_{act}')
    hand_pas, _      = get_xy(df, f'hand_{pas}')
    target, _        = get_xy(df, 'object')
    forehead, _      = get_xy(df, 'forehead')
    chin, _          = get_xy(df, 'chin')

    diag = {'filename': meta['filename'], 'n_frames': T,
            'active_side': active_side, 'shoulder_act_miss': m1, 'shoulder_pas_miss': m2}

    upper_arm_len = np.linalg.norm(elbow_act - shoulder_act, axis=1)
    ref_scale = np.maximum(upper_arm_len, 1e-6)
    mid_shoulder = (shoulder_act + shoulder_pas) / 2.0

    def norm_lm(lm):
        return (lm - mid_shoulder) / ref_scale[:, None]

    n_shoulder_act = norm_lm(shoulder_act)
    n_elbow_act    = norm_lm(elbow_act)
    n_wrist_act    = norm_lm(wrist_act)
    n_hand_act     = norm_lm(hand_act)
    n_shoulder_pas = norm_lm(shoulder_pas)
    n_elbow_pas    = norm_lm(elbow_pas)
    n_wrist_pas    = norm_lm(wrist_pas)
    n_hand_pas     = norm_lm(hand_pas)
    n_target       = norm_lm(target)
    n_forehead     = norm_lm(forehead)

    trunk_up = forehead - chin
    ang_shoulder_act = angle_3pt(elbow_act, shoulder_act, shoulder_act + trunk_up)
    ang_shoulder_pas = angle_3pt(elbow_pas, shoulder_pas, shoulder_pas + trunk_up)
    ang_elbow_act = angle_3pt(shoulder_act, elbow_act, wrist_act)
    ang_elbow_pas = angle_3pt(shoulder_pas, elbow_pas, wrist_pas)
    ang_wrist_act = angle_3pt(elbow_act, wrist_act, hand_act)
    ang_wrist_pas = angle_3pt(elbow_pas, wrist_pas, hand_pas)

    side_map = {'l': 'L', 'r': 'R'}
    precomp_cols = {
        'shoulder': f'{side_map[act]}-Shoulder',
        'elbow':    f'{side_map[act]}-Elbow',
        'wrist':    f'{side_map[act]}-Wrist',
    }
    precomp = {}
    for key, col in precomp_cols.items():
        if col in df.columns and not df[col].isnull().all():
            precomp[key] = df[col].interpolate(limit_direction='both').fillna(0).values
        else:
            precomp[key] = np.zeros(T)

    hand_to_target = np.linalg.norm(hand_act - target, axis=1)
    hand_to_target_norm = hand_to_target / ref_scale
    hand_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_act, axis=0), axis=1)])
    hand_speed = hand_disp / dt
    hand_speed_norm = hand_speed / ref_scale
    hand_speed_smooth = smooth(hand_speed_norm)
    hand_accel = smooth(deriv(hand_speed_norm, dt))

    w_shoulder_act = smooth(deriv(ang_shoulder_act, dt))
    w_elbow_act    = smooth(deriv(ang_elbow_act, dt))
    w_wrist_act    = smooth(deriv(ang_wrist_act, dt))
    w_shoulder_pas = smooth(deriv(ang_shoulder_pas, dt))
    w_elbow_pas    = smooth(deriv(ang_elbow_pas, dt))

    a_shoulder_act = smooth(deriv(w_shoulder_act, dt))
    a_elbow_act    = smooth(deriv(w_elbow_act, dt))
    a_wrist_act    = smooth(deriv(w_wrist_act, dt))

    trunk_mid = (forehead + chin) / 2.0
    trunk_disp_px = np.concatenate([[0], np.linalg.norm(np.diff(trunk_mid, axis=0), axis=1)])
    trunk_disp_norm = trunk_disp_px / ref_scale
    trunk_drift = np.linalg.norm(trunk_mid - trunk_mid[0], axis=1) / ref_scale
    head_x_disp = (forehead[:, 0] - forehead[0, 0]) / ref_scale

    pas_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_pas, axis=0), axis=1)])
    arm_movement_ratio = np.sum(pas_disp) / (np.sum(hand_disp) + 1e-8)

    # ── Time-series matrix (45 features) ──
    ts = np.column_stack([
        n_shoulder_act, n_elbow_act, n_wrist_act, n_hand_act,
        n_shoulder_pas, n_elbow_pas, n_wrist_pas, n_hand_pas,
        n_target, n_forehead,
        ang_shoulder_act[:, None], ang_elbow_act[:, None], ang_wrist_act[:, None],
        ang_shoulder_pas[:, None], ang_elbow_pas[:, None], ang_wrist_pas[:, None],
        precomp['shoulder'][:, None], precomp['elbow'][:, None], precomp['wrist'][:, None],
        w_shoulder_act[:, None], w_elbow_act[:, None], w_wrist_act[:, None],
        w_shoulder_pas[:, None], w_elbow_pas[:, None],
        a_shoulder_act[:, None], a_elbow_act[:, None], a_wrist_act[:, None],
        hand_to_target_norm[:, None], hand_speed_smooth[:, None],
        hand_accel[:, None], (hand_disp / ref_scale)[:, None],
        trunk_disp_norm[:, None], trunk_drift[:, None], head_x_disp[:, None],
        (pas_disp / ref_scale)[:, None],
    ])
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ── Scalar summary features ──
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    total_path = np.sum(hand_disp)
    straight_line = np.linalg.norm(hand_act[-1] - hand_act[0])
    index_of_curvature = total_path / (straight_line + 1e-8)
    endpoint_dist_norm = hand_to_target[-1] / np.mean(ref_scale)
    min_dist_idx = np.argmin(hand_to_target)
    overshoot_ratio = (T - 1 - min_dist_idx) / T
    mean_overshoot = (np.mean(hand_to_target[min_dist_idx:]) / np.mean(ref_scale)
                      if min_dist_idx < T - 1 else 0.0)
    hull_area = convex_hull_area(hand_act) / (np.mean(ref_scale) ** 2)

    stats = {
        'shoulder_act_angle_mean':  np.mean(ang_shoulder_act),
        'shoulder_act_angle_range': np.ptp(ang_shoulder_act),
        'shoulder_act_omega_peak':  np.max(np.abs(w_shoulder_act)),
        'shoulder_act_alpha_rms':   rms(a_shoulder_act),
        'shoulder_act_jerk':        norm_jerk(ang_shoulder_act, dt),
        'shoulder_pas_angle_mean':  np.mean(ang_shoulder_pas),
        'shoulder_pas_angle_range': np.ptp(ang_shoulder_pas),
        'elbow_act_angle_mean':     np.mean(ang_elbow_act),
        'elbow_act_angle_range':    np.ptp(ang_elbow_act),
        'elbow_act_omega_peak':     np.max(np.abs(w_elbow_act)),
        'elbow_act_alpha_rms':      rms(a_elbow_act),
        'elbow_act_jerk':           norm_jerk(ang_elbow_act, dt),
        'elbow_pas_angle_mean':     np.mean(ang_elbow_pas),
        'elbow_pas_angle_range':    np.ptp(ang_elbow_pas),
        'wrist_act_angle_mean':     np.mean(ang_wrist_act),
        'wrist_act_angle_range':    np.ptp(ang_wrist_act),
        'wrist_act_omega_peak':     np.max(np.abs(w_wrist_act)),
        'wrist_act_alpha_rms':      rms(a_wrist_act),
        'wrist_act_jerk':           norm_jerk(ang_wrist_act, dt),
        'hand_speed_mean':          np.mean(hand_speed_norm),
        'hand_speed_peak':          np.max(hand_speed_norm),
        'hand_speed_cv':            np.std(hand_speed_norm) / (np.mean(hand_speed_norm) + 1e-8),
        'hand_accel_rms':           rms(hand_accel),
        'hand_jerk_rms':            jerk_rms(hand_speed_norm, dt),
        'index_of_curvature':       index_of_curvature,
        'endpoint_accuracy':        endpoint_dist_norm,
        'mean_overshoot':           mean_overshoot,
        'overshoot_ratio':          overshoot_ratio,
        'convex_hull_area':         hull_area,
        'total_path_norm':          total_path / np.mean(ref_scale),
        'execution_time':           T / FPS,
        'trunk_total_disp':         np.sum(trunk_disp_norm),
        'trunk_max_drift':          np.max(trunk_drift),
        'head_x_range':             np.ptp(head_x_disp),
        'shoulder_act_variability': np.std(ang_shoulder_act),
        'elbow_act_variability':    np.std(ang_elbow_act),
        'wrist_act_variability':    np.std(ang_wrist_act),
        'arm_movement_ratio':       arm_movement_ratio,
        'shoulder_angle_symmetry':  np.mean(np.abs(ang_shoulder_act - ang_shoulder_pas)),
        'elbow_angle_symmetry':     np.mean(np.abs(ang_elbow_act - ang_elbow_pas)),
        'shoulder_rom':             np.ptp(ang_shoulder_act),
        'elbow_rom':                np.ptp(ang_elbow_act),
        'wrist_rom':                np.ptp(ang_wrist_act),
    }
    return ts, stats, diag

# =============================================================================
# 4. FLIP-FLOP RNN + OTHER RNN CELLS (numpy)
# =============================================================================
class FlipFlopRNNCell:
    """1D JK Flip-Flop RNN Cell — Kumari, Chandrasekaran & Chakravarthy (2023)."""
    def __init__(self, input_size, hidden_size, lr=0.001):
        self.input_size, self.hidden_size, self.lr = input_size, hidden_size, lr
        s = 0.1
        self.W_j = np.random.randn(hidden_size, input_size) * s
        self.U_j = np.random.randn(hidden_size, hidden_size) * s
        self.b_j = np.zeros(hidden_size)
        self.W_k = np.random.randn(hidden_size, input_size) * s
        self.U_k = np.random.randn(hidden_size, hidden_size) * s
        self.b_k = np.zeros(hidden_size)

    def forward_sequence(self, X):
        seq_len = X.shape[0]
        q = np.zeros(self.hidden_size)
        states, cache = [], []
        for t in range(seq_len):
            x_t = X[t]
            j = sigmoid(self.W_j @ x_t + self.U_j @ q + self.b_j)
            k = sigmoid(self.W_k @ x_t + self.U_k @ q + self.b_k)
            q_new = j * (1 - q) + (1 - k) * q
            cache.append((x_t, q.copy(), j, k, q_new))
            q = q_new
            states.append(q.copy())
        return states, cache

    def get_final_state(self, X):
        states, _ = self.forward_sequence(X)
        return states[-1]


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, lr=0.001):
        self.hidden_size = hidden_size
        s = 0.1
        self.W_ih = np.random.randn(hidden_size, input_size) * s
        self.W_hh = np.random.randn(hidden_size, hidden_size) * s
        self.b_h = np.zeros(hidden_size)

    def forward_sequence(self, X):
        h = np.zeros(self.hidden_size)
        states = []
        for t in range(X.shape[0]):
            h = np.tanh(self.W_ih @ X[t] + self.W_hh @ h + self.b_h)
            states.append(h.copy())
        return states, None

    def get_final_state(self, X):
        return self.forward_sequence(X)[0][-1]


class LSTMLiteCell:
    def __init__(self, input_size, hidden_size, lr=0.001):
        self.hidden_size = hidden_size
        s = 0.1
        d = input_size + hidden_size
        self.W_f = np.random.randn(hidden_size, d) * s; self.b_f = np.zeros(hidden_size)
        self.W_i = np.random.randn(hidden_size, d) * s; self.b_i = np.zeros(hidden_size)
        self.W_c = np.random.randn(hidden_size, d) * s; self.b_c = np.zeros(hidden_size)
        self.W_o = np.random.randn(hidden_size, d) * s; self.b_o = np.zeros(hidden_size)

    def forward_sequence(self, X):
        h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        states = []
        for t in range(X.shape[0]):
            xh = np.concatenate([X[t], h])
            f = sigmoid(self.W_f @ xh + self.b_f)
            i = sigmoid(self.W_i @ xh + self.b_i)
            c_hat = np.tanh(self.W_c @ xh + self.b_c)
            o = sigmoid(self.W_o @ xh + self.b_o)
            c = f * c + i * c_hat
            h = o * np.tanh(c)
            states.append(h.copy())
        return states, None

    def get_final_state(self, X):
        return self.forward_sequence(X)[0][-1]


class GRULiteCell:
    def __init__(self, input_size, hidden_size, lr=0.001):
        self.hidden_size = hidden_size
        s = 0.1
        d = input_size + hidden_size
        self.W_z = np.random.randn(hidden_size, d) * s; self.b_z = np.zeros(hidden_size)
        self.W_r = np.random.randn(hidden_size, d) * s; self.b_r = np.zeros(hidden_size)
        self.W_h = np.random.randn(hidden_size, d) * s; self.b_h = np.zeros(hidden_size)

    def forward_sequence(self, X):
        h = np.zeros(self.hidden_size)
        states = []
        for t in range(X.shape[0]):
            xh = np.concatenate([X[t], h])
            z = sigmoid(self.W_z @ xh + self.b_z)
            r = sigmoid(self.W_r @ xh + self.b_r)
            xrh = np.concatenate([X[t], r * h])
            h_hat = np.tanh(self.W_h @ xrh + self.b_h)
            h = (1 - z) * h + z * h_hat
            states.append(h.copy())
        return states, None

    def get_final_state(self, X):
        return self.forward_sequence(X)[0][-1]


class RNNCellClassifier:
    def __init__(self, cell, hidden_size, lr=0.005, epochs=30):
        self.cell, self.hidden_size, self.lr, self.epochs = cell, hidden_size, lr, epochs
        self.W_out = np.random.randn(1, hidden_size) * 0.1
        self.b_out = np.zeros(1)

    def predict_proba_single(self, X):
        h = self.cell.get_final_state(X)
        return sigmoid(self.W_out @ h + self.b_out)[0]

    def train(self, X_seqs, y, X_val=None, y_val=None, verbose=True):
        n = len(X_seqs)
        train_losses, val_accs = [], []
        for epoch in range(self.epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0
            for idx in indices:
                h = self.cell.get_final_state(X_seqs[idx])
                p = np.clip(sigmoid(self.W_out @ h + self.b_out)[0], 1e-7, 1 - 1e-7)
                loss = -(y[idx] * np.log(p) + (1 - y[idx]) * np.log(1 - p))
                epoch_loss += loss
                d_logit = p - y[idx]
                self.W_out -= self.lr * d_logit * h.reshape(1, -1)
                self.b_out -= self.lr * d_logit
            train_losses.append(epoch_loss / n)
            if X_val is not None:
                preds = np.array([1 if self.predict_proba_single(x) > 0.5 else 0 for x in X_val])
                val_accs.append(accuracy_score(y_val, preds))
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"    Ep {epoch+1}/{self.epochs}: loss={epoch_loss/n:.4f}"
                if val_accs:
                    msg += f" val_acc={val_accs[-1]:.3f}"
                print(msg)
        return train_losses, val_accs

    def predict(self, X_seqs):
        return np.array([1 if self.predict_proba_single(x) > 0.5 else 0 for x in X_seqs])

    def predict_proba(self, X_seqs):
        return np.array([self.predict_proba_single(x) for x in X_seqs])


# =============================================================================
# 5. DONN (PyTorch) — only if available
# =============================================================================
if HAS_TORCH:
    class ComplexLinear(nn.Module):
        def __init__(self, inf, outf, bias=True, dtype=torch.cfloat):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(outf, inf, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(outf, dtype=dtype)) if bias else None
            if not bias:
                self.register_parameter('bias', None)
            self._reset()

        def _reset(self):
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fi, _ = init._calculate_fan_in_and_fan_out(self.weight)
                init.uniform_(self.bias, -1 / math.sqrt(fi + 1e-8), 1 / math.sqrt(fi + 1e-8))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    class ResHopf(nn.Module):
        def __init__(self, units, min_omega=0.5, max_omega=5.0, dt=0.01,
                     train_omegas=True, input_scaler=5.0):
            super().__init__()
            self.units, self.min_omega, self.max_omega, self.dt = units, min_omega, max_omega, dt
            self.mu0, self.beta1, self.input_scaler = 1.0, 1.0, input_scaler
            self.omegas = nn.Parameter(torch.randn(1, units), requires_grad=train_omegas)

        def forward(self, X, r=None, phi=None):
            T, B, D = X.shape
            dev = X.device
            if r is None:
                r = torch.ones(B, D, device=dev)
            if phi is None:
                phi = (math.pi / 2) * torch.rand(B, D, device=dev) - (math.pi / 4)
            om = (torch.sigmoid(self.omegas) * (self.max_omega - self.min_omega) + self.min_omega) * 2 * math.pi
            om = om.to(dev)
            rs, ps = [], []
            for t in range(T):
                ir = self.input_scaler * X[t].real * torch.cos(phi)
                ip = self.input_scaler * X[t].imag * torch.sin(phi)
                r = r + ((self.mu0 - self.beta1 * r ** 2) * r + ir) * self.dt
                phi = phi + (om - ip) * self.dt
                rs.append(r)
                ps.append(phi)
            rs, ps = torch.stack(rs, 0), torch.stack(ps, 0)
            return torch.complex(rs * torch.cos(ps), rs * torch.sin(ps)), r, phi

    class DONNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, osc_units=64, num_classes=2,
                     dt=0.01, min_omega=0.5, max_omega=5.0, input_scaler=5.0, dropout=0.3):
            super().__init__()
            self.encoder_r = nn.Linear(input_dim, hidden_dim)
            self.encoder_i = nn.Linear(input_dim, hidden_dim)
            self.proj = ComplexLinear(hidden_dim, osc_units)
            self.hopf = ResHopf(osc_units, min_omega, max_omega, dt, True, input_scaler)
            self.readout = ComplexLinear(osc_units, hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, num_classes))

        def forward(self, x):
            B, T, D = x.shape
            xc = torch.complex(self.encoder_r(x), self.encoder_i(x)).permute(1, 0, 2)
            xp = torch.stack([self.proj(xc[t]) for t in range(T)], 0)
            z, _, _ = self.hopf(xp)
            h = torch.stack([self.readout(z[t]) for t in range(T)], 0)
            hm = h.abs()
            return self.classifier(torch.cat([hm.mean(0), hm[-1]], -1))

    class GaitDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        ls, c, n = 0., 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ls += loss.item() * Xb.size(0)
            c += (out.argmax(1) == yb).sum().item()
            n += Xb.size(0)
        return ls / n, c / n

    @torch.no_grad()
    def evaluate_donn(model, loader, criterion):
        model.eval()
        ls, c, n = 0., 0, 0
        P, Pr, L = [], [], []
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            ls += loss.item() * Xb.size(0)
            pr = torch.softmax(out, 1)
            pd_ = out.argmax(1)
            c += (pd_ == yb).sum().item()
            n += Xb.size(0)
            P.append(pr.cpu().numpy())
            Pr.append(pd_.cpu().numpy())
            L.append(yb.cpu().numpy())
        return ls / n, np.concatenate(Pr), np.concatenate(L), np.concatenate(P)

    def make_donn_model(n_features):
        return DONNClassifier(
            n_features, HIDDEN_DIM, OSC_UNITS, 2, DT, MIN_OMEGA, MAX_OMEGA,
            INPUT_SCALER, DROPOUT).to(device)

# =============================================================================
# 6. DATA LOADING
# =============================================================================
def load_all_data(data_dir):
    # Auto-extract zip
    zip_path = os.path.join(os.path.dirname(data_dir) or '.', 'data.zip')
    if os.path.isfile(zip_path) and not os.path.isdir(data_dir):
        print(f'  Extracting {zip_path}...')
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

    all_xlsx = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
    matched = [f for f in all_xlsx if _FNAME_RE.match(os.path.splitext(os.path.basename(f))[0])]
    if not matched:
        raise FileNotFoundError(f"No matching xlsx in {data_dir}. Found {len(all_xlsx)} xlsx total.")

    records, errors = [], []
    for i, fpath in enumerate(matched):
        try:
            meta = parse_filename(fpath)
            df = pd.read_excel(fpath)
            drop = [c for c in df.columns
                    if re.match(r'^unnamed', str(c), re.I) or str(c).strip().lower() == 'index']
            df = df.drop(columns=drop)
            ts, stats, diag = extract_upper_limb_features(df, meta)
            records.append({**meta, 'ts_data': ts, 'stats': stats, 'raw_df': df})
        except Exception as e:
            errors.append((os.path.basename(fpath), str(e)))
        if (i + 1) % 100 == 0:
            print(f'    Loaded {i+1}/{len(matched)}')

    if errors:
        print(f'  WARNING: {len(errors)} files failed:')
        for fn, err in errors[:3]:
            print(f'    {fn}: {err}')

    labels = np.array([r['label'] for r in records])
    subjects = np.array([r['subject_id'] for r in records])
    td_n, dcp_n = (labels == 0).sum(), (labels == 1).sum()
    td_subj, dcp_subj = len(set(subjects[labels == 0])), len(set(subjects[labels == 1]))
    print(f'  {len(records)} trials loaded | TD: {td_n} ({td_subj} subj) | DCP: {dcp_n} ({dcp_subj} subj)')
    return records, labels, subjects

# =============================================================================
# 7. FEATURE EXTRACTION WRAPPERS
# =============================================================================
def build_stat_features(records):
    stat_keys = sorted(records[0]['stats'].keys())
    X = np.array([[r['stats'][k] for k in stat_keys] for r in records], dtype=np.float64)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, stat_keys

def build_sequence_data(records, max_len):
    seqs = []
    for r in records:
        seqs.append(resample_temporal(r['ts_data'], max_len))
    X = np.stack(seqs).astype(np.float32)
    # Normalize
    n, t, f = X.shape
    flat = X.reshape(-1, f)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    X = ((X.reshape(-1, f) - mu) / sd).reshape(n, t, f)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, mu, sd

def build_regression_data(records):
    all_X, all_y = [], []
    for r in records:
        df = r['raw_df']
        inp_cols = [c for c in LANDMARK_COLS_XY if c in df.columns]
        tgt_cols = [c for c in ANGLE_COLS if c in df.columns]
        if not tgt_cols:
            continue
        inp = df[inp_cols].apply(pd.to_numeric, errors='coerce')
        tgt = df[tgt_cols].apply(pd.to_numeric, errors='coerce')
        mask = tgt.notna().all(axis=1) & inp.notna().all(axis=1)
        if mask.sum() > 0:
            all_X.append(inp[mask].values)
            all_y.append(tgt[mask].values)
    X = np.nan_to_num(np.vstack(all_X), nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(np.vstack(all_y), nan=0, posinf=0, neginf=0)
    return X, y

# =============================================================================
# 8. PLOT FUNCTIONS
# =============================================================================
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  [PLOT] {name}')

def plot_confusion_matrix_pair(y_train, y_train_pred, y_test, y_test_pred,
                               model_name, labels=['NORMAL', 'CEREBRAL PALSY']):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} — Confusion Matrices', fontsize=14, fontweight='bold')
    for ax, yt, yp, title in [(axes[0], y_train, y_train_pred, 'TRAIN'),
                               (axes[1], y_test, y_test_pred, 'TEST')]:
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        acc = accuracy_score(yt, yp)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{title}\nAcc={acc:.3f}')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_fig(fig, f'cm_{safe_name}.png')

def plot_all_classification_bars(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Classification Comparison: TD vs DCP (All Models)',
                 fontsize=14, fontweight='bold', y=1.02)
    models = [r['model'] for r in results]
    colors = ['#e74c3c' if 'FlipFlop' in m or 'DONN' in m else '#3498db' for m in models]

    for ax, metric, label in zip(axes, ['accuracy', 'f1', 'auc'],
                                  ['Accuracy', 'F1 Score', 'AUC-ROC']):
        vals = [r[metric] for r in results]
        errs = [r.get(f'{metric}_std', 0) for r in results]
        bars = ax.barh(models, vals, xerr=errs, color=colors, edgecolor='gray', alpha=0.85)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.1)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.01, 1.05), bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'classification_all_models.png')

def plot_regression_bars(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regression Comparison: Joint Angle Prediction',
                 fontsize=14, fontweight='bold', y=1.02)
    models = [r['model'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for ax, metric, label in zip(axes, ['mse', 'r2', 'mae'], ['MSE', 'R²', 'MAE']):
        vals = [r[metric] for r in results]
        bars = ax.barh(models, vals, color=colors, edgecolor='gray')
        ax.set_xlabel(label)
        for bar, v in zip(bars, vals):
            ax.text(max(v, 0) + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'regression_all_models.png')

def plot_rnn_training_curves(rnn_results, task='classification'):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    for i, res in enumerate(rnn_results):
        if 'train_losses' in res and res['train_losses']:
            ax.plot(res['train_losses'], label=res['model'],
                    color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'RNN Training Curves — {task.title()}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f'rnn_training_{task}.png')

def plot_flipflop_gates(X_seq, y, output_dir):
    input_size = X_seq.shape[2]
    ff = FlipFlopRNNCell(input_size, 8)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('FlipFlop RNN — J/K Gate & Q State Dynamics', fontsize=14, fontweight='bold')
    for row, (idx, label) in enumerate([(np.where(y == 0)[0][0], 'NORMAL'),
                                         (np.where(y == 1)[0][0], 'CEREBRAL PALSY')]):
        _, cache = ff.forward_sequence(X_seq[idx])
        j_vals = np.array([c[2] for c in cache])
        k_vals = np.array([c[3] for c in cache])
        q_vals = np.array([c[4] for c in cache])
        for ax, data, cmap, name in [(axes[row, 0], j_vals, 'Reds', 'J (SET)'),
                                      (axes[row, 1], k_vals, 'Blues', 'K (RESET)'),
                                      (axes[row, 2], q_vals, 'viridis', 'Q (Memory)')]:
            im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f'{label} — {name}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Hidden unit')
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    save_fig(fig, 'flipflop_gate_analysis.png')

def plot_clinical_panels(subj_stat, td_s, dcp_s):
    C = {'TD': '#2196F3', 'DCP': '#E53935'}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('UPPER-LIMB REACHING: NORMAL vs CEREBRAL PALSY\n'
                 '(Subject-level, Vanmechelen 2024)', fontsize=15, fontweight='bold', y=1.02)
    w = 0.35

    # Joint ROM
    ax = axes[0, 0]
    rom_f = ['shoulder_rom', 'elbow_rom', 'wrist_rom']
    rom_l = ['Shoulder', 'Elbow', 'Wrist']
    x = np.arange(len(rom_l))
    ax.bar(x - w / 2, [td_s[f].mean() for f in rom_f], w,
           yerr=[td_s[f].std() for f in rom_f], label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
    ax.bar(x + w / 2, [dcp_s[f].mean() for f in rom_f], w,
           yerr=[dcp_s[f].std() for f in rom_f], label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(rom_l)
    ax.set_ylabel('ROM (°)'); ax.set_title('Joint ROM'); ax.legend()

    # Jerk
    ax = axes[0, 1]
    jf = ['shoulder_act_jerk', 'elbow_act_jerk', 'wrist_act_jerk']
    jl = ['Shoulder', 'Elbow', 'Wrist']
    x = np.arange(len(jl))
    ax.bar(x - w / 2, [td_s[f].mean() for f in jf], w,
           yerr=[td_s[f].std() for f in jf], label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
    ax.bar(x + w / 2, [dcp_s[f].mean() for f in jf], w,
           yerr=[dcp_s[f].std() for f in jf], label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(jl)
    ax.set_ylabel('Normalized Jerk'); ax.set_title('Movement Smoothness'); ax.legend()

    # Peak Hand Speed
    ax = axes[0, 2]
    bp = ax.boxplot([td_s['hand_speed_peak'].values, dcp_s['hand_speed_peak'].values],
                    labels=['NORMAL', 'CEREBRAL PALSY'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C['TD']); bp['boxes'][1].set_facecolor(C['DCP'])
    for b in bp['boxes']:
        b.set_alpha(0.7)
    ax.set_ylabel('Peak Hand Speed (norm.)'); ax.set_title('Peak Hand Speed')

    # Path Efficiency
    ax = axes[1, 0]
    for cls, col, mk in [('TD', C['TD'], 'o'), ('DCP', C['DCP'], 's')]:
        sub = subj_stat[subj_stat['class'] == cls]
        ax.scatter(sub['index_of_curvature'], sub['convex_hull_area'],
                   c=col, marker=mk, alpha=0.7, s=60, label=cls, edgecolors='white')
    ax.set_xlabel('Index of Curvature'); ax.set_ylabel('Convex Hull Area')
    ax.set_title('Path Efficiency vs Spread'); ax.legend()

    # Trunk Compensation
    ax = axes[1, 1]
    bp = ax.boxplot([td_s['trunk_total_disp'].values, dcp_s['trunk_total_disp'].values],
                    labels=['NORMAL', 'CEREBRAL PALSY'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C['TD']); bp['boxes'][1].set_facecolor(C['DCP'])
    for b in bp['boxes']:
        b.set_alpha(0.7)
    ax.set_ylabel('Trunk Disp. (norm.)'); ax.set_title('Trunk Compensation')

    # Endpoint Control
    ax = axes[1, 2]
    x = np.arange(2)
    for cls, col, off in [('TD', C['TD'], -w / 2), ('DCP', C['DCP'], w / 2)]:
        sub = subj_stat[subj_stat['class'] == cls]
        means = [sub['endpoint_accuracy'].mean(), sub['mean_overshoot'].mean()]
        sds = [sub['endpoint_accuracy'].std(), sub['mean_overshoot'].std()]
        ax.bar(x + off, means, w, yerr=sds, label=cls, color=col, alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(['Endpoint\nAccuracy', 'Mean\nOvershoot'])
    ax.set_ylabel('Distance (norm.)'); ax.set_title('Endpoint Control'); ax.legend()

    plt.tight_layout()
    save_fig(fig, 'upper_limb_clinical.png')

def plot_dyskinesia_markers(subj_stat, td_s, dcp_s):
    C = {'TD': '#2196F3', 'DCP': '#E53935'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DYSKINESIA MARKERS: Angular Dynamics & Symmetry', fontsize=14, fontweight='bold')
    w = 0.35

    ax = axes[0]
    af = ['shoulder_act_alpha_rms', 'elbow_act_alpha_rms', 'wrist_act_alpha_rms']
    al = ['Shoulder', 'Elbow', 'Wrist']
    x = np.arange(len(al))
    ax.bar(x - w / 2, [td_s[f].mean() for f in af], w, yerr=[td_s[f].std() for f in af],
           label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
    ax.bar(x + w / 2, [dcp_s[f].mean() for f in af], w, yerr=[dcp_s[f].std() for f in af],
           label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(al)
    ax.set_ylabel('RMS Angular Accel (°/s²)'); ax.set_title('Angular Acceleration'); ax.legend()

    ax = axes[1]
    vf = ['shoulder_act_variability', 'elbow_act_variability', 'wrist_act_variability']
    x = np.arange(3)
    ax.bar(x - w / 2, [td_s[f].mean() for f in vf], w, yerr=[td_s[f].std() for f in vf],
           label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
    ax.bar(x + w / 2, [dcp_s[f].mean() for f in vf], w, yerr=[dcp_s[f].std() for f in vf],
           label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(al)
    ax.set_ylabel('Angular SD (°)'); ax.set_title('Movement Variability'); ax.legend()

    ax = axes[2]
    for cls, col in C.items():
        sub = subj_stat[subj_stat['class'] == cls]
        ax.hist(sub['arm_movement_ratio'], bins=12, alpha=0.6, color=col, label=cls, edgecolor='white')
    ax.set_xlabel('Passive/Active Ratio'); ax.set_ylabel('Count')
    ax.set_title('Inter-Limb Symmetry'); ax.legend()

    plt.tight_layout()
    save_fig(fig, 'dyskinesia_markers.png')

def plot_roc_curve(y_true, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'{model_name} AUC={auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC — {model_name}'); ax.legend()
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_fig(fig, f'roc_{safe_name}.png')

# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("COMPLETE PIPELINE: FFRNN + DONN + Multi-Model Comparison")
    print("Upper Limb Reaching: TD vs Cerebral Palsy")
    print("=" * 80)

    # ── Load data ──
    records, labels, subjects = load_all_data(DATA_DIR)
    N_FEATURES = records[0]['ts_data'].shape[1]
    sides = np.array([r['side'] for r in records])

    # ── Build feature matrices ──
    X_stat, stat_keys = build_stat_features(records)
    X_seq, mu_seq, sd_seq = build_sequence_data(records, MAX_SEQ_LEN)
    X_reg, y_reg = build_regression_data(records)
    print(f'\n  Stat features: {X_stat.shape}')
    print(f'  Sequence data:  {X_seq.shape}')
    print(f'  Regression:     X={X_reg.shape}, y={y_reg.shape}')

    # ── Build subject-level stat table ──
    stat_df = pd.DataFrame(X_stat, columns=stat_keys)
    stat_df['label'] = labels; stat_df['subject'] = subjects; stat_df['side'] = sides
    stat_df['class'] = ['TD' if l == 0 else 'DCP' for l in labels]
    subj_stat = stat_df.groupby(['subject', 'class', 'side'])[stat_keys].mean().reset_index()
    td_s = subj_stat[subj_stat['class'] == 'TD']
    dcp_s = subj_stat[subj_stat['class'] == 'DCP']

    all_clf_results = []
    all_reg_results = []

    # ==================================================================
    # TASK A: CLASSIFICATION — sklearn models (stat features)
    # ==================================================================
    print(f'\n{"=" * 60}')
    print('TASK A: CLASSIFICATION (stat features)')
    print(f'{"=" * 60}')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_stat, labels, test_size=0.2, stratify=labels, random_state=SEED)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    sklearn_clfs = [
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("SVM (RBF)",           SVC(kernel='rbf', probability=True, random_state=SEED)),
        ("MLP Neural Net",      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
    ]

    for name, model in sklearn_clfs:
        model.fit(X_tr_s, y_tr)
        y_tr_pred = model.predict(X_tr_s)
        y_te_pred = model.predict(X_te_s)
        try:
            y_te_prob = model.predict_proba(X_te_s)[:, 1]
            auc = roc_auc_score(y_te, y_te_prob)
        except:
            y_te_prob = None
            auc = 0

        acc_tr = accuracy_score(y_tr, y_tr_pred)
        acc_te = accuracy_score(y_te, y_te_pred)
        f1_te = f1_score(y_te, y_te_pred, average='weighted')

        print(f'  {name:<25s} Train={acc_tr:.3f} Test={acc_te:.3f} F1={f1_te:.3f} AUC={auc:.3f}')
        all_clf_results.append({
            'model': name, 'accuracy': acc_te, 'accuracy_std': 0,
            'f1': f1_te, 'f1_std': 0, 'auc': auc, 'auc_std': 0,
            'train_acc': acc_tr,
        })
        plot_confusion_matrix_pair(y_tr, y_tr_pred, y_te, y_te_pred, name)
        if y_te_prob is not None:
            plot_roc_curve(y_te, y_te_prob, name)

    # ==================================================================
    # TASK A: CLASSIFICATION — numpy RNN cells (sequence data)
    # ==================================================================
    print(f'\n{"=" * 60}')
    print('TASK A: CLASSIFICATION (RNN sequence models)')
    print(f'{"=" * 60}')

    n = len(X_seq)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_i, te_i = idx[:split], idx[split:]
    X_seq_tr, X_seq_te = X_seq[tr_i], X_seq[te_i]
    y_seq_tr, y_seq_te = labels[tr_i], labels[te_i]

    rnn_cells = [
        (FlipFlopRNNCell, "1D FlipFlop RNN"),
        (SimpleRNNCell,   "Simple RNN"),
        (LSTMLiteCell,    "LSTM-lite"),
        (GRULiteCell,     "GRU-lite"),
    ]

    rnn_clf_results = []
    for cell_class, cell_name in rnn_cells:
        print(f'\n  Training {cell_name}...')
        cell = cell_class(X_seq.shape[2], 16, lr=0.01)
        clf = RNNCellClassifier(cell, 16, lr=0.01, epochs=15)
        tl, va = clf.train(X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, verbose=True)

        y_tr_pred = clf.predict(X_seq_tr)
        y_te_pred = clf.predict(X_seq_te)
        y_te_prob = clf.predict_proba(X_seq_te)

        acc_tr = accuracy_score(y_seq_tr, y_tr_pred)
        acc_te = accuracy_score(y_seq_te, y_te_pred)
        f1_te = f1_score(y_seq_te, y_te_pred, average='weighted')
        try:
            auc = roc_auc_score(y_seq_te, y_te_prob)
        except:
            auc = 0

        print(f'  {cell_name:<25s} Train={acc_tr:.3f} Test={acc_te:.3f} F1={f1_te:.3f} AUC={auc:.3f}')
        res = {'model': cell_name, 'accuracy': acc_te, 'accuracy_std': 0,
               'f1': f1_te, 'f1_std': 0, 'auc': auc, 'auc_std': 0,
               'train_acc': acc_tr, 'train_losses': tl, 'val_accs': va}
        rnn_clf_results.append(res)
        all_clf_results.append(res)
        plot_confusion_matrix_pair(y_seq_tr, y_tr_pred, y_seq_te, y_te_pred, cell_name)

    plot_rnn_training_curves(rnn_clf_results, 'classification')

    # ==================================================================
    # TASK A: CLASSIFICATION — DONN (PyTorch, if available)
    # ==================================================================
    if HAS_TORCH:
        print(f'\n{"=" * 60}')
        print('TASK A: CLASSIFICATION — DONN (ResHopf Oscillatory)')
        print(f'{"=" * 60}')

        X_donn = np.stack([resample_temporal(r['ts_data'], SEQ_LEN) for r in records])
        y_donn = labels.copy()

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_results = []
        all_donn_preds = np.zeros(len(y_donn), dtype=int)
        all_donn_probs = np.zeros((len(y_donn), 2))

        for fi, (tri, tei) in enumerate(sgkf.split(X_donn, y_donn, groups=subjects)):
            print(f'\n  FOLD {fi+1}/{N_FOLDS}')
            Xtr, Xte = X_donn[tri], X_donn[tei]
            fl = Xtr.reshape(-1, Xtr.shape[-1])
            mu, sd = fl.mean(0), fl.std(0) + 1e-8

            tr_ld = DataLoader(GaitDataset((Xtr - mu) / sd, y_donn[tri]),
                               batch_size=BATCH_SIZE, shuffle=True)
            te_ld = DataLoader(GaitDataset((Xte - mu) / sd, y_donn[tei]),
                               batch_size=BATCH_SIZE, shuffle=False)

            cc = np.bincount(y_donn[tri], minlength=2).astype(float)
            cw = torch.tensor([1. / c for c in cc], dtype=torch.float32).to(device)
            cw /= cw.sum()
            criterion = nn.CrossEntropyLoss(weight=cw)

            model = make_donn_model(N_FEATURES)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

            best_bal, best_state, pat = 0., None, 0
            for ep in range(1, EPOCHS + 1):
                tl, ta = train_one_epoch(model, tr_ld, opt, criterion)
                vl, vp, vl2, vpr = evaluate_donn(model, te_ld, criterion)
                v_bal = balanced_accuracy_score(vl2, vp)
                sch.step()
                if v_bal > best_bal:
                    best_bal = v_bal
                    best_state = copy.deepcopy(model.state_dict())
                    pat = 0
                else:
                    pat += 1
                if ep % 20 == 0 or ep == 1:
                    print(f'    Ep {ep:3d}/{EPOCHS} TrL={tl:.4f} TrA={ta:.3f} '
                          f'VlBal={v_bal:.3f}{"  *" if pat==0 else ""}')
                if pat >= PATIENCE:
                    print(f'    Early stop ep {ep}.')
                    break

            model.load_state_dict(best_state)
            _, tp, tl2, tpr = evaluate_donn(model, te_ld, criterion)
            all_donn_preds[tei] = tp
            all_donn_probs[tei] = tpr

            f1v = f1_score(tl2, tp, average='weighted')
            bal = balanced_accuracy_score(tl2, tp)
            acc = accuracy_score(tl2, tp)
            try:
                auc = roc_auc_score(tl2, tpr[:, 1])
            except:
                auc = 0
            print(f'    Fold {fi+1}: Acc={acc:.4f} BalAcc={bal:.4f} F1={f1v:.4f} AUC={auc:.4f}')
            fold_results.append({'fold': fi + 1, 'accuracy': acc,
                                 'balanced_accuracy': bal, 'f1': f1v, 'auc': auc})

        # DONN pooled results
        donn_acc = accuracy_score(y_donn, all_donn_preds)
        donn_bal = balanced_accuracy_score(y_donn, all_donn_preds)
        donn_f1 = f1_score(y_donn, all_donn_preds, average='weighted')
        donn_auc = roc_auc_score(y_donn, all_donn_probs[:, 1])
        print(f'\n  DONN Pooled: Acc={donn_acc:.4f} BalAcc={donn_bal:.4f} '
              f'F1={donn_f1:.4f} AUC={donn_auc:.4f}')

        all_clf_results.append({
            'model': 'DONN (ResHopf)', 'accuracy': donn_acc, 'accuracy_std': 0,
            'f1': donn_f1, 'f1_std': 0, 'auc': donn_auc, 'auc_std': 0,
            'train_acc': 0,
        })

        # DONN confusion matrix (pooled CV)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        cm = confusion_matrix(y_donn, all_donn_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NORMAL', 'CEREBRAL PALSY'],
                    yticklabels=['NORMAL', 'CEREBRAL PALSY'], ax=ax)
        ax.set_title(f'DONN (CV Pooled)\nAcc={donn_acc:.3f} BalAcc={donn_bal:.3f}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout()
        save_fig(fig, 'cm_DONN_pooled.png')

        # DONN ROC
        plot_roc_curve(y_donn, all_donn_probs[:, 1], 'DONN (ResHopf)')

        # DONN per-fold bar
        rdf = pd.DataFrame(fold_results)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(N_FOLDS); w = 0.2
        ax.bar(x - 1.5 * w, rdf['accuracy'], w, label='Acc', color='#2196F3')
        ax.bar(x - 0.5 * w, rdf['balanced_accuracy'], w, label='BalAcc', color='#9C27B0')
        ax.bar(x + 0.5 * w, rdf['f1'], w, label='F1', color='#4CAF50')
        ax.bar(x + 1.5 * w, rdf['auc'], w, label='AUC', color='#FF9800')
        ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(N_FOLDS)])
        ax.set_ylim([0, 1.15]); ax.set_ylabel('Score')
        ax.set_title('DONN Per-Fold Performance'); ax.legend(ncol=4)
        plt.tight_layout()
        save_fig(fig, 'donn_per_fold.png')

        # Hopf oscillator frequencies
        with torch.no_grad():
            lf = (torch.sigmoid(model.hopf.omegas.cpu().squeeze())
                  * (MAX_OMEGA - MIN_OMEGA) + MIN_OMEGA).numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lf, bins=20, color='teal', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Learned Hopf Oscillator Frequencies')
        ax.axvline(lf.mean(), color='red', ls='--', label=f'Mean={lf.mean():.2f}')
        ax.legend()
        plt.tight_layout()
        save_fig(fig, 'hopf_frequencies.png')

        # Subject-level majority vote
        sv, st = {}, {}
        for i in range(len(y_donn)):
            s = subjects[i]
            sv.setdefault(s, []).append(all_donn_preds[i])
            st[s] = y_donn[i]
        sids = sorted(sv.keys())
        syt = np.array([st[s] for s in sids])
        syp = np.array([int(np.round(np.mean(sv[s]))) for s in sids])
        sa = accuracy_score(syt, syp)
        sb = balanced_accuracy_score(syt, syp)
        print(f'\n  DONN Subject Vote: Acc={sa:.4f} BalAcc={sb:.4f} '
              f'({(syt == syp).sum()}/{len(sids)} correct)')

        # Subject vote confusion
        fig, ax = plt.subplots(figsize=(6, 5))
        cms = confusion_matrix(syt, syp)
        sns.heatmap(cms, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['NORMAL', 'CEREBRAL PALSY'],
                    yticklabels=['NORMAL', 'CEREBRAL PALSY'], ax=ax)
        ax.set_title(f'DONN Subject Vote\nAcc={sa:.3f} BalAcc={sb:.3f}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout()
        save_fig(fig, 'cm_DONN_subject_vote.png')

        # Classification report
        print('\n' + classification_report(
            y_donn, all_donn_preds, target_names=['NORMAL', 'CEREBRAL PALSY']))

    # ==================================================================
    # TASK B: REGRESSION — sklearn models
    # ==================================================================
    print(f'\n{"=" * 60}')
    print('TASK B: REGRESSION (Joint Angle Prediction)')
    print(f'{"=" * 60}')

    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=SEED)
    sc_r = StandardScaler()
    X_r_tr_s = sc_r.fit_transform(X_r_tr)
    X_r_te_s = sc_r.transform(X_r_te)

    sklearn_regs = [
        ("Random Forest",    RandomForestRegressor(n_estimators=100, random_state=SEED)),
        ("MLP Neural Net",   MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Gradient Boosting", MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=SEED))),
    ]

    for name, model in sklearn_regs:
        model.fit(X_r_tr_s, y_r_tr)
        y_tr_p = model.predict(X_r_tr_s)
        y_te_p = model.predict(X_r_te_s)
        mse_tr = mean_squared_error(y_r_tr, y_tr_p)
        mse_te = mean_squared_error(y_r_te, y_te_p)
        r2_te = r2_score(y_r_te, y_te_p)
        mae_te = mean_absolute_error(y_r_te, y_te_p)
        print(f'  {name:<25s} Train_MSE={mse_tr:.2f} Test_MSE={mse_te:.2f} '
              f'R²={r2_te:.3f} MAE={mae_te:.2f}')
        all_reg_results.append({
            'model': name, 'mse': mse_te, 'r2': r2_te, 'mae': mae_te,
            'mse_train': mse_tr,
        })

    # ==================================================================
    # CLINICAL ANALYSIS — Statistical testing
    # ==================================================================
    print(f'\n{"=" * 60}')
    print('CLINICAL ANALYSIS — Mann-Whitney U (Subject-level)')
    print(f'{"=" * 60}')

    key_feats = [
        ('shoulder_act_angle_range', 'Shoulder ROM'),
        ('elbow_act_angle_range', 'Elbow ROM'),
        ('wrist_act_angle_range', 'Wrist ROM'),
        ('shoulder_act_jerk', 'Shoulder Jerk'),
        ('elbow_act_jerk', 'Elbow Jerk'),
        ('hand_speed_peak', 'Hand Speed Peak'),
        ('index_of_curvature', 'Index of Curvature'),
        ('endpoint_accuracy', 'Endpoint Accuracy'),
        ('trunk_total_disp', 'Trunk Compensation'),
        ('arm_movement_ratio', 'Passive/Active Ratio'),
    ]

    print(f'\n{"Feature":<25s} | {"TD":>16s} | {"DCP":>16s} | {"p":>10s} | Sig')
    print('─' * 80)
    for fk, fn in key_feats:
        tv, dv = td_s[fk].values, dcp_s[fk].values
        try:
            _, pv = mannwhitneyu(tv, dv, alternative='two-sided')
        except:
            pv = 1.0
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
        print(f'{fn:<25s} | {tv.mean():7.3f}±{tv.std():6.3f} | '
              f'{dv.mean():7.3f}±{dv.std():6.3f} | {pv:10.6f} | {sig}')

    # ==================================================================
    # ALL PLOTS
    # ==================================================================
    print(f'\n{"=" * 60}')
    print('GENERATING ALL PLOTS')
    print(f'{"=" * 60}')

    plot_all_classification_bars(all_clf_results)
    plot_regression_bars(all_reg_results)
    plot_flipflop_gates(X_seq, labels, OUTPUT_DIR)
    plot_clinical_panels(subj_stat, td_s, dcp_s)
    plot_dyskinesia_markers(subj_stat, td_s, dcp_s)

    # Per-subject hand path spread
    fig, ax = plt.subplots(figsize=(14, 6))
    psi = subj_stat[['subject', 'class', 'convex_hull_area']].copy()
    psi = psi.sort_values(['class', 'convex_hull_area'])
    x = np.arange(len(psi))
    bcol = ['#2196F3' if c == 'TD' else '#E53935' for c in psi['class']]
    ax.bar(x, psi['convex_hull_area'], color=bcol, alpha=0.8, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(psi['subject'], rotation=45, fontsize=8)
    ax.set_xlabel('Subject'); ax.set_ylabel('Convex Hull Area')
    ax.set_title('Per-Subject Hand Path Spread\nBlue=NORMAL  Red=CEREBRAL PALSY')
    ntd = (psi['class'] == 'TD').sum()
    ax.axvline(ntd - 0.5, color='black', ls='--', lw=2, alpha=0.5)
    plt.tight_layout()
    save_fig(fig, 'per_subject_hand_path.png')

    # Comprehensive summary
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'FFRNN + DONN Multi-Model Comparison — Upper Limb Kinematics (TD vs CP)\n'
        'Kumari, Chandrasekaran & Chakravarthy (2023) | IIT Madras Neuroscience Lab',
        fontsize=14, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    mc = [r['model'] for r in all_clf_results]
    ac = [r['accuracy'] for r in all_clf_results]
    cc = ['#e74c3c' if 'FlipFlop' in m or 'DONN' in m else '#3498db' for m in mc]
    bars = ax1.barh(mc, ac, color=cc, edgecolor='gray', alpha=0.85)
    ax1.set_xlim(0, 1.1); ax1.set_xlabel('Accuracy')
    ax1.set_title('A. Classification Accuracy', fontweight='bold')
    for bar, v in zip(bars, ac):
        ax1.text(min(v + 0.01, 1.05), bar.get_y() + bar.get_height() / 2,
                 f'{v:.3f}', va='center', fontsize=8)
    ax1.grid(axis='x', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    au = [r['auc'] for r in all_clf_results]
    bars = ax2.barh(mc, au, color=cc, edgecolor='gray', alpha=0.85)
    ax2.set_xlim(0, 1.1); ax2.set_xlabel('AUC-ROC')
    ax2.set_title('B. Classification AUC-ROC', fontweight='bold')
    for bar, v in zip(bars, au):
        ax2.text(min(v + 0.01, 1.05), bar.get_y() + bar.get_height() / 2,
                 f'{v:.3f}', va='center', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    mr = [r['model'] for r in all_reg_results]
    r2v = [r['r2'] for r in all_reg_results]
    bars = ax3.barh(mr, r2v, color='#2ecc71', edgecolor='gray', alpha=0.85)
    ax3.set_xlabel('R²'); ax3.set_title('C. Regression R²', fontweight='bold')
    for bar, v in zip(bars, r2v):
        ax3.text(max(v, 0) + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{v:.3f}', va='center', fontsize=9)
    ax3.grid(axis='x', alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    msev = [r['mse'] for r in all_reg_results]
    bars = ax4.barh(mr, msev, color='#e67e22', edgecolor='gray', alpha=0.85)
    ax4.set_xlabel('MSE'); ax4.set_title('D. Regression MSE', fontweight='bold')
    for bar, v in zip(bars, msev):
        ax4.text(v + 1, bar.get_y() + bar.get_height() / 2,
                 f'{v:.1f}', va='center', fontsize=9)
    ax4.grid(axis='x', alpha=0.3)

    save_fig(fig, 'comprehensive_summary.png')

    # ── Save CSV tables ──
    pd.DataFrame(all_clf_results).drop(columns=['train_losses', 'val_accs'], errors='ignore') \
        .to_csv(os.path.join(OUTPUT_DIR, 'classification_results.csv'), index=False)
    pd.DataFrame(all_reg_results).to_csv(
        os.path.join(OUTPUT_DIR, 'regression_results.csv'), index=False)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f'\n{"=" * 80}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"=" * 80}')

    print(f'\n{"Model":<25s} {"Train":>7s} {"Test":>7s} {"F1":>7s} {"AUC":>7s}')
    print('-' * 60)
    for r in sorted(all_clf_results, key=lambda x: x['accuracy'], reverse=True):
        ta = r.get('train_acc', 0)
        print(f'{r["model"]:<25s} {ta:7.3f} {r["accuracy"]:7.3f} {r["f1"]:7.3f} {r["auc"]:7.3f}')

    print(f'\n{"Model":<25s} {"Train_MSE":>10s} {"Test_MSE":>10s} {"R²":>7s} {"MAE":>7s}')
    print('-' * 60)
    for r in sorted(all_reg_results, key=lambda x: x['r2'], reverse=True):
        mt = r.get('mse_train', 0)
        print(f'{r["model"]:<25s} {mt:10.2f} {r["mse"]:10.2f} {r["r2"]:7.3f} {r["mae"]:7.2f}')

    print(f'\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}')
    print(f'Files: {", ".join(sorted(os.listdir(OUTPUT_DIR)))}')
    print('=' * 80)
    print('PIPELINE COMPLETE.')


if __name__ == '__main__':
    main()
