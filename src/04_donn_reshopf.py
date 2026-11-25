#!/usr/bin/env python3
"""
DONN Upper-Limb Reaching Classification — TD vs DCP  (v3)

Dataset : Vanmechelen et al. (2024) J Biomech 173:112233
          doi:10.48804/BZM9SJ  —  reaching-sideways task
          2D markerless (DeepLabCut) stick-figure kinematics
          30 DCP + 21 TD subjects × 8 reps = 408 trials

Task    : Subject reaches laterally toward a target ("object")
          with the arm indicated by filename side (left/right).

Landmarks (per file):
  forehead, chin                         — head/trunk reference
  shoulder_{right,left}                  — shoulder joint
  elbow_{right,left}                     — elbow joint
  wrist_{right,left}                     — wrist joint
  MCP3_{right,left}                      — hand (3rd metacarpal)
  object                                 — reaching target (stationary)
  ankle1, ankle2, hip, knee              — background, NOT used

Pre-computed angles in xlsx:
  R-Shoulder, L-Shoulder                 — shoulder elevation
  R-Elbow, L-Elbow                       — elbow flexion/extension
  R-Wrist, L-Wrist                       — wrist angle

Feature design (upper-limb specific):
  1. Shoulder elevation trajectory         (active + passive)
  2. Elbow flexion/extension trajectory    (active + passive)
  3. Wrist angle trajectory                (active + passive)
  4. Hand (MCP3) path trajectory           (active arm)
  5. Hand-to-target distance profile
  6. Hand velocity / speed profile
  7. Hand acceleration profile
  8. Jerk / movement smoothness
  9. Index of curvature (path length / straight-line)
  10. Endpoint accuracy (final hand-target distance)
  11. Overshoot (hand past target)
  12. Convex hull area of hand path
  13. Trunk displacement (compensatory movement)
  14. Trunk lean / head stability
  15. Active/passive arm symmetry
  16. Range of motion per joint
  17. Movement variability (angular trajectory SD)

Normalization: upper-arm length (shoulder→elbow distance)

Infrastructure (carried from v2):
  - Robust regex filename parsing
  - Robust column detection
  - scipy.signal.resample (not truncate/tile)
  - balanced_accuracy for early stopping
  - Final model retrained on all data
  - Stats tested at subject level
  - Bootstrap CIs
  - Missing-data diagnostics
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX: Force interactive backend BEFORE any other matplotlib import ★
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
# Try TkAgg first, fall back to Qt5Agg, then Agg (non-interactive but saves PNGs)
_backend_set = False
for _backend in ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg']:
    try:
        matplotlib.use(_backend)
        _backend_set = True
        break
    except ImportError:
        continue
if not _backend_set:
    matplotlib.use('Agg')  # fallback: no interactive window, but savefig works
    print('WARNING: No interactive matplotlib backend found. Plots will be saved as PNG only.')
    print('  Install one: pip install pyqt5   OR   pip install tk')

import matplotlib.pyplot as plt
print(f'Matplotlib backend: {matplotlib.get_backend()}')
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob, os, re, math, copy, warnings
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import seaborn as sns
from scipy.signal import savgol_filter, resample
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | PyTorch: {torch.__version__}')

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'), exist_ok=True)
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))  # figures are saved relative to cwd
SEQ_LEN      = 50
N_FOLDS      = 5
SEED         = 42
FPS          = 30.0
N_BOOTSTRAP  = 1000

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
EPOCHS       = 100
PATIENCE     = 20

np.random.seed(SEED); torch.manual_seed(SEED)

# =============================================================================
# 1. DONN LAYERS
# =============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, inf, outf, bias=True, dtype=torch.cfloat):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outf, inf, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(outf, dtype=dtype)) if bias else None
        if not bias: self.register_parameter('bias', None)
        self._reset()
    def _reset(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fi, _ = init._calculate_fan_in_and_fan_out(self.weight)
            init.uniform_(self.bias, -1/math.sqrt(fi+1e-8), 1/math.sqrt(fi+1e-8))
    def forward(self, x): return F.linear(x, self.weight, self.bias)

class ResHopf(nn.Module):
    def __init__(self, units, min_omega=0.5, max_omega=5.0, dt=0.01,
                 train_omegas=True, input_scaler=5.0):
        super().__init__()
        self.units, self.min_omega, self.max_omega, self.dt = units, min_omega, max_omega, dt
        self.mu0, self.beta1, self.input_scaler = 1.0, 1.0, input_scaler
        self.omegas = nn.Parameter(torch.randn(1, units), requires_grad=train_omegas)
    def forward(self, X, r=None, phi=None):
        T, B, D = X.shape; dev = X.device
        if r is None: r = torch.ones(B, D, device=dev)
        if phi is None: phi = (math.pi/2)*torch.rand(B, D, device=dev)-(math.pi/4)
        om = (torch.sigmoid(self.omegas)*(self.max_omega-self.min_omega)+self.min_omega)*2*math.pi
        om = om.to(dev)
        rs, ps = [], []
        for t in range(T):
            ir = self.input_scaler*X[t].real*torch.cos(phi)
            ip = self.input_scaler*X[t].imag*torch.sin(phi)
            r = r + ((self.mu0-self.beta1*r**2)*r + ir)*self.dt
            phi = phi + (om - ip)*self.dt
            rs.append(r); ps.append(phi)
        rs, ps = torch.stack(rs,0), torch.stack(ps,0)
        return torch.complex(rs*torch.cos(ps), rs*torch.sin(ps)), r, phi

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
            nn.Linear(hidden_dim*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes))
    def forward(self, x):
        B, T, D = x.shape
        xc = torch.complex(self.encoder_r(x), self.encoder_i(x)).permute(1,0,2)
        xp = torch.stack([self.proj(xc[t]) for t in range(T)], 0)
        z, _, _ = self.hopf(xp)
        h = torch.stack([self.readout(z[t]) for t in range(T)], 0)
        hm = h.abs()
        return self.classifier(torch.cat([hm.mean(0), hm[-1]], -1))

print('[1] DONN loaded.')

# =============================================================================
# 2. ROBUST FILENAME PARSING
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
    if 'reo' in base.lower():
        print(f'  NOTE: typo in filename "{base}" (reo→rep), loading anyway.')
    return {
        'subject_id': m.group('subject'),
        'side':       m.group('side').lower(),
        'class_str':  cls,
        'label':      0 if cls=='TD' else 1,
        'repetition': int(m.group('rep')),
        'filename':   os.path.basename(fname),
    }

print('[2] Filename parser ready.')

# =============================================================================
# 3. ROBUST COLUMN DETECTION
# =============================================================================
def _detect_xy_col(df_columns, joint_name):
    patterns = [
        (f"('{joint_name}', 'x')", f"('{joint_name}', 'y')"),
        (f'{joint_name}_x', f'{joint_name}_y'),
    ]
    for cx, cy in patterns:
        if cx in df_columns and cy in df_columns:
            return cx, cy
    cols = list(df_columns)
    x_match = [c for c in cols if joint_name.lower() in str(c).lower() and "'x'" in str(c).lower()]
    y_match = [c for c in cols if joint_name.lower() in str(c).lower() and "'y'" in str(c).lower()]
    if x_match and y_match:
        return x_match[0], y_match[0]
    return None, None

JOINT_NAMES = {
    'shoulder_r': 'shoulder_right', 'shoulder_l': 'shoulder_left',
    'elbow_r': 'elbow_right',       'elbow_l': 'elbow_left',
    'wrist_r': 'wrist_right',       'wrist_l': 'wrist_left',
    'hand_r': 'MCP3_right',         'hand_l': 'MCP3_left',
    'forehead': 'forehead',          'chin': 'chin',
    'object': 'object',
}

print('[3] Column detection ready.')

# =============================================================================
# 4. HELPERS
# =============================================================================
def get_xy(df, canonical):
    jname = JOINT_NAMES.get(canonical, canonical)
    cx, cy = _detect_xy_col(df.columns, jname)
    T = len(df)
    if cx is None:
        return np.full((T,2), np.nan, dtype=np.float64), 1.0
    x = df[cx].interpolate(method='linear', limit_direction='both').ffill().bfill()
    y = df[cy].interpolate(method='linear', limit_direction='both').ffill().bfill()
    miss = max(df[cx].isnull().mean(), df[cy].isnull().mean())
    return np.stack([x.values, y.values], 1).astype(np.float64), miss

def angle_3pt(a, b, c):
    ba, bc = a-b, c-b
    cos_a = np.sum(ba*bc, axis=1)/(np.linalg.norm(ba,axis=1)*np.linalg.norm(bc,axis=1)+1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def smooth(s, window=5, polyorder=2):
    if len(s) < max(window, polyorder+1): return s.copy()
    w = window if window%2==1 else window+1
    return savgol_filter(s, w, polyorder)

def deriv(s, dt=1.0/30.0):
    return np.gradient(s, dt)

def jerk_rms(s, dt=1.0/30.0):
    return np.sqrt(np.mean(deriv(deriv(deriv(s, dt), dt), dt)**2))

def norm_jerk(s, dt=1.0/30.0):
    j = deriv(deriv(deriv(s, dt), dt), dt)
    dur = len(s)*dt
    rng = max(np.ptp(s), 1e-8)
    return np.sqrt(0.5*np.trapz(j**2, dx=dt)*dur**5/rng**2)

def convex_hull_area(points_2d):
    try:
        pts = points_2d[~np.isnan(points_2d).any(axis=1)]
        if len(pts) < 3: return 0.0
        if np.linalg.matrix_rank(pts - pts.mean(0)) < 2: return 0.0
        return ConvexHull(pts).volume
    except:
        return 0.0

def resample_temporal(seq, target_len):
    T, D = seq.shape
    if T == target_len: return seq
    return resample(seq, target_len, axis=0).astype(np.float32)

print('[4] Helpers ready.')

# =============================================================================
# 5. UPPER-LIMB FEATURE EXTRACTION
# =============================================================================
def extract_upper_limb_features(df, meta):
    T = len(df)
    dt = 1.0 / FPS
    active_side = meta['side']

    if active_side == 'left':
        act, pas = 'l', 'r'
    else:
        act, pas = 'r', 'l'

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

    diag = {
        'filename': meta['filename'],
        'n_frames': T,
        'active_side': active_side,
        'shoulder_act_miss': m1,
        'shoulder_pas_miss': m2,
    }

    upper_arm_len = np.linalg.norm(elbow_act - shoulder_act, axis=1)
    forearm_len   = np.linalg.norm(wrist_act - elbow_act, axis=1)
    arm_len       = upper_arm_len + forearm_len
    ref_scale     = np.maximum(upper_arm_len, 1e-6)

    mid_shoulder = (shoulder_act + shoulder_pas) / 2.0
    shoulder_width = np.linalg.norm(shoulder_act - shoulder_pas, axis=1)

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
    trunk_total_disp = np.cumsum(trunk_disp_norm)
    trunk_drift = np.linalg.norm(trunk_mid - trunk_mid[0], axis=1) / ref_scale
    head_x_disp = (forehead[:, 0] - forehead[0, 0]) / ref_scale

    pas_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_pas, axis=0), axis=1)])
    pas_total = np.sum(pas_disp)
    act_total = np.sum(hand_disp)
    arm_movement_ratio = pas_total / (act_total + 1e-8)

    ts = np.column_stack([
        n_shoulder_act, n_elbow_act, n_wrist_act, n_hand_act,
        n_shoulder_pas, n_elbow_pas, n_wrist_pas, n_hand_pas,
        n_target,
        n_forehead,
        ang_shoulder_act[:, None], ang_elbow_act[:, None], ang_wrist_act[:, None],
        ang_shoulder_pas[:, None], ang_elbow_pas[:, None], ang_wrist_pas[:, None],
        precomp['shoulder'][:, None],
        precomp['elbow'][:, None],
        precomp['wrist'][:, None],
        w_shoulder_act[:, None], w_elbow_act[:, None], w_wrist_act[:, None],
        w_shoulder_pas[:, None], w_elbow_pas[:, None],
        a_shoulder_act[:, None], a_elbow_act[:, None], a_wrist_act[:, None],
        hand_to_target_norm[:, None],
        hand_speed_smooth[:, None],
        hand_accel[:, None],
        (hand_disp / ref_scale)[:, None],
        trunk_disp_norm[:, None],
        trunk_drift[:, None],
        head_x_disp[:, None],
        (pas_disp / ref_scale)[:, None],
    ])

    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    rms = lambda x: np.sqrt(np.mean(x**2))

    total_path = np.sum(hand_disp)
    straight_line = np.linalg.norm(hand_act[-1] - hand_act[0])
    index_of_curvature = total_path / (straight_line + 1e-8)

    endpoint_dist = hand_to_target[-1]
    endpoint_dist_norm = endpoint_dist / np.mean(ref_scale)

    min_dist_idx = np.argmin(hand_to_target)
    overshoot_frames = T - 1 - min_dist_idx
    overshoot_ratio = overshoot_frames / T

    if min_dist_idx < T - 1:
        mean_overshoot = np.mean(hand_to_target[min_dist_idx:]) / np.mean(ref_scale)
    else:
        mean_overshoot = 0.0

    hull_area = convex_hull_area(hand_act) / (np.mean(ref_scale)**2)
    execution_time = T / FPS

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
        'hand_speed_cv':            np.std(hand_speed_norm)/(np.mean(hand_speed_norm)+1e-8),
        'hand_accel_rms':           rms(hand_accel),
        'hand_jerk_rms':            jerk_rms(hand_speed_norm, dt),
        'index_of_curvature':       index_of_curvature,
        'endpoint_accuracy':        endpoint_dist_norm,
        'mean_overshoot':           mean_overshoot,
        'overshoot_ratio':          overshoot_ratio,
        'convex_hull_area':         hull_area,
        'total_path_norm':          total_path / np.mean(ref_scale),
        'execution_time':           execution_time,
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

print('[5] Upper-limb feature extraction ready.')

# =============================================================================
# 6. LOAD ALL DATA
# =============================================================================
import zipfile
_zip_candidates = [
    os.path.join(os.getcwd(), 'data.zip'),
    os.path.join(DATA_DIR, '..', 'data.zip'),
]
for zp in _zip_candidates:
    if os.path.isfile(zp) and not os.path.isdir(DATA_DIR):
        print(f'  Found {zp}, extracting to {DATA_DIR}/ ...')
        os.makedirs(DATA_DIR, exist_ok=True)
        with zipfile.ZipFile(zp, 'r') as z:
            z.extractall(DATA_DIR)
        print(f'  Extracted {len(os.listdir(DATA_DIR))} files.')
        break

_candidates = [
    DATA_DIR,
    os.path.join(os.getcwd(), 'data'),
    os.path.join(os.path.dirname(os.path.abspath(__file__))
                 if '__file__' in dir() else os.getcwd(), 'data'),
    '.',
    os.path.expanduser('~/data'),
    'data/data',
]

def _find_matching_files(folder):
    all_xlsx = sorted(glob.glob(os.path.join(folder, '*.xlsx')))
    matched = [f for f in all_xlsx if _FNAME_RE.match(
        os.path.splitext(os.path.basename(f))[0])]
    return matched, all_xlsx

all_files = []
_search_log = []
for cand in _candidates:
    matched, total = _find_matching_files(cand)
    _search_log.append((cand, len(total), len(matched)))
    if len(matched) > 0:
        all_files = matched
        DATA_DIR = cand
        break

if len(all_files) == 0:
    print('\n' + '!'*70)
    print('ERROR: No matching dataset files found.')
    print(f'Expected filename pattern: {{digits}}_{{left|right}}_{{TD|DCP}}_rep{{N}}.xlsx')
    print(f'Example: 01_left_TD_rep1.xlsx, 25_right_DCP_rep3.xlsx\n')
    print(f'{"Path":<55s} | {"xlsx":>5s} | {"match":>5s}')
    print('─'*72)
    for path, n_total, n_match in _search_log:
        abs_p = os.path.abspath(path)
        print(f'{abs_p:<55s} | {n_total:>5d} | {n_match:>5d}')

    for path, n_total, n_match in _search_log:
        if n_total > 0 and n_match == 0:
            xlsx_in_dir = sorted(glob.glob(os.path.join(path, '*.xlsx')))
            print(f'\nFiles found in {os.path.abspath(path)} (not matching pattern):')
            for f in xlsx_in_dir[:8]:
                print(f'  {os.path.basename(f)}')
            if len(xlsx_in_dir) > 8:
                print(f'  ... and {len(xlsx_in_dir)-8} more')

    print(f'\nCurrent working directory: {os.getcwd()}')
    print(f'\nFIX — either:')
    print(f'  1) Unzip data.zip into a "data/" folder in your working directory')
    print(f'  2) Set DATA_DIR at the top of this script to the correct path:')
    print(f'     DATA_DIR = r"path\\to\\xlsx_files"')
    print('!'*70)
    raise FileNotFoundError(
        'No dataset files found. See diagnostics above. '
        'Set DATA_DIR to the folder containing your xlsx files.')

print(f'\n[6] Found {len(all_files)} matching files in: {os.path.abspath(DATA_DIR)}')
print(f'  {len(all_files)} valid files to process.')

records = []
all_diag = []
errors = []
for i, fpath in enumerate(all_files):
    try:
        meta = parse_filename(fpath)
        df = pd.read_excel(fpath)
        drop = [c for c in df.columns
                if re.match(r'^unnamed', str(c), re.I) or str(c).strip().lower()=='index']
        df = df.drop(columns=drop)
        ts, stats, diag = extract_upper_limb_features(df, meta)
        records.append({**meta, 'ts_data': ts, 'stats': stats})
        all_diag.append(diag)
    except Exception as e:
        errors.append((os.path.basename(fpath), str(e)))
    if (i+1) % 100 == 0:
        print(f'  Loaded {i+1}/{len(all_files)}')

if errors:
    print(f'\n  WARNING: {len(errors)} files failed during feature extraction:')
    for fn, err in errors[:5]:
        print(f'    {fn}: {err}')

if len(records) == 0:
    raise RuntimeError(
        f'All {len(all_files)} files failed during feature extraction. '
        f'First error: {errors[0] if errors else "unknown"}')

labels   = np.array([r['label'] for r in records])
subjects = np.array([r['subject_id'] for r in records])
sides    = np.array([r['side'] for r in records])

N_FEATURES = records[0]['ts_data'].shape[1]
print(f'  {len(records)} trials loaded | {N_FEATURES} features/frame')
print(f'  TD: {(labels==0).sum()} ({len(set(subjects[labels==0]))} subj) | '
      f'DCP: {(labels==1).sum()} ({len(set(subjects[labels==1]))} subj)')

if all_diag:
    diag_df = pd.DataFrame(all_diag)
    print(f'\n  Active-side distribution: {dict(diag_df["active_side"].value_counts())}')
    frame_lens = diag_df['n_frames']
    print(f'  Frame counts: min={frame_lens.min()}, max={frame_lens.max()}, '
          f'median={frame_lens.median():.0f}')

# =============================================================================
# 7. BUILD MATRICES
# =============================================================================
X_all = np.stack([resample_temporal(r['ts_data'], SEQ_LEN) for r in records])
y_all = labels.copy()

stat_keys = sorted(records[0]['stats'].keys())
stat_matrix = np.array([[r['stats'][k] for k in stat_keys] for r in records])
stat_df = pd.DataFrame(stat_matrix, columns=stat_keys)
stat_df['label'] = labels; stat_df['subject'] = subjects
stat_df['side'] = sides; stat_df['class'] = ['TD' if l==0 else 'DCP' for l in labels]

print(f'  X (resampled): {X_all.shape}')

# =============================================================================
# 8. TRAIN / EVAL
# =============================================================================
class GaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_one_epoch(model, loader, optimizer, criterion):
    model.train(); ls, c, n = 0., 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(); out = model(Xb); loss = criterion(out, yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        ls += loss.item()*Xb.size(0); c += (out.argmax(1)==yb).sum().item(); n += Xb.size(0)
    return ls/n, c/n

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); ls, c, n = 0., 0, 0
    P, Pr, L = [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb); loss = criterion(out, yb)
        ls += loss.item()*Xb.size(0)
        pr = torch.softmax(out, 1); pd_ = out.argmax(1)
        c += (pd_==yb).sum().item(); n += Xb.size(0)
        P.append(pr.cpu().numpy()); Pr.append(pd_.cpu().numpy()); L.append(yb.cpu().numpy())
    return ls/n, np.concatenate(Pr), np.concatenate(L), np.concatenate(P)

def make_model():
    return DONNClassifier(
        N_FEATURES, HIDDEN_DIM, OSC_UNITS, 2, DT, MIN_OMEGA, MAX_OMEGA,
        INPUT_SCALER, DROPOUT).to(device)

# =============================================================================
# 9. CROSS-VALIDATION (balanced_accuracy early stopping)
# =============================================================================
print(f'\n{"="*70}')
print(f'CV: {N_FOLDS}-Fold Subject-Level | {N_FEATURES} upper-limb features')
print(f'Early-stop: balanced_accuracy | Resample: scipy.signal.resample')
print(f'{"="*70}')

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
all_test_preds = np.zeros(len(y_all), dtype=int)
all_test_probs = np.zeros((len(y_all), 2))

for fi, (tri, tei) in enumerate(sgkf.split(X_all, y_all, groups=subjects)):
    print(f'\n{"─"*60}\nFOLD {fi+1}/{N_FOLDS}')
    te_subj = sorted(set(subjects[tei]))
    print(f'  Train: {len(set(subjects[tri]))} subj | Test: {len(te_subj)} subj → {te_subj}')

    Xtr, Xte = X_all[tri], X_all[tei]
    fl = Xtr.reshape(-1, Xtr.shape[-1])
    mu, sd = fl.mean(0), fl.std(0)+1e-8

    tr_ld = DataLoader(GaitDataset((Xtr-mu)/sd, y_all[tri]), batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(GaitDataset((Xte-mu)/sd, y_all[tei]), batch_size=BATCH_SIZE, shuffle=False)

    cc = np.bincount(y_all[tri], minlength=2).astype(float)
    cw = torch.tensor([1./c for c in cc], dtype=torch.float32).to(device); cw /= cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_bal, best_state, pat = 0., None, 0
    for ep in range(1, EPOCHS+1):
        tl, ta = train_one_epoch(model, tr_ld, opt, criterion)
        vl, vp, vl2, vpr = evaluate(model, te_ld, criterion)
        v_bal = balanced_accuracy_score(vl2, vp)
        sch.step()
        if v_bal > best_bal:
            best_bal = v_bal; best_state = copy.deepcopy(model.state_dict()); pat = 0
        else: pat += 1
        if ep % 20 == 0 or ep == 1:
            print(f'  Ep {ep:3d}/{EPOCHS} TrL={tl:.4f} TrA={ta:.3f} '
                  f'VlBal={v_bal:.3f}{"  *" if pat==0 else ""}')
        if pat >= PATIENCE: print(f'  Early stop ep {ep}.'); break

    model.load_state_dict(best_state)
    _, tp, tl2, tpr = evaluate(model, te_ld, criterion)
    all_test_preds[tei] = tp; all_test_probs[tei] = tpr

    f1 = f1_score(tl2, tp, average='weighted')
    bal = balanced_accuracy_score(tl2, tp)
    acc = accuracy_score(tl2, tp)
    try: auc = roc_auc_score(tl2, tpr[:,1])
    except: auc = 0.
    print(f'  → Fold {fi+1}: Acc={acc:.4f} BalAcc={bal:.4f} F1={f1:.4f} AUC={auc:.4f}')
    fold_results.append({'fold':fi+1,'accuracy':acc,'balanced_accuracy':bal,
                         'f1':f1,'auc':auc,'test_subjects':te_subj})

# =============================================================================
# 10. RETRAIN FINAL MODEL ON ALL DATA
# =============================================================================
print('\nRetraining final model on ALL data...')
fl_all = X_all.reshape(-1, X_all.shape[-1])
mu_final, sd_final = fl_all.mean(0), fl_all.std(0)+1e-8
X_all_norm = (X_all - mu_final) / sd_final
full_ld = DataLoader(GaitDataset(X_all_norm, y_all), batch_size=BATCH_SIZE, shuffle=True)
cc = np.bincount(y_all, minlength=2).astype(float)
cw = torch.tensor([1./c for c in cc], dtype=torch.float32).to(device); cw /= cw.sum()
crit_final = nn.CrossEntropyLoss(weight=cw)
final_model = make_model()
opt = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
for ep in range(1, EPOCHS+1):
    tl, ta = train_one_epoch(final_model, full_ld, opt, crit_final)
    sch.step()
    if ep % 20 == 0 or ep == 1:
        print(f'  Ep {ep:3d} Loss={tl:.4f} Acc={ta:.3f}')
print('  Final model trained.')

# =============================================================================
# 11. RESULTS + BOOTSTRAP CI
# =============================================================================
def bootstrap_ci(yt, yp, fn, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed); scores = []
    for _ in range(n):
        idx = rng.randint(0, len(yt), len(yt))
        try: scores.append(fn(yt[idx], yp[idx]))
        except: pass
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

rdf = pd.DataFrame(fold_results)
print(f'\n{"="*70}\nRESULTS\n{"="*70}')
print(rdf[['fold','accuracy','balanced_accuracy','f1','auc']].to_string(index=False))
for m in ['accuracy','balanced_accuracy','f1','auc']:
    print(f'  {m:<20s}: {rdf[m].mean():.4f} ± {rdf[m].std():.4f}')

oa = accuracy_score(y_all, all_test_preds)
ob = balanced_accuracy_score(y_all, all_test_preds)
of1 = f1_score(y_all, all_test_preds, average='weighted')
oauc = roc_auc_score(y_all, all_test_probs[:,1])
print(f'\nPooled: Acc={oa:.4f} BalAcc={ob:.4f} F1={of1:.4f} AUC={oauc:.4f}')

print(f'\nBootstrap 95% CIs:')
for name, fn in [('Accuracy', accuracy_score),
                 ('BalancedAcc', balanced_accuracy_score),
                 ('F1', lambda y,p: f1_score(y,p,average='weighted'))]:
    mean, lo, hi = bootstrap_ci(y_all, all_test_preds, fn)
    print(f'  {name:<14s}: {mean:.4f} [{lo:.4f}, {hi:.4f}]')
auc_bs = []
rng = np.random.RandomState(SEED)
for _ in range(N_BOOTSTRAP):
    idx = rng.randint(0, len(y_all), len(y_all))
    try: auc_bs.append(roc_auc_score(y_all[idx], all_test_probs[idx,1]))
    except: pass
auc_lo, auc_hi = np.percentile(auc_bs, [2.5, 97.5])
print(f'  {"AUC":<14s}: {np.mean(auc_bs):.4f} [{auc_lo:.4f}, {auc_hi:.4f}]')

print('\n'+classification_report(y_all, all_test_preds, target_names=['NORMAL','CEREBRAL PALSY']))

# =============================================================================
# 12. PER-SUBJECT MAJORITY VOTE
# =============================================================================
sv, st = {}, {}
for i in range(len(y_all)):
    s = subjects[i]; sv.setdefault(s,[]).append(all_test_preds[i]); st[s] = y_all[i]
sids = sorted(sv.keys())
syt = np.array([st[s] for s in sids])
syp = np.array([int(np.round(np.mean(sv[s]))) for s in sids])
sa = accuracy_score(syt, syp); sb = balanced_accuracy_score(syt, syp)
print(f'{"="*70}\nSUBJECT VOTE: Acc={sa:.4f} BalAcc={sb:.4f} '
      f'({(syt==syp).sum()}/{len(sids)} correct)\n{"="*70}')

subj_side_map = {r['subject_id']: r['side'] for r in records}
print(f'\n{"Subj":>6s} | {"True":>5s} | {"Pred":>5s} | {"Arm":>5s} | {"Votes":>30s} |')
print('─'*70)
for s in sids:
    tl = 'TD' if st[s]==0 else 'DCP'
    pl = 'TD' if int(np.round(np.mean(sv[s])))==0 else 'DCP'
    mk = '✓' if tl==pl else '✗'
    print(f'{s:>6s} | {tl:>5s} | {pl:>5s} | {subj_side_map.get(s,"?"):>5s} | '
          f'{str(sv[s]):>30s} | {mk}')

# =============================================================================
# 13. CONFUSION MATRICES + ROC
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_all, all_test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL','CEREBRAL PALSY'],
            yticklabels=['NORMAL','CEREBRAL PALSY'], ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Trial-Level\nAcc={oa:.3f} BalAcc={ob:.3f}')
cms = confusion_matrix(syt, syp)
sns.heatmap(cms, annot=True, fmt='d', cmap='Greens',
            xticklabels=['NORMAL','CEREBRAL PALSY'],
            yticklabels=['NORMAL','CEREBRAL PALSY'], ax=axes[1])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title(f'Subject-Level\nAcc={sa:.3f} BalAcc={sb:.3f}')
plt.tight_layout(); plt.savefig('confusion_matrices.png', dpi=150)
plt.show(); plt.close()

fpr, tpr, _ = roc_curve(y_all, all_test_probs[:,1])
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(fpr, tpr, 'b-', lw=2, label=f'DONN AUC={oauc:.3f} [{auc_lo:.3f},{auc_hi:.3f}]')
ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC — DONN Upper-Limb Reaching Classification'); ax.legend()
plt.tight_layout(); plt.savefig('roc_curve.png', dpi=150)
plt.show(); plt.close()

# =============================================================================
# 14. ★ STATISTICAL TESTING AT SUBJECT LEVEL ★
# =============================================================================
print(f'\n{"="*70}')
print('UPPER-LIMB CLINICAL ANALYSIS — SUBJECT LEVEL (Mann-Whitney U)')
print(f'{"="*70}')

subj_stat = stat_df.groupby(['subject','class','side'])[stat_keys].mean().reset_index()
td_s = subj_stat[subj_stat['class']=='TD']
dcp_s = subj_stat[subj_stat['class']=='DCP']

key_feats = [
    ('shoulder_act_angle_range',  'Shoulder ROM (°)'),
    ('elbow_act_angle_range',     'Elbow ROM (°)'),
    ('wrist_act_angle_range',     'Wrist ROM (°)'),
    ('shoulder_act_omega_peak',   'Shoulder Peak ω (°/s)'),
    ('elbow_act_omega_peak',      'Elbow Peak ω (°/s)'),
    ('wrist_act_omega_peak',      'Wrist Peak ω (°/s)'),
    ('shoulder_act_alpha_rms',    'Shoulder Ang Accel RMS'),
    ('elbow_act_alpha_rms',       'Elbow Ang Accel RMS'),
    ('wrist_act_alpha_rms',       'Wrist Ang Accel RMS'),
    ('shoulder_act_jerk',         'Shoulder Jerk'),
    ('elbow_act_jerk',            'Elbow Jerk'),
    ('wrist_act_jerk',            'Wrist Jerk'),
    ('hand_speed_mean',           'Hand Speed (Mean)'),
    ('hand_speed_peak',           'Hand Speed (Peak)'),
    ('hand_speed_cv',             'Hand Speed CV'),
    ('hand_accel_rms',            'Hand Accel RMS'),
    ('hand_jerk_rms',             'Hand Jerk RMS'),
    ('index_of_curvature',        'Index of Curvature'),
    ('endpoint_accuracy',         'Endpoint Accuracy'),
    ('mean_overshoot',            'Mean Overshoot'),
    ('convex_hull_area',          'Convex Hull Area'),
    ('total_path_norm',           'Total Path Length'),
    ('execution_time',            'Execution Time (s)'),
    ('trunk_total_disp',          'Trunk Total Disp.'),
    ('trunk_max_drift',           'Trunk Max Drift'),
    ('head_x_range',              'Head X Range'),
    ('shoulder_act_variability',  'Shoulder Variability'),
    ('elbow_act_variability',     'Elbow Variability'),
    ('wrist_act_variability',     'Wrist Variability'),
    ('arm_movement_ratio',        'Passive/Active Ratio'),
    ('shoulder_angle_symmetry',   'Shoulder Symmetry'),
    ('elbow_angle_symmetry',      'Elbow Symmetry'),
]

print(f'\n{"Feature":<30s} | {"TD (n={len(td_s)})":>18s} | '
      f'{"DCP (n={len(dcp_s)})":>18s} | {"p":>10s} | Sig')
print('─'*100)
sig_feats = []
for fk, fn in key_feats:
    tv, dv = td_s[fk].values, dcp_s[fk].values
    try: _, pv = mannwhitneyu(tv, dv, alternative='two-sided')
    except: pv = 1.0
    sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
    print(f'{fn:<30s} | {tv.mean():8.3f}±{tv.std():6.3f} | '
          f'{dv.mean():8.3f}±{dv.std():6.3f} | {pv:10.6f} | {sig}')
    if pv < 0.05: sig_feats.append((fk, fn, pv))
print(f'\nSignificant: {len(sig_feats)}/{len(key_feats)}')

# =============================================================================
# 15. ★ UPPER-LIMB CLINICAL VISUALIZATION — 6 PANELS ★
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('UPPER-LIMB REACHING: NORMAL vs CEREBRAL PALSY\n'
             '(Subject-level, Vanmechelen 2024 reaching-sideways)',
             fontsize=15, fontweight='bold', y=1.02)
C = {'TD':'#2196F3', 'DCP':'#E53935'}

# 15a. Joint ROM comparison
ax = axes[0,0]
rom_f = ['shoulder_rom','elbow_rom','wrist_rom']
rom_l = ['Shoulder','Elbow','Wrist']
x = np.arange(len(rom_l)); w = 0.35
ax.bar(x-w/2, [td_s[f].mean() for f in rom_f], w,
       yerr=[td_s[f].std() for f in rom_f],
       label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
ax.bar(x+w/2, [dcp_s[f].mean() for f in rom_f], w,
       yerr=[dcp_s[f].std() for f in rom_f],
       label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(rom_l)
ax.set_ylabel('Range of Motion (°)'); ax.set_title('Joint ROM'); ax.legend()

# 15b. Movement Smoothness (Jerk)
ax = axes[0,1]
jf = ['shoulder_act_jerk','elbow_act_jerk','wrist_act_jerk']
jl = ['Shoulder','Elbow','Wrist']
x = np.arange(len(jl))
ax.bar(x-w/2, [td_s[f].mean() for f in jf], w,
       yerr=[td_s[f].std() for f in jf],
       label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
ax.bar(x+w/2, [dcp_s[f].mean() for f in jf], w,
       yerr=[dcp_s[f].std() for f in jf],
       label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(jl)
ax.set_ylabel('Normalized Jerk (lower=smoother)')
ax.set_title('Movement Smoothness (Jerk)'); ax.legend()

# 15c. Hand Speed Profile
ax = axes[0,2]
bp = ax.boxplot([td_s['hand_speed_peak'].values, dcp_s['hand_speed_peak'].values],
                labels=['NORMAL','CEREBRAL PALSY'], patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(C['TD']); bp['boxes'][1].set_facecolor(C['DCP'])
for b in bp['boxes']: b.set_alpha(0.7)
ax.set_ylabel('Peak Hand Speed (norm.)'); ax.set_title('Peak Hand Speed')

# 15d. Index of Curvature + Convex Hull
ax = axes[1,0]
for cls, col, mk in [('TD',C['TD'],'o'),('DCP',C['DCP'],'s')]:
    sub = subj_stat[subj_stat['class']==cls]
    ax.scatter(sub['index_of_curvature'], sub['convex_hull_area'],
               c=col, marker=mk, alpha=0.7, s=60, label=cls, edgecolors='white')
ax.set_xlabel('Index of Curvature (higher=less direct)')
ax.set_ylabel('Convex Hull Area (movement spread)')
ax.set_title('Path Efficiency vs Movement Spread'); ax.legend()

# 15e. Trunk Compensation
ax = axes[1,1]
bp = ax.boxplot([td_s['trunk_total_disp'].values, dcp_s['trunk_total_disp'].values],
                labels=['NORMAL','CEREBRAL PALSY'], patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(C['TD']); bp['boxes'][1].set_facecolor(C['DCP'])
for b in bp['boxes']: b.set_alpha(0.7)
ax.set_ylabel('Trunk Total Displacement (norm.)')
ax.set_title('Trunk Compensation')

# 15f. Endpoint Accuracy + Overshoot
ax = axes[1,2]
x = np.arange(2); w = 0.35
for cls, col, off in [('TD',C['TD'],-w/2),('DCP',C['DCP'],w/2)]:
    sub = subj_stat[subj_stat['class']==cls]
    means = [sub['endpoint_accuracy'].mean(), sub['mean_overshoot'].mean()]
    sds   = [sub['endpoint_accuracy'].std(), sub['mean_overshoot'].std()]
    ax.bar(x+off, means, w, yerr=sds, label=cls, color=col, alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(['Endpoint\nAccuracy','Mean\nOvershoot'])
ax.set_ylabel('Distance (norm.)'); ax.set_title('Endpoint Control'); ax.legend()

plt.tight_layout()
plt.savefig('upper_limb_clinical.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# =============================================================================
# 16. ★ ANGULAR ACCELERATION + VARIABILITY + SYMMETRY ★
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('DYSKINESIA MARKERS: Angular Dynamics & Symmetry', fontsize=14, fontweight='bold')

# 16a. Angular acceleration by joint
ax = axes[0]
af = ['shoulder_act_alpha_rms','elbow_act_alpha_rms','wrist_act_alpha_rms']
al = ['Shoulder','Elbow','Wrist']
x = np.arange(len(al)); w = 0.35
ax.bar(x-w/2, [td_s[f].mean() for f in af], w,
       yerr=[td_s[f].std() for f in af],
       label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
ax.bar(x+w/2, [dcp_s[f].mean() for f in af], w,
       yerr=[dcp_s[f].std() for f in af],
       label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(al)
ax.set_ylabel('RMS Angular Acceleration (°/s²)')
ax.set_title('Angular Acceleration (Dyskinesia Marker)'); ax.legend()

# 16b. Movement variability
ax = axes[1]
vf = ['shoulder_act_variability','elbow_act_variability','wrist_act_variability']
vl = ['Shoulder','Elbow','Wrist']
x = np.arange(len(vl))
ax.bar(x-w/2, [td_s[f].mean() for f in vf], w,
       yerr=[td_s[f].std() for f in vf],
       label='NORMAL', color=C['TD'], alpha=0.8, capsize=4)
ax.bar(x+w/2, [dcp_s[f].mean() for f in vf], w,
       yerr=[dcp_s[f].std() for f in vf],
       label='CEREBRAL PALSY', color=C['DCP'], alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(vl)
ax.set_ylabel('Angular Trajectory SD (°)')
ax.set_title('Movement Variability'); ax.legend()

# 16c. Symmetry (passive/active movement ratio)
ax = axes[2]
for cls, col in C.items():
    sub = subj_stat[subj_stat['class']==cls]
    ax.hist(sub['arm_movement_ratio'], bins=12, alpha=0.6,
            color=col, label=cls, edgecolor='white')
ax.set_xlabel('Passive/Active Arm Movement Ratio')
ax.set_ylabel('Count')
ax.set_title('Inter-Limb Symmetry\n(higher = more mirror/overflow movement)')
ax.legend()

plt.tight_layout()
plt.savefig('dyskinesia_markers.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# =============================================================================
# 17. PER-SUBJECT STACKED BAR
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
psi = subj_stat[['subject','class','convex_hull_area','index_of_curvature']].copy()
psi = psi.sort_values(['class','convex_hull_area'])
x = np.arange(len(psi))
bcol = [C['TD'] if c=='TD' else C['DCP'] for c in psi['class']]
ax.bar(x, psi['convex_hull_area'], color=bcol, alpha=0.8, edgecolor='white',
       label='Convex Hull Area')
ax.set_xticks(x); ax.set_xticklabels(psi['subject'], rotation=45, fontsize=8)
ax.set_xlabel('Subject'); ax.set_ylabel('Convex Hull Area (norm.)')
ax.set_title('Per-Subject Hand Path Spread\nBlue=NORMAL  Red=CEREBRAL PALSY')
ntd = (psi['class']=='TD').sum()
ax.axvline(ntd-0.5, color='black', ls='--', lw=2, alpha=0.5)
plt.tight_layout(); plt.savefig('per_subject_hand_path.png', dpi=150)
plt.show(); plt.close()

# =============================================================================
# 18. PER-FOLD + OSCILLATOR ANALYSIS
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(N_FOLDS); w = 0.2
ax.bar(x-1.5*w, rdf['accuracy'], w, label='Acc', color='#2196F3')
ax.bar(x-0.5*w, rdf['balanced_accuracy'], w, label='BalAcc', color='#9C27B0')
ax.bar(x+0.5*w, rdf['f1'], w, label='F1', color='#4CAF50')
ax.bar(x+1.5*w, rdf['auc'], w, label='AUC', color='#FF9800')
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(N_FOLDS)])
ax.set_ylim([0,1.15]); ax.set_ylabel('Score'); ax.set_title('Per-Fold'); ax.legend(ncol=4)
plt.tight_layout(); plt.savefig('per_fold.png', dpi=150)
plt.show(); plt.close()

with torch.no_grad():
    lf = (torch.sigmoid(final_model.hopf.omegas.cpu().squeeze())*(MAX_OMEGA-MIN_OMEGA)+MIN_OMEGA).numpy()
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(lf, bins=20, color='teal', edgecolor='white', alpha=0.85)
ax.set_xlabel('Frequency (Hz)'); ax.set_title('Learned Hopf Oscillator Frequencies')
ax.axvline(lf.mean(), color='red', ls='--', label=f'Mean={lf.mean():.2f}')
ax.legend(); plt.tight_layout(); plt.savefig('hopf_freq.png', dpi=150)
plt.show(); plt.close()

# =============================================================================
# 19. SAVE FINAL MODEL
# =============================================================================
sp = 'donn_upper_limb_final.pth'
torch.save({
    'model_state_dict': final_model.state_dict(),
    'config': dict(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM, osc_units=OSC_UNITS,
                   dt=DT, min_omega=MIN_OMEGA, max_omega=MAX_OMEGA,
                   input_scaler=INPUT_SCALER, dropout=DROPOUT, seq_len=SEQ_LEN,
                   n_features=N_FEATURES, fps=FPS),
    'norm_stats': {'mu': mu_final, 'std': sd_final},
    'cv_results': fold_results,
    'stat_feature_names': stat_keys,
}, sp)
print(f'\nSaved: {sp} ({os.path.getsize(sp)/1024:.1f} KB)')

# =============================================================================
# 20. INFERENCE
# =============================================================================
def predict_single(model, filepath, mu, std, seq_len, device):
    meta = parse_filename(filepath)
    df = pd.read_excel(filepath)
    drop = [c for c in df.columns
            if re.match(r'^unnamed', str(c), re.I) or str(c).strip().lower()=='index']
    df = df.drop(columns=drop)
    ts, stats, diag = extract_upper_limb_features(df, meta)
    ts_r = resample_temporal(ts, seq_len)
    x = torch.tensor((ts_r-mu)/std, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1).cpu().numpy()[0]
    return {0:'NORMAL',1:'CEREBRAL PALSY'}[logits.argmax(1).item()], probs, stats, diag

demo = all_files[0]
pl, pp, ps, pd_ = predict_single(final_model, demo, mu_final, sd_final, SEQ_LEN, device)
print(f'\nDemo: {os.path.basename(demo)} → {pl}')
print(f'  P(TD)={pp[0]:.4f} P(DCP)={pp[1]:.4f}')
print(f'  Active arm: {pd_["active_side"]}')
print(f'  Shoulder ROM={ps["shoulder_rom"]:.1f}°  Elbow ROM={ps["elbow_rom"]:.1f}°')
print(f'  Index of Curvature={ps["index_of_curvature"]:.3f}')
print(f'  Convex Hull={ps["convex_hull_area"]:.3f}')
print(f'  Endpoint Accuracy={ps["endpoint_accuracy"]:.3f}')
print(f'  Shoulder Jerk={ps["shoulder_act_jerk"]:.1f}  Elbow Jerk={ps["elbow_act_jerk"]:.1f}')
print(f'  Trunk Compensation={ps["trunk_total_disp"]:.3f}')

print('\n✓ PIPELINE COMPLETE.')
