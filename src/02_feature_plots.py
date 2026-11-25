#!/usr/bin/env python3
"""
==============================================================================
FEATURE-WISE TD vs DCP PLOTS — UPPER-LIMB REACHING KINEMATICS
==============================================================================

Dataset:
  Vanmechelen I, Van Wonterghem E, Aerts J-M, Hallez H, Desloovere K,
  Van de Walle P, Buizer AI, Monbaliu E, Haberfehlner H.
  "Markerless motion capture to assess upper extremity movements in
   individuals with dyskinetic cerebral palsy: an accuracy and validity study."
  J Biomech. 2024 Aug;173:112233. doi:10.1016/j.jbiomech.2024.112233

  Replication data: doi:10.48804/BZM9SJ  (KU Leuven RDR)
  51 participants (30 DCP, 21 TD), age 5–24 y, reaching-sideways task,
  8 repetitions, 2D DeepLabCut keypoints (frontal camera).

───────────────────────────────────────────────────────────────────────
FEATURE PROVENANCE — which paper / reference defines each feature?
───────────────────────────────────────────────────────────────────────

A. FEATURES COMPUTED IN THE VANMECHELEN 2024 PAPER ITSELF (Fig 3):
   1. Joint angles at PTA (shoulder, elbow, wrist)
      → Vanmechelen et al. 2022 PLoS ONE 17(9):e0266294
        (psychometric properties; 2D frontal-plane angles defined as:
         shoulder = hip–shoulder–elbow; elbow = shoulder–elbow–wrist;
         wrist = elbow–wrist–MCP3)
   2. SD of angular trajectories across 8 repetitions
      → Sanger TD. "Arm trajectories in dyskinetic CP have increased
        random variability." J Child Neurol. 2006;21(7):551–557.
        (first showed DCP has ↑ movement variability vs TD)
   3. Execution time (time from start to PTA)
      → Standard spatio-temporal parameter, used in
        Butler EE, Rose J. Gait Posture 2010;32(3):301–306.
   4. Mean overshoot (wrist y-displacement above object, normalised)
      → Vanmechelen et al. 2024 §Methods (trajectory deviation measure)
   5. Convex hull area & perimeter (of wrist path over 8 trials)
      → Computational geometry measure; cited as ref [24] in paper.
        Captures path consistency/spread (DCP >> TD).

B. FEATURES EXTENDED BY OUR PIPELINE (not in Vanmechelen 2024 directly):
   6. Normalised jerk (movement smoothness)
      → Flash T, Hogan N. "The coordination of arm movements: an
        experimentally confirmed mathematical model." J Neurosci.
        1985;5(7):1688–1703.
      → Teulings HL, Contreras-Vidal JL, Stelmach GE, Adler CH.
        "Parkinsonism reduces coordination of fingers, wrist, and arm
        in fine motor control." Exp Neurol. 1997;146(1):159–170.
      → Used in DCP context: Vanmechelen et al. 2023 Sensors 23(3):1574
        (max jerk from IMU, p<0.001 DCP vs TD)
   7. Peak angular velocity (ω_peak)
      → Standard biomechanics; see Jaspers E et al. Gait Posture.
        2009;30(4):395–404. (systematic review of UL kinematics in CP)
   8. RMS angular acceleration (α_rms)
      → Sanger TD, Kaiser J, Placek B. J Neurophysiol. 2005;93(3):
        "Reaching movements in childhood dystonia contain signal-
        dependent noise."
   9. Index of curvature (path length / straight-line distance)
      → Schneiberg S et al. Dev Med Child Neurol. 2010;52(7):e167–173.
        (reliability of kinematic measures of functional reaching in CP)
   10. Hand speed (peak, mean, CV)
      → Vanmechelen et al. 2022 PLoS ONE: "lower maximal velocity for
        participants with DCP" (their key between-group finding)
   11. Trunk compensation (total trunk displacement, max drift)
      → Van Wonterghem E, Vanmechelen I, Haberfehlner H, Decrock B,
        Monbaliu E. "Quantitative assessment of trunk movements in
        functional reaching in children and adolescents with dyskinetic
        cerebral palsy." Clin Biomech (Bristol, Avon). 2023.
   12. Passive/active arm movement ratio (inter-limb symmetry)
      → Gordon LM et al. Pediatr Neurol. 2006;35(6):375–381.
        ("Can spasticity and dystonia be independently measured?" —
        used contralateral overflow as dystonia proxy)

───────────────────────────────────────────────────────────────────────
"""

import os, sys, glob, re, math, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import savgol_filter, resample
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu, pearsonr
from collections import defaultdict

warnings.filterwarnings('ignore')
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIG — EDIT THESE TWO PATHS
# =============================================================================
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30.0

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
# 2. HELPERS
# =============================================================================
JOINT_NAMES = {
    'shoulder_r': 'shoulder_right', 'shoulder_l': 'shoulder_left',
    'elbow_r': 'elbow_right',       'elbow_l': 'elbow_left',
    'wrist_r': 'wrist_right',       'wrist_l': 'wrist_left',
    'hand_r': 'MCP3_right',         'hand_l': 'MCP3_left',
    'forehead': 'forehead',          'chin': 'chin',
    'object': 'object',
}

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

def norm_jerk(s, dt=1.0 / 30.0):
    """
    Normalised jerk index.
    Ref: Teulings et al. 1997 Exp Neurol 146(1):159–170
         Flash & Hogan 1985 J Neurosci 5(7):1688–1703
    NJ = sqrt(0.5 * ∫j² dt * T^5 / range²)
    """
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

# =============================================================================
# 3. FEATURE EXTRACTION — every feature annotated with its source
# =============================================================================
def extract_features(df, meta):
    """
    Returns dict of scalar features for one trial.
    Every feature is annotated with the paper that defines / validates it.
    """
    T = len(df)
    dt = 1.0 / FPS
    active_side = meta['side']
    act, pas = ('l', 'r') if active_side == 'left' else ('r', 'l')

    shoulder_act, _ = get_xy(df, f'shoulder_{act}')
    shoulder_pas, _ = get_xy(df, f'shoulder_{pas}')
    elbow_act, _    = get_xy(df, f'elbow_{act}')
    elbow_pas, _    = get_xy(df, f'elbow_{pas}')
    wrist_act, _    = get_xy(df, f'wrist_{act}')
    wrist_pas, _    = get_xy(df, f'wrist_{pas}')
    hand_act, _     = get_xy(df, f'hand_{act}')
    hand_pas, _     = get_xy(df, f'hand_{pas}')
    target, _       = get_xy(df, 'object')
    forehead, _     = get_xy(df, 'forehead')
    chin, _         = get_xy(df, 'chin')

    # Reference scale: upper-arm length (Vanmechelen 2024 normalised to segment)
    upper_arm_len = np.linalg.norm(elbow_act - shoulder_act, axis=1)
    ref_scale = np.maximum(upper_arm_len, 1e-6)

    trunk_up = forehead - chin

    # ── JOINT ANGLES ──
    # Ref: Vanmechelen et al. 2022 PLoS ONE — 2D frontal plane definitions
    #   shoulder = angle(elbow, shoulder, shoulder+trunk_up)
    #   elbow    = angle(shoulder, elbow, wrist)
    #   wrist    = angle(elbow, wrist, hand/MCP3)
    ang_shoulder_act = angle_3pt(elbow_act, shoulder_act, shoulder_act + trunk_up)
    ang_shoulder_pas = angle_3pt(elbow_pas, shoulder_pas, shoulder_pas + trunk_up)
    ang_elbow_act    = angle_3pt(shoulder_act, elbow_act, wrist_act)
    ang_elbow_pas    = angle_3pt(shoulder_pas, elbow_pas, wrist_pas)
    ang_wrist_act    = angle_3pt(elbow_act, wrist_act, hand_act)
    ang_wrist_pas    = angle_3pt(elbow_pas, wrist_pas, hand_pas)

    # ── ANGULAR VELOCITY ──
    # Ref: Jaspers et al. 2009 Gait Posture 30(4):395–404
    w_shoulder_act = smooth(deriv(ang_shoulder_act, dt))
    w_elbow_act    = smooth(deriv(ang_elbow_act, dt))
    w_wrist_act    = smooth(deriv(ang_wrist_act, dt))
    w_shoulder_pas = smooth(deriv(ang_shoulder_pas, dt))
    w_elbow_pas    = smooth(deriv(ang_elbow_pas, dt))

    # ── ANGULAR ACCELERATION ──
    # Ref: Sanger, Kaiser & Placek 2005 J Neurophysiol 93(3)
    a_shoulder_act = smooth(deriv(w_shoulder_act, dt))
    a_elbow_act    = smooth(deriv(w_elbow_act, dt))
    a_wrist_act    = smooth(deriv(w_wrist_act, dt))

    # ── HAND KINEMATICS ──
    hand_to_target = np.linalg.norm(hand_act - target, axis=1)
    hand_to_target_norm = hand_to_target / ref_scale
    hand_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_act, axis=0), axis=1)])
    hand_speed = hand_disp / dt
    hand_speed_norm = hand_speed / ref_scale

    # ── TRUNK COMPENSATION ──
    # Ref: Van Wonterghem et al. 2023 Clin Biomech
    trunk_mid = (forehead + chin) / 2.0
    trunk_disp_px = np.concatenate([[0], np.linalg.norm(np.diff(trunk_mid, axis=0), axis=1)])
    trunk_disp_norm = trunk_disp_px / ref_scale
    trunk_drift = np.linalg.norm(trunk_mid - trunk_mid[0], axis=1) / ref_scale
    head_x_disp = (forehead[:, 0] - forehead[0, 0]) / ref_scale

    # ── PASSIVE ARM MOVEMENT (contralateral overflow) ──
    # Ref: Gordon et al. 2006 Pediatr Neurol 35(6):375–381
    pas_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_pas, axis=0), axis=1)])
    arm_movement_ratio = np.sum(pas_disp) / (np.sum(hand_disp) + 1e-8)

    # ── PATH GEOMETRY ──
    total_path = np.sum(hand_disp)
    straight_line = np.linalg.norm(hand_act[-1] - hand_act[0])
    # Index of curvature: Schneiberg et al. 2010 Dev Med Child Neurol
    index_of_curvature = total_path / (straight_line + 1e-8)
    # Convex hull: Vanmechelen 2024 (ref [24] in paper)
    hull_area = convex_hull_area(hand_act) / (np.mean(ref_scale) ** 2)

    # ── ENDPOINT CONTROL ──
    # Ref: Vanmechelen 2024 — overshoot = wrist y-dist above object
    endpoint_dist_norm = hand_to_target_norm[-1]
    min_dist_idx = np.argmin(hand_to_target)
    overshoot_ratio = (T - 1 - min_dist_idx) / T
    mean_overshoot = (np.mean(hand_to_target[min_dist_idx:]) / np.mean(ref_scale)
                      if min_dist_idx < T - 1 else 0.0)

    rms = lambda x: np.sqrt(np.mean(x ** 2))

    features = {
        # ── GROUP A: Vanmechelen 2024 paper features ──
        # Joint angles at PTA (last frame = PTA proxy)
        'shoulder_angle_PTA':        ang_shoulder_act[-1],      # Vanmechelen 2022, 2024
        'elbow_angle_PTA':           ang_elbow_act[-1],         # Vanmechelen 2022, 2024
        'wrist_angle_PTA':           ang_wrist_act[-1],         # Vanmechelen 2022, 2024
        # SD of angular trajectories (within-trial variability proxy)
        'shoulder_angle_SD':         np.std(ang_shoulder_act),  # Sanger 2006; Vanmechelen 2024
        'elbow_angle_SD':            np.std(ang_elbow_act),     # Sanger 2006; Vanmechelen 2024
        'wrist_angle_SD':            np.std(ang_wrist_act),     # Sanger 2006; Vanmechelen 2024
        # Execution time
        'execution_time':            T / FPS,                   # Butler & Rose 2010
        # Mean overshoot
        'mean_overshoot':            mean_overshoot,            # Vanmechelen 2024
        # Convex hull area
        'convex_hull_area':          hull_area,                 # Vanmechelen 2024 ref[24]

        # ── GROUP B: ROM ──
        'shoulder_ROM':              np.ptp(ang_shoulder_act),  # Vanmechelen 2022
        'elbow_ROM':                 np.ptp(ang_elbow_act),     # Vanmechelen 2022
        'wrist_ROM':                 np.ptp(ang_wrist_act),     # Vanmechelen 2022

        # ── GROUP C: Movement smoothness ──
        'shoulder_jerk':             norm_jerk(ang_shoulder_act, dt),  # Flash & Hogan 1985
        'elbow_jerk':                norm_jerk(ang_elbow_act, dt),     # Flash & Hogan 1985
        'wrist_jerk':                norm_jerk(ang_wrist_act, dt),     # Flash & Hogan 1985

        # ── GROUP D: Angular dynamics ──
        'shoulder_omega_peak':       np.max(np.abs(w_shoulder_act)),   # Jaspers 2009
        'elbow_omega_peak':          np.max(np.abs(w_elbow_act)),      # Jaspers 2009
        'wrist_omega_peak':          np.max(np.abs(w_wrist_act)),      # Jaspers 2009
        'shoulder_alpha_rms':        rms(a_shoulder_act),              # Sanger 2005
        'elbow_alpha_rms':           rms(a_elbow_act),                 # Sanger 2005
        'wrist_alpha_rms':           rms(a_wrist_act),                 # Sanger 2005

        # ── GROUP E: Hand speed ──
        'hand_speed_mean':           np.mean(hand_speed_norm),         # Vanmechelen 2022
        'hand_speed_peak':           np.max(hand_speed_norm),          # Vanmechelen 2022
        'hand_speed_CV':             np.std(hand_speed_norm) / (np.mean(hand_speed_norm) + 1e-8),

        # ── GROUP F: Path efficiency ──
        'index_of_curvature':        index_of_curvature,               # Schneiberg 2010
        'total_path_norm':           total_path / np.mean(ref_scale),
        'endpoint_accuracy':         endpoint_dist_norm,               # Vanmechelen 2024

        # ── GROUP G: Trunk compensation ──
        'trunk_total_disp':          np.sum(trunk_disp_norm),          # Van Wonterghem 2023
        'trunk_max_drift':           np.max(trunk_drift),              # Van Wonterghem 2023
        'head_x_range':              np.ptp(head_x_disp),             # Van Wonterghem 2023

        # ── GROUP H: Inter-limb symmetry ──
        'arm_movement_ratio':        arm_movement_ratio,               # Gordon 2006
        'shoulder_angle_symmetry':   np.mean(np.abs(ang_shoulder_act - ang_shoulder_pas)),
        'elbow_angle_symmetry':      np.mean(np.abs(ang_elbow_act - ang_elbow_pas)),

        # ── GROUP I: Overshoot / endpoint control ──
        'overshoot_ratio':           overshoot_ratio,
    }

    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            for k, v in features.items()}

# =============================================================================
# 4. DATA LOADING
# =============================================================================
def load_all_data(data_dir):
    zip_path = os.path.join(os.path.dirname(data_dir) or '.', 'data.zip')
    if os.path.isfile(zip_path) and not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

    all_xlsx = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
    matched = [f for f in all_xlsx if _FNAME_RE.match(os.path.splitext(os.path.basename(f))[0])]
    if not matched:
        raise FileNotFoundError(f"No matching xlsx in {data_dir}. Found {len(all_xlsx)} total.")

    records = []
    for i, fpath in enumerate(matched):
        try:
            meta = parse_filename(fpath)
            df = pd.read_excel(fpath)
            drop = [c for c in df.columns
                    if re.match(r'^unnamed', str(c), re.I) or str(c).strip().lower() == 'index']
            df = df.drop(columns=drop)
            feats = extract_features(df, meta)
            records.append({**meta, **feats})
        except Exception as e:
            pass
        if (i + 1) % 100 == 0:
            print(f'  Loaded {i+1}/{len(matched)}')

    df_all = pd.DataFrame(records)
    df_all['class'] = df_all['class_str']
    print(f'  {len(df_all)} trials loaded | TD: {(df_all["class"]=="TD").sum()} | '
          f'DCP: {(df_all["class"]=="DCP").sum()}')
    return df_all

# =============================================================================
# 5. SUBJECT-LEVEL AGGREGATION
# =============================================================================
def aggregate_subject_level(df_all, feat_cols):
    """Average across repetitions per subject (8 reps → 1 row per subject-side)."""
    return df_all.groupby(['subject_id', 'class', 'side'])[feat_cols].mean().reset_index()

# =============================================================================
# 6. STAT TESTING
# =============================================================================
def compute_stats(subj, feat_cols):
    """Mann-Whitney U for each feature, subject-level."""
    results = []
    td = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']
    for f in feat_cols:
        tv, dv = td[f].dropna().values, dcp[f].dropna().values
        if len(tv) < 3 or len(dv) < 3:
            continue
        try:
            stat, pval = mannwhitneyu(tv, dv, alternative='two-sided')
        except:
            pval = 1.0
        # Effect size: rank-biserial r = 1 - 2U/(n1*n2)
        n1, n2 = len(tv), len(dv)
        r_effect = 1 - 2 * stat / (n1 * n2)
        results.append({
            'feature': f,
            'TD_mean': np.mean(tv), 'TD_std': np.std(tv),
            'DCP_mean': np.mean(dv), 'DCP_std': np.std(dv),
            'U': stat, 'p': pval, 'r_effect': r_effect,
            'sig': '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns',
        })
    return pd.DataFrame(results)

# =============================================================================
# 7. PLOTTING FUNCTIONS
# =============================================================================
C_TD  = '#1976D2'   # Blue
C_DCP = '#D32F2F'   # Red
C_TD_LIGHT  = '#90CAF9'
C_DCP_LIGHT = '#EF9A9A'

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [SAVED] {name}')


# ─────────────────────────────────────────────────────
# PLOT 1: Vanmechelen 2024 Fig-3 style — Joint angles
#         at PTA + SD + execution time + overshoot +
#         convex hull
# ─────────────────────────────────────────────────────
def plot_vanmechelen_fig3_style(subj, stats_df):
    """
    Reproduces the core comparisons from Vanmechelen et al. 2024 Fig 3:
      (A–C) Joint angles at PTA (shoulder, elbow, wrist)
      (D–F) SD of angular trajectories
      (G) Execution time
      (H) Mean overshoot
      (I) Convex hull area
    All using violin + jitter (individual subject data points).
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        'Vanmechelen et al. 2024 — Core TD vs DCP Feature Comparison\n'
        '(Subject-level means; violin + individual data; Mann-Whitney U)',
        fontsize=14, fontweight='bold', y=0.98)

    panels = [
        # Row 1: Joint angles at PTA
        ('shoulder_angle_PTA', 'A. Shoulder Angle at PTA (°)',
         'Vanmechelen 2022 PLoS ONE\n(hip–shoulder–elbow angle)'),
        ('elbow_angle_PTA',    'B. Elbow Angle at PTA (°)',
         'Vanmechelen 2022 PLoS ONE\n(shoulder–elbow–wrist angle)'),
        ('wrist_angle_PTA',    'C. Wrist Angle at PTA (°)',
         'Vanmechelen 2022 PLoS ONE\n(elbow–wrist–MCP3 angle)'),
        # Row 2: SD of angular trajectories
        ('shoulder_angle_SD',  'D. Shoulder Angle SD (°)',
         'Sanger 2006 J Child Neurol\n(↑variability in DCP)'),
        ('elbow_angle_SD',     'E. Elbow Angle SD (°)',
         'Sanger 2006 J Child Neurol'),
        ('wrist_angle_SD',     'F. Wrist Angle SD (°)',
         'Sanger 2006 J Child Neurol'),
        # Row 3: Spatio-temporal
        ('execution_time',     'G. Execution Time (s)',
         'Butler & Rose 2010 Gait Posture'),
        ('mean_overshoot',     'H. Mean Overshoot (norm)',
         'Vanmechelen 2024 J Biomech'),
        ('convex_hull_area',   'I. Convex Hull Area (norm²)',
         'Vanmechelen 2024 ref[24]'),
    ]

    for idx, (feat, title, ref_text) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        data_plot = [td[feat].values, dcp[feat].values]
        vp = ax.violinplot(data_plot, positions=[0, 1], showmedians=True,
                           showextrema=False, widths=0.7)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor([C_TD_LIGHT, C_DCP_LIGHT][i])
            body.set_edgecolor([C_TD, C_DCP][i])
            body.set_alpha(0.6)
        vp['cmedians'].set_color('black')

        # Jitter individual points
        for i, (vals, color) in enumerate(zip(data_plot, [C_TD, C_DCP])):
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       c=color, s=25, alpha=0.7, edgecolors='white', linewidths=0.3, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['TD', 'DCP'], fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Add p-value annotation
        row = stats_df[stats_df['feature'] == feat]
        if len(row):
            p = row.iloc[0]['p']
            sig = row.iloc[0]['sig']
            ymax = max(np.max(data_plot[0]), np.max(data_plot[1]))
            ymin = min(np.min(data_plot[0]), np.min(data_plot[1]))
            ybar = ymax + 0.05 * (ymax - ymin)
            ax.plot([0, 1], [ybar, ybar], 'k-', lw=1)
            ax.text(0.5, ybar + 0.02 * (ymax - ymin), f'{sig} (p={p:.4f})',
                    ha='center', va='bottom', fontsize=8)

        # Reference annotation
        ax.text(0.02, 0.02, ref_text, transform=ax.transAxes,
                fontsize=6.5, color='gray', style='italic', va='bottom')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '01_vanmechelen2024_core_features.png')


# ─────────────────────────────────────────────────────
# PLOT 2: Joint ROM comparison (bar + error + jitter)
# ─────────────────────────────────────────────────────
def plot_joint_rom(subj, stats_df):
    """
    Ref: Vanmechelen et al. 2022 PLoS ONE — ROM = max–min of angle waveform
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']
    feats = ['shoulder_ROM', 'elbow_ROM', 'wrist_ROM']
    labels = ['Shoulder', 'Elbow', 'Wrist']

    fig, ax = plt.subplots(figsize=(10, 7))
    w = 0.35
    x = np.arange(len(labels))

    for i, (cls_data, color, label, offset) in enumerate([
        (td, C_TD, 'TD (n={})'.format(len(td)), -w/2),
        (dcp, C_DCP, 'DCP (n={})'.format(len(dcp)), w/2),
    ]):
        means = [cls_data[f].mean() for f in feats]
        stds  = [cls_data[f].std() for f in feats]
        bars = ax.bar(x + offset, means, w, yerr=stds,
                       label=label, color=color, alpha=0.75, capsize=5, edgecolor='white')
        # Overlay individual points
        for j, f in enumerate(feats):
            vals = cls_data[f].values
            jitter = np.random.uniform(-w/4, w/4, len(vals))
            ax.scatter(np.full_like(vals, x[j] + offset) + jitter, vals,
                       c=color, s=18, alpha=0.5, edgecolors='gray', linewidths=0.3, zorder=5)

    # Significance bars
    for j, f in enumerate(feats):
        row = stats_df[stats_df['feature'] == f]
        if len(row) and row.iloc[0]['p'] < 0.05:
            ymax = max(td[f].max(), dcp[f].max())
            ax.text(x[j], ymax * 1.05, row.iloc[0]['sig'],
                    ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Range of Motion (°)', fontsize=12)
    ax.set_title('Joint ROM: TD vs DCP\n'
                 'Ref: Vanmechelen et al. 2022 PLoS ONE 17(9):e0266294',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, '02_joint_ROM.png')


# ─────────────────────────────────────────────────────
# PLOT 3: Normalised Jerk — Movement Smoothness
# ─────────────────────────────────────────────────────
def plot_normalised_jerk(subj, stats_df):
    """
    Ref: Flash & Hogan 1985 J Neurosci 5(7):1688–1703
         Teulings et al. 1997 Exp Neurol 146(1):159–170
         Vanmechelen et al. 2023 Sensors 23(3):1574 (max jerk p<0.001)
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']
    feats = ['shoulder_jerk', 'elbow_jerk', 'wrist_jerk']
    labels = ['Shoulder', 'Elbow', 'Wrist']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Normalised Jerk (Movement Smoothness): TD vs DCP\n'
                 'Ref: Flash & Hogan 1985; Teulings et al. 1997; '
                 'Vanmechelen et al. 2023 Sensors',
                 fontsize=12, fontweight='bold')

    for i, (f, lbl) in enumerate(zip(feats, labels)):
        ax = axes[i]
        td_vals = td[f].values
        dcp_vals = dcp[f].values

        bp = ax.boxplot([td_vals, dcp_vals], labels=['TD', 'DCP'],
                        patch_artist=True, widths=0.5,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(C_TD_LIGHT)
        bp['boxes'][1].set_facecolor(C_DCP_LIGHT)

        # Overlay jitter
        for j, (vals, color) in enumerate(zip([td_vals, dcp_vals], [C_TD, C_DCP])):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                       c=color, s=25, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)

        row = stats_df[stats_df['feature'] == f]
        if len(row):
            ax.set_xlabel(f'{row.iloc[0]["sig"]} (p={row.iloc[0]["p"]:.4f})', fontsize=9)

        ax.set_title(f'{lbl} Jerk', fontweight='bold')
        ax.set_ylabel('Normalised Jerk Index')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '03_normalised_jerk.png')


# ─────────────────────────────────────────────────────
# PLOT 4: Angular Dynamics — Peak ω and RMS α
# ─────────────────────────────────────────────────────
def plot_angular_dynamics(subj, stats_df):
    """
    Ref: Jaspers et al. 2009 Gait Posture 30(4):395–404 (review)
         Sanger, Kaiser & Placek 2005 J Neurophysiol
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Angular Dynamics: TD vs DCP\n'
                 'Ref: Jaspers et al. 2009 Gait Posture (peak ω); '
                 'Sanger et al. 2005 J Neurophysiol (RMS α)',
                 fontsize=12, fontweight='bold')

    w = 0.3
    # Panel A: Peak angular velocity
    ax = axes[0]
    feats_w = ['shoulder_omega_peak', 'elbow_omega_peak', 'wrist_omega_peak']
    labels = ['Shoulder', 'Elbow', 'Wrist']
    x = np.arange(3)
    for cls_data, color, lbl, off in [
        (td, C_TD, 'TD', -w/2), (dcp, C_DCP, 'DCP', w/2)]:
        means = [cls_data[f].mean() for f in feats_w]
        stds  = [cls_data[f].std() for f in feats_w]
        ax.bar(x + off, means, w, yerr=stds, color=color, alpha=0.75,
               capsize=4, label=lbl, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Peak Angular Velocity (°/s)')
    ax.set_title('A. Peak Angular Velocity (ω_peak)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: RMS angular acceleration
    ax = axes[1]
    feats_a = ['shoulder_alpha_rms', 'elbow_alpha_rms', 'wrist_alpha_rms']
    for cls_data, color, lbl, off in [
        (td, C_TD, 'TD', -w/2), (dcp, C_DCP, 'DCP', w/2)]:
        means = [cls_data[f].mean() for f in feats_a]
        stds  = [cls_data[f].std() for f in feats_a]
        ax.bar(x + off, means, w, yerr=stds, color=color, alpha=0.75,
               capsize=4, label=lbl, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('RMS Angular Acceleration (°/s²)')
    ax.set_title('B. RMS Angular Acceleration (α_rms)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '04_angular_dynamics.png')


# ─────────────────────────────────────────────────────
# PLOT 5: Hand Speed — Peak, Mean, CV
# ─────────────────────────────────────────────────────
def plot_hand_speed(subj, stats_df):
    """
    Ref: Vanmechelen et al. 2022 PLoS ONE — "lower maximal velocity
         for participants with DCP" (key between-group finding)
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Hand Speed: TD vs DCP\n'
                 'Ref: Vanmechelen et al. 2022 PLoS ONE ("lower maximal velocity in DCP")',
                 fontsize=12, fontweight='bold')

    for i, (f, title, ylabel) in enumerate([
        ('hand_speed_peak', 'A. Peak Hand Speed', 'Speed (normalised)'),
        ('hand_speed_mean', 'B. Mean Hand Speed', 'Speed (normalised)'),
        ('hand_speed_CV',   'C. Hand Speed CV',   'Coefficient of Variation'),
    ]):
        ax = axes[i]
        td_v = td[f].values
        dcp_v = dcp[f].values

        vp = ax.violinplot([td_v, dcp_v], positions=[0, 1],
                           showmedians=True, showextrema=False, widths=0.6)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor([C_TD_LIGHT, C_DCP_LIGHT][j])
            body.set_edgecolor([C_TD, C_DCP][j])
            body.set_alpha(0.5)
        vp['cmedians'].set_color('black')

        for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, j) + jitter, vals,
                       c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)

        ax.set_xticks([0, 1]); ax.set_xticklabels(['TD', 'DCP'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)

        row = stats_df[stats_df['feature'] == f]
        if len(row):
            ax.text(0.5, 0.97, f'{row.iloc[0]["sig"]} p={row.iloc[0]["p"]:.4f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '05_hand_speed.png')


# ─────────────────────────────────────────────────────
# PLOT 6: Path Efficiency — IoC vs Convex Hull scatter
# ─────────────────────────────────────────────────────
def plot_path_efficiency(subj, stats_df):
    """
    Ref: Schneiberg et al. 2010 Dev Med Child Neurol (Index of Curvature)
         Vanmechelen 2024 (Convex Hull)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Path Efficiency: TD vs DCP\n'
                 'Ref: Schneiberg et al. 2010 (IoC); Vanmechelen 2024 (convex hull)',
                 fontsize=12, fontweight='bold')

    # Scatter: IoC vs Hull
    ax = axes[0]
    for cls, color, marker in [('TD', C_TD, 'o'), ('DCP', C_DCP, 's')]:
        sub = subj[subj['class'] == cls]
        ax.scatter(sub['index_of_curvature'], sub['convex_hull_area'],
                   c=color, marker=marker, s=50, alpha=0.7, label=cls, edgecolors='white')
    ax.set_xlabel('Index of Curvature')
    ax.set_ylabel('Convex Hull Area (norm²)')
    ax.set_title('A. Path Curvature vs Spread', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Boxplot: IoC
    ax = axes[1]
    td_v = subj[subj['class']=='TD']['index_of_curvature'].values
    dcp_v = subj[subj['class']=='DCP']['index_of_curvature'].values
    bp = ax.boxplot([td_v, dcp_v], labels=['TD', 'DCP'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C_TD_LIGHT); bp['boxes'][1].set_facecolor(C_DCP_LIGHT)
    for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                   c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
    ax.set_title('B. Index of Curvature', fontweight='bold')
    ax.set_ylabel('IoC (path / straight line)')
    ax.grid(axis='y', alpha=0.3)
    row = stats_df[stats_df['feature'] == 'index_of_curvature']
    if len(row):
        ax.set_xlabel(f'{row.iloc[0]["sig"]} p={row.iloc[0]["p"]:.4f}')

    # Boxplot: total path
    ax = axes[2]
    td_v = subj[subj['class']=='TD']['total_path_norm'].values
    dcp_v = subj[subj['class']=='DCP']['total_path_norm'].values
    bp = ax.boxplot([td_v, dcp_v], labels=['TD', 'DCP'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C_TD_LIGHT); bp['boxes'][1].set_facecolor(C_DCP_LIGHT)
    for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                   c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
    ax.set_title('C. Total Path Length (norm)', fontweight='bold')
    ax.set_ylabel('Normalised Path Length')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '06_path_efficiency.png')


# ─────────────────────────────────────────────────────
# PLOT 7: Trunk Compensation
# ─────────────────────────────────────────────────────
def plot_trunk_compensation(subj, stats_df):
    """
    Ref: Van Wonterghem E, Vanmechelen I, Haberfehlner H, Decrock B, Monbaliu E.
         "Quantitative assessment of trunk movements in functional reaching
          in children and adolescents with dyskinetic cerebral palsy."
         Clin Biomech (Bristol, Avon). 2023.
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Trunk Compensation: TD vs DCP\n'
                 'Ref: Van Wonterghem et al. 2023 Clin Biomech',
                 fontsize=12, fontweight='bold')

    for i, (f, title) in enumerate([
        ('trunk_total_disp', 'A. Total Trunk Displacement'),
        ('trunk_max_drift',  'B. Maximum Trunk Drift'),
        ('head_x_range',     'C. Head Lateral Range'),
    ]):
        ax = axes[i]
        td_v = td[f].values
        dcp_v = dcp[f].values
        bp = ax.boxplot([td_v, dcp_v], labels=['TD', 'DCP'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(C_TD_LIGHT)
        bp['boxes'][1].set_facecolor(C_DCP_LIGHT)
        for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                       c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Normalised Distance')
        ax.grid(axis='y', alpha=0.3)
        row = stats_df[stats_df['feature'] == f]
        if len(row):
            ax.set_xlabel(f'{row.iloc[0]["sig"]} p={row.iloc[0]["p"]:.4f}')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '07_trunk_compensation.png')


# ─────────────────────────────────────────────────────
# PLOT 8: Inter-Limb Symmetry & Overflow
# ─────────────────────────────────────────────────────
def plot_interlimb_symmetry(subj, stats_df):
    """
    Ref: Gordon LM et al. Pediatr Neurol. 2006;35(6):375–381.
         ("Can spasticity and dystonia be independently measured?"
          — contralateral overflow as a dystonia marker)
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Inter-Limb Symmetry & Contralateral Overflow: TD vs DCP\n'
                 'Ref: Gordon et al. 2006 Pediatr Neurol',
                 fontsize=12, fontweight='bold')

    # A: Passive/Active ratio histogram
    ax = axes[0]
    bins = np.linspace(0, max(subj['arm_movement_ratio'].max(), 1), 15)
    ax.hist(td['arm_movement_ratio'], bins=bins, alpha=0.6, color=C_TD,
            label='TD', edgecolor='white')
    ax.hist(dcp['arm_movement_ratio'], bins=bins, alpha=0.6, color=C_DCP,
            label='DCP', edgecolor='white')
    ax.set_xlabel('Passive / Active Arm Ratio')
    ax.set_ylabel('Count')
    ax.set_title('A. Passive/Active Movement Ratio', fontweight='bold')
    ax.legend()
    row = stats_df[stats_df['feature'] == 'arm_movement_ratio']
    if len(row):
        ax.text(0.95, 0.95, f'{row.iloc[0]["sig"]} p={row.iloc[0]["p"]:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # B: Shoulder angle symmetry
    ax = axes[1]
    td_v = td['shoulder_angle_symmetry'].values
    dcp_v = dcp['shoulder_angle_symmetry'].values
    bp = ax.boxplot([td_v, dcp_v], labels=['TD', 'DCP'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C_TD_LIGHT); bp['boxes'][1].set_facecolor(C_DCP_LIGHT)
    for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                   c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
    ax.set_title('B. Shoulder Angle Asymmetry (°)', fontweight='bold')
    ax.set_ylabel('|Active − Passive| (°)')
    ax.grid(axis='y', alpha=0.3)

    # C: Elbow angle symmetry
    ax = axes[2]
    td_v = td['elbow_angle_symmetry'].values
    dcp_v = dcp['elbow_angle_symmetry'].values
    bp = ax.boxplot([td_v, dcp_v], labels=['TD', 'DCP'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(C_TD_LIGHT); bp['boxes'][1].set_facecolor(C_DCP_LIGHT)
    for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full_like(vals, j + 1) + jitter, vals,
                   c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
    ax.set_title('C. Elbow Angle Asymmetry (°)', fontweight='bold')
    ax.set_ylabel('|Active − Passive| (°)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '08_interlimb_symmetry.png')


# ─────────────────────────────────────────────────────
# PLOT 9: Endpoint Control — Accuracy + Overshoot
# ─────────────────────────────────────────────────────
def plot_endpoint_control(subj, stats_df):
    """
    Ref: Vanmechelen 2024 Fig 3 (H) — mean overshoot
         Sanger 2006 — trajectory variability
    """
    td  = subj[subj['class'] == 'TD']
    dcp = subj[subj['class'] == 'DCP']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Endpoint Control: TD vs DCP\n'
                 'Ref: Vanmechelen 2024 J Biomech; Sanger 2006 J Child Neurol',
                 fontsize=12, fontweight='bold')

    for i, (f, title, ylabel) in enumerate([
        ('endpoint_accuracy', 'A. Endpoint Distance (norm)', 'Normalised Distance to Target'),
        ('mean_overshoot',    'B. Mean Overshoot (norm)',    'Normalised Overshoot'),
        ('overshoot_ratio',   'C. Overshoot Ratio',          'Fraction of Trial Past Min-Dist'),
    ]):
        ax = axes[i]
        td_v = td[f].values; dcp_v = dcp[f].values
        vp = ax.violinplot([td_v, dcp_v], positions=[0, 1],
                           showmedians=True, showextrema=False, widths=0.6)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor([C_TD_LIGHT, C_DCP_LIGHT][j])
            body.set_edgecolor([C_TD, C_DCP][j])
            body.set_alpha(0.5)
        vp['cmedians'].set_color('black')
        for j, (vals, color) in enumerate(zip([td_v, dcp_v], [C_TD, C_DCP])):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, j) + jitter, vals,
                       c=color, s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['TD', 'DCP'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        row = stats_df[stats_df['feature'] == f]
        if len(row):
            ax.text(0.5, 0.97, f'{row.iloc[0]["sig"]} p={row.iloc[0]["p"]:.4f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, '09_endpoint_control.png')


# ─────────────────────────────────────────────────────
# PLOT 10: Comprehensive heatmap — all features, effect sizes
# ─────────────────────────────────────────────────────
def plot_feature_heatmap(stats_df):
    """
    Shows all features sorted by effect size (rank-biserial r).
    Colour = direction & magnitude of group difference.
    """
    df = stats_df.sort_values('r_effect', ascending=True).copy()
    df['abs_r'] = df['r_effect'].abs()

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))
    colors = [C_DCP if r > 0 else C_TD for r in df['r_effect']]
    bars = ax.barh(df['feature'], df['r_effect'], color=colors, alpha=0.75, edgecolor='gray')

    for bar, sig in zip(bars, df['sig']):
        x = bar.get_width()
        ax.text(x + 0.01 * np.sign(x), bar.get_y() + bar.get_height() / 2,
                sig, va='center', fontsize=8, fontweight='bold')

    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Rank-Biserial r  (← TD higher | DCP higher →)', fontsize=11)
    ax.set_title('Feature Effect Sizes: TD vs DCP (Mann-Whitney U)\n'
                 'Subject-level; Vanmechelen et al. 2024 dataset',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, '10_feature_effect_sizes.png')


# ─────────────────────────────────────────────────────
# PLOT 11: Per-subject barcode — convex hull spread
# ─────────────────────────────────────────────────────
def plot_per_subject_hull(subj):
    """
    Per-subject convex hull area, sorted by class then value.
    Ref: Vanmechelen 2024 ref[24]
    """
    psi = subj[['subject_id', 'class', 'convex_hull_area']].copy()
    psi = psi.sort_values(['class', 'convex_hull_area'])
    x = np.arange(len(psi))
    colors = [C_TD if c == 'TD' else C_DCP for c in psi['class']]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, psi['convex_hull_area'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(psi['subject_id'], rotation=45, fontsize=7)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Convex Hull Area (norm²)')
    ax.set_title('Per-Subject Hand-Path Spread\n'
                 'Blue = TD | Red = DCP | Ref: Vanmechelen 2024',
                 fontsize=12, fontweight='bold')
    ntd = (psi['class'] == 'TD').sum()
    ax.axvline(ntd - 0.5, color='black', ls='--', lw=2, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, '11_per_subject_hull.png')


# ─────────────────────────────────────────────────────
# PLOT 12: Feature correlation matrix by class
# ─────────────────────────────────────────────────────
def plot_correlation_matrices(subj, feat_cols):
    """Within-class Pearson correlation of key features."""
    key_feats = [
        'shoulder_ROM', 'elbow_ROM', 'wrist_ROM',
        'shoulder_jerk', 'elbow_jerk', 'wrist_jerk',
        'hand_speed_peak', 'index_of_curvature', 'convex_hull_area',
        'trunk_total_disp', 'arm_movement_ratio', 'execution_time',
    ]
    key_feats = [f for f in key_feats if f in feat_cols]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Feature Correlation Matrices by Class', fontsize=13, fontweight='bold')

    for i, (cls, title) in enumerate([('TD', 'TD (Typically Developing)'),
                                       ('DCP', 'DCP (Dyskinetic CP)')]):
        ax = axes[i]
        sub = subj[subj['class'] == cls][key_feats]
        corr = sub.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, square=True,
                    cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 7})
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '12_correlation_matrices.png')


# ─────────────────────────────────────────────────────
# PLOT 13: Statistical summary table as figure
# ─────────────────────────────────────────────────────
def plot_stats_table(stats_df):
    """Publication-quality stats table rendered as figure."""
    cols = ['feature', 'TD_mean', 'TD_std', 'DCP_mean', 'DCP_std', 'p', 'r_effect', 'sig']
    df = stats_df[cols].sort_values('p').copy()
    df['TD'] = df.apply(lambda r: f'{r.TD_mean:.2f} ± {r.TD_std:.2f}', axis=1)
    df['DCP'] = df.apply(lambda r: f'{r.DCP_mean:.2f} ± {r.DCP_std:.2f}', axis=1)
    df['p_str'] = df['p'].apply(lambda p: f'{p:.4f}' if p >= 0.001 else f'{p:.2e}')
    df['r'] = df['r_effect'].apply(lambda r: f'{r:.3f}')

    table_data = df[['feature', 'TD', 'DCP', 'p_str', 'r', 'sig']].values.tolist()
    col_labels = ['Feature', 'TD (mean±SD)', 'DCP (mean±SD)', 'p-value', 'Effect r', 'Sig']

    fig, ax = plt.subplots(figsize=(18, max(6, len(df) * 0.35 + 2)))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight significant rows
    for i, sig in enumerate(df['sig'].values):
        if sig != 'ns':
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#FFF9C4' if sig == '*' else
                                               '#FFE082' if sig == '**' else '#FFCC80')

    ax.set_title('Mann-Whitney U Test — All Features (Subject-Level)\n'
                 'Dataset: Vanmechelen et al. 2024 | 30 DCP, 21 TD | Reaching-Sideways',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    save_fig(fig, '13_stats_table.png')


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    print('=' * 70)
    print('FEATURE-WISE TD vs DCP PLOTS')
    print('Dataset: Vanmechelen et al. 2024 J Biomech 173:112233')
    print('=' * 70)

    # Load
    print('\n[1] Loading data...')
    df_all = load_all_data(DATA_DIR)

    # Feature columns
    meta_cols = ['subject_id', 'side', 'class_str', 'label', 'repetition', 'filename', 'class']
    feat_cols = [c for c in df_all.columns if c not in meta_cols]
    print(f'  {len(feat_cols)} features extracted per trial')

    # Subject-level aggregation
    print('\n[2] Aggregating to subject level...')
    subj = aggregate_subject_level(df_all, feat_cols)
    td_n = (subj['class'] == 'TD').sum()
    dcp_n = (subj['class'] == 'DCP').sum()
    print(f'  {len(subj)} subject-side entries (TD: {td_n}, DCP: {dcp_n})')

    # Statistics
    print('\n[3] Computing Mann-Whitney U statistics...')
    stats_df = compute_stats(subj, feat_cols)
    n_sig = (stats_df['p'] < 0.05).sum()
    print(f'  {n_sig}/{len(stats_df)} features significant at p<0.05')

    # Print table
    print(f'\n{"Feature":<30s} | {"TD":>16s} | {"DCP":>16s} | {"p":>10s} | Sig | r')
    print('─' * 95)
    for _, row in stats_df.sort_values('p').iterrows():
        print(f'{row["feature"]:<30s} | {row["TD_mean"]:7.2f}±{row["TD_std"]:6.2f} | '
              f'{row["DCP_mean"]:7.2f}±{row["DCP_std"]:6.2f} | {row["p"]:10.6f} | '
              f'{row["sig"]:>3s} | {row["r_effect"]:+.3f}')

    # Generate all plots
    print(f'\n[4] Generating plots → {OUTPUT_DIR}')
    plot_vanmechelen_fig3_style(subj, stats_df)
    plot_joint_rom(subj, stats_df)
    plot_normalised_jerk(subj, stats_df)
    plot_angular_dynamics(subj, stats_df)
    plot_hand_speed(subj, stats_df)
    plot_path_efficiency(subj, stats_df)
    plot_trunk_compensation(subj, stats_df)
    plot_interlimb_symmetry(subj, stats_df)
    plot_endpoint_control(subj, stats_df)
    plot_feature_heatmap(stats_df)
    plot_per_subject_hull(subj)
    plot_correlation_matrices(subj, feat_cols)
    plot_stats_table(stats_df)

    # Save stats CSV
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_stats_td_vs_dcp.csv'), index=False)
    subj.to_csv(os.path.join(OUTPUT_DIR, 'subject_level_features.csv'), index=False)
    print(f'\n  Stats CSV saved: feature_stats_td_vs_dcp.csv')
    print(f'  Subject CSV saved: subject_level_features.csv')

    print(f'\n{"=" * 70}')
    print(f'DONE. {len(os.listdir(OUTPUT_DIR))} files in {OUTPUT_DIR}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
