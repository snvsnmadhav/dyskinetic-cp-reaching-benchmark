#!/usr/bin/env python3
"""
==============================================================================
MEAN WAVEFORM PLOTS — EVERY FEATURE SEPARATELY — TD vs DCP
==============================================================================
Dataset: Vanmechelen et al. 2024 J Biomech 173:112233 (doi:10.48804/BZM9SJ)
         30 DCP + 21 TD, reaching-sideways, 8 reps, DeepLabCut 2D keypoints

For each kinematic feature:
  • Compute frame-by-frame time-series for every trial
  • Resample all trials to 100% movement cycle (0–100%)
  • Plot mean ± 1 SD band for TD (blue) and DCP (red)
  • Display the exact mathematical formula on the plot
  • Mann-Whitney U at each % point → shade significant regions
==============================================================================
"""

import os, sys, glob, re, math, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter, resample
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from collections import defaultdict

warnings.filterwarnings('ignore')
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'waveform_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS         = 30.0
RESAMPLE_N  = 101   # 0% to 100% movement cycle
SG_WINDOW   = 7
SG_POLY     = 3

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
        return np.full((T, 2), np.nan, dtype=np.float64)
    x = df[cx].interpolate(method='linear', limit_direction='both').ffill().bfill()
    y = df[cy].interpolate(method='linear', limit_direction='both').ffill().bfill()
    return np.stack([x.values, y.values], 1).astype(np.float64)

def smooth(s, window=SG_WINDOW, polyorder=SG_POLY):
    if len(s) < max(window, polyorder + 1):
        return s.copy()
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(s, w, polyorder)

def resample_to_cycle(arr, n=RESAMPLE_N):
    """Resample 1-D array to n points (0–100% cycle)."""
    if len(arr) < 2:
        return np.zeros(n)
    return resample(arr.astype(np.float64), n)

# =============================================================================
# 3. PER-TRIAL WAVEFORM EXTRACTION — 26 features, each returns (T,) array
# =============================================================================
def extract_waveforms(df, meta):
    """
    Returns dict: feature_name → 1-D numpy array of length T (raw frames).
    Each feature has an associated formula string and reference.
    """
    T = len(df)
    dt = 1.0 / FPS
    active_side = meta['side']
    act, pas = ('l', 'r') if active_side == 'left' else ('r', 'l')

    # ── Raw keypoints ──
    shoulder_act = get_xy(df, f'shoulder_{act}')
    shoulder_pas = get_xy(df, f'shoulder_{pas}')
    elbow_act    = get_xy(df, f'elbow_{act}')
    elbow_pas    = get_xy(df, f'elbow_{pas}')
    wrist_act    = get_xy(df, f'wrist_{act}')
    wrist_pas    = get_xy(df, f'wrist_{pas}')
    hand_act     = get_xy(df, f'hand_{act}')
    hand_pas     = get_xy(df, f'hand_{pas}')
    target       = get_xy(df, 'object')
    forehead     = get_xy(df, 'forehead')
    chin         = get_xy(df, 'chin')

    # ── Reference scale: upper-arm length ──
    upper_arm_len = np.linalg.norm(elbow_act - shoulder_act, axis=1)
    ref_scale = np.maximum(upper_arm_len, 1e-6)

    trunk_up = forehead - chin
    mid_shoulder = (shoulder_act + shoulder_pas) / 2.0

    # ── Helper: 3-point angle ──
    def angle_3pt(a, b, c):
        ba, bc = a - b, c - b
        cos_a = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    # ================================================================
    # FEATURE 1: SHOULDER ANGLE (active side)
    # ================================================================
    # Formula: θ_shoulder = arccos( (SE⃗ · ST⃗) / (|SE⃗| · |ST⃗|) )
    #   where SE⃗ = elbow − shoulder, ST⃗ = (shoulder + trunk_up) − shoulder = trunk_up
    # Ref: Vanmechelen et al. 2022 PLoS ONE 17(9):e0266294
    ang_shoulder_act = angle_3pt(elbow_act, shoulder_act, shoulder_act + trunk_up)

    # ================================================================
    # FEATURE 2: ELBOW ANGLE (active side)
    # ================================================================
    # Formula: θ_elbow = arccos( (ES⃗ · EW⃗) / (|ES⃗| · |EW⃗|) )
    #   where ES⃗ = shoulder − elbow, EW⃗ = wrist − elbow
    # Ref: Vanmechelen et al. 2022 PLoS ONE
    ang_elbow_act = angle_3pt(shoulder_act, elbow_act, wrist_act)

    # ================================================================
    # FEATURE 3: WRIST ANGLE (active side)
    # ================================================================
    # Formula: θ_wrist = arccos( (WE⃗ · WH⃗) / (|WE⃗| · |WH⃗|) )
    #   where WE⃗ = elbow − wrist, WH⃗ = hand(MCP3) − wrist
    # Ref: Vanmechelen et al. 2022 PLoS ONE
    ang_wrist_act = angle_3pt(elbow_act, wrist_act, hand_act)

    # ================================================================
    # FEATURE 4: SHOULDER ANGLE (passive side)
    # ================================================================
    ang_shoulder_pas = angle_3pt(elbow_pas, shoulder_pas, shoulder_pas + trunk_up)

    # ================================================================
    # FEATURE 5: ELBOW ANGLE (passive side)
    # ================================================================
    ang_elbow_pas = angle_3pt(shoulder_pas, elbow_pas, wrist_pas)

    # ================================================================
    # FEATURE 6: WRIST ANGLE (passive side)
    # ================================================================
    ang_wrist_pas = angle_3pt(elbow_pas, wrist_pas, hand_pas)

    # ================================================================
    # FEATURE 7: SHOULDER ANGULAR VELOCITY (active)
    # ================================================================
    # Formula: ω_shoulder(t) = dθ_shoulder/dt  (central difference)
    # Ref: Jaspers et al. 2009 Gait Posture 30(4):395–404
    w_shoulder_act = smooth(np.gradient(ang_shoulder_act, dt))

    # ================================================================
    # FEATURE 8: ELBOW ANGULAR VELOCITY (active)
    # ================================================================
    w_elbow_act = smooth(np.gradient(ang_elbow_act, dt))

    # ================================================================
    # FEATURE 9: WRIST ANGULAR VELOCITY (active)
    # ================================================================
    w_wrist_act = smooth(np.gradient(ang_wrist_act, dt))

    # ================================================================
    # FEATURE 10: SHOULDER ANGULAR ACCELERATION (active)
    # ================================================================
    # Formula: α_shoulder(t) = d²θ_shoulder/dt² = dω_shoulder/dt
    # Ref: Sanger, Kaiser & Placek 2005 J Neurophysiol 93(3)
    a_shoulder_act = smooth(np.gradient(w_shoulder_act, dt))

    # ================================================================
    # FEATURE 11: ELBOW ANGULAR ACCELERATION (active)
    # ================================================================
    a_elbow_act = smooth(np.gradient(w_elbow_act, dt))

    # ================================================================
    # FEATURE 12: WRIST ANGULAR ACCELERATION (active)
    # ================================================================
    a_wrist_act = smooth(np.gradient(w_wrist_act, dt))

    # ================================================================
    # FEATURE 13: HAND SPEED (normalised)
    # ================================================================
    # Formula: v_hand(t) = ||hand(t) − hand(t−1)|| / (Δt · L_upperarm(t))
    # Ref: Vanmechelen et al. 2022 PLoS ONE ("lower maximal velocity in DCP")
    hand_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_act, axis=0), axis=1)])
    hand_speed = hand_disp / dt
    hand_speed_norm = smooth(hand_speed / ref_scale)

    # ================================================================
    # FEATURE 14: HAND ACCELERATION (normalised)
    # ================================================================
    # Formula: a_hand(t) = dv_hand/dt
    # Ref: Butler & Rose 2012 J Biomech 45(6):945–951
    hand_accel_norm = smooth(np.gradient(hand_speed_norm, dt))

    # ================================================================
    # FEATURE 15: HAND JERK (normalised)
    # ================================================================
    # Formula: j_hand(t) = d³position/dt³ = da_hand/dt
    # Ref: Flash & Hogan 1985 J Neurosci 5(7):1688–1703
    hand_jerk_norm = smooth(np.gradient(hand_accel_norm, dt))

    # ================================================================
    # FEATURE 16: HAND-TO-TARGET DISTANCE (normalised)
    # ================================================================
    # Formula: d_target(t) = ||hand(t) − target(t)|| / L_upperarm(t)
    # Ref: Vanmechelen 2024 J Biomech (overshoot = wrist above object)
    hand_to_target = np.linalg.norm(hand_act - target, axis=1) / ref_scale

    # ================================================================
    # FEATURE 17: CUMULATIVE PATH LENGTH (normalised)
    # ================================================================
    # Formula: S(t) = Σ_{i=1}^{t} ||hand(i) − hand(i−1)|| / L_upperarm(i)
    # Ref: Schneiberg et al. 2010 Dev Med Child Neurol (path efficiency)
    cum_path = np.cumsum(hand_disp / ref_scale)

    # ================================================================
    # FEATURE 18: TRUNK DISPLACEMENT (normalised, cumulative)
    # ================================================================
    # Formula: d_trunk(t) = ||trunk_mid(t) − trunk_mid(0)|| / L_upperarm(t)
    #   where trunk_mid = (forehead + chin) / 2
    # Ref: Van Wonterghem et al. 2023 Clin Biomech
    trunk_mid = (forehead + chin) / 2.0
    trunk_drift = np.linalg.norm(trunk_mid - trunk_mid[0], axis=1) / ref_scale

    # ================================================================
    # FEATURE 19: TRUNK INSTANTANEOUS SPEED (normalised)
    # ================================================================
    # Formula: v_trunk(t) = ||trunk_mid(t) − trunk_mid(t−1)|| / (Δt · L_upperarm(t))
    # Ref: Van Wonterghem et al. 2023 Clin Biomech
    trunk_disp = np.concatenate([[0], np.linalg.norm(np.diff(trunk_mid, axis=0), axis=1)])
    trunk_speed = smooth(trunk_disp / (dt * ref_scale))

    # ================================================================
    # FEATURE 20: HEAD LATERAL DISPLACEMENT (normalised)
    # ================================================================
    # Formula: Δx_head(t) = (forehead_x(t) − forehead_x(0)) / L_upperarm(t)
    # Ref: Van Wonterghem et al. 2023 Clin Biomech
    head_x_disp = (forehead[:, 0] - forehead[0, 0]) / ref_scale

    # ================================================================
    # FEATURE 21: SHOULDER ANGLE ASYMMETRY (instantaneous)
    # ================================================================
    # Formula: Δθ_shoulder(t) = |θ_shoulder_active(t) − θ_shoulder_passive(t)|
    # Ref: Gordon et al. 2006 Pediatr Neurol 35(6):375–381
    shoulder_asym = np.abs(ang_shoulder_act - ang_shoulder_pas)

    # ================================================================
    # FEATURE 22: ELBOW ANGLE ASYMMETRY (instantaneous)
    # ================================================================
    # Formula: Δθ_elbow(t) = |θ_elbow_active(t) − θ_elbow_passive(t)|
    # Ref: Gordon et al. 2006 Pediatr Neurol
    elbow_asym = np.abs(ang_elbow_act - ang_elbow_pas)

    # ================================================================
    # FEATURE 23: PASSIVE ARM SPEED (normalised)
    # ================================================================
    # Formula: v_passive(t) = ||hand_passive(t) − hand_passive(t−1)|| / (Δt · L_upperarm(t))
    # Ref: Gordon et al. 2006 (contralateral overflow as dystonia marker)
    pas_disp = np.concatenate([[0], np.linalg.norm(np.diff(hand_pas, axis=0), axis=1)])
    pas_speed = smooth(pas_disp / (dt * ref_scale))

    # ================================================================
    # FEATURE 24: NORMALISED HAND X (frontal-plane horizontal)
    # ================================================================
    # Formula: x_norm(t) = (hand_x(t) − mid_shoulder_x(t)) / L_upperarm(t)
    # Ref: Vanmechelen 2024 (normalised coordinates)
    hand_x_norm = (hand_act[:, 0] - mid_shoulder[:, 0]) / ref_scale

    # ================================================================
    # FEATURE 25: NORMALISED HAND Y (frontal-plane vertical)
    # ================================================================
    # Formula: y_norm(t) = (hand_y(t) − mid_shoulder_y(t)) / L_upperarm(t)
    hand_y_norm = (hand_act[:, 1] - mid_shoulder[:, 1]) / ref_scale

    # ================================================================
    # FEATURE 26: INSTANTANEOUS INDEX OF CURVATURE
    # ================================================================
    # Formula: IoC(t) = S(t) / ||hand(t) − hand(0)||
    #   S(t) = cumulative path up to frame t
    # Ref: Schneiberg et al. 2010 Dev Med Child Neurol
    straight = np.linalg.norm(hand_act - hand_act[0], axis=1) / ref_scale
    ioc_inst = cum_path / (straight + 1e-8)
    ioc_inst[0] = 1.0  # by definition at t=0

    # ── Clean all features ──
    waveforms = {
        'shoulder_angle_active':      ang_shoulder_act,
        'elbow_angle_active':         ang_elbow_act,
        'wrist_angle_active':         ang_wrist_act,
        'shoulder_angle_passive':     ang_shoulder_pas,
        'elbow_angle_passive':        ang_elbow_pas,
        'wrist_angle_passive':        ang_wrist_pas,
        'shoulder_angular_velocity':  w_shoulder_act,
        'elbow_angular_velocity':     w_elbow_act,
        'wrist_angular_velocity':     w_wrist_act,
        'shoulder_angular_accel':     a_shoulder_act,
        'elbow_angular_accel':        a_elbow_act,
        'wrist_angular_accel':        a_wrist_act,
        'hand_speed_norm':            hand_speed_norm,
        'hand_accel_norm':            hand_accel_norm,
        'hand_jerk_norm':             hand_jerk_norm,
        'hand_to_target_dist':        hand_to_target,
        'cumulative_path_length':     cum_path,
        'trunk_drift':                trunk_drift,
        'trunk_speed':                trunk_speed,
        'head_lateral_disp':          head_x_disp,
        'shoulder_angle_asymmetry':   shoulder_asym,
        'elbow_angle_asymmetry':      elbow_asym,
        'passive_arm_speed':          pas_speed,
        'hand_x_norm':                hand_x_norm,
        'hand_y_norm':                hand_y_norm,
        'instantaneous_IoC':          ioc_inst,
    }

    for k in waveforms:
        waveforms[k] = np.nan_to_num(waveforms[k], nan=0.0, posinf=0.0, neginf=0.0)

    return waveforms

# =============================================================================
# FORMULA + REFERENCE STRINGS FOR EACH FEATURE
# =============================================================================
FEATURE_META = {
    'shoulder_angle_active': {
        'title': 'Shoulder Angle (Active Side)',
        'ylabel': 'Angle (°)',
        'formula': (
            r'$\theta_{sh}(t) = \arccos\!\left(\frac{\vec{SE}\,\cdot\,\vec{ST}}'
            r'{|\vec{SE}|\;|\vec{ST}|}\right)$'
            '\n'
            r'$\vec{SE} = \mathbf{elbow} - \mathbf{shoulder}$, '
            r'$\vec{ST} = \mathbf{trunk\_up}$'
        ),
        'ref': 'Vanmechelen et al. 2022 PLoS ONE 17(9):e0266294',
    },
    'elbow_angle_active': {
        'title': 'Elbow Angle (Active Side)',
        'ylabel': 'Angle (°)',
        'formula': (
            r'$\theta_{el}(t) = \arccos\!\left(\frac{\vec{ES}\,\cdot\,\vec{EW}}'
            r'{|\vec{ES}|\;|\vec{EW}|}\right)$'
            '\n'
            r'$\vec{ES} = \mathbf{shoulder} - \mathbf{elbow}$, '
            r'$\vec{EW} = \mathbf{wrist} - \mathbf{elbow}$'
        ),
        'ref': 'Vanmechelen et al. 2022 PLoS ONE',
    },
    'wrist_angle_active': {
        'title': 'Wrist Angle (Active Side)',
        'ylabel': 'Angle (°)',
        'formula': (
            r'$\theta_{wr}(t) = \arccos\!\left(\frac{\vec{WE}\,\cdot\,\vec{WH}}'
            r'{|\vec{WE}|\;|\vec{WH}|}\right)$'
            '\n'
            r'$\vec{WE} = \mathbf{elbow} - \mathbf{wrist}$, '
            r'$\vec{WH} = \mathbf{MCP3} - \mathbf{wrist}$'
        ),
        'ref': 'Vanmechelen et al. 2022 PLoS ONE',
    },
    'shoulder_angle_passive': {
        'title': 'Shoulder Angle (Passive Side)',
        'ylabel': 'Angle (°)',
        'formula': r'Same as active: $\theta_{sh,pas}(t) = \arccos(...)$ on passive-side joints',
        'ref': 'Gordon et al. 2006 Pediatr Neurol 35(6):375–381\n(contralateral overflow)',
    },
    'elbow_angle_passive': {
        'title': 'Elbow Angle (Passive Side)',
        'ylabel': 'Angle (°)',
        'formula': r'Same as active: $\theta_{el,pas}(t)$ on passive-side joints',
        'ref': 'Gordon et al. 2006 Pediatr Neurol',
    },
    'wrist_angle_passive': {
        'title': 'Wrist Angle (Passive Side)',
        'ylabel': 'Angle (°)',
        'formula': r'Same as active: $\theta_{wr,pas}(t)$ on passive-side joints',
        'ref': 'Gordon et al. 2006 Pediatr Neurol',
    },
    'shoulder_angular_velocity': {
        'title': 'Shoulder Angular Velocity (Active)',
        'ylabel': 'Angular Velocity (°/s)',
        'formula': (
            r'$\omega_{sh}(t) = \frac{d\theta_{sh}}{dt}$'
            '\n'
            r'Central diff: $\omega(t) = \frac{\theta(t+1) - \theta(t-1)}{2\Delta t}$'
        ),
        'ref': 'Jaspers et al. 2009 Gait Posture 30(4):395–404',
    },
    'elbow_angular_velocity': {
        'title': 'Elbow Angular Velocity (Active)',
        'ylabel': 'Angular Velocity (°/s)',
        'formula': r'$\omega_{el}(t) = \frac{d\theta_{el}}{dt}$',
        'ref': 'Jaspers et al. 2009 Gait Posture',
    },
    'wrist_angular_velocity': {
        'title': 'Wrist Angular Velocity (Active)',
        'ylabel': 'Angular Velocity (°/s)',
        'formula': r'$\omega_{wr}(t) = \frac{d\theta_{wr}}{dt}$',
        'ref': 'Jaspers et al. 2009 Gait Posture',
    },
    'shoulder_angular_accel': {
        'title': 'Shoulder Angular Acceleration (Active)',
        'ylabel': 'Angular Accel (°/s²)',
        'formula': (
            r'$\alpha_{sh}(t) = \frac{d^2\theta_{sh}}{dt^2} = \frac{d\omega_{sh}}{dt}$'
        ),
        'ref': 'Sanger, Kaiser & Placek 2005 J Neurophysiol 93(3)',
    },
    'elbow_angular_accel': {
        'title': 'Elbow Angular Acceleration (Active)',
        'ylabel': 'Angular Accel (°/s²)',
        'formula': r'$\alpha_{el}(t) = \frac{d^2\theta_{el}}{dt^2}$',
        'ref': 'Sanger et al. 2005 J Neurophysiol',
    },
    'wrist_angular_accel': {
        'title': 'Wrist Angular Acceleration (Active)',
        'ylabel': 'Angular Accel (°/s²)',
        'formula': r'$\alpha_{wr}(t) = \frac{d^2\theta_{wr}}{dt^2}$',
        'ref': 'Sanger et al. 2005 J Neurophysiol',
    },
    'hand_speed_norm': {
        'title': 'Hand Speed (Normalised)',
        'ylabel': 'Speed (body-lengths/s)',
        'formula': (
            r'$v_{hand}(t) = \frac{\|\mathbf{hand}(t) - \mathbf{hand}(t\!-\!1)\|}'
            r'{\Delta t \;\cdot\; L_{upperarm}(t)}$'
        ),
        'ref': 'Vanmechelen et al. 2022 PLoS ONE\n("lower maximal velocity in DCP")',
    },
    'hand_accel_norm': {
        'title': 'Hand Acceleration (Normalised)',
        'ylabel': 'Accel (body-lengths/s²)',
        'formula': r'$a_{hand}(t) = \frac{dv_{hand}}{dt}$',
        'ref': 'Butler & Rose 2012 J Biomech 45(6):945–951',
    },
    'hand_jerk_norm': {
        'title': 'Hand Jerk (Normalised)',
        'ylabel': 'Jerk (body-lengths/s³)',
        'formula': (
            r'$j_{hand}(t) = \frac{d^3 \mathbf{position}}{dt^3} = \frac{da_{hand}}{dt}$'
            '\n'
            r'NJ $= \sqrt{\frac{T^5}{2\,\mathrm{range}^2}\int_0^T j^2\,dt}$'
        ),
        'ref': 'Flash & Hogan 1985 J Neurosci 5(7):1688–1703\nTeulings et al. 1997 Exp Neurol',
    },
    'hand_to_target_dist': {
        'title': 'Hand-to-Target Distance (Normalised)',
        'ylabel': 'Distance (body-lengths)',
        'formula': (
            r'$d_{target}(t) = \frac{\|\mathbf{hand}(t) - \mathbf{target}(t)\|}'
            r'{L_{upperarm}(t)}$'
        ),
        'ref': 'Vanmechelen 2024 J Biomech 173:112233',
    },
    'cumulative_path_length': {
        'title': 'Cumulative Path Length (Normalised)',
        'ylabel': 'Path Length (body-lengths)',
        'formula': (
            r'$S(t) = \sum_{i=1}^{t}\frac{\|\mathbf{hand}(i) - \mathbf{hand}(i\!-\!1)\|}'
            r'{L_{upperarm}(i)}$'
        ),
        'ref': 'Schneiberg et al. 2010 Dev Med Child Neurol 52(7):e167',
    },
    'trunk_drift': {
        'title': 'Trunk Drift from Start (Normalised)',
        'ylabel': 'Drift (body-lengths)',
        'formula': (
            r'$d_{trunk}(t) = \frac{\|\mathbf{trunk}_{mid}(t) - \mathbf{trunk}_{mid}(0)\|}'
            r'{L_{upperarm}(t)}$'
            '\n'
            r'$\mathbf{trunk}_{mid} = \frac{\mathbf{forehead} + \mathbf{chin}}{2}$'
        ),
        'ref': 'Van Wonterghem et al. 2023 Clin Biomech',
    },
    'trunk_speed': {
        'title': 'Trunk Speed (Normalised)',
        'ylabel': 'Speed (body-lengths/s)',
        'formula': (
            r'$v_{trunk}(t) = \frac{\|\mathbf{trunk}_{mid}(t) - \mathbf{trunk}_{mid}(t\!-\!1)\|}'
            r'{\Delta t \;\cdot\; L_{upperarm}(t)}$'
        ),
        'ref': 'Van Wonterghem et al. 2023 Clin Biomech',
    },
    'head_lateral_disp': {
        'title': 'Head Lateral Displacement (Normalised)',
        'ylabel': 'Displacement (body-lengths)',
        'formula': (
            r'$\Delta x_{head}(t) = \frac{x_{forehead}(t) - x_{forehead}(0)}'
            r'{L_{upperarm}(t)}$'
        ),
        'ref': 'Van Wonterghem et al. 2023 Clin Biomech',
    },
    'shoulder_angle_asymmetry': {
        'title': 'Shoulder Angle Asymmetry (|Active − Passive|)',
        'ylabel': 'Asymmetry (°)',
        'formula': (
            r'$\Delta\theta_{sh}(t) = |\theta_{sh,active}(t) - \theta_{sh,passive}(t)|$'
        ),
        'ref': 'Gordon et al. 2006 Pediatr Neurol 35(6):375\n(contralateral overflow as dystonia)',
    },
    'elbow_angle_asymmetry': {
        'title': 'Elbow Angle Asymmetry (|Active − Passive|)',
        'ylabel': 'Asymmetry (°)',
        'formula': r'$\Delta\theta_{el}(t) = |\theta_{el,active}(t) - \theta_{el,passive}(t)|$',
        'ref': 'Gordon et al. 2006 Pediatr Neurol',
    },
    'passive_arm_speed': {
        'title': 'Passive Arm Speed (Contralateral Overflow)',
        'ylabel': 'Speed (body-lengths/s)',
        'formula': (
            r'$v_{passive}(t) = \frac{\|\mathbf{hand}_{pas}(t) - \mathbf{hand}_{pas}(t\!-\!1)\|}'
            r'{\Delta t \;\cdot\; L_{upperarm}(t)}$'
        ),
        'ref': 'Gordon et al. 2006 Pediatr Neurol\n(overflow = involuntary passive-arm movement)',
    },
    'hand_x_norm': {
        'title': 'Hand X (Frontal Horizontal, Normalised)',
        'ylabel': 'x (body-lengths)',
        'formula': (
            r'$x_{hand}^{norm}(t) = \frac{x_{hand}(t) - x_{mid\_shoulder}(t)}'
            r'{L_{upperarm}(t)}$'
        ),
        'ref': 'Vanmechelen 2024 (normalised to segment length)',
    },
    'hand_y_norm': {
        'title': 'Hand Y (Frontal Vertical, Normalised)',
        'ylabel': 'y (body-lengths)',
        'formula': (
            r'$y_{hand}^{norm}(t) = \frac{y_{hand}(t) - y_{mid\_shoulder}(t)}'
            r'{L_{upperarm}(t)}$'
        ),
        'ref': 'Vanmechelen 2024 (normalised to segment length)',
    },
    'instantaneous_IoC': {
        'title': 'Instantaneous Index of Curvature',
        'ylabel': 'IoC (ratio)',
        'formula': (
            r'$\mathrm{IoC}(t) = \frac{S(t)}{\|\mathbf{hand}(t) - \mathbf{hand}(0)\| / L_{ua}}$'
            '\n'
            r'IoC $= 1 \Rightarrow$ straight line; IoC $\gg 1 \Rightarrow$ curved path'
        ),
        'ref': 'Schneiberg et al. 2010 Dev Med Child Neurol\n(path efficiency in CP reaching)',
    },
}

# =============================================================================
# 4. DATA LOADING
# =============================================================================
def load_all_waveforms(data_dir):
    zip_path = os.path.join(os.path.dirname(data_dir) or '.', 'data.zip')
    if os.path.isfile(zip_path) and not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

    all_xlsx = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
    matched = [f for f in all_xlsx if _FNAME_RE.match(os.path.splitext(os.path.basename(f))[0])]
    if not matched:
        raise FileNotFoundError(f"No matching xlsx in {data_dir}.")

    # Dict: feature_name → {'TD': list of (RESAMPLE_N,) arrays, 'DCP': list}
    all_waves = {k: {'TD': [], 'DCP': []} for k in FEATURE_META}

    for i, fpath in enumerate(matched):
        try:
            meta = parse_filename(fpath)
            df = pd.read_excel(fpath)
            drop = [c for c in df.columns
                    if re.match(r'^unnamed', str(c), re.I) or str(c).strip().lower() == 'index']
            df = df.drop(columns=drop)
            waveforms = extract_waveforms(df, meta)
            cls = meta['class_str']
            for fname, arr in waveforms.items():
                if fname in all_waves:
                    resampled = resample_to_cycle(arr, RESAMPLE_N)
                    all_waves[fname][cls].append(resampled)
        except Exception as e:
            pass
        if (i + 1) % 50 == 0:
            print(f'  Loaded {i+1}/{len(matched)}')

    td_n = len(all_waves['shoulder_angle_active']['TD'])
    dcp_n = len(all_waves['shoulder_angle_active']['DCP'])
    print(f'  {td_n} TD trials, {dcp_n} DCP trials loaded')
    return all_waves

# =============================================================================
# 5. PLOT SINGLE WAVEFORM
# =============================================================================
C_TD  = '#1565C0'
C_DCP = '#C62828'
C_TD_BAND  = '#90CAF9'
C_DCP_BAND = '#EF9A9A'
C_SIG      = '#FFF176'  # yellow shading for significant regions

def plot_single_waveform(feature_name, td_array, dcp_array, output_dir, idx):
    """
    Plot mean ± SD waveform for one feature.
    td_array, dcp_array: list of (RESAMPLE_N,) arrays.
    """
    meta = FEATURE_META[feature_name]
    td_mat  = np.array(td_array)    # (n_td, RESAMPLE_N)
    dcp_mat = np.array(dcp_array)   # (n_dcp, RESAMPLE_N)

    x = np.linspace(0, 100, RESAMPLE_N)
    td_mean  = np.mean(td_mat, axis=0)
    td_std   = np.std(td_mat, axis=0)
    dcp_mean = np.mean(dcp_mat, axis=0)
    dcp_std  = np.std(dcp_mat, axis=0)

    # Point-wise Mann-Whitney U significance
    sig_mask = np.zeros(RESAMPLE_N, dtype=bool)
    for t in range(RESAMPLE_N):
        try:
            _, p = mannwhitneyu(td_mat[:, t], dcp_mat[:, t], alternative='two-sided')
            sig_mask[t] = (p < 0.05)
        except:
            pass

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(14, 7))

    # Shade significant regions
    sig_regions = []
    in_region = False
    for t in range(RESAMPLE_N):
        if sig_mask[t] and not in_region:
            start = x[t]
            in_region = True
        elif not sig_mask[t] and in_region:
            sig_regions.append((start, x[t - 1]))
            in_region = False
    if in_region:
        sig_regions.append((start, x[-1]))

    for s, e in sig_regions:
        ax.axvspan(s, e, alpha=0.15, color=C_SIG, zorder=0)

    # Mean ± SD bands
    ax.fill_between(x, td_mean - td_std, td_mean + td_std,
                     color=C_TD_BAND, alpha=0.35, zorder=1)
    ax.fill_between(x, dcp_mean - dcp_std, dcp_mean + dcp_std,
                     color=C_DCP_BAND, alpha=0.35, zorder=1)

    # Mean lines
    ax.plot(x, td_mean,  color=C_TD,  linewidth=2.5, label=f'TD  (n={len(td_array)})',  zorder=3)
    ax.plot(x, dcp_mean, color=C_DCP, linewidth=2.5, label=f'DCP (n={len(dcp_array)})', zorder=3)

    # Axis labels
    ax.set_xlabel('Movement Cycle (%)', fontsize=12)
    ax.set_ylabel(meta['ylabel'], fontsize=12)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, zorder=0)

    # Title
    ax.set_title(
        f'Feature {idx:02d}: {meta["title"]}\n'
        f'Mean ± 1 SD  |  Yellow = significant region (p<0.05, point-wise Mann-Whitney U)',
        fontsize=13, fontweight='bold')

    # Legend
    sig_patch = mpatches.Patch(color=C_SIG, alpha=0.3, label='p < 0.05 region')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sig_patch)
    ax.legend(handles=handles, fontsize=10, loc='upper right')

    # ── FORMULA BOX ──
    formula_text = (
        f"FORMULA:\n{meta['formula']}\n\n"
        f"REF: {meta['ref']}"
    )
    ax.text(
        0.02, 0.97, formula_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#616161', alpha=0.92),
        family='serif', usetex=False
    )

    # ── Summary stats in bottom-right ──
    n_sig_pts = sig_mask.sum()
    pct_sig = 100 * n_sig_pts / RESAMPLE_N
    diff_at_peak = np.max(np.abs(dcp_mean - td_mean))
    peak_loc = x[np.argmax(np.abs(dcp_mean - td_mean))]

    stats_text = (
        f'Significant: {pct_sig:.0f}% of cycle\n'
        f'Max |Δmean|: {diff_at_peak:.2f} at {peak_loc:.0f}%\n'
        f'TD range: [{td_mean.min():.1f}, {td_mean.max():.1f}]\n'
        f'DCP range: [{dcp_mean.min():.1f}, {dcp_mean.max():.1f}]'
    )
    ax.text(
        0.98, 0.03, stats_text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                  edgecolor='#9E9E9E', alpha=0.9),
        family='monospace'
    )

    plt.tight_layout()
    fname = f'{idx:02d}_{feature_name}.png'
    fig.savefig(os.path.join(output_dir, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [{idx:02d}] {meta["title"]}: {pct_sig:.0f}% sig, max Δ={diff_at_peak:.2f}')

# =============================================================================
# 6. COMBINED OVERVIEW GRID
# =============================================================================
def plot_overview_grid(all_waves, output_dir):
    """6x5 grid of all 26 waveforms (small multiples)."""
    feature_names = list(FEATURE_META.keys())
    n_feats = len(feature_names)
    ncols = 5
    nrows = math.ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(28, nrows * 4))
    fig.suptitle(
        'ALL FEATURE WAVEFORMS: TD (blue) vs DCP (red)\n'
        'Mean ± 1 SD | Vanmechelen et al. 2024 dataset | 0–100% movement cycle',
        fontsize=16, fontweight='bold', y=0.995)

    x = np.linspace(0, 100, RESAMPLE_N)

    for i, fname in enumerate(feature_names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        td_mat  = np.array(all_waves[fname]['TD'])
        dcp_mat = np.array(all_waves[fname]['DCP'])
        td_mean  = np.mean(td_mat, axis=0)
        td_std   = np.std(td_mat, axis=0)
        dcp_mean = np.mean(dcp_mat, axis=0)
        dcp_std  = np.std(dcp_mat, axis=0)

        ax.fill_between(x, td_mean - td_std, td_mean + td_std,
                         color=C_TD_BAND, alpha=0.3)
        ax.fill_between(x, dcp_mean - dcp_std, dcp_mean + dcp_std,
                         color=C_DCP_BAND, alpha=0.3)
        ax.plot(x, td_mean,  color=C_TD,  linewidth=1.5)
        ax.plot(x, dcp_mean, color=C_DCP, linewidth=1.5)

        meta = FEATURE_META[fname]
        ax.set_title(f'{i+1}. {meta["title"]}', fontsize=8, fontweight='bold')
        ax.set_ylabel(meta['ylabel'], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for j in range(n_feats, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, '00_overview_all_waveforms.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  [00] Overview grid saved')

# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    print('=' * 70)
    print('MEAN WAVEFORM PLOTS — EACH FEATURE SEPARATELY')
    print('TD vs DCP  |  Vanmechelen et al. 2024  |  doi:10.48804/BZM9SJ')
    print('=' * 70)

    print(f'\n[1] Loading & extracting waveforms from {DATA_DIR}...')
    all_waves = load_all_waveforms(DATA_DIR)

    print(f'\n[2] Generating individual waveform plots → {OUTPUT_DIR}')
    feature_names = list(FEATURE_META.keys())
    for i, fname in enumerate(feature_names):
        td_arr  = all_waves[fname]['TD']
        dcp_arr = all_waves[fname]['DCP']
        if len(td_arr) < 2 or len(dcp_arr) < 2:
            print(f'  [{i+1:02d}] SKIP {fname}: insufficient data')
            continue
        plot_single_waveform(fname, td_arr, dcp_arr, OUTPUT_DIR, i + 1)

    print(f'\n[3] Generating overview grid...')
    plot_overview_grid(all_waves, OUTPUT_DIR)

    print(f'\n{"=" * 70}')
    print(f'DONE. {len(os.listdir(OUTPUT_DIR))} files in {OUTPUT_DIR}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
