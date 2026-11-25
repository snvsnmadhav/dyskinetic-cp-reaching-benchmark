#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
END-TO-END PIPELINE: DCP vs TD Upper-Limb Reaching-Sideways Classification
& Pose Skeleton Video Generation from DeepLabCut Keypoints
==============================================================================

DATASET: Vanmechelen I, Van Wonterghem E, Aerts J-M, Hallez H, Desloovere K,
         Van de Walle P, Buizer AI, Monbaliu E, Haberfehlner H (2024).
         "Markerless motion analysis to assess reaching-sideways in individuals
         with dyskinetic cerebral palsy: A validity study."
         Journal of Biomechanics, 173, 112233.
         DOI: 10.1016/j.jbiomech.2024.112233

DATA REPOSITORY: KU Leuven RDR, DOI: 10.48804/BZM9SJ

LITERATURE SUMMARY:
-------------------
- 51 participants: 30 DCP (dyskinetic cerebral palsy), 21 TD (typically developing)
- Age range: 5-24 years
- Task: reaching-sideways (start on upper leg -> PTA sidewards at shoulder height)
- 8 repetitions per participant
- Keypoints extracted via DeepLabCut v2.3 (ResNet-101, MPII pretrained)
- 10 keypoints: ankle, knee, hip, shoulder, elbow, wrist, forehead, chin, MCP3, target
- Each xlsx contains per-frame x,y pixel coordinates + likelihood for each keypoint
- Concurrent validity: 2DMMA vs 3DMA (gold standard)
  - Wrist angle: rho=0.810, p<0.001
  - Elbow angle: rho=0.483, p<0.001
  - Shoulder angle: rho=0.247, p=0.102 (not significant)
- Construct validity: DCP vs TD differences
  - Wrist/elbow angles, execution time, variability features all differed (ES=0.35-0.81, p<0.05)
- Tracking accuracy: ~1.5 cm MAE against human labels

RELATED PUBLICATIONS:
---------------------
[1] Haberfehlner H, Vanmechelen I, et al. (2023). Frontiers in Robotics and AI, 10, 1108114.
[2] Haberfehlner H, Roth Z, Vanmechelen I, et al. (2024). Neurorehabil Neural Repair, 38(7), 479-492.
[3] Vanmechelen I, Bekteshi S, et al. (2022). Dev Med Child Neurol.
[4] Vanmechelen I, Desloovere K, et al. (2023). Gait & Posture.
[5] Mathis A, et al. (2018). Nature Neuroscience, 21(9), 1281-1289.
[6] Nath T, et al. (2019). Nature Protocols, 14(7), 2152-2176.
==============================================================================
"""

import os
import re
import sys
import glob
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pose_videos')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

KEYPOINT_NAMES = ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist',
                  'forehead', 'chin', 'MCP3', 'target']

# NOTE: DeepLabCut outputs pixel coordinates in IMAGE space:
#   - x increases left → right
#   - y increases TOP → BOTTOM (inverted vs. standard Cartesian)
# All plots invert y-axis to display anatomically correct orientation.
# Unsigned joint angles (via arccos of dot product) are invariant to y-flip.

SKELETON_EDGES = [
    ('ankle', 'knee'),
    ('knee', 'hip'),
    ('hip', 'shoulder'),
    ('shoulder', 'elbow'),
    ('elbow', 'wrist'),
    ('wrist', 'MCP3'),
    ('shoulder', 'forehead'),
    ('forehead', 'chin'),
]

GROUP_COLORS = {'DCP': '#E74C3C', 'TD': '#2ECC71'}

KP_COLORS = {
    'ankle': '#1f77b4', 'knee': '#ff7f0e', 'hip': '#2ca02c',
    'shoulder': '#d62728', 'elbow': '#9467bd', 'wrist': '#8c564b',
    'forehead': '#e377c2', 'chin': '#7f7f7f', 'MCP3': '#bcbd22',
    'target': '#17becf'
}

FPS = 30
DPI = 100
FIGSIZE = (8, 6)
LIKELIHOOD_THRESH = 0.6
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3


# =============================================================================
# 1. DIAGNOSTIC — INSPECT XLSX STRUCTURE
# =============================================================================
def diagnose_xlsx(filepath):
    """
    Print the raw structure of the first xlsx file so we know what we are dealing with.
    Returns the detected format string: 'dlc3', 'dlc2', 'flat', 'unknown'
    """
    print("\n" + "=" * 70)
    print(f"DIAGNOSING FILE STRUCTURE: {os.path.basename(filepath)}")
    print("=" * 70)

    # Read raw — no header interpretation
    try:
        df_raw = pd.read_excel(filepath, header=None, nrows=8)
        print(f"\nRaw shape (first 8 rows): {df_raw.shape}")
        print(f"Raw first 5 rows:\n{df_raw.iloc[:5, :min(15, df_raw.shape[1])].to_string()}")
        print(f"...(showing first 15 of {df_raw.shape[1]} columns)")
    except Exception as e:
        print(f"[ERROR] Cannot read file at all: {e}")
        return 'unknown'

    # Detect format by inspecting first few rows
    row0 = df_raw.iloc[0].astype(str).str.lower().tolist()
    row1 = df_raw.iloc[1].astype(str).str.lower().tolist() if df_raw.shape[0] > 1 else []
    row2 = df_raw.iloc[2].astype(str).str.lower().tolist() if df_raw.shape[0] > 2 else []

    # Check for DLC 3-row header: row0=scorer, row1=bodyparts, row2=x/y/likelihood
    has_xy_row2 = any('x' == v.strip() for v in row2) and any('y' == v.strip() for v in row2)
    has_bodyparts_row1 = any(bp in ' '.join(row1) for bp in ['ankle', 'knee', 'hip', 'shoulder',
                                                              'elbow', 'wrist', 'forehead', 'chin',
                                                              'mcp3', 'mcp', 'target'])
    has_bodyparts_row0 = any(bp in ' '.join(row0) for bp in ['ankle', 'knee', 'hip', 'shoulder',
                                                              'elbow', 'wrist', 'forehead', 'chin',
                                                              'mcp3', 'mcp', 'target'])
    has_xy_row1 = any('x' == v.strip() for v in row1) and any('y' == v.strip() for v in row1)

    fmt = 'unknown'
    if has_xy_row2 and has_bodyparts_row1:
        fmt = 'dlc3'
        print(f"\n>>> DETECTED FORMAT: DLC 3-header (scorer / bodyparts / x,y,likelihood)")
        print(f"    Scorer row: {row0[:5]}")
        print(f"    Bodyparts row: {row1[:15]}")
        print(f"    Coords row: {row2[:15]}")
    elif has_xy_row1 and has_bodyparts_row0:
        fmt = 'dlc2'
        print(f"\n>>> DETECTED FORMAT: DLC 2-header (bodyparts / x,y,likelihood)")
        print(f"    Bodyparts row: {row0[:15]}")
        print(f"    Coords row: {row1[:15]}")
    elif has_bodyparts_row0 and not has_xy_row1:
        fmt = 'flat_named'
        print(f"\n>>> DETECTED FORMAT: Flat columns with bodypart names in row 0")
        print(f"    Columns: {row0[:15]}")
    else:
        # Check if column names have patterns like 'ankle_x', 'knee_y', etc.
        combined = ' '.join(row0)
        if '_x' in combined or '_y' in combined:
            fmt = 'flat_underscore'
            print(f"\n>>> DETECTED FORMAT: Flat columns with underscore pattern (ankle_x, ankle_y)")
            print(f"    Columns: {row0[:15]}")
        else:
            # Could be numeric columns — maybe bodyparts are just x,y,likelihood repeated
            # Check if all data rows are numeric starting from some row
            fmt = 'numeric_only'
            print(f"\n>>> DETECTED FORMAT: Possibly numeric-only or custom layout")
            print(f"    Row 0: {row0[:15]}")
            print(f"    Row 1: {row1[:15]}")
            print(f"    Row 2: {row2[:15]}")

    # Also try the standard pandas multi-header read and show what it produces
    for n_headers in [3, 2, 1, 0]:
        try:
            if n_headers == 0:
                df_test = pd.read_excel(filepath, header=None, nrows=3)
            else:
                df_test = pd.read_excel(filepath, header=list(range(n_headers)), nrows=3)
            print(f"\n--- With header={list(range(n_headers)) if n_headers > 0 else None} ---")
            print(f"    Columns ({len(df_test.columns)}): {list(df_test.columns[:10])}...")
            print(f"    Dtypes: {df_test.dtypes.value_counts().to_dict()}")
            print(f"    First row values: {df_test.iloc[0, :min(10, df_test.shape[1])].tolist()}")
        except Exception as e:
            print(f"\n--- With header={list(range(n_headers))} --- FAILED: {e}")

    return fmt


# =============================================================================
# 2. FILE DISCOVERY & METADATA PARSING
# =============================================================================
def parse_filename(filepath):
    fname = os.path.basename(filepath).replace('.xlsx', '')
    fname = fname.replace('_reo', '_rep')
    pattern = r'^(\d+)_(left|right)_(DCP|TD)_rep(\d+)$'
    match = re.match(pattern, fname)
    if match:
        return {
            'subject_id': match.group(1),
            'hand': match.group(2),
            'group': match.group(3),
            'rep': int(match.group(4)),
            'filepath': filepath
        }
    return None


def discover_files(data_dir):
    files = glob.glob(os.path.join(data_dir, '*.xlsx'))
    records = []
    for f in sorted(files):
        meta = parse_filename(f)
        if meta:
            records.append(meta)
        else:
            print(f"[WARN] Could not parse: {os.path.basename(f)}")
    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f"Discovered {len(df)} files: "
              f"{df['group'].value_counts().to_dict()}, "
              f"{df['subject_id'].nunique()} unique subjects")
    else:
        print(f"[FATAL] No valid files found in {data_dir}")
    return df


# =============================================================================
# 3. UNIVERSAL XLSX LOADER — HANDLES ALL DLC FORMATS + CUSTOM
# =============================================================================
def load_dlc_xlsx(filepath, detected_format=None):
    """
    Universal loader. Tries every known DLC xlsx layout until one works.
    Returns: dict of {bodypart_name: {'x': np.array, 'y': np.array, 'likelihood': np.array}}
    or None if all attempts fail.
    """
    data = None

    # ---- ATTEMPT 1: DLC 3-row header (scorer / bodyparts / coords) ----
    data = _try_dlc3_header(filepath)
    if data and len(data) >= 3:
        return data

    # ---- ATTEMPT 2: DLC 2-row header (bodyparts / coords) ----
    data = _try_dlc2_header(filepath)
    if data and len(data) >= 3:
        return data

    # ---- ATTEMPT 3: Flat columns with underscore pattern ----
    data = _try_flat_underscore(filepath)
    if data and len(data) >= 3:
        return data

    # ---- ATTEMPT 4: Flat columns with bodypart names as-is ----
    data = _try_flat_named(filepath)
    if data and len(data) >= 3:
        return data

    # ---- ATTEMPT 5: Numeric-only — columns are just 0,1,2,... ----
    data = _try_numeric_positional(filepath)
    if data and len(data) >= 3:
        return data

    # ---- ATTEMPT 6: Brute force — read everything, guess by column count ----
    data = _try_bruteforce(filepath)
    if data and len(data) >= 3:
        return data

    print(f"[ERROR] All parse attempts failed for: {os.path.basename(filepath)}")
    return None


def _match_bodypart(name, target_bp):
    """Fuzzy match a column or level name to a target bodypart."""
    name_clean = str(name).lower().strip()
    target_clean = target_bp.lower().strip()
    # Exact match
    if name_clean == target_clean:
        return True
    # MCP3 variants: 'mcp3', 'mcp_3', 'mcpiii', 'metacarpophalangeal'
    if target_clean == 'mcp3' and any(v in name_clean for v in ['mcp3', 'mcp_3', 'mcp iii', 'mcpiii', 'metacar']):
        return True
    # shoulder variants
    if target_clean == 'shoulder' and 'shoulder' in name_clean:
        return True
    # Generic substring
    if target_clean in name_clean:
        return True
    return False


def _try_dlc3_header(filepath):
    """DLC format: 3 header rows (scorer, bodyparts, x/y/likelihood), index_col=0."""
    try:
        df = pd.read_excel(filepath, header=[0, 1, 2], index_col=0)
        data = {}
        level0_vals = df.columns.get_level_values(0).unique()
        level1_vals = df.columns.get_level_values(1).unique()

        # The scorer is the first level
        scorer = level0_vals[0]

        for bp in KEYPOINT_NAMES:
            matched_bp = None
            for col_bp in level1_vals:
                if _match_bodypart(col_bp, bp):
                    matched_bp = col_bp
                    break
            if matched_bp is not None:
                try:
                    data[bp] = {
                        'x': df[(scorer, matched_bp, 'x')].values.astype(np.float64),
                        'y': df[(scorer, matched_bp, 'y')].values.astype(np.float64),
                        'likelihood': df[(scorer, matched_bp, 'likelihood')].values.astype(np.float64)
                    }
                except KeyError:
                    # Try without scorer level
                    pass
        return data if len(data) >= 3 else None
    except Exception:
        return None


def _try_dlc3_header_no_index(filepath):
    """DLC 3-header but without index_col."""
    try:
        df = pd.read_excel(filepath, header=[0, 1, 2])
        data = {}
        level0_vals = df.columns.get_level_values(0).unique()
        level1_vals = df.columns.get_level_values(1).unique()
        scorer = level0_vals[0]

        for bp in KEYPOINT_NAMES:
            matched_bp = None
            for col_bp in level1_vals:
                if _match_bodypart(col_bp, bp):
                    matched_bp = col_bp
                    break
            if matched_bp is not None:
                try:
                    data[bp] = {
                        'x': df[(scorer, matched_bp, 'x')].values.astype(np.float64),
                        'y': df[(scorer, matched_bp, 'y')].values.astype(np.float64),
                        'likelihood': df[(scorer, matched_bp, 'likelihood')].values.astype(np.float64)
                    }
                except KeyError:
                    pass
        return data if len(data) >= 3 else None
    except Exception:
        return None


def _try_dlc2_header(filepath):
    """DLC format: 2 header rows (bodyparts, x/y/likelihood)."""
    try:
        df = pd.read_excel(filepath, header=[0, 1], index_col=0)
        data = {}
        level0_vals = df.columns.get_level_values(0).unique()
        for bp in KEYPOINT_NAMES:
            for col_bp in level0_vals:
                if _match_bodypart(col_bp, bp):
                    try:
                        data[bp] = {
                            'x': df[(col_bp, 'x')].values.astype(np.float64),
                            'y': df[(col_bp, 'y')].values.astype(np.float64),
                            'likelihood': df[(col_bp, 'likelihood')].values.astype(np.float64)
                        }
                    except KeyError:
                        pass
                    break
        return data if len(data) >= 3 else None
    except Exception:
        return None


def _try_flat_underscore(filepath):
    """Flat columns like 'ankle_x', 'ankle_y', 'ankle_likelihood'."""
    try:
        df = pd.read_excel(filepath)
        cols = df.columns.tolist()
        data = {}
        for bp in KEYPOINT_NAMES:
            bp_lower = bp.lower()
            x_col = None
            y_col = None
            l_col = None
            for c in cols:
                cl = str(c).lower().strip()
                if _match_bodypart(cl.split('_')[0] if '_' in cl else cl, bp):
                    if cl.endswith('_x') or cl.endswith(' x'):
                        x_col = c
                    elif cl.endswith('_y') or cl.endswith(' y'):
                        y_col = c
                    elif 'like' in cl:
                        l_col = c
            if x_col is not None and y_col is not None:
                data[bp] = {
                    'x': pd.to_numeric(df[x_col], errors='coerce').values.astype(np.float64),
                    'y': pd.to_numeric(df[y_col], errors='coerce').values.astype(np.float64),
                    'likelihood': pd.to_numeric(df[l_col], errors='coerce').values.astype(np.float64) if l_col else np.ones(len(df))
                }
        return data if len(data) >= 3 else None
    except Exception:
        return None


def _try_flat_named(filepath):
    """Flat columns where bodypart names appear somewhere in column headers."""
    try:
        df = pd.read_excel(filepath)
        cols = [str(c).lower().strip() for c in df.columns]
        data = {}

        # Build mapping: for each bodypart, find columns containing its name
        for bp in KEYPOINT_NAMES:
            bp_lower = bp.lower()
            matching_cols = [(i, c) for i, c in enumerate(cols) if _match_bodypart(c, bp)]
            if len(matching_cols) >= 2:
                # Assume first matching = x, second = y, third = likelihood
                orig_cols = df.columns.tolist()
                x_idx = matching_cols[0][0]
                y_idx = matching_cols[1][0]
                l_idx = matching_cols[2][0] if len(matching_cols) >= 3 else None
                data[bp] = {
                    'x': pd.to_numeric(df.iloc[:, x_idx], errors='coerce').values.astype(np.float64),
                    'y': pd.to_numeric(df.iloc[:, y_idx], errors='coerce').values.astype(np.float64),
                    'likelihood': pd.to_numeric(df.iloc[:, l_idx], errors='coerce').values.astype(np.float64) if l_idx else np.ones(len(df))
                }
        return data if len(data) >= 3 else None
    except Exception:
        return None


def _try_numeric_positional(filepath):
    """
    If columns are just numbers or generic names, try to read raw and
    figure out bodypart mapping from the actual row data.
    DLC standard: col0=frame, then for each bodypart 3 columns (x,y,likelihood).
    So total columns = 1 + 10*3 = 31 (for 10 keypoints).
    Or without frame col = 30.
    Could also have extra columns for 3DMA angles at the end.
    """
    try:
        # Read skipping potential header rows
        for skip in [3, 2, 1, 0]:
            try:
                df = pd.read_excel(filepath, header=None, skiprows=skip)
                # Filter to only numeric rows
                df_num = df.apply(pd.to_numeric, errors='coerce')
                # Drop rows that are all NaN (header remnants)
                df_num = df_num.dropna(how='all').reset_index(drop=True)
                # Drop columns that are all NaN
                df_num = df_num.dropna(axis=1, how='all')

                ncols = df_num.shape[1]
                if ncols < 6:
                    continue

                # Check if first column looks like a frame index (0,1,2,3...)
                first_col = df_num.iloc[:, 0].values
                has_frame_idx = False
                if len(first_col) > 3:
                    diffs = np.diff(first_col[:20])
                    if np.all(np.isclose(diffs, 1.0, atol=0.1)):
                        has_frame_idx = True

                start_col = 1 if has_frame_idx else 0
                data_cols = ncols - start_col

                # How many bodyparts can we fit? (3 cols each: x, y, likelihood)
                n_bodyparts = data_cols // 3
                remainder = data_cols % 3

                if n_bodyparts >= 3:
                    data = {}
                    bp_list = KEYPOINT_NAMES[:n_bodyparts]
                    for i, bp in enumerate(bp_list):
                        col_x = start_col + i * 3
                        col_y = start_col + i * 3 + 1
                        col_l = start_col + i * 3 + 2
                        data[bp] = {
                            'x': df_num.iloc[:, col_x].values.astype(np.float64),
                            'y': df_num.iloc[:, col_y].values.astype(np.float64),
                            'likelihood': df_num.iloc[:, col_l].values.astype(np.float64)
                        }
                    # Validate: likelihood should be in [0,1] range
                    sample_lk = data[bp_list[0]]['likelihood']
                    valid_lk = sample_lk[~np.isnan(sample_lk)]
                    if len(valid_lk) > 0 and np.nanmax(valid_lk) <= 1.1 and np.nanmin(valid_lk) >= -0.1:
                        return data
                    # Maybe x,y only (no likelihood) — 2 cols per bodypart
                    n_bodyparts_2 = data_cols // 2
                    if n_bodyparts_2 >= 3:
                        data = {}
                        bp_list = KEYPOINT_NAMES[:n_bodyparts_2]
                        for i, bp in enumerate(bp_list):
                            col_x = start_col + i * 2
                            col_y = start_col + i * 2 + 1
                            data[bp] = {
                                'x': df_num.iloc[:, col_x].values.astype(np.float64),
                                'y': df_num.iloc[:, col_y].values.astype(np.float64),
                                'likelihood': np.ones(len(df_num))
                            }
                        return data
            except Exception:
                continue
        return None
    except Exception:
        return None


def _try_bruteforce(filepath):
    """
    Last resort: read the file with every possible header combination,
    look for anything resembling x,y coordinate pairs.
    """
    try:
        # Try with header=None, skip first 3 rows (DLC typical)
        df = pd.read_excel(filepath, header=None)

        # Find the first fully numeric row (that's where data starts)
        data_start = 0
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            numeric_count = sum(1 for v in row if isinstance(v, (int, float)) and not pd.isna(v))
            if numeric_count >= 6:
                data_start = i
                break
            data_start = i + 1

        # Also try to extract bodypart names from pre-data rows
        bp_names_found = []
        for i in range(data_start):
            row = df.iloc[i].astype(str).str.lower().tolist()
            for bp in KEYPOINT_NAMES:
                if any(_match_bodypart(v, bp) for v in row):
                    bp_names_found.append(bp)

        df_data = df.iloc[data_start:].apply(pd.to_numeric, errors='coerce')
        df_data = df_data.dropna(how='all').reset_index(drop=True)
        df_data = df_data.dropna(axis=1, how='all')

        ncols = df_data.shape[1]
        if ncols < 6:
            return None

        # Check for frame index column
        first_col = df_data.iloc[:, 0].values
        has_frame_idx = False
        if len(first_col) > 3:
            diffs = np.diff(first_col[:20])
            if np.nanmean(np.abs(diffs - 1.0)) < 0.5:
                has_frame_idx = True

        start_col = 1 if has_frame_idx else 0
        data_cols = ncols - start_col

        # Try 3 cols per bodypart first
        n_bp3 = data_cols // 3
        if n_bp3 >= 3:
            data = {}
            bp_list = KEYPOINT_NAMES[:n_bp3]
            for i, bp in enumerate(bp_list):
                cx = start_col + i * 3
                cy = start_col + i * 3 + 1
                cl = start_col + i * 3 + 2
                if cl < ncols:
                    data[bp] = {
                        'x': df_data.iloc[:, cx].values.astype(np.float64),
                        'y': df_data.iloc[:, cy].values.astype(np.float64),
                        'likelihood': df_data.iloc[:, cl].values.astype(np.float64)
                    }
            if len(data) >= 3:
                return data

        # Try 2 cols per bodypart
        n_bp2 = data_cols // 2
        if n_bp2 >= 3:
            data = {}
            bp_list = KEYPOINT_NAMES[:n_bp2]
            for i, bp in enumerate(bp_list):
                cx = start_col + i * 2
                cy = start_col + i * 2 + 1
                data[bp] = {
                    'x': df_data.iloc[:, cx].values.astype(np.float64),
                    'y': df_data.iloc[:, cy].values.astype(np.float64),
                    'likelihood': np.ones(len(df_data))
                }
            if len(data) >= 3:
                return data

        return None
    except Exception:
        return None


# =============================================================================
# 4. PREPROCESSING
# =============================================================================
def preprocess_keypoints(kp_data, likelihood_thresh=LIKELIHOOD_THRESH):
    cleaned = {}
    for bp, coords in kp_data.items():
        x = coords['x'].copy()
        y = coords['y'].copy()
        lk = coords['likelihood'].copy()

        # Mask low-likelihood
        bad = lk < likelihood_thresh
        x[bad] = np.nan
        y[bad] = np.nan

        # Interpolate NaNs
        nframes = len(x)
        if np.sum(~np.isnan(x)) >= 4:
            idx = np.arange(nframes)
            valid = ~np.isnan(x)
            x = np.interp(idx, idx[valid], x[valid])
            y = np.interp(idx, idx[valid], y[valid])

            win = min(SAVGOL_WINDOW, nframes)
            if win % 2 == 0:
                win -= 1
            if win >= SAVGOL_POLY + 2:
                x = savgol_filter(x, win, SAVGOL_POLY)
                y = savgol_filter(y, win, SAVGOL_POLY)

        cleaned[bp] = {'x': x, 'y': y, 'likelihood': lk}
    return cleaned


# =============================================================================
# 5. FEATURE EXTRACTION
# =============================================================================
def compute_angle(p1, vertex, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def find_pta_frame(kp_data):
    bp_for_wrist = 'wrist' if 'wrist' in kp_data else ('MCP3' if 'MCP3' in kp_data else None)
    if bp_for_wrist is None:
        # Fallback: use elbow
        bp_for_wrist = 'elbow' if 'elbow' in kp_data else list(kp_data.keys())[0]

    wrist_x = kp_data[bp_for_wrist]['x']
    target_x = kp_data['target']['x'] if 'target' in kp_data else None

    if target_x is not None:
        dist = np.abs(wrist_x - target_x)
        mid = len(dist) // 3
        pta_frame = mid + np.argmin(dist[mid:])
    else:
        pta_frame = np.argmax(np.abs(wrist_x - wrist_x[0]))

    return int(pta_frame)


def extract_trial_features(kp_data, pta_frame):
    features = {}

    def get_pt(bp, frame):
        return (kp_data[bp]['x'][frame], kp_data[bp]['y'][frame])

    # Joint angles at PTA
    if all(bp in kp_data for bp in ['hip', 'shoulder', 'elbow']):
        features['shoulder_angle_PTA'] = compute_angle(
            get_pt('hip', pta_frame), get_pt('shoulder', pta_frame), get_pt('elbow', pta_frame))

    if all(bp in kp_data for bp in ['shoulder', 'elbow', 'wrist']):
        features['elbow_angle_PTA'] = compute_angle(
            get_pt('shoulder', pta_frame), get_pt('elbow', pta_frame), get_pt('wrist', pta_frame))

    if all(bp in kp_data for bp in ['elbow', 'wrist', 'MCP3']):
        features['wrist_angle_PTA'] = compute_angle(
            get_pt('elbow', pta_frame), get_pt('wrist', pta_frame), get_pt('MCP3', pta_frame))

    features['execution_time'] = pta_frame

    if 'wrist' in kp_data:
        wx = kp_data['wrist']['x'][:pta_frame + 1]
        wy = kp_data['wrist']['y'][:pta_frame + 1]
        dx, dy = np.diff(wx), np.diff(wy)
        features['wrist_path_length'] = np.sum(np.sqrt(dx**2 + dy**2))
        start = get_pt('wrist', 0)
        end = get_pt('wrist', pta_frame)
        features['wrist_displacement'] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        if features['wrist_path_length'] > 0:
            features['path_efficiency'] = features['wrist_displacement'] / features['wrist_path_length']

    if all(bp in kp_data for bp in ['wrist', 'target']):
        post_pta = kp_data['wrist']['x'][pta_frame:] - kp_data['target']['x'][pta_frame:]
        features['overshoot'] = np.max(np.abs(post_pta)) if len(post_pta) > 0 else 0.0

    # Angular trajectory features
    for angle_name, bps in [('shoulder_elev', ['hip', 'shoulder', 'elbow']),
                             ('elbow_flex', ['shoulder', 'elbow', 'wrist']),
                             ('wrist_flex', ['elbow', 'wrist', 'MCP3'])]:
        if all(bp in kp_data for bp in bps):
            nf = min(len(kp_data[bps[0]]['x']), pta_frame + 1)
            angles = np.array([
                compute_angle(
                    (kp_data[bps[0]]['x'][f], kp_data[bps[0]]['y'][f]),
                    (kp_data[bps[1]]['x'][f], kp_data[bps[1]]['y'][f]),
                    (kp_data[bps[2]]['x'][f], kp_data[bps[2]]['y'][f])
                ) for f in range(nf)
            ])
            features[f'{angle_name}_mean'] = np.mean(angles)
            features[f'{angle_name}_std'] = np.std(angles)
            features[f'{angle_name}_range'] = np.ptp(angles)
            if len(angles) > 1:
                ang_vel = np.abs(np.diff(angles))
                features[f'{angle_name}_vel_mean'] = np.mean(ang_vel)
                features[f'{angle_name}_vel_max'] = np.max(ang_vel)

    if all(bp in kp_data for bp in ['hip', 'shoulder']):
        trunk = kp_data['shoulder']['x'][:pta_frame+1] - kp_data['hip']['x'][:pta_frame+1]
        features['trunk_lean_mean'] = np.mean(trunk)
        features['trunk_lean_std'] = np.std(trunk)

    if 'forehead' in kp_data:
        fx = kp_data['forehead']['x'][:pta_frame+1]
        fy = kp_data['forehead']['y'][:pta_frame+1]
        features['head_movement'] = np.sum(np.sqrt(np.diff(fx)**2 + np.diff(fy)**2))

    return features


def extract_subject_variability_features(all_rep_kp_data):
    features = {}
    wrist_endpoints = []
    shoulder_angles_at_pta = []
    elbow_angles_at_pta = []

    for kp_data in all_rep_kp_data:
        pta = find_pta_frame(kp_data)
        if 'wrist' in kp_data:
            wrist_endpoints.append((kp_data['wrist']['x'][pta], kp_data['wrist']['y'][pta]))
        if all(bp in kp_data for bp in ['hip', 'shoulder', 'elbow']):
            shoulder_angles_at_pta.append(compute_angle(
                (kp_data['hip']['x'][pta], kp_data['hip']['y'][pta]),
                (kp_data['shoulder']['x'][pta], kp_data['shoulder']['y'][pta]),
                (kp_data['elbow']['x'][pta], kp_data['elbow']['y'][pta])))
        if all(bp in kp_data for bp in ['shoulder', 'elbow', 'wrist']):
            elbow_angles_at_pta.append(compute_angle(
                (kp_data['shoulder']['x'][pta], kp_data['shoulder']['y'][pta]),
                (kp_data['elbow']['x'][pta], kp_data['elbow']['y'][pta]),
                (kp_data['wrist']['x'][pta], kp_data['wrist']['y'][pta])))

    if len(wrist_endpoints) >= 3:
        pts = np.array(wrist_endpoints)
        try:
            hull = ConvexHull(pts)
            features['convex_hull_area'] = hull.volume
            features['convex_hull_perimeter'] = hull.area
        except Exception:
            features['convex_hull_area'] = 0.0
            features['convex_hull_perimeter'] = 0.0
    else:
        features['convex_hull_area'] = 0.0
        features['convex_hull_perimeter'] = 0.0

    if len(shoulder_angles_at_pta) > 1:
        features['shoulder_angle_SD_across_reps'] = np.std(shoulder_angles_at_pta)
    if len(elbow_angles_at_pta) > 1:
        features['elbow_angle_SD_across_reps'] = np.std(elbow_angles_at_pta)

    return features


# =============================================================================
# 6. POSE SKELETON VIDEO GENERATION
# =============================================================================
def render_pose_video(kp_data, output_path, title="", fps=FPS):
    first_bp = list(kp_data.keys())[0]
    nframes = len(kp_data[first_bp]['x'])

    all_x, all_y = [], []
    for bp in kp_data:
        all_x.extend(kp_data[bp]['x'])
        all_y.extend(kp_data[bp]['y'])
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    valid = ~(np.isnan(all_x) | np.isnan(all_y))
    if valid.sum() == 0:
        print(f"    [SKIP] No valid keypoints for {os.path.basename(output_path)}")
        return

    x_min, x_max = np.nanmin(all_x[valid]) - 30, np.nanmax(all_x[valid]) + 30
    y_min, y_max = np.nanmin(all_y[valid]) - 30, np.nanmax(all_y[valid]) + 30

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Inverted: DLC uses image coordinates (y increases downward)
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)', color='white', fontsize=9)
    ax.set_ylabel('Y (inverted image coordinates)', color='white', fontsize=9)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333355')

    joint_plots = {}
    for bp in kp_data:
        color = KP_COLORS.get(bp, '#ffffff')
        ms = 10 if bp == 'target' else 7
        mk = 'D' if bp == 'target' else 'o'
        joint_plots[bp], = ax.plot([], [], mk, color=color, markersize=ms,
                                    markeredgecolor='white', markeredgewidth=0.5, label=bp)

    bone_lines = {}
    for (bp1, bp2) in SKELETON_EDGES:
        if bp1 in kp_data and bp2 in kp_data:
            bone_lines[(bp1, bp2)], = ax.plot([], [], '-', color='#cccccc', linewidth=2, alpha=0.7)

    target_line = None
    if 'MCP3' in kp_data and 'target' in kp_data:
        target_line, = ax.plot([], [], '--', color='#17becf', linewidth=1.5, alpha=0.5)

    trail_line, = ax.plot([], [], '-', color='#8c564b', linewidth=1, alpha=0.4)
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                         fontsize=10, verticalalignment='top')
    ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e',
              edgecolor='#333355', labelcolor='white', ncol=2)

    def update(frame):
        artists = []
        for bp, plot_obj in joint_plots.items():
            x_val = kp_data[bp]['x'][frame]
            y_val = kp_data[bp]['y'][frame]
            lk = kp_data[bp]['likelihood'][frame] if 'likelihood' in kp_data[bp] else 1.0
            if not np.isnan(x_val) and not np.isnan(y_val) and lk > LIKELIHOOD_THRESH:
                plot_obj.set_data([x_val], [y_val])
                plot_obj.set_alpha(min(1.0, lk + 0.3))
            else:
                plot_obj.set_data([], [])
            artists.append(plot_obj)

        for (bp1, bp2), line_obj in bone_lines.items():
            x1, y1 = kp_data[bp1]['x'][frame], kp_data[bp1]['y'][frame]
            x2, y2 = kp_data[bp2]['x'][frame], kp_data[bp2]['y'][frame]
            if not any(np.isnan([x1, y1, x2, y2])):
                line_obj.set_data([x1, x2], [y1, y2])
            else:
                line_obj.set_data([], [])
            artists.append(line_obj)

        if target_line and 'MCP3' in kp_data and 'target' in kp_data:
            mx, my = kp_data['MCP3']['x'][frame], kp_data['MCP3']['y'][frame]
            tx, ty = kp_data['target']['x'][frame], kp_data['target']['y'][frame]
            if not any(np.isnan([mx, my, tx, ty])):
                target_line.set_data([mx, tx], [my, ty])
            artists.append(target_line)

        if 'wrist' in kp_data:
            ts = max(0, frame - 20)
            trail_line.set_data(kp_data['wrist']['x'][ts:frame+1], kp_data['wrist']['y'][ts:frame+1])
        artists.append(trail_line)

        frame_text.set_text(f'Frame {frame}/{nframes-1}')
        artists.append(frame_text)
        return artists

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)

    saved = False
    # Try mp4
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': title})
        mp4_path = output_path if output_path.endswith('.mp4') else output_path.replace('.gif', '.mp4')
        anim.save(mp4_path, writer=writer)
        print(f"    Saved: {os.path.basename(mp4_path)}")
        saved = True
    except Exception:
        pass

    # Fallback gif
    if not saved:
        try:
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer=PillowWriter(fps=min(fps, 15)))
            print(f"    Saved: {os.path.basename(gif_path)}")
            saved = True
        except Exception as e:
            print(f"    [ERROR] Could not save video: {e}")

    plt.close(fig)
    return saved


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("DCP vs TD REACHING-SIDEWAYS PIPELINE")
    print("Vanmechelen et al. 2024 — J Biomech 173:112233")
    print("=" * 70)

    # --- Step 1: Discover files ---
    file_df = discover_files(DATA_DIR)
    if len(file_df) == 0:
        print("[FATAL] No files found. Check DATA_DIR path.")
        return

    print(f"\nSubjects per group:")
    for grp, sub_df in file_df.groupby('group'):
        subs = sorted(sub_df['subject_id'].unique())
        print(f"  {grp}: {len(subs)} subjects — IDs: {subs}")

    # --- Step 2: DIAGNOSE first file to understand xlsx structure ---
    first_file = file_df.iloc[0]['filepath']
    detected_fmt = diagnose_xlsx(first_file)

    # --- Step 3: Test-load first file and verify ---
    print("\n" + "-" * 70)
    print("TEST LOADING FIRST FILE...")
    print("-" * 70)
    test_data = load_dlc_xlsx(first_file, detected_format=detected_fmt)
    if test_data is None:
        print(f"\n[FATAL] Cannot parse the xlsx files.")
        print(f"Please upload one sample xlsx file to Claude so I can inspect the exact structure.")
        return
    else:
        print(f"  SUCCESS — Loaded {len(test_data)} bodyparts: {list(test_data.keys())}")
        for bp, coords in test_data.items():
            n = len(coords['x'])
            xr = (np.nanmin(coords['x']), np.nanmax(coords['x']))
            yr = (np.nanmin(coords['y']), np.nanmax(coords['y']))
            lr = (np.nanmin(coords['likelihood']), np.nanmax(coords['likelihood']))
            print(f"    {bp:12s}: {n} frames, x=[{xr[0]:.1f}, {xr[1]:.1f}], "
                  f"y=[{yr[0]:.1f}, {yr[1]:.1f}], likelihood=[{lr[0]:.3f}, {lr[1]:.3f}]")

    # --- Step 4: Load all files, preprocess, extract features, render videos ---
    all_trial_features = []
    subject_rep_data = defaultdict(list)
    load_success = 0
    load_fail = 0

    print(f"\nProcessing {len(file_df)} files...")
    for idx, row in file_df.iterrows():
        fpath = row['filepath']
        fname = os.path.basename(fpath)
        sid = row['subject_id']
        grp = row['group']
        rep = row['rep']

        # Load
        kp_data = load_dlc_xlsx(fpath, detected_format=detected_fmt)
        if kp_data is None:
            load_fail += 1
            if load_fail <= 5:
                print(f"  [FAIL] {fname}")
            continue
        load_success += 1

        # Preprocess
        kp_data = preprocess_keypoints(kp_data)

        # Store for variability
        subject_rep_data[sid].append(kp_data)

        # PTA
        pta_frame = find_pta_frame(kp_data)

        # Features
        feats = extract_trial_features(kp_data, pta_frame)
        feats['subject_id'] = sid
        feats['group'] = grp
        feats['hand'] = row['hand']
        feats['rep'] = rep
        feats['label'] = 1 if grp == 'DCP' else 0
        feats['pta_frame'] = pta_frame
        feats['n_frames'] = len(kp_data[list(kp_data.keys())[0]]['x'])
        all_trial_features.append(feats)

        # Render pose video
        video_name = fname.replace('.xlsx', '.mp4')
        video_path = os.path.join(VIDEO_DIR, video_name)
        title_str = f"{grp} | Subject {sid} | {row['hand']} | Rep {rep}"
        if load_success <= 3 or load_success % 50 == 0:
            print(f"  [{load_success}/{len(file_df)}] {fname} — {len(kp_data)} bodyparts, "
                  f"{len(kp_data[list(kp_data.keys())[0]]['x'])} frames")
        render_pose_video(kp_data, video_path, title=title_str, fps=FPS)

    print(f"\nLoading summary: {load_success} success, {load_fail} failed out of {len(file_df)} files")

    if load_success == 0:
        print("[FATAL] No files could be loaded. Check the xlsx column structure.")
        print("Run diagnose_xlsx() output above and share it so I can fix the parser.")
        return

    # --- Step 5: Build feature matrix ---
    feat_df = pd.DataFrame(all_trial_features)
    print(f"\nFeature matrix: {feat_df.shape[0]} trials x {feat_df.shape[1]} columns")
    print(f"Columns: {list(feat_df.columns)}")

    # Variability features
    for sid, rep_list in subject_rep_data.items():
        if len(rep_list) >= 3:
            var_feats = extract_subject_variability_features(rep_list)
            for k, v in var_feats.items():
                feat_df.loc[feat_df['subject_id'] == sid, k] = v

    feat_csv = os.path.join(OUTPUT_DIR, 'trial_features.csv')
    feat_df.to_csv(feat_csv, index=False)
    print(f"Saved features to: {feat_csv}")

    # --- Step 6: Classification ---
    print("\n" + "=" * 70)
    print("CLASSIFICATION: DCP vs TD")
    print("=" * 70)

    meta_cols = ['subject_id', 'group', 'hand', 'rep', 'label', 'pta_frame', 'n_frames']
    feature_cols = [c for c in feat_df.columns if c not in meta_cols and feat_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    if len(feature_cols) == 0:
        print("[ERROR] No numeric feature columns found. Dumping dtypes:")
        print(feat_df.dtypes)
        return

    X = feat_df[feature_cols].copy()
    y = feat_df['label'].values
    groups = feat_df['subject_id'].values

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Drop columns that are still all NaN
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Class distribution: DCP={np.sum(y==1)}, TD={np.sum(y==0)}")

    n_unique_groups = len(np.unique(groups))
    n_splits = min(5, n_unique_groups)
    if n_splits < 2:
        print("[ERROR] Not enough unique subjects for cross-validation.")
        return

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                       class_weight='balanced', random_state=42),
    }

    results = {}
    for clf_name, clf in classifiers.items():
        fold_acc, fold_f1, fold_auc = [], [], []
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)
            y_prob = clf.predict_proba(X_test_sc)[:, 1]

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
            try:
                fold_auc.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_auc.append(0.5)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

        mean_acc = np.mean(fold_acc)
        mean_f1 = np.mean(fold_f1)
        mean_auc = np.mean(fold_auc)
        results[clf_name] = {'Accuracy': mean_acc, 'F1': mean_f1, 'AUC': mean_auc}

        print(f"\n--- {clf_name} ---")
        print(f"  Accuracy: {mean_acc:.3f} +/- {np.std(fold_acc):.3f}")
        print(f"  F1 Score: {mean_f1:.3f} +/- {np.std(fold_f1):.3f}")
        print(f"  AUC:      {mean_auc:.3f} +/- {np.std(fold_auc):.3f}")
        cm = confusion_matrix(all_y_true, all_y_pred)
        print(f"  Confusion Matrix (aggregated):\n    {cm}")
        print(classification_report(all_y_true, all_y_pred, target_names=['TD', 'DCP'], zero_division=0))

    results_df = pd.DataFrame(results).T
    results_csv = os.path.join(OUTPUT_DIR, 'classification_results.csv')
    results_df.to_csv(results_csv)
    print(f"Saved classification results to: {results_csv}")

    # --- Step 7: Feature importance ---
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (RandomForest)")
    print("=" * 70)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_all, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(15).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(importances))
    importances.head(top_n).plot(kind='barh', ax=ax, color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances (RandomForest)')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close(fig)

    # --- Step 8: Group comparison plots ---
    print("\n" + "=" * 70)
    print("GROUP COMPARISON PLOTS")
    print("=" * 70)

    plot_features = ['shoulder_angle_PTA', 'elbow_angle_PTA', 'wrist_angle_PTA',
                     'execution_time', 'overshoot', 'wrist_path_length',
                     'path_efficiency', 'convex_hull_area']
    plot_features = [f for f in plot_features if f in feat_df.columns]

    if len(plot_features) > 0:
        n_plots = len(plot_features)
        ncols = min(4, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feat_name in enumerate(plot_features):
            ax = axes[i]
            dcp_vals = feat_df.loc[feat_df['group'] == 'DCP', feat_name].dropna()
            td_vals = feat_df.loc[feat_df['group'] == 'TD', feat_name].dropna()
            bp_plot = ax.boxplot([td_vals, dcp_vals], labels=['TD', 'DCP'],
                                patch_artist=True, widths=0.6)
            bp_plot['boxes'][0].set_facecolor(GROUP_COLORS['TD'])
            bp_plot['boxes'][1].set_facecolor(GROUP_COLORS['DCP'])
            for box in bp_plot['boxes']:
                box.set_alpha(0.6)
            ax.set_title(feat_name, fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('DCP vs TD Group Comparisons', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'group_comparison.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {os.path.join(OUTPUT_DIR, 'group_comparison.png')}")

    # --- Step 9: Wrist trajectory overlay ---
    print("\nGenerating wrist trajectory overlay plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grp in zip(axes, ['TD', 'DCP']):
        ax.set_title(f'{grp} — Wrist Trajectories (8 reps/subject)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (inverted image coordinates)')
        ax.set_facecolor('#f8f8f8')
        grp_subs = file_df.loc[file_df['group'] == grp, 'subject_id'].unique()
        plotted = 0
        for sid in grp_subs:
            for rep_data in subject_rep_data.get(sid, []):
                if 'wrist' in rep_data:
                    ax.plot(rep_data['wrist']['x'], rep_data['wrist']['y'],
                            alpha=0.3, linewidth=0.8, color=GROUP_COLORS[grp])
                    plotted += 1
            if plotted > 40:
                break
        ax.invert_yaxis()  # DLC image coordinates: y increases downward
        ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'wrist_trajectories.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'wrist_trajectories.png')}")

    # --- DONE ---
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Features CSV:     {feat_csv}")
    print(f"  Results CSV:      {results_csv}")
    print(f"  Plots:            {OUTPUT_DIR}")
    print(f"  Pose videos:      {VIDEO_DIR}")
    print(f"  Files loaded:     {load_success}/{len(file_df)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
