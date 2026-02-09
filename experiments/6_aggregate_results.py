"""Aggregate perturbation results.

Loads XAI and UQ perturbation pickles, computes degradation ratios and
correlations, and generates summary tables and plots for reporting.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


# CONFIGURATION

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT_DIR / "results"
XAI_RESULTS_DIR = RESULTS_ROOT / "5_run_perturbation_explainers"
UQ_RESULTS_DIR = RESULTS_ROOT / "4_run_perturbation_uncertainty"
OUTPUT_DIR = RESULTS_ROOT / Path(__file__).stem

# LATEX TABLE SIZE TINY SCRIPTSIZE FOOTNOTESIZE SMALL NORMALSIZE
TABLE_SIZE = "tiny"
STRONG_CORR_THRESHOLD = -0.6
PLOT_CORRELATION_HEATMAPS = True
PLOT_CORRELATION_HEATMAP_MEAN = False
HEATMAP_DPI = 300
HEATMAP_COLOR_MIN = -1.0
HEATMAP_COLOR_CENTER = -0.6
HEATMAP_COLOR_MAX = 0.2
HEATMAP_WIDTH_PER_COL = 0.32
HEATMAP_HEIGHT_PER_ROW = 0.1
HEATMAP_ROW_HEIGHT_MIN = 1.2
HEATMAP_MEAN_HEIGHT_MIN = 1.2
HEATMAP_FILE_EXT = "pdf"
HEATMAP_SHOW_COLORBAR_NATURAL = False
HEATMAP_SHOW_COLORBAR_ADVERSARIAL = True
HEATMAP_LEFT_PAD = 1.2
HEATMAP_RIGHT_PAD = 1.2
HEATMAP_TOP_PAD = 0.6
HEATMAP_BOTTOM_PAD = 0.4
HEATMAP_CBAR_WIDTH = 0.1
HEATMAP_CBAR_PAD = 0.1
HEATMAP_CBAR_LABEL_SIZE = 6
HEATMAP_CELL_FONT_SIZE = 5
HEATMAP_LABEL_FONT_SIZE = 8
HEATMAP_BLOCK_GAP = 0.2
HEATMAP_WSPACE = 0.1
HEATMAP_HSPACE = 0.08
HEATMAP_ROW_LABEL_PAD = 7
HEATMAP_CMAP = "coolwarm"

MODEL_LABELS = {
    "Logistic": "LR",
    "RF": "RF",
    "MLP": "MLP",
    "LightGBM": "LGBM",
    "CatBoost": "CB",
}

# DATASET CONFIGURATIONS NAME UCI ID INTERNAL NAME IN DESIRED ORDER
DATASETS = [
    #("Iris", 53, "iris"),
    ("Wine", 186, "wine_binary"),
    ("Bean", 602, "bean"),
    ("Rice", 545, "rice"),
    ("Ecoli", 39, "ecoli"),
]

MODELS = ['Logistic', 'RF', 'MLP', 'LightGBM', 'CatBoost']
NATURAL_PERTURBATIONS = ['gaussian', 'missing', 'permutation']
ADVERSARIAL_PERTURBATIONS = ['adversarial_bim', 'adversarial_pgd', 'adversarial_cw']
ADVERSARIAL_PERTURBATIONS_UQ = ['bim', 'pgd', 'cw']  # For UQ data (different naming)
FIXED_PERT_LEVELS = {'gaussian': 0.5, 'missing': 0.1, 'permutation': 0.1}
FIXED_ADV_LEVELS = {'adversarial_bim': 0.1, 'adversarial_pgd': 0.1, 'adversarial_cw': 1.0}
FIXED_ADV_LEVELS_UQ = {'bim': 0.1, 'pgd': 0.1, 'cw': 1.0}
DEFAULT_UQ_METRICS = ['epi', 'ale', 'conf']
REDUCED_UQ_METRICS = ['ale', 'conf']
# PER MODEL METRIC SELECTION FOR THE UQ TABLE REMOVE ITEMS TO HIDE COLUMNS PER MODEL
UQ_MODEL_METRICS = {
    'Logistic': DEFAULT_UQ_METRICS,
    'RF': DEFAULT_UQ_METRICS,
    'MLP': DEFAULT_UQ_METRICS,
    'LightGBM': REDUCED_UQ_METRICS,
    'CatBoost': REDUCED_UQ_METRICS,
}


# HELPER FUNCTIONS

def fixed_levels_caption() -> str:
    """Return a short caption snippet describing fixed perturbation levels."""
    g = FIXED_PERT_LEVELS['gaussian']
    m = FIXED_PERT_LEVELS['missing']
    p = FIXED_PERT_LEVELS['permutation']
    bim = FIXED_ADV_LEVELS['adversarial_bim']
    pgd = FIXED_ADV_LEVELS['adversarial_pgd']
    cw = FIXED_ADV_LEVELS['adversarial_cw']
    return f"G={g}, M/P={m}, BIM/PGD={bim}, C\\&W={cw}"

def load_xai_results(dataset_name: str, uci_id: int) -> List[Dict]:
    """Load XAI explainer results from pickle."""
    pickle_file = XAI_RESULTS_DIR / f"explainers_perturbations_{dataset_name}_{uci_id}.pkl"
    
    if not pickle_file.exists():
        print(f"WARNING: {pickle_file} not found")
        return []
    
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    
    return results


def load_uq_results(dataset_name: str, uci_id: int) -> List[Dict]:
    """Load UQ perturbation results from pickle."""
    pickle_file = UQ_RESULTS_DIR / f"perturbation_uncertainty_{dataset_name}_{uci_id}.pkl"
    
    if not pickle_file.exists():
        print(f"WARNING: {pickle_file} not found")
        return []
    
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    
    return results


def compute_degradation_ratio(clean_val: float, max_pert_val: float, 
                              metric_type: str = 'rank_correlation') -> Tuple[float, str]:
    """Compute the degradation or growth ratio."""
    if clean_val == 0 or max_pert_val == 0:
        return 0.0, "$-$"
    
    if metric_type == 'rank_correlation':
        ratio = clean_val / max_pert_val
        direction = "$\\downarrow$"
    else:
        ratio = max_pert_val / clean_val
        if ratio > 1.0:
            direction = "$\\uparrow$"
        elif ratio < 1.0:
            direction = "$\\downarrow$"
        else:
            direction = "$-$"
    
    return ratio, direction


# TABLE 1 XAI DEGRADATION SUMMARY NO AVERAGING

def generate_xai_degradation_table() -> Tuple[str, str]:
    """Generate XAI degradation tables (natural + adversarial)."""
    
    print("\n" + "="*80)
    print("GENERATING TABLE 1: XAI DEGRADATION SUMMARY")
    print("="*80)
    
    data = []
    
    for display_name, uci_id, dataset_name in DATASETS:
        print(f"\nProcessing {display_name}...")
        
        xai_results = load_xai_results(dataset_name, uci_id)
        if not xai_results:
            continue
        
        df = pd.DataFrame(xai_results)
        row = {'Dataset': display_name}
        
        for model in MODELS:
            model_results = df[df['model'] == model]
            
            if len(model_results) == 0:
                continue
            
            # NATURAL PERTURBATIONS SHAP LIME FOR ALL MODELS IG SMOOTHGRAD FOR MLP FIXED LEVELS
            for pert in NATURAL_PERTURBATIONS:
                target_level = FIXED_PERT_LEVELS.get(pert, None)
                # SHAP
                shap_results = model_results[model_results['explainer'].str.contains('SHAP', na=False)]
                pert_results = shap_results[shap_results['perturbation'] == pert]

                if len(pert_results) > 0:
                    clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                    pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                    pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                    if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                        ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                        row[f"{model}_SHAP_{pert}"] = ratio
                    else:
                        row[f"{model}_SHAP_{pert}"] = np.nan
                else:
                    row[f"{model}_SHAP_{pert}"] = np.nan

                # LIME
                lime_results = model_results[model_results['explainer'] == 'LIME']
                pert_results = lime_results[lime_results['perturbation'] == pert]

                if len(pert_results) > 0:
                    clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                    pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                    pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                    if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                        ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                        row[f"{model}_LIME_{pert}"] = ratio
                    else:
                        row[f"{model}_LIME_{pert}"] = np.nan
                else:
                    row[f"{model}_LIME_{pert}"] = np.nan

                # IG AND SMOOTHGRAD FOR MLP ONLY ON NATURAL PERTURBATIONS
                if model == 'MLP':
                    # INTGRAD
                    ig_results = model_results[model_results['explainer'] == 'IntGrad']
                    pert_results = ig_results[ig_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_IG_{pert}"] = ratio
                        else:
                            row[f"{model}_IG_{pert}"] = np.nan
                    else:
                        row[f"{model}_IG_{pert}"] = np.nan

                    # SMOOTHGRAD
                    sg_results = model_results[model_results['explainer'] == 'SmoothGrad']
                    pert_results = sg_results[sg_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_SmoothGrad_{pert}"] = ratio
                        else:
                            row[f"{model}_SmoothGrad_{pert}"] = np.nan
                    else:
                        row[f"{model}_SmoothGrad_{pert}"] = np.nan
            
            # ADVERSARIAL PERTURBATIONS IG SMOOTHGRAD MLP ONLY
            if model == 'MLP':
                for pert in ADVERSARIAL_PERTURBATIONS:
                    target_level = FIXED_ADV_LEVELS.get(pert, None)
                    # SHAP
                    shap_results = model_results[model_results['explainer'].str.contains('SHAP', na=False)]
                    pert_results = shap_results[shap_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_SHAP_{pert}"] = ratio
                        else:
                            row[f"{model}_SHAP_{pert}"] = np.nan
                    else:
                        row[f"{model}_SHAP_{pert}"] = np.nan

                    # LIME
                    lime_results = model_results[model_results['explainer'] == 'LIME']
                    pert_results = lime_results[lime_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_LIME_{pert}"] = ratio
                        else:
                            row[f"{model}_LIME_{pert}"] = np.nan
                    else:
                        row[f"{model}_LIME_{pert}"] = np.nan

                    # INTGRAD
                    ig_results = model_results[model_results['explainer'] == 'IntGrad']
                    pert_results = ig_results[ig_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_IG_{pert}"] = ratio
                        else:
                            row[f"{model}_IG_{pert}"] = np.nan
                    else:
                        row[f"{model}_IG_{pert}"] = np.nan

                    # SMOOTHGRAD
                    sg_results = model_results[model_results['explainer'] == 'SmoothGrad']
                    pert_results = sg_results[sg_results['perturbation'] == pert]

                    if len(pert_results) > 0:
                        clean = pert_results[pert_results['level'] == 0.0]['rank_correlation'].values
                        pert_level = pert_results[np.isclose(pert_results['level'], target_level)]['rank_correlation'].values if target_level is not None else []
                        pert_val = pert_level[0] if len(pert_level) > 0 else np.nan

                        if len(clean) > 0 and not np.isnan(pert_val) and pert_val > 0:
                            ratio, _ = compute_degradation_ratio(clean[0], pert_val, 'rank_correlation')
                            row[f"{model}_SmoothGrad_{pert}"] = ratio
                        else:
                            row[f"{model}_SmoothGrad_{pert}"] = np.nan
                    else:
                        row[f"{model}_SmoothGrad_{pert}"] = np.nan
        
        data.append(row)
    
    df_summary = pd.DataFrame(data)
    latex_natural = generate_xai_latex_natural(df_summary)
    latex_adversarial = generate_xai_latex_adversarial(df_summary)
    
    return latex_natural, latex_adversarial


def generate_xai_latex_natural(df: pd.DataFrame) -> str:
    """Generate LaTeX code for XAI degradation table on natural perturbations."""

    xai_columns = [
        ("Logistic", "SHAP", NATURAL_PERTURBATIONS),
        ("Logistic", "LIME", NATURAL_PERTURBATIONS),
        ("RF", "SHAP", NATURAL_PERTURBATIONS),
        ("RF", "LIME", NATURAL_PERTURBATIONS),
        ("MLP", "SHAP", NATURAL_PERTURBATIONS),
        ("MLP", "LIME", NATURAL_PERTURBATIONS),
        ("MLP", "IG", NATURAL_PERTURBATIONS),
        ("MLP", "SmoothGrad", NATURAL_PERTURBATIONS),
        ("LightGBM", "SHAP", NATURAL_PERTURBATIONS),
        ("LightGBM", "LIME", NATURAL_PERTURBATIONS),
        ("CatBoost", "SHAP", NATURAL_PERTURBATIONS),
        ("CatBoost", "LIME", NATURAL_PERTURBATIONS),
    ]

    fixed_caption = fixed_levels_caption()

    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{XAI Degradation Summary (Natural Perturbations): Rank Correlation Degradation Ratio = Clean $\div$ Perturbed@fixed level ({fixed_caption}). 
Higher ratio ($\uparrow$) indicates stronger degradation. 
SHAP/LIME tested on Gaussian, Missing, Permutation. 
MLP: IG/SmoothGrad reported for natural perturbations.}}
\label{{tab:xai-degradation-natural}}
\{TABLE_SIZE}
\begin{{tabular}}{{ll|cc|cc|cccc|cc|cc}}
\toprule
\multicolumn{{2}}{{c|}}{{\textbf{{Dataset/Pert}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{Logistic}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{RF}}}} & 
\multicolumn{{4}}{{c|}}{{\textbf{{MLP}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{LightGBM}}}} & 
\multicolumn{{2}}{{c}}{{\textbf{{CatBoost}}}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}} \cmidrule(lr){{7-10}} \cmidrule(lr){{11-12}} \cmidrule(lr){{13-14}}
& & SHAP & LIME & SHAP & LIME & SHAP & LIME & IG & SG & SHAP & LIME & SHAP & LIME \\
\midrule
"""

    for idx, row in df.iterrows():
        dataset_name = row['Dataset']

        for pert_idx, (pert_short, pert_full) in enumerate([
            ('Gaussian', 'gaussian'),
            ('Missing', 'missing'),
            ('Permutation', 'permutation')
        ]):
            if pert_idx == 0:
                latex += rf"\multirow{{3}}{{*}}{{{dataset_name}}} & {pert_short}"
            else:
                latex += f" & {pert_short}"

            for model in ['Logistic']:
                for method in ['SHAP', 'LIME']:
                    col = f"{model}_{method}_{pert_full}"
                    val = row.get(col, np.nan)

                    if not pd.isna(val):
                        if val >= 10:
                            val_str = f"{val:.1f}$\\uparrow$"
                        else:
                            val_str = f"{val:.2f}$\\uparrow$"
                    else:
                        val_str = "---"

                    latex += f" & {val_str}"

            for model in ['RF']:
                for method in ['SHAP', 'LIME']:
                    col = f"{model}_{method}_{pert_full}"
                    val = row.get(col, np.nan)

                    if not pd.isna(val):
                        if val >= 10:
                            val_str = f"{val:.1f}$\\uparrow$"
                        else:
                            val_str = f"{val:.2f}$\\uparrow$"
                    else:
                        val_str = "---"

                    latex += f" & {val_str}"

            for method in ['SHAP', 'LIME', 'IG', 'SmoothGrad']:
                col = f"MLP_{method}_{pert_full}"
                val = row.get(col, np.nan)

                if not pd.isna(val):
                    if val >= 10:
                        val_str = f"{val:.1f}$\\uparrow$"
                    else:
                        val_str = f"{val:.2f}$\\uparrow$"
                else:
                    val_str = "---"

                latex += f" & {val_str}"

            for model in ['LightGBM']:
                for method in ['SHAP', 'LIME']:
                    col = f"{model}_{method}_{pert_full}"
                    val = row.get(col, np.nan)

                    if not pd.isna(val):
                        if val >= 10:
                            val_str = f"{val:.1f}$\\uparrow$"
                        else:
                            val_str = f"{val:.2f}$\\uparrow$"
                    else:
                        val_str = "---"

                    latex += f" & {val_str}"

            for model in ['CatBoost']:
                for method in ['SHAP', 'LIME']:
                    col = f"{model}_{method}_{pert_full}"
                    val = row.get(col, np.nan)

                    if not pd.isna(val):
                        if val >= 10:
                            val_str = f"{val:.1f}$\\uparrow$"
                        else:
                            val_str = f"{val:.2f}$\\uparrow$"
                    else:
                        val_str = "---"

                    latex += f" & {val_str}"

            latex += r" \\" + "\n"

        latex += r"\multicolumn{2}{l|}{\textbf{Mean}}"
        for model, method, perts in xai_columns:
            vals = []
            for pert in perts:
                col = f"{model}_{method}_{pert}"
                if col in row and not pd.isna(row[col]):
                    vals.append(row[col])
            if len(vals) > 0:
                mean_val = np.mean(vals)
                if mean_val >= 10:
                    val_str = f"{mean_val:.1f}$\\uparrow$"
                else:
                    val_str = f"{mean_val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"
        latex += r" \\" + "\n"

        if idx < len(df) - 1:
            latex += r"\cmidrule{1-14}" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table*}" + "\n"

    return latex


def generate_xai_latex_adversarial(df: pd.DataFrame) -> str:
    """Generate LaTeX code for XAI degradation table on adversarial perturbations."""

    xai_columns = [
        ("MLP", "SHAP", ADVERSARIAL_PERTURBATIONS),
        ("MLP", "LIME", ADVERSARIAL_PERTURBATIONS),
        ("MLP", "IG", ADVERSARIAL_PERTURBATIONS),
        ("MLP", "SmoothGrad", ADVERSARIAL_PERTURBATIONS),
    ]

    fixed_caption = fixed_levels_caption()

    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{XAI Degradation Summary (Adversarial Attacks): Rank Correlation Degradation Ratio = Clean $\div$ Perturbed@fixed level ({fixed_caption}). 
Higher ratio ($\uparrow$) indicates stronger degradation. 
Adversarial attacks (BIM, PGD, C\&W) evaluated only for MLP using IG and SmoothGrad.}}
\label{{tab:xai-degradation-adv}}
\{TABLE_SIZE}
\begin{{tabular}}{{ll|cccc}}
\toprule
\multicolumn{{2}}{{c|}}{{\textbf{{Dataset/Pert}}}} & 
\multicolumn{{4}}{{c}}{{\textbf{{MLP}}}} \\
\cmidrule(lr){{3-6}}
& & SHAP & LIME & IG & SG \\
\midrule
"""

    for idx, row in df.iterrows():
        dataset_name = row['Dataset']

        for pert_idx, (pert_short, pert_full) in enumerate([
            ('BIM', 'adversarial_bim'),
            ('PGD', 'adversarial_pgd'),
            ('C\\&W', 'adversarial_cw')
        ]):
            if pert_idx == 0:
                latex += rf"\multirow{{3}}{{*}}{{{dataset_name}}} & {pert_short}"
            else:
                latex += f" & {pert_short}"

            col = f"MLP_SHAP_{pert_full}"
            val = row.get(col, np.nan)
            if not pd.isna(val):
                if val >= 10:
                    val_str = f"{val:.1f}$\\uparrow$"
                else:
                    val_str = f"{val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"

            col = f"MLP_LIME_{pert_full}"
            val = row.get(col, np.nan)
            if not pd.isna(val):
                if val >= 10:
                    val_str = f"{val:.1f}$\\uparrow$"
                else:
                    val_str = f"{val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"

            col = f"MLP_IG_{pert_full}"
            val = row.get(col, np.nan)
            if not pd.isna(val):
                if val >= 10:
                    val_str = f"{val:.1f}$\\uparrow$"
                else:
                    val_str = f"{val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"

            col = f"MLP_SmoothGrad_{pert_full}"
            val = row.get(col, np.nan)
            if not pd.isna(val):
                if val >= 10:
                    val_str = f"{val:.1f}$\\uparrow$"
                else:
                    val_str = f"{val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"

            latex += r" \\" + "\n"

        latex += r"\multicolumn{2}{l|}{\textbf{Mean}}"
        for model, method, perts in xai_columns:
            vals = []
            for pert in perts:
                col = f"{model}_{method}_{pert}"
                if col in row and not pd.isna(row[col]):
                    vals.append(row[col])
            if len(vals) > 0:
                mean_val = np.mean(vals)
                if mean_val >= 10:
                    val_str = f"{mean_val:.1f}$\\uparrow$"
                else:
                    val_str = f"{mean_val:.2f}$\\uparrow$"
            else:
                val_str = "---"
            latex += f" & {val_str}"
        latex += r" \\" + "\n"

        if idx < len(df) - 1:
            latex += r"\cmidrule{1-6}" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table*}" + "\n"

    return latex


# TABLE 2 UQ GROWTH SUMMARY NO AVERAGING

def generate_uq_growth_table() -> Tuple[str, str]:
    """Generate UQ growth tables (natural + adversarial) with no averaging."""

    print("\n" + "="*80)
    print("GENERATING TABLE 2: UNCERTAINTY GROWTH SUMMARY")
    print("="*80)

    data = []

    for display_name, uci_id, dataset_name in DATASETS:
        print(f"\nProcessing {display_name}...")

        uq_results = load_uq_results(dataset_name, uci_id)
        if not uq_results:
            continue

        df = pd.DataFrame(uq_results)
        row = {'Dataset': display_name}

        for model in MODELS:
            model_results = df[df['model'] == model]

            if len(model_results) == 0:
                continue

            # NATURAL PERTURBATIONS ALL MODELS FIXED LEVELS
            for pert in NATURAL_PERTURBATIONS:
                target_level = FIXED_PERT_LEVELS.get(pert, None)
                pert_results = model_results[model_results['perturbation'] == pert]

                if len(pert_results) > 0:
                    # EPISTEMIC
                    clean_epi = pert_results[pert_results['level'] == 0.0]['epistemic'].values
                    pert_epi_vals = pert_results[np.isclose(pert_results['level'], target_level)]['epistemic'].values if target_level is not None else []
                    pert_epi = pert_epi_vals[0] if len(pert_epi_vals) > 0 else np.nan

                    if len(clean_epi) > 0 and not np.isnan(pert_epi) and clean_epi[0] > 0:
                        ratio_epi, dir_epi = compute_degradation_ratio(clean_epi[0], pert_epi, 'uncertainty')
                        row[f"{model}_epi_{pert}"] = ratio_epi
                        row[f"{model}_epi_{pert}_dir"] = dir_epi
                    else:
                        row[f"{model}_epi_{pert}"] = np.nan
                        row[f"{model}_epi_{pert}_dir"] = "$-$"

                    # ALEATORIC
                    clean_ale = pert_results[pert_results['level'] == 0.0]['aleatoric'].values
                    pert_ale_vals = pert_results[np.isclose(pert_results['level'], target_level)]['aleatoric'].values if target_level is not None else []
                    pert_ale = pert_ale_vals[0] if len(pert_ale_vals) > 0 else np.nan

                    if len(clean_ale) > 0 and not np.isnan(pert_ale) and clean_ale[0] > 0:
                        ratio_ale, dir_ale = compute_degradation_ratio(clean_ale[0], pert_ale, 'uncertainty')
                        row[f"{model}_ale_{pert}"] = ratio_ale
                        row[f"{model}_ale_{pert}_dir"] = dir_ale
                    else:
                        row[f"{model}_ale_{pert}"] = np.nan
                        row[f"{model}_ale_{pert}_dir"] = "$-$"

                    # CONFIDENCE LOWER IS WORSE SO WE LOOK FOR MINIMUM
                    clean_conf = pert_results[pert_results['level'] == 0.0]['confidence'].values
                    pert_conf_vals = pert_results[np.isclose(pert_results['level'], target_level)]['confidence'].values if target_level is not None else []
                    pert_conf = pert_conf_vals[0] if len(pert_conf_vals) > 0 else np.nan

                    if len(clean_conf) > 0 and not np.isnan(pert_conf) and pert_conf > 0:
                        # For confidence: clean/perturbed (degradation, like rank_correlation)
                        ratio_conf, _ = compute_degradation_ratio(clean_conf[0], pert_conf, 'rank_correlation')
                        row[f"{model}_conf_{pert}"] = ratio_conf
                    else:
                        row[f"{model}_conf_{pert}"] = np.nan
                else:
                    row[f"{model}_epi_{pert}"] = np.nan
                    row[f"{model}_epi_{pert}_dir"] = "$-$"
                    row[f"{model}_ale_{pert}"] = np.nan
                    row[f"{model}_ale_{pert}_dir"] = "$-$"
                    row[f"{model}_conf_{pert}"] = np.nan

            # ADVERSARIAL PERTURBATIONS MLP ONLY FIXED LEVELS
            if model == 'MLP':
                # MAP SHORT NAMES FROM UQ DATA TO FULL NAMES FOR COLUMN KEYS
                adv_map = {'bim': 'adversarial_bim', 'pgd': 'adversarial_pgd', 'cw': 'adversarial_cw'}
                for pert_short in ADVERSARIAL_PERTURBATIONS_UQ:
                    pert_full = adv_map[pert_short]
                    target_level = FIXED_ADV_LEVELS_UQ.get(pert_short, None)
                    pert_results = model_results[model_results['perturbation'] == pert_short]

                    if len(pert_results) > 0:
                        # EPISTEMIC
                        clean_epi = pert_results[pert_results['level'] == 0.0]['epistemic'].values
                        pert_epi_vals = pert_results[np.isclose(pert_results['level'], target_level)]['epistemic'].values if target_level is not None else []
                        pert_epi = pert_epi_vals[0] if len(pert_epi_vals) > 0 else np.nan

                        if len(clean_epi) > 0 and not np.isnan(pert_epi) and clean_epi[0] > 0:
                            ratio_epi, dir_epi = compute_degradation_ratio(clean_epi[0], pert_epi, 'uncertainty')
                            row[f"{model}_epi_{pert_full}"] = ratio_epi
                            row[f"{model}_epi_{pert_full}_dir"] = dir_epi
                        else:
                            row[f"{model}_epi_{pert_full}"] = np.nan
                            row[f"{model}_epi_{pert_full}_dir"] = "$-$"

                        # ALEATORIC
                        clean_ale = pert_results[pert_results['level'] == 0.0]['aleatoric'].values
                        pert_ale_vals = pert_results[np.isclose(pert_results['level'], target_level)]['aleatoric'].values if target_level is not None else []
                        pert_ale = pert_ale_vals[0] if len(pert_ale_vals) > 0 else np.nan

                        if len(clean_ale) > 0 and not np.isnan(pert_ale) and clean_ale[0] > 0:
                            ratio_ale, dir_ale = compute_degradation_ratio(clean_ale[0], pert_ale, 'uncertainty')
                            row[f"{model}_ale_{pert_full}"] = ratio_ale
                            row[f"{model}_ale_{pert_full}_dir"] = dir_ale
                        else:
                            row[f"{model}_ale_{pert_full}"] = np.nan
                            row[f"{model}_ale_{pert_full}_dir"] = "$-$"

                        # CONFIDENCE
                        clean_conf = pert_results[pert_results['level'] == 0.0]['confidence'].values
                        pert_conf_vals = pert_results[np.isclose(pert_results['level'], target_level)]['confidence'].values if target_level is not None else []
                        pert_conf = pert_conf_vals[0] if len(pert_conf_vals) > 0 else np.nan

                        if len(clean_conf) > 0 and not np.isnan(pert_conf) and pert_conf > 0:
                            ratio_conf, _ = compute_degradation_ratio(clean_conf[0], pert_conf, 'rank_correlation')
                            row[f"{model}_conf_{pert_full}"] = ratio_conf
                        else:
                            row[f"{model}_conf_{pert_full}"] = np.nan
                    else:
                        row[f"{model}_epi_{pert_full}"] = np.nan
                        row[f"{model}_epi_{pert_full}_dir"] = "$-$"
                        row[f"{model}_ale_{pert_full}"] = np.nan
                        row[f"{model}_ale_{pert_full}_dir"] = "$-$"
                        row[f"{model}_conf_{pert_full}"] = np.nan

        data.append(row)

    df_summary = pd.DataFrame(data)
    natural_latex = generate_uq_latex_natural(df_summary)
    adversarial_latex = generate_uq_latex_adversarial(df_summary)

    return natural_latex, adversarial_latex


def generate_uq_latex_natural(df: pd.DataFrame) -> str:
    """Generate LaTeX code for UQ growth table (natural perturbations only)."""

    perturbations = [
        ("Gaussian", "gaussian", False),
        ("Missing", "missing", False),
        ("Permutation", "permutation", False),
    ]
    rows_per_dataset = len(perturbations)
    metric_labels = {'epi': 'Epi', 'ale': 'Ale', 'conf': 'Conf'}
    included_models = [(m, UQ_MODEL_METRICS.get(m, DEFAULT_UQ_METRICS)) for m in MODELS if len(UQ_MODEL_METRICS.get(m, DEFAULT_UQ_METRICS)) > 0]
    total_metric_cols = sum(len(metrics) for _, metrics in included_models)
    total_columns = 2 + total_metric_cols

    def format_cell(row: pd.Series, model: str, metric_short: str, pert_key: str, is_adv: bool) -> str:
        col = f"{model}_{metric_short}_{pert_key}"
        val = row.get(col, np.nan)

        if pd.isna(val):
            return "---"

        fmt = "{:.1f}" if val >= 10 else "{:.2f}"

        if metric_short in ['epi', 'ale']:
            dir_col = f"{model}_{metric_short}_{pert_key}_dir"
            direction = row.get(dir_col, "$\\uparrow$")
            return f"{fmt.format(val)}{direction}"

        return f"{fmt.format(val)}$\\uparrow$"

    fixed_caption = fixed_levels_caption()

    col_spec_segments = []
    for model, metrics in included_models:
        col_spec_segments.append("".join(["c"] * len(metrics)))
    col_spec = "ll|" + "|".join(col_spec_segments)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Uncertainty Growth Ratios (Natural Perturbations at fixed levels: {fixed_caption}). Epi/Ale = Perturbed $\div$ Clean ($\uparrow$ growth); Conf = Clean $\div$ Perturbed ($\uparrow$ degradation). Columns grouped by model with Epistemic, Aleatoric, and Confidence.}}",
        r"\label{tab:uq-growth-natural}",
        f"\\{TABLE_SIZE}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_cells = [r"\multicolumn{2}{c|}{\textbf{Dataset / Perturbation}}"]
    for idx, (model, metrics) in enumerate(included_models):
        right_bar = "|" if idx < len(included_models) - 1 else ""
        header_cells.append(rf"\multicolumn{{{len(metrics)}}}{{c{right_bar}}}{{\textbf{{{model}}}}}")
    latex_lines.append(" & ".join(header_cells) + r" \\")

    cmidrules = []
    start_col = 3
    for _, metrics in included_models:
        end_col = start_col + len(metrics) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    latex_lines.append(" ".join(cmidrules))

    metric_header = [r"\textbf{Dataset}", r"\textbf{Pert.}"]
    for _, metrics in included_models:
        metric_header.extend([metric_labels[m] for m in metrics])
    latex_lines.append(" & ".join(metric_header) + r" \\")
    latex_lines.append(r"\midrule")

    for idx, row in df.iterrows():
        dataset_name = row['Dataset']
        for pert_idx, (pert_label, pert_key, is_adv) in enumerate(perturbations):
            dataset_cell = rf"\multirow{{{rows_per_dataset}}}{{*}}{{{dataset_name}}}" if pert_idx == 0 else ""
            line_parts = [dataset_cell, pert_label]

            for model, metrics in included_models:
                for metric_short in metrics:
                    line_parts.append(format_cell(row, model, metric_short, pert_key, is_adv))

            latex_lines.append(" & ".join(line_parts) + r" \\")

        mean_parts = [r"\multicolumn{2}{l|}{\textbf{Mean}}"]
        for model, metrics in included_models:
            for metric_short in metrics:
                vals = []
                dirs = []
                for _, pert_key, _ in perturbations:
                    col = f"{model}_{metric_short}_{pert_key}"
                    if col in df.columns and not pd.isna(row.get(col, np.nan)):
                        vals.append(row[col])
                        dir_col = f"{model}_{metric_short}_{pert_key}_dir"
                        if dir_col in row:
                            dirs.append(row[dir_col])

                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    fmt = "{:.1f}" if mean_val >= 10 else "{:.2f}"
                    if metric_short in ['epi', 'ale']:
                        from collections import Counter
                        mean_dir = Counter(dirs).most_common(1)[0][0] if len(dirs) > 0 else "$\\uparrow$"
                        mean_parts.append(f"{fmt.format(mean_val)}{mean_dir}")
                    else:
                        mean_parts.append(f"{fmt.format(mean_val)}$\\uparrow$")
                else:
                    mean_parts.append("---")

        latex_lines.append(" & ".join(mean_parts) + r" \\")

        if idx < len(df) - 1:
            latex_lines.append(rf"\cmidrule{{1-{total_columns}}}")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(latex_lines)


def generate_uq_latex_adversarial(df: pd.DataFrame) -> str:
    """Generate LaTeX code for UQ growth table (adversarial perturbations, MLP only)."""

    perturbations = [
        ("BIM", "adversarial_bim", True),
        ("PGD", "adversarial_pgd", True),
        ("C\\&W", "adversarial_cw", True),
    ]
    rows_per_dataset = len(perturbations)
    metric_labels = {'epi': 'Epi', 'ale': 'Ale', 'conf': 'Conf'}
    included_models = [("MLP", UQ_MODEL_METRICS.get("MLP", DEFAULT_UQ_METRICS))]
    total_metric_cols = sum(len(metrics) for _, metrics in included_models)
    total_columns = 2 + total_metric_cols

    def format_cell(row: pd.Series, model: str, metric_short: str, pert_key: str, is_adv: bool) -> str:
        col = f"{model}_{metric_short}_{pert_key}"
        val = row.get(col, np.nan)

        if pd.isna(val):
            return "---"

        fmt = "{:.1f}" if val >= 10 else "{:.2f}"

        if metric_short in ['epi', 'ale']:
            dir_col = f"{model}_{metric_short}_{pert_key}_dir"
            direction = row.get(dir_col, "$\\uparrow$")
            return f"{fmt.format(val)}{direction}"

        return f"{fmt.format(val)}$\\uparrow$"

    fixed_caption = fixed_levels_caption()

    col_spec_segments = []
    for model, metrics in included_models:
        col_spec_segments.append("".join(["c"] * len(metrics)))
    col_spec = "ll|" + "|".join(col_spec_segments)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Uncertainty Growth Ratios (Adversarial Attacks at fixed levels: {fixed_caption}). Epi/Ale = Perturbed $\div$ Clean ($\uparrow$ growth); Conf = Clean $\div$ Perturbed ($\uparrow$ degradation). Adversarial BIM/PGD/C\&W evaluated only for the MLP model.}}",
        r"\label{tab:uq-growth-adv}",
        f"\\{TABLE_SIZE}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_cells = [r"\multicolumn{2}{c|}{\textbf{Dataset / Perturbation}}"]
    for idx, (model, metrics) in enumerate(included_models):
        right_bar = "|" if idx < len(included_models) - 1 else ""
        header_cells.append(rf"\multicolumn{{{len(metrics)}}}{{c{right_bar}}}{{\textbf{{{model}}}}}")
    latex_lines.append(" & ".join(header_cells) + r" \\")

    cmidrules = []
    start_col = 3
    for _, metrics in included_models:
        end_col = start_col + len(metrics) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    latex_lines.append(" ".join(cmidrules))

    metric_header = [r"\textbf{Dataset}", r"\textbf{Pert.}"]
    for _, metrics in included_models:
        metric_header.extend([metric_labels[m] for m in metrics])
    latex_lines.append(" & ".join(metric_header) + r" \\")
    latex_lines.append(r"\midrule")

    for idx, row in df.iterrows():
        dataset_name = row['Dataset']
        for pert_idx, (pert_label, pert_key, is_adv) in enumerate(perturbations):
            dataset_cell = rf"\multirow{{{rows_per_dataset}}}{{*}}{{{dataset_name}}}" if pert_idx == 0 else ""
            line_parts = [dataset_cell, pert_label]

            for model, metrics in included_models:
                for metric_short in metrics:
                    line_parts.append(format_cell(row, model, metric_short, pert_key, is_adv))

            latex_lines.append(" & ".join(line_parts) + r" \\")

        mean_parts = [r"\multicolumn{2}{l|}{\textbf{Mean}}"]
        for model, metrics in included_models:
            for metric_short in metrics:
                vals = []
                dirs = []
                for _, pert_key, _ in perturbations:
                    col = f"{model}_{metric_short}_{pert_key}"
                    if col in df.columns and not pd.isna(row.get(col, np.nan)):
                        vals.append(row[col])
                        dir_col = f"{model}_{metric_short}_{pert_key}_dir"
                        if dir_col in row:
                            dirs.append(row[dir_col])

                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    fmt = "{:.1f}" if mean_val >= 10 else "{:.2f}"
                    if metric_short in ['epi', 'ale']:
                        from collections import Counter
                        mean_dir = Counter(dirs).most_common(1)[0][0] if len(dirs) > 0 else "$\\uparrow$"
                        mean_parts.append(f"{fmt.format(mean_val)}{mean_dir}")
                    else:
                        mean_parts.append(f"{fmt.format(mean_val)}$\\uparrow$")
                else:
                    mean_parts.append("---")

        latex_lines.append(" & ".join(mean_parts) + r" \\")

        if idx < len(df) - 1:
            latex_lines.append(rf"\cmidrule{{1-{total_columns}}}")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(latex_lines)


# TABLE 3 CORRELATION SUMMARY EPISTEMIC XAI STABILITY

def generate_correlation_table() -> Tuple[str, str, pd.DataFrame]:
    """Generate correlation tables (natural + adversarial)."""
    
    print("\n" + "="*80)
    print("GENERATING TABLE 3: CORRELATION SUMMARY")
    print("="*80)
    
    data = []
    
    for display_name, uci_id, dataset_name in DATASETS:
        print(f"\nProcessing {display_name}...")
        
        xai_results = load_xai_results(dataset_name, uci_id)
        uq_results = load_uq_results(dataset_name, uci_id)
        
        if not xai_results or not uq_results:
            continue
        
        df_xai = pd.DataFrame(xai_results)
        df_uq = pd.DataFrame(uq_results)
        
        row = {'Dataset': display_name}
        
        # Get RF epistemic for fallback (LightGBM/CatBoost)
        rf_epistemic = {}
        rf_uq = df_uq[df_uq['model'] == 'RF']
        for pert in NATURAL_PERTURBATIONS + ['bim', 'pgd', 'cw']:
            pert_data = rf_uq[rf_uq['perturbation'] == pert]
            if len(pert_data) > 0:
                rf_epistemic[pert] = pert_data[['level', 'epistemic']].copy()
        
        for model in MODELS:
            xai_model = df_xai[df_xai['model'] == model]
            uq_model = df_uq[df_uq['model'] == model]
            
            if len(xai_model) == 0:
                continue
            
            # NATURAL PERTURBATIONS USE ALL LEVELS NOT JUST FIXED
            for pert in NATURAL_PERTURBATIONS:
                # Get epistemic (use RF fallback for LightGBM/CatBoost)
                if model in ['LightGBM', 'CatBoost']:
                    epistemic_data = rf_epistemic.get(pert, pd.DataFrame())
                else:
                    pert_uq = uq_model[uq_model['perturbation'] == pert]
                    epistemic_data = pert_uq[['level', 'epistemic']].copy() if len(pert_uq) > 0 else pd.DataFrame()
                
                if len(epistemic_data) == 0:
                    continue
                
                # SHAP ALL LEVELS
                shap_results = xai_model[xai_model['explainer'].str.contains('SHAP', na=False)]
                shap_pert = shap_results[shap_results['perturbation'] == pert]
                shap_subset = shap_pert[['level', 'rank_correlation']]
                
                merged_shap = pd.merge(epistemic_data, shap_subset, on='level', how='inner')
                if len(merged_shap) >= 3:  # Need at least 3 points
                    if merged_shap['epistemic'].std() > 0 and merged_shap['rank_correlation'].std() > 0:
                        rho, pval = spearmanr(merged_shap['epistemic'], merged_shap['rank_correlation'])
                        row[f"{model}_SHAP_{pert}_rho"] = rho
                        row[f"{model}_SHAP_{pert}_pval"] = pval
                    else:
                        row[f"{model}_SHAP_{pert}_rho"] = np.nan
                        row[f"{model}_SHAP_{pert}_pval"] = np.nan
                else:
                    row[f"{model}_SHAP_{pert}_rho"] = np.nan
                    row[f"{model}_SHAP_{pert}_pval"] = np.nan
                
                # LIME ALL LEVELS
                lime_results = xai_model[xai_model['explainer'] == 'LIME']
                lime_pert = lime_results[lime_results['perturbation'] == pert]
                lime_subset = lime_pert[['level', 'rank_correlation']]
                
                merged_lime = pd.merge(epistemic_data, lime_subset, on='level', how='inner')
                if len(merged_lime) >= 3:
                    if merged_lime['epistemic'].std() > 0 and merged_lime['rank_correlation'].std() > 0:
                        rho, pval = spearmanr(merged_lime['epistemic'], merged_lime['rank_correlation'])
                        row[f"{model}_LIME_{pert}_rho"] = rho
                        row[f"{model}_LIME_{pert}_pval"] = pval
                    else:
                        row[f"{model}_LIME_{pert}_rho"] = np.nan
                        row[f"{model}_LIME_{pert}_pval"] = np.nan
                else:
                    row[f"{model}_LIME_{pert}_rho"] = np.nan
                    row[f"{model}_LIME_{pert}_pval"] = np.nan
                
                # IG AND SMOOTHGRAD FOR MLP
                if model == 'MLP':
                    # INTGRAD ALL LEVELS
                    ig_results = xai_model[xai_model['explainer'] == 'IntGrad']
                    ig_pert = ig_results[ig_results['perturbation'] == pert]
                    ig_subset = ig_pert[['level', 'rank_correlation']]
                    
                    merged_ig = pd.merge(epistemic_data, ig_subset, on='level', how='inner')
                    if len(merged_ig) >= 3:
                        if merged_ig['epistemic'].std() > 0 and merged_ig['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_ig['epistemic'], merged_ig['rank_correlation'])
                            row[f"{model}_IG_{pert}_rho"] = rho
                            row[f"{model}_IG_{pert}_pval"] = pval
                        else:
                            row[f"{model}_IG_{pert}_rho"] = np.nan
                            row[f"{model}_IG_{pert}_pval"] = np.nan
                    else:
                        row[f"{model}_IG_{pert}_rho"] = np.nan
                        row[f"{model}_IG_{pert}_pval"] = np.nan
                    
                    # SMOOTHGRAD ALL LEVELS
                    sg_results = xai_model[xai_model['explainer'] == 'SmoothGrad']
                    sg_pert = sg_results[sg_results['perturbation'] == pert]
                    sg_subset = sg_pert[['level', 'rank_correlation']]
                    
                    merged_sg = pd.merge(epistemic_data, sg_subset, on='level', how='inner')
                    if len(merged_sg) >= 3:
                        if merged_sg['epistemic'].std() > 0 and merged_sg['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_sg['epistemic'], merged_sg['rank_correlation'])
                            row[f"{model}_SmoothGrad_{pert}_rho"] = rho
                            row[f"{model}_SmoothGrad_{pert}_pval"] = pval
                        else:
                            row[f"{model}_SmoothGrad_{pert}_rho"] = np.nan
                            row[f"{model}_SmoothGrad_{pert}_pval"] = np.nan
                    else:
                        row[f"{model}_SmoothGrad_{pert}_rho"] = np.nan
                        row[f"{model}_SmoothGrad_{pert}_pval"] = np.nan
            
            # ADVERSARIAL PERTURBATIONS MLP ONLY USE ALL LEVELS
            if model == 'MLP':
                adv_map = {'bim': 'adversarial_bim', 'pgd': 'adversarial_pgd', 'cw': 'adversarial_cw'}
                for pert_short, pert_full in adv_map.items():
                    # EPISTEMIC
                    pert_uq = uq_model[uq_model['perturbation'] == pert_short]
                    epistemic_data = pert_uq[['level', 'epistemic']].copy() if len(pert_uq) > 0 else pd.DataFrame()
                    
                    if len(epistemic_data) == 0:
                        continue
                    
                    # SHAP ALL LEVELS
                    shap_results = xai_model[xai_model['explainer'].str.contains('SHAP', na=False)]
                    shap_pert = shap_results[shap_results['perturbation'] == pert_full]
                    shap_subset = shap_pert[['level', 'rank_correlation']]
                    
                    merged_shap = pd.merge(epistemic_data, shap_subset, on='level', how='inner')
                    if len(merged_shap) >= 3:
                        if merged_shap['epistemic'].std() > 0 and merged_shap['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_shap['epistemic'], merged_shap['rank_correlation'])
                            row[f"{model}_SHAP_{pert_full}_rho"] = rho
                            row[f"{model}_SHAP_{pert_full}_pval"] = pval
                        else:
                            row[f"{model}_SHAP_{pert_full}_rho"] = np.nan
                            row[f"{model}_SHAP_{pert_full}_pval"] = np.nan
                    else:
                        row[f"{model}_SHAP_{pert_full}_rho"] = np.nan
                        row[f"{model}_SHAP_{pert_full}_pval"] = np.nan
                    
                    # LIME ALL LEVELS
                    lime_results = xai_model[xai_model['explainer'] == 'LIME']
                    lime_pert = lime_results[lime_results['perturbation'] == pert_full]
                    lime_subset = lime_pert[['level', 'rank_correlation']]
                    
                    merged_lime = pd.merge(epistemic_data, lime_subset, on='level', how='inner')
                    if len(merged_lime) >= 3:
                        if merged_lime['epistemic'].std() > 0 and merged_lime['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_lime['epistemic'], merged_lime['rank_correlation'])
                            row[f"{model}_LIME_{pert_full}_rho"] = rho
                            row[f"{model}_LIME_{pert_full}_pval"] = pval
                        else:
                            row[f"{model}_LIME_{pert_full}_rho"] = np.nan
                            row[f"{model}_LIME_{pert_full}_pval"] = np.nan
                    else:
                        row[f"{model}_LIME_{pert_full}_rho"] = np.nan
                        row[f"{model}_LIME_{pert_full}_pval"] = np.nan
                    
                    # INTGRAD ALL LEVELS
                    ig_results = xai_model[xai_model['explainer'] == 'IntGrad']
                    ig_pert = ig_results[ig_results['perturbation'] == pert_full]
                    ig_subset = ig_pert[['level', 'rank_correlation']]
                    
                    merged_ig = pd.merge(epistemic_data, ig_subset, on='level', how='inner')
                    if len(merged_ig) >= 3:
                        if merged_ig['epistemic'].std() > 0 and merged_ig['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_ig['epistemic'], merged_ig['rank_correlation'])
                            row[f"{model}_IG_{pert_full}_rho"] = rho
                            row[f"{model}_IG_{pert_full}_pval"] = pval
                        else:
                            row[f"{model}_IG_{pert_full}_rho"] = np.nan
                            row[f"{model}_IG_{pert_full}_pval"] = np.nan
                    else:
                        row[f"{model}_IG_{pert_full}_rho"] = np.nan
                        row[f"{model}_IG_{pert_full}_pval"] = np.nan
                    
                    # SMOOTHGRAD ALL LEVELS
                    sg_results = xai_model[xai_model['explainer'] == 'SmoothGrad']
                    sg_pert = sg_results[sg_results['perturbation'] == pert_full]
                    sg_subset = sg_pert[['level', 'rank_correlation']]
                    
                    merged_sg = pd.merge(epistemic_data, sg_subset, on='level', how='inner')
                    if len(merged_sg) >= 3:
                        if merged_sg['epistemic'].std() > 0 and merged_sg['rank_correlation'].std() > 0:
                            rho, pval = spearmanr(merged_sg['epistemic'], merged_sg['rank_correlation'])
                            row[f"{model}_SmoothGrad_{pert_full}_rho"] = rho
                            row[f"{model}_SmoothGrad_{pert_full}_pval"] = pval
                        else:
                            row[f"{model}_SmoothGrad_{pert_full}_rho"] = np.nan
                            row[f"{model}_SmoothGrad_{pert_full}_pval"] = np.nan
                    else:
                        row[f"{model}_SmoothGrad_{pert_full}_rho"] = np.nan
                        row[f"{model}_SmoothGrad_{pert_full}_pval"] = np.nan
        
        data.append(row)
    
    df_summary = pd.DataFrame(data)
    latex_natural = generate_correlation_latex_natural(df_summary)
    latex_adversarial = generate_correlation_latex_adversarial(df_summary)
    
    return latex_natural, latex_adversarial, df_summary


def generate_correlation_latex_natural(df: pd.DataFrame) -> str:
    """Generate LaTeX for the correlation table (natural perturbations)."""
    
    fixed_caption = fixed_levels_caption()
    
    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{Correlation Summary (Natural Perturbations): Spearman $\rho$ between Epistemic Uncertainty and XAI Stability (Rank Correlation) at fixed levels ({fixed_caption}). 
More negative ($\downarrow$) indicates higher epistemic uncertainty aligns with lower explanation stability. 
For LightGBM/CatBoost, RF epistemic used (no native epistemic).}}
\label{{tab:correlation-natural}}
\{TABLE_SIZE}
\begin{{tabular}}{{ll|cc|cc|cccc|cc|cc}}
\toprule
\multicolumn{{2}}{{c|}}{{\textbf{{Dataset/Pert}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{Logistic}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{RF}}}} & 
\multicolumn{{4}}{{c|}}{{\textbf{{MLP}}}} & 
\multicolumn{{2}}{{c|}}{{\textbf{{LightGBM}}}} & 
\multicolumn{{2}}{{c}}{{\textbf{{CatBoost}}}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}} \cmidrule(lr){{7-10}} \cmidrule(lr){{11-12}} \cmidrule(lr){{13-14}}
& & SHAP & LIME & SHAP & LIME & SHAP & LIME & IG & SG & SHAP & LIME & SHAP & LIME \\
\midrule
"""
    
    def format_corr(row, model, method, pert):
        rho_col = f"{model}_{method}_{pert}_rho"
        
        if rho_col not in row or pd.isna(row[rho_col]):
            return "---"
        
        rho = row[rho_col]
        return f"{rho:.2f}"
    
    for idx, row in df.iterrows():
        dataset_name = row['Dataset']
        
        for pert_idx, (pert_short, pert_full) in enumerate([
            ('Gaussian', 'gaussian'),
            ('Missing', 'missing'),
            ('Permutation', 'permutation')
        ]):
            if pert_idx == 0:
                latex += rf"\multirow{{3}}{{*}}{{{dataset_name}}} & {pert_short}"
            else:
                latex += f" & {pert_short}"
            
            # LOGISTIC SHAP LIME
            for method in ['SHAP', 'LIME']:
                latex += f" & {format_corr(row, 'Logistic', method, pert_full)}"
            
            # RF SHAP LIME
            for method in ['SHAP', 'LIME']:
                latex += f" & {format_corr(row, 'RF', method, pert_full)}"
            
            # MLP SHAP LIME IG SG
            for method in ['SHAP', 'LIME', 'IG', 'SmoothGrad']:
                latex += f" & {format_corr(row, 'MLP', method, pert_full)}"
            
            # LIGHTGBM SHAP LIME
            for method in ['SHAP', 'LIME']:
                latex += f" & {format_corr(row, 'LightGBM', method, pert_full)}"
            
            # CATBOOST SHAP LIME
            for method in ['SHAP', 'LIME']:
                latex += f" & {format_corr(row, 'CatBoost', method, pert_full)}"
            
            latex += r" \\" + "\n"
        
        # MEAN ROW PER DATASET
        latex += r"\multicolumn{2}{l|}{\textbf{Mean}}"
        
        xai_columns = [
            ('Logistic', 'SHAP'), ('Logistic', 'LIME'),
            ('RF', 'SHAP'), ('RF', 'LIME'),
            ('MLP', 'SHAP'), ('MLP', 'LIME'), ('MLP', 'IG'), ('MLP', 'SmoothGrad'),
            ('LightGBM', 'SHAP'), ('LightGBM', 'LIME'),
            ('CatBoost', 'SHAP'), ('CatBoost', 'LIME'),
        ]
        
        for model, method in xai_columns:
            vals = []
            for pert in NATURAL_PERTURBATIONS:
                col = f"{model}_{method}_{pert}_rho"
                if col in row and not pd.isna(row[col]):
                    vals.append(row[col])
            
            if len(vals) > 0:
                mean_val = np.mean(vals)
                val_str = f"{mean_val:.2f}"
            else:
                val_str = "---"
            
            latex += f" & {val_str}"
        
        latex += r" \\" + "\n"
        
        if idx < len(df) - 1:
            latex += r"\cmidrule{1-14}" + "\n"
    
    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table*}" + "\n"
    
    return latex


def generate_correlation_latex_adversarial(df: pd.DataFrame) -> str:
    """Generate LaTeX for the correlation table (adversarial perturbations)."""
    
    fixed_caption = fixed_levels_caption()
    
    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{Correlation Summary (Adversarial Attacks): Spearman $\rho$ between Epistemic Uncertainty and XAI Stability at fixed levels ({fixed_caption}). 
More negative ($\downarrow$) indicates higher epistemic uncertainty aligns with lower explanation stability. 
Adversarial attacks evaluated only for MLP.}}
\label{{tab:correlation-adv}}
\{TABLE_SIZE}
\begin{{tabular}}{{ll|cccc}}
\toprule
\multicolumn{{2}}{{c|}}{{\textbf{{Dataset/Pert}}}} & 
\multicolumn{{4}}{{c}}{{\textbf{{MLP}}}} \\
\cmidrule(lr){{3-6}}
& & SHAP & LIME & IG & SG \\
\midrule
"""
    
    def format_corr(row, model, method, pert):
        rho_col = f"{model}_{method}_{pert}_rho"
        
        if rho_col not in row or pd.isna(row[rho_col]):
            return "---"
        
        rho = row[rho_col]
        return f"{rho:.2f}"
    
    for idx, row in df.iterrows():
        dataset_name = row['Dataset']
        
        for pert_idx, (pert_short, pert_full) in enumerate([
            ('BIM', 'adversarial_bim'),
            ('PGD', 'adversarial_pgd'),
            ('C\\&W', 'adversarial_cw')
        ]):
            if pert_idx == 0:
                latex += rf"\multirow{{3}}{{*}}{{{dataset_name}}} & {pert_short}"
            else:
                latex += f" & {pert_short}"
            
            # MLP SHAP LIME IG SG
            for method in ['SHAP', 'LIME', 'IG', 'SmoothGrad']:
                latex += f" & {format_corr(row, 'MLP', method, pert_full)}"
            
            latex += r" \\" + "\n"
        
        # MEAN ROW PER DATASET
        latex += r"\multicolumn{2}{l|}{\textbf{Mean}}"
        
        for method in ['SHAP', 'LIME', 'IG', 'SmoothGrad']:
            vals = []
            for pert in ['adversarial_bim', 'adversarial_pgd', 'adversarial_cw']:
                col = f"MLP_{method}_{pert}_rho"
                if col in row and not pd.isna(row[col]):
                    vals.append(row[col])
            
            if len(vals) > 0:
                mean_val = np.mean(vals)
                val_str = f"{mean_val:.2f}"
            else:
                val_str = "---"
            
            latex += f" & {val_str}"
        
        latex += r" \\" + "\n"
        
        if idx < len(df) - 1:
            latex += r"\cmidrule{1-6}" + "\n"
    
    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table*}" + "\n"
    
    return latex

# PLOT CORRELATION HEATMAPS NATURAL ADVERSARIAL

def _format_col_labels(col_defs: List[Tuple[str, str, str]]) -> List[str]:
    return [label for _, _, label in col_defs]

def _collect_model_groups(col_defs: List[Tuple[str, str, str]]) -> List[Tuple[str, str, int, int]]:
    if not col_defs:
        return []
    groups = []
    current_model = col_defs[0][0]
    start = 0
    for idx, (model, _, _) in enumerate(col_defs):
        if model != current_model:
            groups.append((current_model, MODEL_LABELS.get(current_model, current_model), start, idx))
            current_model = model
            start = idx
    groups.append((current_model, MODEL_LABELS.get(current_model, current_model), start, len(col_defs)))
    return groups

def _build_corr_matrices_natural(df_summary: pd.DataFrame):
    perts = [
        ("Gaussian", "gaussian"),
        ("Missing", "missing"),
        ("Permutation", "permutation"),
    ]
    pert_labels = [label for label, _ in perts]
    col_defs = [
        ("Logistic", "SHAP", "LR-SHAP"),
        ("Logistic", "LIME", "LR-LIME"),
        ("RF", "SHAP", "RF-SHAP"),
        ("RF", "LIME", "RF-LIME"),
        ("MLP", "SHAP", "MLP-SHAP"),
        ("MLP", "LIME", "MLP-LIME"),
        ("MLP", "IG", "MLP-IG"),
        ("MLP", "SmoothGrad", "MLP-SIG"),
        ("LightGBM", "SHAP", "LGBM-SHAP"),
        ("LightGBM", "LIME", "LGBM-LIME"),
        ("CatBoost", "SHAP", "CB-SHAP"),
        ("CatBoost", "LIME", "CB-LIME"),
    ]
    row_labels = []
    data_rows = []
    mean_rows = []
    dataset_boundaries = []
    datasets_present = set(df_summary["Dataset"].tolist())
    dataset_names = []

    for display_name, _, _ in DATASETS:
        if display_name not in datasets_present:
            continue
        dataset_names.append(display_name)
        row = df_summary[df_summary["Dataset"] == display_name].iloc[0]
        for pert_label, pert_key in perts:
            row_labels.append(f"{display_name} | {pert_label}")
            vals = []
            for model, method, _ in col_defs:
                col = f"{model}_{method}_{pert_key}_rho"
                val = row[col] if col in row else np.nan
                vals.append(val if pd.notna(val) else np.nan)
            data_rows.append(vals)
        dataset_boundaries.append(len(row_labels))

        mean_vals = []
        for model, method, _ in col_defs:
            vals = []
            for _, pert_key in perts:
                col = f"{model}_{method}_{pert_key}_rho"
                if col in row and pd.notna(row[col]):
                    vals.append(row[col])
            mean_vals.append(float(np.mean(vals)) if vals else np.nan)
        mean_rows.append(mean_vals)

    data = np.array(data_rows, dtype=float)
    mean_data = np.array(mean_rows, dtype=float)
    col_labels = _format_col_labels(col_defs)
    model_groups = _collect_model_groups(col_defs)
    mean_row_labels = dataset_names

    return (data, mean_data, mean_row_labels, col_labels,
            dataset_names, pert_labels, model_groups)

def _build_corr_matrices_adversarial(df_summary: pd.DataFrame):
    perts = [
        ("BIM", "adversarial_bim"),
        ("PGD", "adversarial_pgd"),
        ("C&W", "adversarial_cw"),
    ]
    pert_labels = [label for label, _ in perts]
    col_defs = [
        ("MLP", "SHAP", "MLP-SHAP"),
        ("MLP", "LIME", "MLP-LIME"),
        ("MLP", "IG", "MLP-IG"),
        ("MLP", "SmoothGrad", "MLP-SIG"),
    ]
    row_labels = []
    data_rows = []
    mean_rows = []
    dataset_boundaries = []
    datasets_present = set(df_summary["Dataset"].tolist())
    dataset_names = []

    for display_name, _, _ in DATASETS:
        if display_name not in datasets_present:
            continue
        dataset_names.append(display_name)
        row = df_summary[df_summary["Dataset"] == display_name].iloc[0]
        for pert_label, pert_key in perts:
            row_labels.append(f"{display_name} | {pert_label}")
            vals = []
            for model, method, _ in col_defs:
                col = f"{model}_{method}_{pert_key}_rho"
                val = row[col] if col in row else np.nan
                vals.append(val if pd.notna(val) else np.nan)
            data_rows.append(vals)
        dataset_boundaries.append(len(row_labels))

        mean_vals = []
        for model, method, _ in col_defs:
            vals = []
            for _, pert_key in perts:
                col = f"{model}_{method}_{pert_key}_rho"
                if col in row and pd.notna(row[col]):
                    vals.append(row[col])
            mean_vals.append(float(np.mean(vals)) if vals else np.nan)
        mean_rows.append(mean_vals)

    data = np.array(data_rows, dtype=float)
    mean_data = np.array(mean_rows, dtype=float)
    col_labels = _format_col_labels(col_defs)
    model_groups = _collect_model_groups(col_defs)
    mean_row_labels = dataset_names

    return (data, mean_data, mean_row_labels, col_labels,
            dataset_names, pert_labels, model_groups)

def _annotate_cells(ax, data: np.ndarray, threshold: float) -> None:
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val <= threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_CELL_FONT_SIZE,
                color=color,
                fontweight="bold",
            )

def _outline_strong_cells(ax, data: np.ndarray, threshold: float) -> None:
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isfinite(val):
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.6)
                ax.add_patch(rect)

def _split_by_perturbation(
    data: np.ndarray,
    dataset_names: List[str],
    pert_labels: List[str],
) -> List[Tuple[str, np.ndarray, List[str]]]:
    n_datasets = len(dataset_names)
    n_perts = len(pert_labels)
    expected_rows = n_datasets * n_perts
    if data.shape[0] != expected_rows:
        print(f"WARNING: data rows ({data.shape[0]}) != datasets*perts ({expected_rows}); skipping split")
        return []
    reshaped = data.reshape(n_datasets, n_perts, data.shape[1])
    slices = []
    for pert_idx, pert_name in enumerate(pert_labels):
        pert_data = reshaped[:, pert_idx, :]
        slices.append((pert_name, pert_data, dataset_names))
    return slices

def _strip_model_prefix(labels: List[str]) -> List[str]:
    trimmed = []
    for label in labels:
        if "-" in label:
            trimmed.append(label.split("-", 1)[1])
        else:
            trimmed.append(label)
    return trimmed

def _render_corr_heatmap_block(
    axes: np.ndarray,
    data: np.ndarray,
    mean_data: np.ndarray,
    mean_row_labels: List[str],
    col_labels: List[str],
    dataset_names: List[str],
    pert_labels: List[str],
    model_groups: List[Tuple[str, str, int, int]],
    strong_threshold: float,
    show_mean: bool,
    show_dataset_labels: bool,
    cmap,
    norm,
    pert_slices: Optional[List[Tuple[str, np.ndarray, List[str]]]] = None,
):
    if pert_slices is None:
        pert_slices = _split_by_perturbation(data, dataset_names, pert_labels)
    if not pert_slices:
        return None

    pert_count = len(pert_slices)
    first_im = None
    for row_idx, (pert_name, pert_data, ds_labels) in enumerate(pert_slices):
        for col_idx, (_model_name, model_label, start, end) in enumerate(model_groups):
            ax = axes[row_idx, col_idx]
            sub = pert_data[:, start:end]
            masked = np.ma.masked_invalid(sub)
            im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            if first_im is None:
                first_im = im
            if row_idx == 0:
                ax.set_title(model_label, fontsize=HEATMAP_LABEL_FONT_SIZE)
            if col_idx == 0:
                if show_dataset_labels:
                    ax.set_yticks(np.arange(len(ds_labels)))
                    ax.set_yticklabels(ds_labels, fontsize=HEATMAP_LABEL_FONT_SIZE)
                else:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", which="both", left=False)
                ax.set_ylabel(
                    pert_name,
                    rotation=90,
                    labelpad=HEATMAP_ROW_LABEL_PAD,
                    fontsize=HEATMAP_LABEL_FONT_SIZE,
                    va="center",
                )
            else:
                ax.set_yticks([])
                ax.tick_params(axis="y", which="both", left=False)
            ax.set_xticks(np.arange(end - start))
            if not show_mean and row_idx == pert_count - 1:
                ax.set_xticklabels(
                    _strip_model_prefix(col_labels[start:end]),
                    ha="center",
                    fontsize=HEATMAP_LABEL_FONT_SIZE,
                )
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=True)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            _annotate_cells(ax, sub, strong_threshold)
            _outline_strong_cells(ax, sub, strong_threshold)

    if show_mean:
        for col_idx, (_model_name, _model_label, start, end) in enumerate(model_groups):
            ax_mean = axes[-1, col_idx]
            sub_mean = mean_data[:, start:end]
            masked_mean = np.ma.masked_invalid(sub_mean)
            ax_mean.imshow(masked_mean, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            if col_idx == 0:
                ax_mean.set_ylabel(
                    "Mean",
                    rotation=90,
                    labelpad=HEATMAP_ROW_LABEL_PAD,
                    fontsize=HEATMAP_LABEL_FONT_SIZE,
                    va="center",
                )
            if col_idx == 0 and show_dataset_labels:
                ax_mean.set_yticks(np.arange(len(mean_row_labels)))
                ax_mean.set_yticklabels(mean_row_labels, fontsize=HEATMAP_LABEL_FONT_SIZE)
            else:
                ax_mean.set_yticks([])
                ax_mean.tick_params(axis="y", which="both", left=False)
            ax_mean.set_xticks(np.arange(end - start))
            ax_mean.set_xticklabels(
                _strip_model_prefix(col_labels[start:end]),
                ha="center",
                fontsize=HEATMAP_LABEL_FONT_SIZE,
            )
            _annotate_cells(ax_mean, sub_mean, strong_threshold)
            _outline_strong_cells(ax_mean, sub_mean, strong_threshold)

    return first_im

def _plot_corr_heatmap_by_perturbation_and_model(
    data: np.ndarray,
    mean_data: np.ndarray,
    mean_row_labels: List[str],
    col_labels: List[str],
    dataset_names: List[str],
    pert_labels: List[str],
    model_groups: List[Tuple[str, str, int, int]],
    title: str,
    out_file: Path,
    strong_threshold: float,
    show_mean: bool,
    show_colorbar: bool,
) -> None:
    if data.size == 0:
        print(f"WARNING: no data for {title} heatmap; skipping {out_file}")
        return

    pert_slices = _split_by_perturbation(data, dataset_names, pert_labels)
    if not pert_slices or not model_groups:
        print(f"WARNING: no perturbation/model slices for {title} heatmap; skipping {out_file}")
        return

    pert_count = len(pert_slices)
    n_rows = pert_count + 1 if show_mean else pert_count
    n_cols = len(model_groups)
    per_row_height = max(HEATMAP_ROW_HEIGHT_MIN, HEATMAP_HEIGHT_PER_ROW * len(dataset_names))
    mean_height = max(HEATMAP_MEAN_HEIGHT_MIN, HEATMAP_HEIGHT_PER_ROW * len(mean_row_labels))
    fig_height = per_row_height * pert_count + (mean_height if show_mean else 0.0)
    width_units = [end - start for _, _, start, end in model_groups]
    fig_width = HEATMAP_WIDTH_PER_COL * sum(width_units) + HEATMAP_LEFT_PAD + HEATMAP_RIGHT_PAD
    left = HEATMAP_LEFT_PAD / fig_width
    right = 1.0 - (HEATMAP_RIGHT_PAD / fig_width)
    top = 1.0 - (HEATMAP_TOP_PAD / fig_height)
    bottom = HEATMAP_BOTTOM_PAD / fig_height
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            "height_ratios": [per_row_height] * pert_count + ([mean_height] if show_mean else []),
            "width_ratios": width_units,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "wspace": HEATMAP_WSPACE,
            "hspace": HEATMAP_HSPACE,
        },
        constrained_layout=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    cmap = plt.get_cmap(HEATMAP_CMAP)
    cmap.set_bad(color="lightgray")
    norm = TwoSlopeNorm(
        vcenter=HEATMAP_COLOR_CENTER,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
    )

    title_x = (left + right) / 2.0
    fig.suptitle(f"{title}", fontsize=11, x=title_x)

    first_im = _render_corr_heatmap_block(
        axes,
        data,
        mean_data,
        mean_row_labels,
        col_labels,
        dataset_names,
        pert_labels,
        model_groups,
        strong_threshold,
        show_mean,
        show_dataset_labels=True,
        cmap=cmap,
        norm=norm,
        pert_slices=pert_slices,
    )

    if show_colorbar:
        cbar_left = right + (HEATMAP_CBAR_PAD / fig_width)
        cbar_width = HEATMAP_CBAR_WIDTH / fig_width
        cax = fig.add_axes([cbar_left, bottom, cbar_width, top - bottom])
        cbar = fig.colorbar(first_im, cax=cax)
        cbar.ax.tick_params(labelsize=HEATMAP_CBAR_LABEL_SIZE)
        cbar.set_ticks([HEATMAP_COLOR_MIN, HEATMAP_COLOR_CENTER, 0.0, HEATMAP_COLOR_MAX])
        #cbar.set_label("Spearman rho")

    fig.savefig(out_file, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"OK Saved: {out_file}")

def _plot_corr_heatmap_combined(
    nat_data: np.ndarray,
    nat_mean: np.ndarray,
    nat_mean_labels: List[str],
    nat_col_labels: List[str],
    nat_dataset_names: List[str],
    nat_pert_labels: List[str],
    nat_model_groups: List[Tuple[str, str, int, int]],
    adv_data: np.ndarray,
    adv_mean: np.ndarray,
    adv_mean_labels: List[str],
    adv_col_labels: List[str],
    adv_dataset_names: List[str],
    adv_pert_labels: List[str],
    adv_model_groups: List[Tuple[str, str, int, int]],
    out_file: Path,
    strong_threshold: float,
    show_mean: bool,
) -> None:
    if nat_data.size == 0 or adv_data.size == 0:
        print(f"WARNING: no data for combined heatmap; skipping {out_file}")
        return
    if not nat_model_groups or not adv_model_groups:
        print(f"WARNING: no model groups for combined heatmap; skipping {out_file}")
        return
    if nat_dataset_names != adv_dataset_names:
        print(f"WARNING: dataset mismatch for combined heatmap; skipping {out_file}")
        return
    if len(nat_pert_labels) != len(adv_pert_labels):
        print(f"WARNING: perturbation mismatch for combined heatmap; skipping {out_file}")
        return
    if show_mean and nat_mean_labels != adv_mean_labels:
        print(f"WARNING: mean labels mismatch for combined heatmap; skipping {out_file}")
        return

    nat_slices = _split_by_perturbation(nat_data, nat_dataset_names, nat_pert_labels)
    adv_slices = _split_by_perturbation(adv_data, adv_dataset_names, adv_pert_labels)
    if not nat_slices or not adv_slices:
        print(f"WARNING: no perturbation slices for combined heatmap; skipping {out_file}")
        return

    pert_count = len(nat_slices)
    n_rows = pert_count + 1 if show_mean else pert_count
    nat_width_units = [end - start for _, _, start, end in nat_model_groups]
    adv_width_units = [end - start for _, _, start, end in adv_model_groups]
    gap_units = HEATMAP_BLOCK_GAP / HEATMAP_WIDTH_PER_COL
    width_units = nat_width_units + [gap_units] + adv_width_units
    n_cols = len(width_units)
    per_row_height = max(HEATMAP_ROW_HEIGHT_MIN, HEATMAP_HEIGHT_PER_ROW * len(nat_dataset_names))
    mean_height = max(HEATMAP_MEAN_HEIGHT_MIN, HEATMAP_HEIGHT_PER_ROW * len(nat_mean_labels))
    fig_height = per_row_height * pert_count + (mean_height if show_mean else 0.0)
    fig_width = (HEATMAP_WIDTH_PER_COL * (sum(nat_width_units) + sum(adv_width_units))
                 + HEATMAP_LEFT_PAD + HEATMAP_RIGHT_PAD + HEATMAP_BLOCK_GAP)
    left = HEATMAP_LEFT_PAD / fig_width
    right = 1.0 - (HEATMAP_RIGHT_PAD / fig_width)
    top = 1.0 - (HEATMAP_TOP_PAD / fig_height)
    bottom = HEATMAP_BOTTOM_PAD / fig_height
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            "height_ratios": [per_row_height] * pert_count + ([mean_height] if show_mean else []),
            "width_ratios": width_units,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "wspace": HEATMAP_WSPACE,
            "hspace": HEATMAP_HSPACE,
        },
        constrained_layout=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    gap_idx = len(nat_width_units)
    for row_idx in range(n_rows):
        axes[row_idx, gap_idx].axis("off")

    nat_axes = axes[:, :gap_idx]
    adv_axes = axes[:, gap_idx + 1:]

    cmap = plt.get_cmap(HEATMAP_CMAP)
    cmap.set_bad(color="lightgray")
    norm = TwoSlopeNorm(
        vcenter=HEATMAP_COLOR_CENTER,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
    )

    title_x = (left + right) / 2.0
    fig.suptitle("XEC, Natural and Adversarial", fontsize=9, x=title_x, y=0.94)

    nat_im = _render_corr_heatmap_block(
        nat_axes,
        nat_data,
        nat_mean,
        nat_mean_labels,
        nat_col_labels,
        nat_dataset_names,
        nat_pert_labels,
        nat_model_groups,
        strong_threshold,
        show_mean,
        show_dataset_labels=True,
        cmap=cmap,
        norm=norm,
        pert_slices=nat_slices,
    )
    adv_im = _render_corr_heatmap_block(
        adv_axes,
        adv_data,
        adv_mean,
        adv_mean_labels,
        adv_col_labels,
        adv_dataset_names,
        adv_pert_labels,
        adv_model_groups,
        strong_threshold,
        show_mean,
        show_dataset_labels=False,
        cmap=cmap,
        norm=norm,
        pert_slices=adv_slices,
    )

    first_im = nat_im if nat_im is not None else adv_im
    if first_im is None:
        plt.close(fig)
        print(f"WARNING: no images for combined heatmap; skipping {out_file}")
        return

    cbar_left = right + (HEATMAP_CBAR_PAD / fig_width)
    cbar_width = HEATMAP_CBAR_WIDTH / fig_width
    cax = fig.add_axes([cbar_left, bottom, cbar_width, top - bottom])
    cbar = fig.colorbar(first_im, cax=cax)
    cbar.ax.tick_params(labelsize=HEATMAP_CBAR_LABEL_SIZE)
    cbar.set_ticks([HEATMAP_COLOR_MIN, HEATMAP_COLOR_CENTER, 0.0, HEATMAP_COLOR_MAX])

    fig.savefig(out_file, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"OK Saved: {out_file}")

def plot_correlation_heatmaps(df_summary: pd.DataFrame, output_dir: Path) -> None:
    nat = _build_corr_matrices_natural(df_summary)
    adv = _build_corr_matrices_adversarial(df_summary)

    (nat_data, nat_mean, nat_mean_labels,
     nat_col_labels, nat_dataset_names, nat_pert_labels, nat_model_groups) = nat
    (adv_data, adv_mean, adv_mean_labels,
     adv_col_labels, adv_dataset_names, adv_pert_labels, adv_model_groups) = adv

    _plot_corr_heatmap_by_perturbation_and_model(
        nat_data,
        nat_mean,
        nat_mean_labels,
        nat_col_labels,
        nat_dataset_names,
        nat_pert_labels,
        nat_model_groups,
        title="XEC, Natural",
        out_file=output_dir / f"correlation_heatmap.{HEATMAP_FILE_EXT}",
        strong_threshold=STRONG_CORR_THRESHOLD,
        show_mean=PLOT_CORRELATION_HEATMAP_MEAN,
        show_colorbar=HEATMAP_SHOW_COLORBAR_NATURAL,
    )
    _plot_corr_heatmap_by_perturbation_and_model(
        adv_data,
        adv_mean,
        adv_mean_labels,
        adv_col_labels,
        adv_dataset_names,
        adv_pert_labels,
        adv_model_groups,
        title="XEC, Adversarial",
        out_file=output_dir / f"correlation_heatmap_adversarial.{HEATMAP_FILE_EXT}",
        strong_threshold=STRONG_CORR_THRESHOLD,
        show_mean=PLOT_CORRELATION_HEATMAP_MEAN,
        show_colorbar=HEATMAP_SHOW_COLORBAR_ADVERSARIAL,
    )
    _plot_corr_heatmap_combined(
        nat_data,
        nat_mean,
        nat_mean_labels,
        nat_col_labels,
        nat_dataset_names,
        nat_pert_labels,
        nat_model_groups,
        adv_data,
        adv_mean,
        adv_mean_labels,
        adv_col_labels,
        adv_dataset_names,
        adv_pert_labels,
        adv_model_groups,
        out_file=output_dir / f"correlation_heatmap_combined.{HEATMAP_FILE_EXT}",
        strong_threshold=STRONG_CORR_THRESHOLD,
        show_mean=PLOT_CORRELATION_HEATMAP_MEAN,
    )

# MAIN EXECUTION

def main():
    """Generate summary tables (natural + adversarial splits)."""
    
    print("\n" + "="*80)
    print("AGGREGATE RESULTS ACROSS DATASETS - VERSION 2")
    print("="*80)
    print(f"\nXAI results directory: {XAI_RESULTS_DIR}")
    print(f"UQ results directory: {UQ_RESULTS_DIR}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Table size: {TABLE_SIZE}")
    
    if not XAI_RESULTS_DIR.exists():
        print(f"\nERROR: XAI results directory not found: {XAI_RESULTS_DIR}")
        return
    if not UQ_RESULTS_DIR.exists():
        print(f"\nERROR: UQ results directory not found: {UQ_RESULTS_DIR}")
        return
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TABLE 1 XAI DEGRADATION
    xai_latex_nat, xai_latex_adv = generate_xai_degradation_table()
    xai_file_nat = output_dir / "table_xai_degradation.tex"
    with open(xai_file_nat, 'w') as f:
        f.write(xai_latex_nat)
    print(f"\nOK Saved: {xai_file_nat}")

    xai_file_adv = output_dir / "table_xai_degradation_adversarial.tex"
    with open(xai_file_adv, 'w') as f:
        f.write(xai_latex_adv)
    print(f"OK Saved: {xai_file_adv}")
    
    # TABLE 2 UQ GROWTH
    uq_latex_nat, uq_latex_adv = generate_uq_growth_table()
    uq_file_nat = output_dir / "table_uq_growth.tex"
    with open(uq_file_nat, 'w') as f:
        f.write(uq_latex_nat)
    print(f"\nOK Saved: {uq_file_nat}")

    uq_file_adv = output_dir / "table_uq_growth_adversarial.tex"
    with open(uq_file_adv, 'w') as f:
        f.write(uq_latex_adv)
    print(f"OK Saved: {uq_file_adv}")
    
    # TABLE 3 CORRELATION
    corr_latex_nat, corr_latex_adv, corr_df = generate_correlation_table()
    corr_file_nat = output_dir / "table_correlation.tex"
    with open(corr_file_nat, 'w') as f:
        f.write(corr_latex_nat)
    print(f"\nOK Saved: {corr_file_nat}")
    
    corr_file_adv = output_dir / "table_correlation_adversarial.tex"
    with open(corr_file_adv, 'w') as f:
        f.write(corr_latex_adv)
    print(f"OK Saved: {corr_file_adv}")

    if PLOT_CORRELATION_HEATMAPS:
        plot_correlation_heatmaps(corr_df, output_dir)
    
    # COMBINED FILE
    combined_file = output_dir / "tables_all.tex"
    combined_sections = [
        f"% table_xai_degradation.tex\n{xai_latex_nat}",
        f"% table_xai_degradation_adversarial.tex\n{xai_latex_adv}",
        f"% table_uq_growth.tex\n{uq_latex_nat}",
        f"% table_uq_growth_adversarial.tex\n{uq_latex_adv}",
        f"% table_correlation.tex\n{corr_latex_nat}",
        f"% table_correlation_adversarial.tex\n{corr_latex_adv}"
    ]
    with open(combined_file, 'w') as f:
        f.write("\n\n".join(combined_sections))
    print(f"\nOK Saved: {combined_file}")
    
    print("\n" + "="*80)
    print("ALL TABLES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Table size: \\{TABLE_SIZE}")
    print("\nFiles generated:")
    print(f"  1. {xai_file_nat.name}")
    print(f"  2. {xai_file_adv.name}")
    print(f"  3. {uq_file_nat.name}")
    print(f"  4. {uq_file_adv.name}")
    print(f"  5. {corr_file_nat.name}")
    print(f"  6. {corr_file_adv.name}")
    print(f"  7. {combined_file.name}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
