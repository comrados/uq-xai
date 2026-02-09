import sys
import os
sys.path.append(os.getcwd())

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT_DIR / "results"
INPUT_DIR = RESULTS_ROOT / "5_run_perturbation_explainers"
OUTPUT_DIR = RESULTS_ROOT / Path(__file__).stem


# CONFIGURATION
FONT_SIZE = "tiny"  # Options: "tiny", "scriptsize", "footnotesize", "small", "normalsize"

PERTURBATION_DISPLAY = {
    "gaussian": "Gauss.",
    "missing": "Miss.",
    "permutation": "Perm.",
    "adversarial_bim": "BIM",
    "adversarial_pgd": "PGD",
    "adversarial_cw": "C\\&W"
}

EXPLAINER_DISPLAY = {
    "SHAP": "SHAP",
    "SHAP (Tree)": "SHAP",
    "SHAP (Kernel)": "SHAP",
    "LIME": "LIME",
    "IntGrad": "IntGrad",
    "SmoothGrad": "SmoothG"
}


def load_explainer_results(dataset_name, uci_id):
    """Load explainer results from a pickle file."""
    results_file = INPUT_DIR / f"explainers_perturbations_{dataset_name}_{uci_id}.pkl"
    if not results_file.exists():
        print(f"ERROR: {results_file} not found!")
        return None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return pd.DataFrame(results)


def create_natural_explainer_table(df, dataset_name, output_dir, font_size=FONT_SIZE):
    """Create LaTeX tables for natural perturbations (split to keep tables shorter)."""
    
    natural_perts = ['gaussian', 'missing', 'permutation']
    df_natural = df[df['perturbation'].isin(natural_perts)].copy()
    
    if len(df_natural) == 0:
        print("  No natural perturbation data found - skipping")
        return None
    
    # MERGE SHAP VARIANTS UNDER A SINGLE LABEL TO SHARE ONE EXPLAINER ROW
    df_natural['explainer'] = df_natural['explainer'].replace({
        "SHAP (Kernel)": "SHAP",
        "SHAP (Tree)": "SHAP"
    })
    
    """
    groups = [
        {
            "name": "log_mlp",
            "models": ["Logistic", "MLP"],
            "explainers": ["LIME", "SHAP (Kernel)"],
            "caption": "XAI Degradation - Natural Perturbations (Logistic + MLP)",
        },
        {
            "name": "trees",
            "models": ["RF", "LightGBM", "CatBoost"],
            "explainers": ["LIME", "SHAP (Tree)"],
            "caption": "XAI Degradation - Natural Perturbations (Tree Models)",
        },
    ]
    """
    groups = [
        {
            "name": "all",
            "models": ["Logistic", "RF", "MLP", "LightGBM", "CatBoost"],
            "explainers": ["LIME", "SHAP"],
            "caption": "XAI Degradation - Natural Perturbations",
        },
    ]
    
    output_files = []
    
    for group in groups:
        models = [m for m in group["models"] if m in df_natural['model'].unique()]
        explainers_allowed = [e for e in group["explainers"] if e in df_natural['explainer'].unique()]
        
        if not models or not explainers_allowed:
            continue
        
        df_group = df_natural[
            df_natural['model'].isin(models) &
            df_natural['explainer'].isin(explainers_allowed) &
            (~np.isclose(df_natural['level'], 0.0))
        ].copy()
        
        if len(df_group) == 0:
            continue
        
        col_spec = "|l|l|l||" + "".join("cc|" for _ in models)
        total_cols = 3 + 2 * len(models)
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(f"\\{font_size}")
        latex.append(r"\caption{" + group["caption"] + r": " + dataset_name.replace("_", " ").title() + r"}")
        latex.append(r"\label{tab:xai_natural_" + group["name"] + "_" + dataset_name + "}")
        latex.append(r"\begin{tabular}{" + col_spec + r"}")
        latex.append(r"\toprule")
        
        header1 = r"\textbf{Explainer} & \textbf{Pert.} & \textbf{Level}"
        for model in models:
            header1 += r" & \multicolumn{2}{|c|}{\textbf{" + model + r"}}"
        header1 += r" \\"
        latex.append(header1)
        
        header2 = r" &  & "
        for _ in models:
            header2 += r"& \textbf{Rank $\rho$} & \textbf{Attr Var} "
        header2 += r"\\"
        latex.append(header2)
        latex.append(r"\midrule")
        
        explainers = sorted(explainers_allowed)
        
        for explainer in explainers:
            df_exp = df_group[df_group['explainer'] == explainer].copy()
            if len(df_exp) == 0:
                continue
            
            exp_display = EXPLAINER_DISPLAY.get(explainer, explainer)
            
            # COUNT ROWS FOR THIS EXPLAINER ACROSS ALL PERTURBATIONS
            rows_for_exp = sum(
                len(sorted(l for l in df_exp[df_exp['perturbation'] == pert]['level'].unique() if not np.isclose(l, 0.0)))
                for pert in natural_perts
            )
            if rows_for_exp == 0:
                continue
            
            exp_started = False
            
            for pert in sorted(natural_perts):
                df_pert = df_exp[df_exp['perturbation'] == pert].copy()
                levels = sorted(l for l in df_pert['level'].unique() if not np.isclose(l, 0.0))
                if len(df_pert) == 0 or not levels:
                    continue
                
                pert_display = PERTURBATION_DISPLAY.get(pert, pert)
                pert_started = False
                
                for level in levels:
                    cells = []
                    
                    if not exp_started:
                        cells.append(r"\multirow{" + str(rows_for_exp) + r"}{*}{" + exp_display + r"}")
                        exp_started = True
                    else:
                        cells.append("")
                    
                    if not pert_started:
                        cells.append(r"\multirow{" + str(len(levels)) + r"}{*}{" + pert_display + r"}")
                        pert_started = True
                    else:
                        cells.append("")
                    
                    cells.append(f"{level:.2f}")
                    
                    for model in models:
                        df_level = df_pert[(df_pert['model'] == model) & (np.isclose(df_pert['level'], level))]
                        if len(df_level) > 0:
                            row_data = df_level.iloc[0]
                            rank_corr = row_data['rank_correlation']
                            attr_var = row_data['attribution_variance']
                            cells.append(f"{rank_corr:.3f}")
                            cells.append(f"{attr_var:.4f}")
                        else:
                            cells.append("-")
                            cells.append("-")
                    
                    latex.append(" & ".join(cells) + r" \\")
                
                latex.append(r"\cline{2-" + str(total_cols) + r"}")
            
            latex.append(r"\midrule")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        output_file = output_dir / f"xai_natural_{group['name']}_{dataset_name}.tex"
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))
        
        print(f"OK Natural XAI degradation table saved: {output_file}")
        output_files.append(output_file)
    
    return output_files if output_files else None


def create_adversarial_explainer_table(df, dataset_name, output_dir, font_size=FONT_SIZE):
    """Create a LaTeX table for adversarial explainer degradation (MLP only)."""
    
    # FILTER ADVERSARIAL ATTACKS ONLY
    adversarial_attacks = ['adversarial_bim', 'adversarial_pgd', 'adversarial_cw']
    df_adv = df[df['perturbation'].isin(adversarial_attacks)].copy()
    
    if len(df_adv) == 0:
        print("  No adversarial data found - skipping")
        return None
    
    # ONLY MLP FOR ADVERSARIAL
    df_adv = df_adv[df_adv['model'] == 'MLP'].copy()
    
    if len(df_adv) == 0:
        print("  No MLP adversarial data found - skipping")
        return None
    
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(f"\\{font_size}")
    latex.append(r"\caption{XAI Degradation - Adversarial Attacks (MLP only): " + dataset_name.replace("_", " ").title() + r"}")
    latex.append(r"\label{tab:xai_adversarial_" + dataset_name + "}")
    
    # TABLE STRUCTURE EXPLAINER ATTACK LEVEL RANK CORR ATTR VAR
    latex.append(r"\begin{tabular}{|l|l|l|cc|}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Explainer} & \textbf{Attack} & \textbf{Level} & \textbf{Rank $\rho$} & \textbf{Attr Var} \\")
    latex.append(r"\midrule")
    
    # DATA ROWS GROUPED BY EXPLAINER THEN ATTACK
    explainers = sorted(df_adv['explainer'].unique())
    
    for explainer in explainers:
        df_exp = df_adv[(df_adv['explainer'] == explainer) & (~np.isclose(df_adv['level'], 0.0))].copy()
        if len(df_exp) == 0:
            continue
        
        explainer_first_row = True
        
        for attack in sorted(adversarial_attacks):
            df_attack = df_exp[df_exp['perturbation'] == attack].copy()
            if len(df_attack) == 0:
                continue
            
            levels = sorted(l for l in df_attack['level'].unique() if not np.isclose(l, 0.0))
            
            for i, level in enumerate(levels):
                df_level = df_attack[np.isclose(df_attack['level'], level)]
                
                if len(df_level) == 0:
                    continue
                
                row_data = df_level.iloc[0]
                
                # FIRST COLUMN EXPLAINER NAME MULTIROW
                if explainer_first_row:
                    total_exp_rows = len(df_exp)
                    exp_display = EXPLAINER_DISPLAY.get(explainer, explainer)
                    row = r"\multirow{" + str(total_exp_rows) + r"}{*}{\textbf{" + exp_display + r"}}"
                    explainer_first_row = False
                else:
                    row = ""
                
                # SECOND COLUMN ATTACK NAME MULTIROW
                if i == 0:
                    attack_display = PERTURBATION_DISPLAY.get(attack, attack.replace('adversarial_', '').upper())
                    row += r" & \multirow{" + str(len(levels)) + r"}{*}{" + attack_display + r"}"
                else:
                    row += " & "
                
                # THIRD COLUMN LEVEL
                row += f" & {level:.2f}"
                
                # DATA COLUMNS
                rank_corr = row_data['rank_correlation']
                attr_var = row_data['attribution_variance']
                
                row += f" & {rank_corr:.3f} & {attr_var:.4f}"
                
                row += r" \\"
                latex.append(row)
            
            # ADD SEPARATOR AFTER EACH ATTACK TYPE
            latex.append(r"\cline{2-5}")
        
        # ADD SEPARATOR AFTER EACH EXPLAINER
        latex.append(r"\midrule")
    
    # CLOSE TABLE
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # SAVE TO FILE
    output_file = output_dir / f"xai_adversarial_{dataset_name}.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"OK Adversarial XAI degradation table saved: {output_file}")
    return output_file


def create_markdown_tables(df, dataset_name, output_dir):
    """Create Markdown tables for explainer degradation."""
    
    md = []
    md.append(f"# XAI Degradation: {dataset_name.replace('_', ' ').title()}\n")
    
    # NATURAL PERTURBATIONS
    natural_perts = ['gaussian', 'missing', 'permutation']
    df_natural = df[df['perturbation'].isin(natural_perts)].copy()
    
    if len(df_natural) > 0:
        md.append("## Natural Perturbations\n")
        
        models = sorted(df_natural['model'].unique())
        
        for model in models:
            df_model = df_natural[df_natural['model'] == model].copy()
            explainers = sorted(df_model['explainer'].unique())
            
            md.append(f"\n### {model}\n")
            
            for explainer in explainers:
                df_exp = df_model[df_model['explainer'] == explainer].copy()
                
                md.append(f"\n**{explainer}**\n")
                
                # HEADER
                header = "| Perturbation | Level | Rank Correlation | Attribution Variance |"
                separator = "|--------------|------:|-----------------:|---------------------:|"
                md.append(header)
                md.append(separator)
                
                for pert in sorted(natural_perts):
                    df_pert = df_exp[df_exp['perturbation'] == pert].copy()
                    levels = sorted(df_pert['level'].unique())
                    
                    for level in levels:
                        if np.isclose(level, 0.0):
                            continue
                        df_level = df_pert[np.isclose(df_pert['level'], level)]
                        
                        if len(df_level) > 0:
                            row_data = df_level.iloc[0]
                            rank_corr = row_data['rank_correlation']
                            attr_var = row_data['attribution_variance']
                            
                            row = f"| {pert} | {level:.2f} | {rank_corr:.3f} | {attr_var:.4f} |"
                            md.append(row)
    
    # ADVERSARIAL ATTACKS
    adversarial_attacks = ['adversarial_bim', 'adversarial_pgd', 'adversarial_cw']
    df_adv = df[df['perturbation'].isin(adversarial_attacks)].copy()
    df_adv = df_adv[df_adv['model'] == 'MLP'].copy()
    
    if len(df_adv) > 0:
        md.append("\n## Adversarial Attacks (MLP only)\n")
        
        explainers = sorted(df_adv['explainer'].unique())
        
        for explainer in explainers:
            df_exp = df_adv[df_adv['explainer'] == explainer].copy()
            
            md.append(f"\n### {explainer}\n")
            
            # HEADER
            header = "| Attack | Level | Rank Correlation | Attribution Variance |"
            separator = "|--------|------:|-----------------:|---------------------:|"
            md.append(header)
            md.append(separator)
            
            for attack in sorted(adversarial_attacks):
                df_attack = df_exp[df_exp['perturbation'] == attack].copy()
                levels = sorted(df_attack['level'].unique())
                
                attack_name = attack.replace('adversarial_', '').upper()
                
                for level in levels:
                    if np.isclose(level, 0.0):
                        continue
                    df_level = df_attack[np.isclose(df_attack['level'], level)]
                    
                    if len(df_level) > 0:
                        row_data = df_level.iloc[0]
                        rank_corr = row_data['rank_correlation']
                        attr_var = row_data['attribution_variance']
                        
                        row = f"| {attack_name} | {level:.2f} | {rank_corr:.3f} | {attr_var:.4f} |"
                        md.append(row)
    
    # SAVE TO FILE
    output_file = output_dir / f"xai_degradation_{dataset_name}.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"OK Markdown XAI degradation saved: {output_file}")
    return output_file


def combine_tex_tables(tex_paths, output_dir, output_filename="explainer_all.tex"):
    """Combine all generated LaTeX table files into a single file."""
    existing_files = [Path(p) for p in tex_paths if p is not None]
    existing_files = [p for p in existing_files if p.exists()]
    
    if not existing_files:
        print("No LaTeX tables found to combine - skipping")
        return None
    
    combined_sections = []
    for path in existing_files:
        content = path.read_text().strip()
        if not content:
            continue
        combined_sections.append(f"% ---- {path.name} ----\n{content}")
    
    if not combined_sections:
        print("LaTeX files were empty - skipping combined output")
        return None
    
    output_path = output_dir / output_filename
    output_path.write_text("\n\n".join(combined_sections) + "\n")
    print(f"OK Combined LaTeX tables saved: {output_path}")
    return output_path


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        ("wine_binary", "186"),
        ("bean", "602"),
        ("iris", "53"),
        ("rice", "545"),
        ("ecoli", "39"),
    ]
    
    tex_files = []
    
    print("=" * 70)
    print("GENERATING XAI DEGRADATION TABLES")
    print(f"Font size: {FONT_SIZE}")
    print("=" * 70)
    
    for dataset_name, uci_id in datasets:
        print(f"\n--- {dataset_name.upper()} ---")
        
        # LOAD EXPLAINER DATA
        df_xai = load_explainer_results(dataset_name, uci_id)
        if df_xai is None:
            continue
        
        print(f"Loaded {len(df_xai)} XAI rows")
        print(f"Models: {sorted(df_xai['model'].unique())}")
        print(f"Explainers: {sorted(df_xai['explainer'].unique())}")
        print(f"Perturbations: {sorted(df_xai['perturbation'].unique())}")
        
        # GENERATE NATURAL PERTURBATION TABLE
        print("\nGenerating natural XAI degradation table...")
        nat_tex = create_natural_explainer_table(df_xai, dataset_name, output_dir, font_size=FONT_SIZE)
        if nat_tex:
            if isinstance(nat_tex, list):
                tex_files.extend(nat_tex)
            else:
                tex_files.append(nat_tex)
        
        # GENERATE ADVERSARIAL TABLE
        print("Generating adversarial XAI degradation table...")
        adv_tex = create_adversarial_explainer_table(df_xai, dataset_name, output_dir, font_size=FONT_SIZE)
        if adv_tex:
            tex_files.append(adv_tex)
        
        # GENERATE MARKDOWN TABLES
        print("Generating Markdown tables...")
        create_markdown_tables(df_xai, dataset_name, output_dir)
    
    print("\nCombining LaTeX tables into a single file...")
    combine_tex_tables(tex_files, output_dir)
    
    print("\n" + "=" * 70)
    print(f"DONE! Tables saved to {output_dir}/")
    print("=" * 70)
    print("\nFiles generated per dataset:")
    print("  - xai_natural_*.tex              (Natural perturbations)")
    print("  - xai_adversarial_*.tex          (Adversarial attacks - MLP only)")
    print("  - xai_degradation_*.md           (Markdown preview)")
    print("\nAdditional combined file:")
    print("  - explainer_all.tex              (All LaTeX tables concatenated)")


if __name__ == "__main__":
    main()
