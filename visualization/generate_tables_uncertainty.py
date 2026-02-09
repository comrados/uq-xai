import sys
import os
sys.path.append(os.getcwd())

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT_DIR / "results"
INPUT_DIR = RESULTS_ROOT / "4_run_perturbation_uncertainty"
OUTPUT_DIR = RESULTS_ROOT / Path(__file__).stem


# CONFIGURATION
FONT_SIZE = "tiny"  # Options: "tiny", "scriptsize", "footnotesize", "small", "normalsize"
SHOW_STD_NATURAL = False  # Show ±std for natural perturbations
SHOW_STD_ADVERSARIAL = True  # Show ±std for adversarial attacks

PERTURBATION_DISPLAY = {
    "gaussian": "Gauss.",
    "missing": "Miss.",
    "permutation": "Perm."
}


def load_results(dataset_name, uci_id):
    """Load results from a pickle file."""
    results_file = INPUT_DIR / f"perturbation_uncertainty_{dataset_name}_{uci_id}.pkl"
    if not results_file.exists():
        print(f"ERROR: {results_file} not found!")
        return None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return pd.DataFrame(results)


def create_natural_perturbation_table(df, dataset_name, output_dir, font_size=FONT_SIZE, show_std=SHOW_STD_NATURAL):
    """Create a LaTeX table for natural perturbations (Gaussian, Missing, Permutation)."""
    
    # FILTER NATURAL PERTURBATIONS ONLY
    natural_perts = ['gaussian', 'missing', 'permutation']
    df_natural = df[df['perturbation'].isin(natural_perts)].copy()
    
    models = ['Logistic', 'RF', 'MLP', 'LightGBM', 'CatBoost']
    models = [m for m in models if m in df_natural['model'].unique()]
    
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(f"\\{font_size}")  # Font size control
    latex.append(r"\caption{Natural Perturbations: " + dataset_name.replace("_", " ").title() + r"}")
    latex.append(r"\label{tab:natural_" + dataset_name + "}")
    
    # Table structure with vertical separators around perturbation/level and per-model groups
    col_spec = "|l|l|" + "".join("cccc|" for _ in models)
    latex.append(r"\begin{tabular}{" + col_spec + r"}")
    latex.append(r"\toprule")
    
    # HEADER ROW 1 MODEL NAMES MULTICOLUMN
    header1 = r"\textbf{Pert.} & \textbf{Level}"
    for model in models:
        header1 += r" & \multicolumn{4}{|c|}{\textbf{" + model + "}}"
    header1 += r" \\"
    latex.append(header1)
    
    # HEADER ROW 2 METRICS
    header2 = r"& "
    for _ in models:
        header2 += r"& \textbf{Err} & \textbf{Ale} & \textbf{Epi} & \textbf{Conf}"
    header2 += r" \\"
    latex.append(header2)
    latex.append(r"\midrule")

    # Clean baseline row (level == 0)
    clean_rows = df_natural[np.isclose(df_natural['level'], 0.0)]
    if not clean_rows.empty:
        row = r"\textbf{Clean} & -"
        for model in models:
            df_model = clean_rows[clean_rows['model'] == model]
            if len(df_model) > 0:
                row_data = df_model.iloc[0]
                error = row_data['error']
                ale = row_data['aleatoric']
                ale_std = row_data['aleatoric_std']
                epi = row_data['epistemic']
                epi_std = row_data['epistemic_std']
                conf = row_data['confidence']
                conf_std = row_data['confidence_std']
                
                if show_std:
                    row += f" & {error:.3f} & {ale:.3f}$\\pm${ale_std:.3f} & {epi:.3f}$\\pm${epi_std:.3f} & {conf:.3f}$\\pm${conf_std:.3f}"
                else:
                    row += f" & {error:.3f} & {ale:.3f} & {epi:.3f} & {conf:.3f}"
            else:
                row += r" & - & - & - & -"
        row += r" \\"
        latex.append(row)
        latex.append(r"\midrule")
    
    # DATA ROWS
    for pert in sorted(natural_perts):
        if pert not in df_natural['perturbation'].unique():
            continue
            
        df_pert = df_natural[df_natural['perturbation'] == pert].copy()
        levels = sorted(l for l in df_pert['level'].unique() if not np.isclose(l, 0.0))
        
        for i, level in enumerate(levels):
            df_level = df_pert[df_pert['level'] == level]
            
            # FIRST COLUMN PERTURBATION NAME MULTIROW
            if i == 0:
                pert_name = PERTURBATION_DISPLAY.get(pert, pert.upper().replace('_', ' '))
                row = r"\multirow{" + str(len(levels)) + r"}{*}{\textbf{" + pert_name + r"}}"
            else:
                row = ""
            
            # SECOND COLUMN LEVEL
            row += f" & {level:.2f}"
            
            # DATA COLUMNS
            for model in models:
                df_model = df_level[df_level['model'] == model]
                
                if len(df_model) > 0:
                    row_data = df_model.iloc[0]
                    error = row_data['error']
                    ale = row_data['aleatoric']
                    ale_std = row_data['aleatoric_std']
                    epi = row_data['epistemic']
                    epi_std = row_data['epistemic_std']
                    conf = row_data['confidence']
                    conf_std = row_data['confidence_std']
                    
                    if show_std:
                        row += f" & {error:.3f} & {ale:.3f}$\\pm${ale_std:.3f} & {epi:.3f}$\\pm${epi_std:.3f} & {conf:.3f}$\\pm${conf_std:.3f}"
                    else:
                        row += f" & {error:.3f} & {ale:.3f} & {epi:.3f} & {conf:.3f}"
                else:
                    row += r" & - & - & - & -"
            
            row += r" \\"
            latex.append(row)
        
        # ADD SEPARATOR BETWEEN PERTURBATIONS
        latex.append(r"\midrule")
    
    # CLOSE TABLE
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # SAVE TO FILE
    output_file = output_dir / f"natural_perturbation_{dataset_name}.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"OK Natural perturbation table saved: {output_file}")
    return output_file


def create_adversarial_table(df, dataset_name, output_dir, font_size=FONT_SIZE, show_std=SHOW_STD_ADVERSARIAL):
    """Create a LaTeX table for adversarial attacks (MLP only)."""
    
    # FILTER ADVERSARIAL ATTACKS ONLY
    adversarial_attacks = ['bim', 'pgd', 'cw']
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
    latex.append(f"\\{font_size}")  # Font size control
    latex.append(r"\caption{Adversarial Attacks (MLP only): " + dataset_name.replace("_", " ").title() + r"}")
    latex.append(r"\label{tab:adversarial_" + dataset_name + "}")
    
    # TABLE STRUCTURE ATTACK LEVEL ERROR ALEATORIC EPISTEMIC CONFIDENCE
    latex.append(r"\begin{tabular}{|l|l|cccc|}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Attack} & \textbf{Level} & \textbf{Error} & \textbf{Aleatoric} & \textbf{Epistemic} & \textbf{Confidence} \\")
    latex.append(r"\midrule")
    
    # DATA ROWS
    # Clean baseline (level == 0)
    clean_rows = df_adv[np.isclose(df_adv['level'], 0.0)]
    if not clean_rows.empty:
        row_data = clean_rows.iloc[0]
        error = row_data['error']
        ale = row_data['aleatoric']
        ale_std = row_data['aleatoric_std']
        epi = row_data['epistemic']
        epi_std = row_data['epistemic_std']
        conf = row_data['confidence']
        conf_std = row_data['confidence_std']
        if show_std:
            row = r"\textbf{Clean} & -"
            row += f" & {error:.3f} & {ale:.3f}$\\pm${ale_std:.3f} & {epi:.3f}$\\pm${epi_std:.3f} & {conf:.3f}$\\pm${conf_std:.3f}"
        else:
            row = r"\textbf{Clean} & -"
            row += f" & {error:.3f} & {ale:.3f} & {epi:.3f} & {conf:.3f}"
        row += r" \\"
        latex.append(row)
        latex.append(r"\midrule")

    for attack in sorted(adversarial_attacks):
        if attack not in df_adv['perturbation'].unique():
            continue
            
        df_attack = df_adv[df_adv['perturbation'] == attack].copy()
        levels = sorted(l for l in df_attack['level'].unique() if not np.isclose(l, 0.0))
        
        for i, level in enumerate(levels):
            df_level = df_attack[df_attack['level'] == level]
            
            if len(df_level) == 0:
                continue
            
            row_data = df_level.iloc[0]
            
            # FIRST COLUMN ATTACK NAME MULTIROW
            if i == 0:
                attack_name = attack.upper()
                row = r"\multirow{" + str(len(levels)) + r"}{*}{\textbf{" + attack_name + r"}}"
            else:
                row = ""
            
            # SECOND COLUMN LEVEL
            row += f" & {level:.2f}"
            
            # DATA COLUMNS
            error = row_data['error']
            ale = row_data['aleatoric']
            ale_std = row_data['aleatoric_std']
            epi = row_data['epistemic']
            epi_std = row_data['epistemic_std']
            conf = row_data['confidence']
            conf_std = row_data['confidence_std']
            
            if show_std:
                row += f" & {error:.3f} & {ale:.3f}$\\pm${ale_std:.3f} & {epi:.3f}$\\pm${epi_std:.3f} & {conf:.3f}$\\pm${conf_std:.3f}"
            else:
                row += f" & {error:.3f} & {ale:.3f} & {epi:.3f} & {conf:.3f}"
            
            row += r" \\"
            latex.append(row)
        
        # ADD SEPARATOR BETWEEN ATTACKS
        latex.append(r"\midrule")
    
    # CLOSE TABLE
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # SAVE TO FILE
    output_file = output_dir / f"adversarial_attacks_{dataset_name}.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"OK Adversarial attacks table saved: {output_file}")
    return output_file


def create_markdown_tables(df, dataset_name, output_dir):
    """Create Markdown tables (separate for natural and adversarial)."""
    
    md = []
    md.append(f"# Uncertainty Metrics: {dataset_name.replace('_', ' ').title()}\n")
    
    # NATURAL PERTURBATIONS
    md.append("## Natural Perturbations\n")
    
    natural_perts = ['gaussian', 'missing', 'permutation']
    models = ['Logistic', 'RF', 'MLP', 'LightGBM', 'CatBoost']
    models = [m for m in models if m in df['model'].unique()]
    
    for pert in sorted(natural_perts):
        df_pert = df[df['perturbation'] == pert].copy()
        if len(df_pert) == 0:
            continue
            
        levels = sorted(df_pert['level'].unique())
        
        md.append(f"\n### {pert.upper().replace('_', ' ')}\n")
        
        # HEADER
        header = "| Level |"
        separator = "|-------|"
        for model in models:
            header += f" {model} Err | {model} Ale | {model} Epi | {model} Conf |"
            separator += "------:|------:|------:|------:|"
        md.append(header)
        md.append(separator)
        
        # DATA ROWS
        for level in levels:
            df_level = df_pert[df_pert['level'] == level]
            row = f"| {level:.2f} |"
            
            for model in models:
                df_model = df_level[df_level['model'] == model]
                
                if len(df_model) > 0:
                    row_data = df_model.iloc[0]
                    error = row_data['error']
                    ale = row_data['aleatoric']
                    epi = row_data['epistemic']
                    conf = row_data['confidence']
                    
                    row += f" {error:.3f} | {ale:.3f} | {epi:.3f} | {conf:.3f} |"
                else:
                    row += " - | - | - | - |"
            
            md.append(row)
    
    # ADVERSARIAL ATTACKS
    adversarial_attacks = ['bim', 'pgd', 'cw']
    df_adv = df[(df['perturbation'].isin(adversarial_attacks)) & (df['model'] == 'MLP')].copy()
    
    if len(df_adv) > 0:
        md.append("\n## Adversarial Attacks (MLP only)\n")
        
        for attack in sorted(adversarial_attacks):
            df_attack = df_adv[df_adv['perturbation'] == attack].copy()
            if len(df_attack) == 0:
                continue
                
            levels = sorted(df_attack['level'].unique())
            
            md.append(f"\n### {attack.upper()}\n")
            
            # HEADER
            header = "| Level | Error | Aleatoric | Epistemic | Confidence |"
            separator = "|-------|------:|----------:|----------:|-----------:|"
            md.append(header)
            md.append(separator)
            
            # DATA ROWS
            for level in levels:
                df_level = df_attack[df_attack['level'] == level]
                
                if len(df_level) > 0:
                    row_data = df_level.iloc[0]
                    error = row_data['error']
                    ale = row_data['aleatoric']
                    epi = row_data['epistemic']
                    conf = row_data['confidence']
                    
                    row = f"| {level:.2f} | {error:.3f} | {ale:.3f} | {epi:.3f} | {conf:.3f} |"
                    md.append(row)
    
    # SAVE TO FILE
    output_file = output_dir / f"uncertainty_tables_{dataset_name}.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"OK Markdown tables saved: {output_file}")
    return output_file


def combine_tex_tables(tex_paths, output_dir, output_filename="perturbation_all.tex"):
    """Combine all generated LaTeX table files into a single file for easy copy/paste."""
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
    print("GENERATING UNCERTAINTY TABLES")
    print(f"Font size: {FONT_SIZE}")
    print(f"Show std (natural): {SHOW_STD_NATURAL}")
    print(f"Show std (adversarial): {SHOW_STD_ADVERSARIAL}")
    print("=" * 70)
    
    for dataset_name, uci_id in datasets:
        print(f"\n--- {dataset_name.upper()} ---")
        
        # LOAD DATA
        df = load_results(dataset_name, uci_id)
        if df is None:
            continue
        
        print(f"Loaded {len(df)} rows")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Perturbations: {sorted(df['perturbation'].unique())}")
        
        # GENERATE NATURAL PERTURBATION TABLE
        print("\nGenerating natural perturbation table...")
        nat_tex = create_natural_perturbation_table(df, dataset_name, output_dir, 
                                                   font_size=FONT_SIZE, 
                                                   show_std=SHOW_STD_NATURAL)
        if nat_tex:
            tex_files.append(nat_tex)
        
        # GENERATE ADVERSARIAL TABLE
        print("Generating adversarial attacks table...")
        adv_tex = create_adversarial_table(df, dataset_name, output_dir,
                                           font_size=FONT_SIZE,
                                           show_std=SHOW_STD_ADVERSARIAL)
        if adv_tex:
            tex_files.append(adv_tex)
        
        # GENERATE MARKDOWN TABLES FOR PREVIEW
        print("Generating Markdown tables...")
        create_markdown_tables(df, dataset_name, output_dir)
    
    print("\nCombining LaTeX tables into a single file...")
    combine_tex_tables(tex_files, output_dir)
    
    print("\n" + "=" * 70)
    print(f"DONE! Tables saved to {output_dir}/")
    print("=" * 70)
    print("\nFiles generated per dataset:")
    print("  - natural_perturbation_*.tex     (Gaussian, Missing, Permutation)")
    print("  - adversarial_attacks_*.tex      (BIM, PGD, C&W - MLP only)")
    print("  - uncertainty_tables_*.md        (Both tables in Markdown)")
    print("\nAdditional combined file:")
    print("  - perturbation_all.tex           (All LaTeX tables concatenated)")
    print("\nTo change font size, edit FONT_SIZE at top of script:")
    print("  Options: 'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize'")
    print("\nTo toggle ±std display, edit SHOW_STD_NATURAL and SHOW_STD_ADVERSARIAL")


if __name__ == "__main__":
    main()
