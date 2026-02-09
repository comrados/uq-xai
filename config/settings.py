import random
import numpy as np
from pathlib import Path

# REPRODUCIBILITY

GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    # PYTORCH SEEDS IMPORT ONLY IF AVAILABLE
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# PATHS

PROJECT_ROOT = Path(__file__).parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODELS_CACHE_DIR = PROJECT_ROOT / "models" / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# DATA SETTINGS

# TRAIN VAL TEST SPLIT RATIOS
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# DATASET UCI IDS
WINE_UCI_ID = 186
COVERTYPE_UCI_ID = 31
BEAN_UCI_ID = 602
IRIS_UCI_ID = 53
RICE_UCI_ID = 545
ECOLI_UCI_ID = 39

# PERTURBATION SETTINGS

PERTURBATION_CONFIG = {
    'gaussian': [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    'missing': [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'permutation': [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
}

ADVERSARIAL_CONFIG = {
    'bim': {
        'n_iter': 10,
        'epsilons': [0.01, 0.05, 0.1, 0.2]
    },
    'pgd': {
        'n_iter': 20,
        'n_restarts': 1,
        'epsilons': [0.01, 0.05, 0.1, 0.2]
    },
    'cw': {
        'n_iter': 100,
        'lr': 0.01,
        'c_values': [0.1, 1.0, 10.0],
        'kappa': 0.0
    }
}

# MODEL SETTINGS

# DEFAULT HYPERPARAMETERS
LOGISTIC_DEFAULTS = {
    'C': 1.0
}

RANDOM_FOREST_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4
}

MLP_DEFAULTS = {
    'hidden_dims': [128, 64],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'max_epochs': 100,
    'early_stopping_patience': 20
}

LIGHTGBM_DEFAULTS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63,
    'early_stopping_rounds': 20,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7
}

CATBOOST_DEFAULTS = {
    # TRAINING HYPERPARAMETERS
    'iterations': 200,              # Number of boosting rounds (same as LightGBM)
    'learning_rate': 0.05,          # Boosting learning rate
    'depth': 6,                     # Maximum tree depth (CatBoost default)
    'l2_leaf_reg': 3.0,             # L2 regularization (CatBoost default)
    
    # EARLY STOPPING
    'early_stopping_rounds': 50,    # Rounds without improvement before stopping
    
    # UNCERTAINTY QUANTIFICATION VIRTUAL ENSEMBLES
    'n_virtual_ensembles': 10,      # Number of posterior samples for epistemic UQ
                                    # Higher = more accurate but slower
                                    # RECOMMENDED 10 20
                                    # COST 1 N FORWARD PASSES
                                    # Example: n=10 -> ~11 FP vs SHAP Tree (~1 FP)
                                    # STILL MUCH CHEAPER THAN SHAP KERNEL 32 FP
}

# UQ SETTINGS

UQ_CONFIG = {
    'bootstrap_n_models': 20,
    'mc_dropout_n_samples': 50,
    'ensemble_n_models': 20,
    # Bootstrap (100% samples with replacement) for epistemic uncertainty estimation.
    # Note: Using full bootstrap (1.0) instead of bagging (<1.0) to get proper epistemic.
    # Bagging would reduce overfitting but artificially lower epistemic uncertainty.
    # For UQ research, we prioritize interpretable epistemic over regularization.
    'ensemble_bag_fraction': 1.0,
    'quantile_levels': [0.1, 0.9]  # [q_low, q_high]
}

# XAI SETTINGS

XAI_CONFIG = {
    'shap_background_samples': 100,
    'lime_num_features': 10,
    'lime_num_samples': 5000,
    'ig_n_steps': 50,
    'smoothgrad_n_samples': 50,
    'smoothgrad_noise_level': 0.1,
    'stability_n_runs': 10,
    'explanation_subsample_size': 1000  # For large datasets
}

# EXPERIMENT SETTINGS

EXPERIMENT_CONFIG = {
    'n_explanation_runs': 10,  # For stability measurement
    'save_intermediate': True
}
