from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

PAMAP2_SEARCH_SPACE_CONFIG_PATH = CONFIGS_DIR / "PAMAP2_search_space.yaml"
PAMAP2_FIXED_ARCH_CONFIG_PATH = CONFIGS_DIR / "PAMAP2_fixed_architecture.yaml"

PAMAP2_DIR = DATA_DIR / "PAMAP2"
PAMAP2_DATA_LOADERS_DIR = PAMAP2_DIR / "DataLoaders"

BASELINE_EXPERIMENT_DB_PATH = EXPERIMENTS_DIR / "PAMAP2_Baseline_NAS_Experiment.db"
SPOS_EXPERIMENT_DB_PATH = EXPERIMENTS_DIR / "PAMAP2_SPOS_NAS_Experiment.db"
BASELINE_RANDOM_EXPERIMENT_DB_PATH = EXPERIMENTS_DIR / "PAMAP2_Baseline_Random_Experiment.db"
SPOS_RANDOM_EXPERIMENT_DB_PATH = EXPERIMENTS_DIR / "PAMAP2_SPOS_Random_Experiment.db"

RANKING_CORRELATION_DATA_PATH = LOGS_DIR / "ranking_correlation" / "ranking_correlation.csv"
BENCHMARK_SUMMARY_PATH = EXPERIMENTS_DIR / "benchmark_summary.csv"

SUPERNET_PATH = MODELS_DIR / "supernet" / "Supernet.pth"
SPOS_BEST_MODEL_PATH = MODELS_DIR / "supernet" / "Supernet_NAS_Best_Model.pth"
SPOS_RANDOM_BEST_MODEL_PATH = MODELS_DIR / "supernet" / "Supernet_Random_NAS_Best_Model.pth"
BASELINE_BEST_MODEL_PATH = MODELS_DIR / "baseline" / "baseline_NAS_Best_Model.pth"

SEARCH_PLOTS_DIR = PLOTS_DIR / "search_progress"
HEATMAPS_DIR = PLOTS_DIR / "heatmaps"
LEARNING_CURVES_DIR = PLOTS_DIR / "learning_curves"
BAR_CHARTS_DIR = PLOTS_DIR / "bar_charts"
SCATTER_PLOTS_DIR = PLOTS_DIR / "scatter_plots"
