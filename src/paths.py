from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

HEATMAPS_DIR = PLOTS_DIR / "heatmaps"
PAMAP2_DIR = DATA_DIR / "PAMAP2"
PAMAP2_DATA_LOADERS_DIR = PAMAP2_DIR / "DataLoaders"
SAMPLES_DIR = MODELS_DIR / "samples"
SUPERNET_DIR = MODELS_DIR / "supernet"

SAMPLED_ADL_MODEL_PATH = SAMPLES_DIR / "adl_model_86_acc.pth"
SAMPLED_PROTOCOL_MODEL_PATH = SAMPLES_DIR / "protocol_model_81_acc.pth"
