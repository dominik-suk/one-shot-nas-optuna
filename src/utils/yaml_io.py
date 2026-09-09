import yaml

from src.paths import PAMAP2_SEARCH_SPACE_PATH


def load_search_space(search_space_path):
    with open(search_space_path, "r") as search_space_file:
        return yaml.safe_load(search_space_file)


def load_pamap2_search_space():
    return load_search_space(PAMAP2_SEARCH_SPACE_PATH)
