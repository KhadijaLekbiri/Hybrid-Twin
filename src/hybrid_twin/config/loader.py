import yaml
import numpy as np
from pathlib import Path


class Config:
    """
    Loads and exposes all configuration sections from config.yaml.
    """

    def __init__(self, model: dict, paths: dict, simulation: dict):
        self.model      = model
        self.paths      = paths
        self.simulation = simulation

    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {config_file.resolve()}")

        with open(config_file, "r") as f:
            raw = yaml.safe_load(f)

        ROOT = config_file.resolve().parent

        m = raw["model"]
        model_params = {
            "INPUT_SIZE":  m["input_size"],
            "HIDDEN_SIZE": m["hidden_size"],
            "NUM_LAYERS":  m["num_layers"],
            "DROPOUT":     m["dropout"],
            "BATCH_SIZE":  m["batch_size"],
            "EPOCHS":      m["epochs"],
            "LR":          m["lr"],
            "ALPHA":       m["alpha"],
            "BETA":        m["beta"],
        }

        p = raw["paths"]
        path_params = {
            "ROOT":        ROOT,
            "DATASET_DIR": ROOT / p["dataset_dir"],
            "OUTPUT_FILE": ROOT / p["output_file"],
            "DATA_DIR":    ROOT / p["data_dir"],
            "OUTPUT_DIR":  ROOT / p["output_dir"],
        }

        s = raw["simulation"]
        simulation_params = {
            "SEED":           s["seed"],
            "N_SIMULATIONS":  s["n_simulations"],
            "SEQ_LENGTH":     s["seq_length"],
            "DT":             s["dt"],
            "R":              s["r"],
            "AK_MODES":       np.array(s["ak_modes"]),
            "MAPPING_MATRIX": np.array(s["mapping_matrix"]),
        }

        return cls(model=model_params, paths=path_params, simulation=simulation_params)

    def __getitem__(self, key):
        """Backward-compat: config['model'], config['paths'], config['simulation']."""
        return {"model": self.model, "paths": self.paths, "simulation": self.simulation}[key]


# Backward-compat alias — existing callers of load_config() still work unchanged
def load_config(config_path: str = "config.yaml") -> "Config":
    return Config.from_yaml(config_path)
