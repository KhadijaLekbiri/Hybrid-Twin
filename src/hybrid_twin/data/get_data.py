from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Fx", "Fy", "Fz", "sigma_sim", "delta"]
FEATURE_COLUMNS  = ["Fx", "Fy", "Fz", "sigma_sim"]


class DataLoader:
    """
    Loads simulation data from CSV or Excel and returns sequence arrays.
    """

    def __init__(
        self,
        feature_cols: list = None,
        required_cols: list = None,
    ):
        self.feature_cols  = feature_cols  or FEATURE_COLUMNS
        self.required_cols = required_cols or REQUIRED_COLUMNS

    def load(self, file_path, seq_length: int = 1) -> tuple:
        """
        Returns:
            X  (N_sim, T, 4)   features per timestep
            y  (N_sim, T)      delta per timestep
            df                 cleaned dataframe
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Format non supporté : '{ext}'.")

        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        df = df.dropna(subset=self.required_cols).reset_index(drop=True)
        X, y = [], []
        for _, group in df.groupby("simulation_id"):
            group = group.sort_values("timestep").reset_index(drop=True)
            if len(group) < seq_length:
                continue
            X.append(group[self.feature_cols].to_numpy(dtype=np.float32))
            y.append(group["delta"].to_numpy(dtype=np.float32))

        return np.stack(X), np.stack(y), df


# Backward-compat alias
def get_data(file_path, seq_length=1):
    return DataLoader().load(file_path, seq_length)
