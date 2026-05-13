import numpy as np
import torch
import joblib
from pathlib import Path

from hybrid_twin.config.loader import load_config
from hybrid_twin.model.model   import BiasLearner


class Predictor:
    """
    Loads a trained BiasLearner checkpoint and runs inference on test data.
    """

    def __init__(self, model: BiasLearner, cfg, device: torch.device):
        self.model  = model
        self.cfg    = cfg
        self.device = device

    def from_checkpoint(cls, config_path: str = "config.yaml") -> "Predictor":
        cfg        = load_config(Path(config_path).resolve())
        OUTPUT_DIR = cfg["paths"]["OUTPUT_DIR"]
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            OUTPUT_DIR / "best_bias_learner.pt",
            map_location=device,
            weights_only=False,
        )
        model_cfg = checkpoint["config"]
        model = BiasLearner(**{k: model_cfg[k] for k in
                               ["input_size", "hidden_size", "num_layers", "dropout"]}).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return cls(model=model, cfg=cfg, device=device)

    def predict(self, config_path: str = None) -> tuple:
        """
        Returns:
            sigma_sim    (N, T)
            sigma_hybrid (N, T)
            sigma_real   (N, T)
            delta_pred   (N, T)
            delta_true   (N, T)
        """
        cfg      = self.cfg
        DATA_DIR = cfg["paths"]["DATA_DIR"]

        X_test   = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
        y_test   = np.load(DATA_DIR / "y_test.npy").astype(np.float32)
        scaler_X = joblib.load(DATA_DIR / "scaler_X.pkl")
        scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")

        if y_test.ndim == 1:
            y_test = y_test[:, None]

        with torch.no_grad():
            delta_pred_scaled = (
                self.model(torch.from_numpy(X_test).to(self.device))
                .cpu().numpy().squeeze(-1)
            )

        N, T, _ = X_test.shape

        delta_pred = scaler_y.inverse_transform(delta_pred_scaled.reshape(-1, 1)).reshape(N, T)
        delta_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(N, -1)

        if delta_true.shape[1] != T:
            padded = np.zeros((N, T), dtype=delta_true.dtype)
            padded[:, -1] = delta_true[:, 0]
            delta_true = padded

        X_denorm     = scaler_X.inverse_transform(X_test.reshape(-1, 4)).reshape(N, T, 4)
        sigma_sim    = X_denorm[:, :, 3]
        sigma_real   = sigma_sim + delta_true
        sigma_hybrid = sigma_sim + delta_pred

        return sigma_sim, sigma_hybrid, sigma_real, delta_pred, delta_true


# Backward-compat alias
def load_model_and_predict(config_path: str = "config.yaml"):
    return Predictor.from_checkpoint(config_path).predict()
