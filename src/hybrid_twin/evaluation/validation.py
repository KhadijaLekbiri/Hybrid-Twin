import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hybrid_twin.inference.predict import Predictor
from hybrid_twin.durability.fatigue import FatigueAnalyser

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 130, 'axes.titleweight': 'bold', 'axes.titlesize': 13})


class Validator:
    """
    Runs end-to-end evaluation: inference → fatigue damage → KPI reporting → plots.
    """

    def __init__(self, project_root: str = PROJECT_ROOT):
        self.project_root = Path(project_root)
        self.analyser     = FatigueAnalyser()

    def run(self):
        print("Loading model and predicting...")
        predictor = Predictor.from_checkpoint(self.project_root / "config.yaml")
        sigma_sim, sigma_hybrid, sigma_real, delta_pred, delta_true = predictor.predict()
        N, T = sigma_sim.shape
        print(f"Test samples : {N}   |   sequence length : {T}")

        print("Computing fatigue damage...")
        D_real   = self.analyser.batch_damage(sigma_real)
        D_sim    = self.analyser.batch_damage(sigma_sim)
        D_hybrid = self.analyser.batch_damage(sigma_hybrid)

        err_sim    = np.abs(D_sim    - D_real) / (np.abs(D_real) + 1e-12) * 100
        err_hybrid = np.abs(D_hybrid - D_real) / (np.abs(D_real) + 1e-12) * 100
        mse_delta  = np.mean((delta_pred - delta_true) ** 2)
        mae_delta  = np.mean(np.abs(delta_pred - delta_true))

        self._print_kpis(err_sim, err_hybrid, mse_delta, mae_delta)
        self.plot_damage_distribution(err_sim, err_hybrid, D_real, D_sim, D_hybrid)
        self.plot_bias_prediction(delta_pred, delta_true, N, T)
        self.plot_signals(sigma_sim, sigma_real, sigma_hybrid, N, T)

        reduction = (err_sim.mean() - err_hybrid.mean()) / err_sim.mean() * 100
        print(f"Réduction de l'erreur de dommage : {reduction:.1f}%")
        print(" KPI atteint : erreur dommage hybride < 5%" if err_hybrid.mean() < 5
              else f"KPI non atteint : erreur = {err_hybrid.mean():.1f}% (cible < 5%)")

    def _print_kpis(self, err_sim, err_hybrid, mse_delta, mae_delta):
        print("\n========== KPIs Phase 5 (seq-to-seq) ==========")
        print(f"Erreur dommage — ROM pur : {err_sim.mean():.1f}% ")
        print(f"Erreur dommage — Hybride : {err_hybrid.mean():.1f}%  (cible < 5%)")
        print(f"MSE biais delta : {mse_delta:.6f}")
        print(f"MAE biais delta : {mae_delta:.6f}\n")

    def plot_damage_distribution(self, err_sim, err_hybrid, D_real, D_sim, D_hybrid) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(err_sim,    bins=40, color='tomato',    alpha=0.8, label='ROM pur')
        axes[0].hist(err_hybrid, bins=40, color='steelblue', alpha=0.8, label='Hybride')
        axes[0].axvline(5,  color='green',  linestyle='--', lw=1.8, label='Cible 5%')
        axes[0].axvline(25, color='orange', linestyle='--', lw=1.8, label='Seuil ROM 25%')
        axes[0].set_xlabel("Erreur relative dommage (%)"); axes[0].set_ylabel("Nombre de simulations")
        axes[0].set_title("Distribution des erreurs de dommage"); axes[0].legend()
        axes[1].scatter(D_real, D_sim,    alpha=0.4, s=18, color='tomato',    label='ROM pur')
        axes[1].scatter(D_real, D_hybrid, alpha=0.4, s=18, color='steelblue', label='Hybride')
        lims = [min(D_real.min(), D_sim.min()), max(D_real.max(), D_sim.max())]
        axes[1].plot(lims, lims, 'k--', lw=1.2, label='Parfait')
        axes[1].set_xlabel("Dommage réel D"); axes[1].set_ylabel("Dommage prédit D")
        axes[1].set_title("Dommage prédit vs réel"); axes[1].legend()
        plt.tight_layout()
        out = self.project_root / "outputs" / "figures" / "phase5_damage_distribution.png"
        plt.savefig(out, bbox_inches='tight'); plt.show()
        return out

    def plot_bias_prediction(self, delta_pred, delta_true, N, T) -> Path:
        np.random.seed(0)
        examples = np.random.choice(N, min(3, N), replace=False)
        fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4), sharey=True)
        if len(examples) == 1: axes = [axes]
        t = np.arange(T)
        for ax, i in zip(axes, examples):
            ax.plot(t, delta_true[i], 'k-',  lw=1.5, label=r'$\delta$ réel')
            ax.plot(t, delta_pred[i], 'b--', lw=1.5, label=r'$\hat{\delta}$ prédit')
            ax.fill_between(t, delta_true[i], delta_pred[i], alpha=0.15, color='blue')
            ax.set_title(f"Simulation #{i}"); ax.set_xlabel("Timestep")
        axes[0].set_ylabel(r"Biais $\delta(t)$"); axes[0].legend(fontsize=9)
        plt.suptitle("Biais prédit vs réel — séquence complète", fontweight='bold')
        plt.tight_layout()
        out = self.project_root / "outputs" / "figures" / "phase5_bias_prediction.png"
        plt.savefig(out, bbox_inches='tight'); plt.show()
        return out

    def plot_signals(self, sigma_sim, sigma_real, sigma_hybrid, N, T) -> Path:
        np.random.seed(0)
        examples = np.random.choice(N, min(3, N), replace=False)
        fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4), sharey=True)
        if len(examples) == 1: axes = [axes]
        t = np.arange(T)
        for ax, i in zip(axes, examples):
            ax.plot(t, sigma_sim[i],    'tomato',    lw=1.4, label='ROM pur')
            ax.plot(t, sigma_real[i],   'black',     lw=1.4, label='Réel')
            ax.plot(t, sigma_hybrid[i], 'steelblue', lw=1.4, linestyle='--', label='Hybride')
            ax.set_title(f"Simulation #{i}"); ax.set_xlabel("Timestep")
        axes[0].set_ylabel(r"Contrainte $\sigma$"); axes[0].legend(fontsize=9)
        plt.suptitle("Signaux de contrainte — ROM / Réel / Hybride", fontweight='bold')
        plt.tight_layout()
        out = self.project_root / "outputs" / "figures" / "phase5_signals.png"
        plt.savefig(out, bbox_inches='tight'); plt.show()
        return out


if __name__ == "__main__":
    Validator().run()
