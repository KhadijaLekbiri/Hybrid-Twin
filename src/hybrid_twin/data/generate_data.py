import numpy as np
import pandas as pd
from pathlib import Path

from hybrid_twin.config.loader import load_config
from hybrid_twin.physics.signal_generator import RealSignalGenerator
from hybrid_twin.physics.rom_model import ROMModel

cfg = load_config(Path(__file__).resolve().parent.parent.parent.parent / "config.yaml")

np.random.seed(cfg["simulation"]["SEED"])
N_SIMULATIONS = cfg["simulation"]["N_SIMULATIONS"]
SEQ_LENGTH = cfg["simulation"]["SEQ_LENGTH"]
AK_MODES = cfg["simulation"]["AK_MODES"]
MAPPING_MATRIX = cfg["simulation"]["MAPPING_MATRIX"]
OUTPUT_FILE = cfg["paths"]["OUTPUT_FILE"]


class DataGenerator:
    def __init__(self, ak_modes, mapping_matrix, output_file, noise_lvl: float = 0.01, seed: int = None):
        self.rom = ROMModel(ak_modes, mapping_matrix)
        self.synthesiser = RealSignalGenerator(noise_lvl)
        self.output_file = Path(output_file)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate(self, n_simulations, seq_length):
        all_x, all_y, rows = [], [], []
        print(f"Generating {n_simulations} sequences of length {seq_length}...")

        for i in range(n_simulations):
            forces = np.zeros((seq_length, 3))
            for j in range(3):
                forces[:, j] = np.cumsum(np.random.normal(0, 0.1, seq_length))

            sigma_sim, _ = self.rom.simulate(forces)
            sigma_real = self.synthesiser.generate(sigma_sim)
            delta = sigma_real - sigma_sim

            x_frame = np.column_stack((forces, sigma_sim))
            all_x.append(x_frame)
            all_y.append(delta.astype(np.float32))

            for t in range(seq_length):
                rows.append({
                    'simulation_id': i,
                    'timestep': t,
                    'Fx': forces[t, 0],
                    'Fy': forces[t, 1],
                    'Fz': forces[t, 2],
                    'sigma_sim': sigma_sim[t],
                    'sigma_real': sigma_real[t],
                    'delta': delta[t],
                })

        X = np.array(all_x, dtype=np.float32)
        y = np.array(all_y, dtype=np.float32)
        df = pd.DataFrame(rows)
        self.save(df, self.output_file)
        return X, y, df

    def save(self, df: pd.DataFrame, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def generate_data(n_simulations, seq_length, AK_MODES, MAPPING_MATRIX):
    generator = DataGenerator(
        ak_modes=AK_MODES,
        mapping_matrix=MAPPING_MATRIX,
        output_file=OUTPUT_FILE,
        seed=cfg["simulation"]["SEED"],
    )
    return generator.generate(n_simulations, seq_length)


if __name__ == "__main__":
    X, y, df = generate_data(N_SIMULATIONS, SEQ_LENGTH, AK_MODES, MAPPING_MATRIX)
    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}     <-- (N, T) instead of (N,)")
    print(f"y stats : mean={y.mean():.4f}  std={y.std():.4f}")
