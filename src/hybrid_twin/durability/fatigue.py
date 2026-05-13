import rainflow
import numpy as np


class FatigueAnalyser:
    """
    Rainflow-based fatigue damage calculator.
    """

    def __init__(self, k: float = 1000, m: float = 5):
        self.k = k
        self.m = m

    def calculate_damage(self, signal: np.ndarray) -> float:
        """Compute cumulative fatigue damage for a single stress signal."""
        damage = 0.0
        for rng, mean, count, i_start, i_end in rainflow.extract_cycles(signal):
            amp = rng / 2
            if amp > 0:
                Ni = self.k * (amp ** (-self.m))
                if Ni > 0:
                    damage += count / Ni
        return damage

    def batch_damage(self, signals: np.ndarray) -> np.ndarray:
        """Compute damage for every row in a (N, T) array."""
        return np.array([self.calculate_damage(signals[i]) for i in range(len(signals))])


# Backward-compat alias
def calculate_damage(signal: np.ndarray, k: float = 1000, m: float = 5) -> float:
    return FatigueAnalyser(k, m).calculate_damage(signal)
