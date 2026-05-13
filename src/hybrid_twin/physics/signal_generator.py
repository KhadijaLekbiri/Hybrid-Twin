import numpy as np


class RealSignalGenerator:
    """
    Generates a realistic signal from simulated stress via tanh saturation
    plus small Gaussian measurement noise.
    """

    def __init__(self, noise_lvl: float = 0.01):
        self.noise_lvl = noise_lvl

    def generate(self, sigma_sim) -> np.ndarray:
        """
        Args:
            sigma_sim: float or ndarray of shape (SEQ_LENGTH,)

        Returns:
            Noisy non-linear version of the input signal (same shape).
        """
        sigma_sim    = np.asarray(sigma_sim, dtype=float)
        scalar_input = sigma_sim.ndim == 0
        sigma_sim    = np.atleast_1d(sigma_sim)

        sig_nl = 2.0 * np.tanh(sigma_sim / 1.5)
        noise  = np.random.normal(0, self.noise_lvl, len(sigma_sim))

        result = sig_nl + noise
        return float(result[0]) if scalar_input else result


# Backward-compat alias
def generate_real_signal(sigma_sim, noise_lvl=0.01):
    return RealSignalGenerator(noise_lvl).generate(sigma_sim)
