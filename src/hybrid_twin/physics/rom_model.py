import numpy as np


class ROMModel:
    """
    PGD Reduced-Order Model.
    Holds the spatial modes and mapping matrix; exposes simulate().
    """

    def __init__(self, ak_modes: np.ndarray, mapping_matrix: np.ndarray):
        self.ak_modes       = np.asarray(ak_modes)
        self.mapping_matrix = np.asarray(mapping_matrix)

    def simulate(self, forces_t: np.ndarray) -> tuple:
        """
        Reconstruct stress using the PGD ROM.

        Args:
            forces_t: shape (3,) for a single point or (SEQ_LENGTH, 3) for a sequence.

        Returns:
            (sigma_sim, betas)
        """
        forces_t = np.asarray(forces_t)

        if forces_t.ndim == 1:
            betas     = np.dot(self.mapping_matrix, forces_t)          # (r,)
            sigma_sim = float(np.sum(self.ak_modes * betas))           # scalar
        else:
            betas     = np.dot(self.mapping_matrix, forces_t.T)        # (r, SEQ_LENGTH)
            sigma_sim = np.sum(self.ak_modes[:, None] * betas, axis=0) # (SEQ_LENGTH,)

        return sigma_sim, betas


# Backward-compat alias
def simulate_sigma_pgd(forces_t, Ak_modes, mapping_matrix):
    return ROMModel(Ak_modes, mapping_matrix).simulate(forces_t)
