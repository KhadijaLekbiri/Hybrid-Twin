import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss for the seq-to-seq Hybrid Twin bias learner.

    Two terms:
      * MSE term      : data-fit on the full predicted delta sequence
      * Gradient term : penalises rapid variations of delta along the
                        TIME axis (encourages temporal smoothness).

    Important: in the previous (seq-to-one) version the gradient penalty
    was computed across the BATCH axis, which is meaningless.
    Now it is computed along the TIME axis, which is the physical one.
    """

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.mse   = nn.MSELoss()

    def forward(self, delta_pred: torch.Tensor, delta_true: torch.Tensor) -> torch.Tensor:
        if delta_pred.dim() == 2:
            delta_pred = delta_pred.unsqueeze(-1)
        if delta_true.dim() == 2:
            delta_true = delta_true.unsqueeze(-1)

        mse_loss = self.mse(delta_pred, delta_true)

        if delta_pred.size(1) > 1:
            d_dt          = delta_pred[:, 1:, :] - delta_pred[:, :-1, :]
            grad_penalty  = torch.mean(d_dt ** 2)
        else:
            grad_penalty  = torch.tensor(0.0, device=delta_pred.device)

        return self.alpha * mse_loss + self.beta * grad_penalty