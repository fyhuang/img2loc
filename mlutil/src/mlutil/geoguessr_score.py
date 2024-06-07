"""PyTorch metric implementing Geoguessr score.

>>> import torch
>>> s = GeoguessrScore()
>>> s.update(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]]))
>>> s.compute()
tensor(25000.)
>>> # Distance is 1034.6608 km, score should be ~12500
>>> s.update(torch.tensor([[40.74847, -73.98570]]), torch.tensor([[35.75040, -83.99380]]))
>>> torch.round(s.compute())
tensor(18750.)
"""

import torch
from torchmetrics import Metric

def _haversine(ll1, ll2):
    lat1, lon1 = torch.split(ll1, 1, dim=1)
    lat2, lon2 = torch.split(ll2, 1, dim=1)
    r = 6371  # Radius of Earth in kilometers
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    return torch.mean(2 * r * torch.asin(torch.sqrt(a)))

class GeoguessrScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Both preds and target should be a Mx2 tensor of lat/lng pairs."""
        if preds.shape != target.shape:
            raise ValueError(f"Preds ({preds.shape}) and target ({target.shape}) should have the same shape")

        diffs = _haversine(preds, target)
        # Using 25k instead of 5k to project total score from game (of 5 rounds)
        scores = 25000.0 * torch.exp(-diffs / 1492.7)

        self.score += scores.sum()
        self.count += preds.numel() // 2

    def compute(self) -> torch.Tensor:
        return self.score.float() / self.count


if __name__ == "__main__":
    import doctest
    doctest.testmod()