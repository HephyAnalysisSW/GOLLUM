import os
import glob
import torch
import torch.nn as nn

class SoftForest(nn.Module):
    """
    A collection of soft differentiable decision trees, summed together.
    Each tree is expected to be a torch.nn.Module implementing a forward(X)->(N,) output.
    """
    def __init__(self, trees: list, config: dict):
        super().__init__()
        # ModuleList ensures subtrees are registered and their parameters tracked
        self.trees = nn.ModuleList(trees)
        self.config = config

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the ensemble prediction.

        Args:
            X: Tensor of shape (N, D)
        Returns:
            Tensor of shape (N,) equal to sum_t tree_t(X)
        """
        # collect each tree's soft output
        outputs = [tree(X) for tree in self.trees]
        # stack to shape (T, N) and sum over trees
        stacked = torch.stack(outputs, dim=0)
        return torch.sum(stacked, dim=0)

    def save(self, path: str, epoch: int):
        """
        Save the forest's state_dict to a file for this epoch.
        """
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"forest_epoch_{epoch}.pt")
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, fname)

    @classmethod
    def load(cls, path: str, trees: list, epoch: int = None):
        """
        Load a saved forest from disk. Requires passing in the same list of instantiated tree modules.
        If epoch is None, loads the latest checkpoint.

        Returns:
            A Forest instance with parameters loaded.
        """
        # find most recent file
        if epoch is None:
            files = sorted(glob.glob(os.path.join(path, 'forest_epoch_*.pt')))
            if not files:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            checkpoint = files[-1]
        else:
            checkpoint = os.path.join(path, f"forest_epoch_{epoch}.pt")
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Checkpoint {{checkpoint}} not found")

        data = torch.load(checkpoint, map_location='cpu')
        config = data['config']
        forest = cls(trees, config)
        forest.load_state_dict(data['state_dict'])
        return forest

