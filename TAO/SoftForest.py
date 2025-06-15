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
        # configure and cast to desired dtype
        dt = config.get('dtype', 'float32')
        if isinstance(dt, str):
            self.dtype = getattr(torch, dt)
        else:
            self.dtype = dt
        # cast entire module (and its submodules) to this dtype
        self.to(self.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the ensemble prediction.

        Args:
            X: Tensor of shape (N, D)
        Returns:
            Tensor of shape (N,) equal to sum_t tree_t(X)
        """
        # ensure input is correct dtype
        X = X.to(self.dtype)
        # collect each tree's soft output
        outputs = [tree(X) for tree in self.trees]  # list of (N,) tensors
        stacked = torch.stack(outputs, dim=0)       # shape (T, N)
        return torch.sum(stacked, dim=0)

    def save(self, path: str, epoch: int):
        """
        Save the forest's state_dict and config to a checkpoint file for this epoch.
        """
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"forest_epoch_{epoch}.pt")
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, fname)

    @classmethod
    def load(cls, path: str, epoch: int = None):
        """
        Load a saved forest from disk. Instantiates trees from stored config.
        If epoch is None, loads the latest checkpoint.

        Returns:
            A SoftForest instance with parameters loaded.
        """
        # find checkpoint file
        if epoch is None:
            files = sorted(glob.glob(os.path.join(path, 'forest_epoch_*.pt')))
            if not files:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            checkpoint_file = files[-1]
        else:
            checkpoint_file = os.path.join(path, f"forest_epoch_{epoch}.pt")
            if not os.path.isfile(checkpoint_file):
                raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found")

        # load data
        data = torch.load(checkpoint_file, map_location='cpu')
        config = data['config']

        # instantiate trees from config
        # assumes a SoftTree class available in scope
        ntrees = config.get('ntrees', 1)
        from SoftTree import SoftTree  # or appropriate import
        trees = [SoftTree(config) for _ in range(ntrees)]

        # create forest and load weights
        forest = cls(trees, config)
        forest.load_state_dict(data['state_dict'])
        return forest

