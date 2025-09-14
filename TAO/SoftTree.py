import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTree(nn.Module):
    """
    A soft decision tree implemented in PyTorch.
    Splits are parameterized by (W, b) with a temperature hyperparameter for sigmoid.
    Leaves predict a linear response W_leaf x + b_leaf.
    Configuration is loaded from a YAML file or dict.
    """
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        # Load config
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # Hyperparameters from config
        self.depth       = self.config['max_depth']
        self.input_dim   = self.config['input_dim']
        self.temperature = self.config.get('temperature', 1.0)
        self.dtype       = dtype

        # Number of internal and leaf nodes
        # Full binary tree of depth D has (2^D - 1) internal and 2^D leaves
        self.num_internal = 2**self.depth - 1
        self.num_leaves   = 2**self.depth

        # Standardization buffers
        self.register_buffer('X_mean', torch.zeros(self.input_dim, dtype=self.dtype))
        self.register_buffer('X_std',  torch.ones(self.input_dim, dtype=self.dtype))

        # Create split parameters: one weight vector and bias per internal node
        self.split_W = nn.Parameter(torch.randn(self.num_internal, self.input_dim, dtype=self.dtype))
        self.split_b = nn.Parameter(torch.randn(self.num_internal, dtype=self.dtype))

        # Create leaf parameters: one weight vector and bias per leaf
        self.leaf_W = nn.Parameter(torch.randn(self.num_leaves, self.input_dim, dtype=self.dtype))
        self.leaf_b = nn.Parameter(torch.randn(self.num_leaves, dtype=self.dtype))

    def set_standardization(self, X_mean, X_std):
        """Store dataset mean and std for feature standardization."""
        self.X_mean.copy_(torch.tensor(X_mean, dtype=self.dtype))
        self.X_std.copy_(torch.tensor(X_std, dtype=self.dtype))

    def standardize_input(self, X):
        return (X.to(self.dtype) - self.X_mean) / self.X_std

    def forward(self, X):
        """
        Compute soft-tree prediction for input batch X: (N, input_dim) -> (N,)
        """
        # Standardize
        X = self.standardize_input(X)
        N = X.shape[0]

        # Compute raw split decisions: (N, num_internal)
        # d = split_W x + b --> sigmoid(d / temperature)
        d = F.linear(X, self.split_W, self.split_b)  # (N, num_internal)
        probs = torch.sigmoid(d / self.temperature)

        # Build routing probabilities for each leaf
        mask = X.new_ones(N, 1, dtype=self.dtype)
        routing = []  # list of (N, 1) masks per leaf
        decisions = probs

        def recurse(node_idx, curr_mask):
            if node_idx >= self.num_internal:
                routing.append(curr_mask)
            else:
                p = decisions[:, node_idx:node_idx+1]
                recurse(2*node_idx+1, curr_mask * (1 - p))  # left
                recurse(2*node_idx+2, curr_mask * p)        # right

        recurse(0, mask)
        routing = torch.cat(routing, dim=1)  # (N, num_leaves)

        # Leaf predictions: (N, num_leaves)
        leaf_out = X @ self.leaf_W.t() + self.leaf_b

        # Final output = sum over leaves of routing * leaf_out
        out = (routing * leaf_out).sum(dim=1)
        return out

    def save(self, path, epoch):
        """
        Save model state and config. Creates directory if needed.
        """
        os.makedirs(path, exist_ok=True)
        cfg_file = os.path.join(path, 'tree_config.yaml')
        if not os.path.exists(cfg_file):
            with open(cfg_file, 'w') as f:
                yaml.safe_dump(self.config, f)
        model_file = os.path.join(path, f'softtree_epoch_{epoch}.pt')
        torch.save(self.state_dict(), model_file)

    @classmethod
    def load(cls, path, epoch=None, dtype=torch.float32):
        """
        Load the latest or specified epoch model from directory.
        """
        cfg_file = os.path.join(path, 'tree_config.yaml')
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)

        if epoch is None:
            files = [f for f in os.listdir(path) if f.startswith('softtree_epoch_') and f.endswith('.pt')]
            files.sort()
            if not files:
                raise FileNotFoundError(f'No model files in {path}')
            epoch = int(files[-1].split('_')[-1].split('.')[0])

        model_file = os.path.join(path, f'softtree_epoch_{epoch}.pt')
        model = cls(config, dtype=dtype)
        state = torch.load(model_file, map_location=lambda s,t: s)
        model.load_state_dict(state)
        return model
