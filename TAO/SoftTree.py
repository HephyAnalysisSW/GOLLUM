import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantize import quantize_torch as quantize

class STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantization):
        # here you implement your quantization, e.g. sign or ternary
        return quantize(x, quantization)
    @staticmethod
    def backward(ctx, grad_output):
        # pretend quantize_int was identity:
        return grad_output, None

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
        self.temperature = 1.0 
        self.quantization = self.config.get('quantization', None)
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
        # Reduce the variance by 1/dim because we don't want Wx+b to populate large values.
        #self.leaf_W = nn.Parameter(1./self.input_dim*torch.randn(self.num_leaves, self.input_dim, dtype=self.dtype))
        self.leaf_W = nn.Parameter(1./self.input_dim*torch.zeros(self.num_leaves, self.input_dim, dtype=self.dtype))
        #self.leaf_b = nn.Parameter(-log(2)+1./self.input_dim*torch.randn(self.num_leaves, dtype=self.dtype))
        self.leaf_b = nn.Parameter(
            torch.full(
                (self.num_leaves,), 
                -math.log(2), 
                dtype=self.dtype
            ))

    def print(self):
        """
        Print the tree structure with right-first indentation.
        """
        threshold = 1e-3
        def _recurse(node_idx, depth):
            prefix = '  ' * depth
            if node_idx < self.num_internal:
                w = self.split_W[node_idx].detach().cpu().numpy()
                terms = [f"{v:.3f}*x{j}" for j, v in enumerate(w) if abs(v) >= threshold]
                w_str = " + ".join(terms) if terms else "0"
                b = float(self.split_b[node_idx].item())
                print(f"{prefix}[Split] Node {node_idx} at depth {depth} → {w_str} + {b:.3f} >= 0")
                # right then left
                _recurse(2*node_idx+2, depth+1)
                _recurse(2*node_idx+1, depth+1)
            else:
                leaf_idx = node_idx - self.num_internal
                w = self.leaf_W[leaf_idx].detach().cpu().numpy()
                terms = [f"{v:.3f}*x{j}" for j, v in enumerate(w) if abs(v) >= threshold]
                w_str = " + ".join(terms) if terms else "0"
                b = float(self.leaf_b[leaf_idx].item())
                print(f"{prefix}[Leaf] Node {node_idx} at depth {self.depth} → prediction = {w_str} + {b:.3f}")
        _recurse(0, 0)

    def set_standardization(self, X_mean, X_std):
        """Store dataset mean and std for feature standardization."""
        self.X_mean.copy_(torch.tensor(X_mean, dtype=self.dtype))
        self.X_std.copy_(torch.tensor(X_std, dtype=self.dtype))

    def standardize_input(self, X):
        return (X.to(self.dtype) - self.X_mean) / self.X_std

    def set_temperature( self, temperature):
        self.temperature = temperature

    def forward(self, X):
        """
        Compute soft-tree prediction in log‐space for numerical stability.
        """
        # Standardize
        X = self.standardize_input(X)
        N = X.size(0)

        # --- SPLIT QUANTIZATION (via STEQuant) ---
        b_split_q = self.split_b
        if self.quantization is not None:
            W_split_q = STEQuant.apply(self.split_W, self.quantization)
        else:
            W_split_q = self.split_W

        # 1) compute split probabilities
        d = F.linear(X, W_split_q, b_split_q)
        #d     = F.linear(X, self.split_W, self.split_b)  # (N, num_internal)
        raw_p = d / self.temperature
        eps   = 1e-6
        probs = torch.sigmoid(raw_p).clamp(eps, 1 - eps)  # avoid exact 0 or 1

        # 2) log‐space
        logp   = probs.log()
        log1mp = torch.log1p(-probs)  # log(1 - p)

        # 3) build log‐routing masks recursively
        log_masks = []
        def recurse(i, curr_log):
            if i >= self.num_internal:
                log_masks.append(curr_log)
            else:
                recurse(2*i+2, curr_log + logp[:, i:i+1])      # right
                recurse(2*i+1, curr_log + log1mp[:, i:i+1])    # left

        recurse(0, X.new_zeros(N, 1, dtype=self.dtype))
        log_routing = torch.cat(log_masks, dim=1)   # (N, num_leaves)
        routing     = log_routing.exp()             # back to normal space

        # --- LEAF QUANTIZATION (via STEQuant) ---
        b_leaf_q = self.leaf_b
        if self.quantization is not None:
            W_leaf_q = STEQuant.apply(self.leaf_W, self.quantization)
        else:
            W_leaf_q = self.leaf_W

        # Leaf predictions and final output
        leaf_out = X @ W_leaf_q.t() + b_leaf_q

        #print( self.leaf_W, self.leaf_b, W_leaf_q, b_leaf_q, leaf_out )

        out = (routing * leaf_out.exp()).sum(dim=1)
        #print ("leaf_out", leaf_out.shape, leaf_out)
        #print ("routing", routing.shape, routing, routing.sum(axis=1), routing.sum(axis=0))
        #print ("log_routing", log_routing.shape, log_routing, log_routing.sum(axis=1), log_routing.sum(axis=0))

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
