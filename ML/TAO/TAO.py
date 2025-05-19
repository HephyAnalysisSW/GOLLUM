import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression


def generate_overlapping_gaussians(n_per_class=100, d=2, separation=1.0, rng=None, weight_mode="uniform"):
    """
    Generate labeled and weighted data from two overlapping Gaussian distributions.

    Args:
        n_per_class (int): Number of samples per class.
        d (int): Dimensionality of the feature space.
        separation (float): Distance between the class means.
        rng (np.random.Generator or None): Optional random number generator.
        weight_mode (str): One of ["uniform", "random", "class_imbalance"]

    Returns:
        X (np.ndarray): Array of shape (2 * n_per_class, d) with feature vectors.
        y (np.ndarray): Array of shape (2 * n_per_class,) with labels {0, 1}.
        w (np.ndarray): Array of shape (2 * n_per_class,) with sample weights.
    """
    rng = np.random.default_rng(rng)
    
    # Define means
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[0] = separation

    # Covariance
    cov = np.eye(d)

    # Generate samples
    X0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    # Assign weights
    if weight_mode == "uniform":
        w = np.ones(2 * n_per_class)
    elif weight_mode == "random":
        w = rng.uniform(0.5, 2.0, size=2 * n_per_class)
    elif weight_mode == "class_imbalance":
        w = np.concatenate([np.ones(n_per_class), np.full(n_per_class, 2.0)])
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return X, y, w

class TreeNode:
    def __init__(self, node_id, depth, is_leaf=False):
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = is_leaf

        self.W = None  # (1, d) for oblique splits
        self.b = None

        self.prediction = None  # scalar value for regression
        self.n_instances = None  # Number of training instances 

        self.left = None
        self.right = None

    def is_internal(self):
        return not self.is_leaf

    def set_split(self, W, b):
        self.W = W
        self.b = b

    def set_prediction(self, value):
        self.prediction = value

class Tree:
    def __init__(self, max_depth, input_dim, rng=None, order = "reverse_bfs"):
        self.max_depth = max_depth
        self.input_dim = input_dim
        self.root = None
        self.nodes = {}
        self.rng = np.random.default_rng(rng)

        self._build_tree()

        self.X_mean = None # Standardization
        self.X_std  = None

        self.order  = order 

    def _build_tree(self):
        def add_node(current_depth, node_id):
            is_leaf = (current_depth == self.max_depth)
            node = TreeNode(node_id=node_id, depth=current_depth, is_leaf=is_leaf)

            if is_leaf:
                node.set_prediction(0.0)  # will be set during training
            else:
                W = self.rng.normal(size=(1, self.input_dim))
                b = self.rng.normal()
                node.set_split(W, b)

                node.left = add_node(current_depth + 1, 2 * node_id + 1)
                node.right = add_node(current_depth + 1, 2 * node_id + 2)

            self.nodes[node_id] = node
            return node

        self.root = add_node(current_depth=0, node_id=0)

    # Placeholder for later implementation
    def predict(self, X):
        raise NotImplementedError

    def standardize_input(self, X):
        """
        Standardize input features: subtract mean and divide by std.
        Stores the mean and std in the tree instance for use during inference.
        """
        if self.X_mean is None or self.X_std is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1.0  # avoid division by zero

        return (X - self.X_mean) / self.X_std

    def route_all(self, X):
        """
        Returns:
            result: boolean array of shape (N_samples, N_nodes),
                    where result[i, j] is True if sample i passes through the j-th node
                    as defined by self._get_nodes().
            ordered_nodes: list of nodes (internal or all), in traversal order.
        """
        N = X.shape[0]
        ordered_nodes = self._get_nodes(internal=False)
        result = np.zeros((N, len(ordered_nodes)), dtype=bool)

        def emit_mask(node, current_mask):
            idx = node_to_index[node]
            result[:, idx] = current_mask

            if node.is_leaf:
                return

            decision_values = (X @ node.W.T).flatten() + node.b
            go_right = decision_values > 0
            go_left = ~go_right

            emit_mask(node.left, current_mask & go_left)
            emit_mask(node.right, current_mask & go_right)

        node_to_index = {node: i for i, node in enumerate(ordered_nodes)}
        emit_mask(self.root, np.ones(N, dtype=bool))

        # Tell each node how many entries it has
        for i_node, node in enumerate(ordered_nodes):
            node.n_instances = np.count_nonzero( result[:, i_node] )

        return result, ordered_nodes


    #def prune_by_min_node_size(self, min_size):
    #    """
    #    Prune internal nodes based on cached leaf_indices.
    #    """
    #    print(f"→ Pruning with min node size: {min_size}")
    #    keep = set(self.leaf_indices.keys())

    #    for node in self._get_nodes():
    #        if node.is_leaf:
    #            continue
    #        if (node.left is None or node.left.node_id not in keep or
    #            node.right is None or node.right.node_id not in keep):
    #            continue
    #        left_count = len(self.leaf_indices.get(node.left.node_id, []))
    #        right_count = len(self.leaf_indices.get(node.right.node_id, []))

    #        if left_count < min_size or right_count < min_size:
    #            print(f"  ↳ Pruning node {node.node_id} (left: {left_count}, right: {right_count})")

    #            left_id = node.left.node_id
    #            right_id = node.right.node_id

    #            node.is_leaf = True
    #            node.left = None
    #            node.right = None
    #            node.W = None
    #            node.b = None

    #            keep.discard(left_id)
    #            keep.discard(right_id)
    #            self.leaf_indices[node.node_id] = (
    #                self.leaf_indices.get(left_id, []) + self.leaf_indices.get(right_id, [])
    #            )

    #    print("✓ Pruning complete.")

    def train_step(self, X, y, w, alpha=0.01, mode="regression", min_node_size=None):
        """
        Full TAO-style training step with single routing pass.
        """
        print("→ Standardizing input...")
        X = self.standardize_input(X)
        
        #self.loss_cache = np.zeros( X.shape[0] )

        self.print()
        print("→ Routing all data...")
        routing, ordered_nodes = self.route_all(X)

        print("→ Computing new predictions")
        for i_node, node in enumerate(ordered_nodes):
            if node.is_leaf:
                self._update_leaf_from_routing(node, routing[:, i_node], y, w, mode=mode)
            else:
                self._update_split_node_tao(X, y, w, node, routing[:, i_node], alpha=alpha)
        self.print()


    def _update_leaf_from_routing(self, node, mask, y, w, mode="regression"):
        """
        Update prediction and loss cache for a single leaf node using a mask over the samples.
        
        Args:
            node (TreeNode): the leaf node
            mask (np.ndarray): boolean mask of shape (N,) indicating which samples reach this leaf
            y (np.ndarray): labels
            w (np.ndarray): weights
            mode (str): "regression" or "classification"
        """
        
        print( f"Fitting leaf node  {node.node_id}")

        if not np.any(mask):
            node.set_prediction(0.0)
            return

        y_sub = y[mask]
        w_sub = w[mask]

        if mode == "regression":
            weighted_mean = np.sum(w_sub * y_sub) / np.sum(w_sub)
            node.set_prediction(weighted_mean)
            #residuals = (y_sub - weighted_mean) ** 2
            #self.loss_cache[mask] = w_sub * residuals

        elif mode == "classification":
            y_bar = np.sum(w_sub * y_sub) / np.sum(w_sub)
            eps = 1e-8
            y_bar = np.clip(y_bar, eps, 1 - eps)
            v = np.log(y_bar / (1 - y_bar))
            node.set_prediction(v)
            #pi = 1 / (1 + np.exp(-v))
            #ce_loss = -y_sub * np.log(pi + eps) - (1 - y_sub) * np.log(1 - pi + eps)
            #self.loss_cache[mask] = w_sub * ce_loss

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def print(self):
        threshold = 1e-3
        def _print(node, prefix=""):
            if node is None:
                return

            if hasattr( node, "n_instances" ):
                n_inst_str = f"(n_inst = {node.n_instances})"
            else:
                n_inst_str = f"(n_inst = (None))"

            if node.is_leaf:
                print(f"{prefix}[Leaf] Node {node.node_id} at depth {node.depth} → prediction = {node.prediction:.4f} {n_inst_str}")
            else:
                terms = [
                    f"{w:.3f}*x{j}"
                    for j, w in enumerate(node.W.flatten())
                    if abs(w) >= threshold 
                ]
                w_str = " + ".join(terms) if terms else "0"
                decision_str = f"{w_str} + {node.b:.3f} >= 0"
                print(f"{prefix}[Split] Node {node.node_id} at depth {node.depth} → {decision_str} {n_inst_str}")
                _print(node.left, prefix + "  ")
                _print(node.right, prefix + "  ")
        _print(self.root)

    def _get_nodes(self, internal = True):
        if self.order == "reverse_bfs":
            return sorted(
                [n for n in self.nodes.values() if (n.is_internal() or not internal)],
                key=lambda n: -n.depth
            )
        elif self.order == "bfs":
            return sorted(
                [n for n in self.nodes.values() if (n.is_internal() or not internal)],
                key=lambda n: n.depth
            )
        elif self.order == "preorder":
            result = []
            def dfs(node):
                if node is None:
                    return
                if node.is_internal() or not internal:
                    result.append(node)
                dfs(node.left)
                dfs(node.right)
            dfs(self.root)
            return result
        else:
            raise ValueError(f"Unknown order: {self.order}")

    def _accumulate_leaf_losses(self, root, X, y, w, base_mask, mode):
        """
        Traverse tree rooted at `root`, emitting per-sample losses into a vector.
        Only computes losses for samples in `base_mask`.
        """
        N = X.shape[0]
        losses = np.zeros(N)

        def emit_loss(node, mask):
            if not np.any(mask):
                return
            if node.is_leaf:
                if mode == "regression":
                    residual = y[mask] - node.prediction
                    losses[mask] = w[mask] * residual**2
                elif mode == "classification":
                    v = node.prediction
                    pi = 1 / (1 + np.exp(-v))
                    eps = 1e-8
                    ce_loss = -y[mask] * np.log(pi + eps) - (1 - y[mask]) * np.log(1 - pi + eps)
                    losses[mask] = w[mask] * ce_loss
                return

            decision = (X @ node.W.T).flatten() + node.b
            go_right = decision > 0
            go_left = ~go_right
            emit_loss(node.left, mask & go_left)
            emit_loss(node.right, mask & go_right)

        emit_loss(root, base_mask)
        return losses

    def _update_split_node_tao(self, X, y, w, node, mask, alpha=0.01, mode="regression"):

        if node.is_leaf or not np.any(mask):
            return

        X_sub = X[mask]
        indices = np.where(mask)[0]

        print(f"Fitting internal node {node.node_id} with {len(indices)} events")

        left_losses = self._accumulate_leaf_losses(node.left, X, y, w, mask, mode)
        right_losses = self._accumulate_leaf_losses(node.right, X, y, w, mask, mode)

        delta_loss = left_losses[mask] - right_losses[mask]
        y_target = np.sign(delta_loss).astype(int)
        sample_weight = np.abs(delta_loss)

        if np.sum(sample_weight) == 0:
            return

        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1 / alpha)
        clf.fit(X_sub, y_target, sample_weight=sample_weight)

        node.W = clf.coef_.reshape(1, -1)
        node.b = clf.intercept_[0]

    def predict(self, X):
        """
        Predict output for new data using emit-style vectorized tree traversal.

        Args:
            X (np.ndarray): input data of shape (N, d)

        Returns:
            predictions (np.ndarray): array of shape (N,) with predicted values
        """
        X = (X - self.X_mean) / self.X_std  # standardize
        N = X.shape[0]
        predictions = np.zeros(N)

        def emit_prediction(node, mask):
            if not np.any(mask):
                return
            if node.is_leaf:
                predictions[mask] = node.prediction
                return
            decision = (X @ node.W.T).flatten() + node.b
            go_right = decision > 0
            go_left = ~go_right
            emit_prediction(node.left, mask & go_left)
            emit_prediction(node.right, mask & go_right)

        emit_prediction(self.root, np.ones(N, dtype=bool))
        return predictions

# random seed
rng = 40
#rng = 42
X, y, w = generate_overlapping_gaussians( n_per_class=10000, d=3, separation=1.0, rng=rng)
t = Tree( max_depth = 3, input_dim=3, rng=rng)

print("Tree before fit:")

for iteration in range(10):
    
    t.train_step(X,y,w, alpha=0.01, mode="regression", min_node_size=25)
#t.print()
#t.train_step(X,y,w, alpha=0.01, mode="regression", min_node_size=25)
#print("After before fit:")
#t.print()
