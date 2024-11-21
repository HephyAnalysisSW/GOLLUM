import torch
import numpy as np

default_cfg = {
    "learning_rate" : 0.001,
    "hidden_layers" : [64, 32]
}

class PNN(torch.nn.Module):
    def __init__(self, in_features, out_features, VkA,  **kwargs ):
        super(PNN, self).__init__()

        self.VkA = VkA

        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        for (key, val) in self.cfg.items():
                setattr( self, key, val )

        # Build the neural network dynamically
        layers = []
        self.in_features = in_features
        self.out_features = out_features
        for out_features in self.hidden_layers:
            layers.append(torch.nn.Linear(in_features, out_features))  # Add linear layer
            layers.append(torch.nn.ReLU())  # Add activation
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, self.out_features))  # Final output layer

        self.net = torch.nn.Sequential(*layers)

    #def __setstate__(self, state):
    #    self.__dict__ = state

    #def save(self, filename):
    #    with open(filename,'wb') as file_:
    #        pickle.dump( self, file_ )

    def layer_size( self ):
        return [self.in_features]+self.hidden_layers+[self.out_features]

    def forward(self, x):
        """
        Compute \(\Delta_A(x)\) for all A simultaneously.
        Args:
            x (Tensor): Input features of shape (batch_size, input_dim).
        Returns:
            Tensor: Outputs of shape (batch_size, nu_dim), where each column corresponds to a \(\Delta_A(x)\).
        """
        return self.net(x)
