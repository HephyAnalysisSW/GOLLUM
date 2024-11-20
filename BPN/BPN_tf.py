import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

default_cfg = {
    "learning_rate" : 0.001,
    "hidden_layers" : [64, 32]
}


class BPN(Model):
    def __init__(self, in_features, out_features, VkA, **kwargs):
        """
        TensorFlow implementation of the BPN model.
        
        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension (number of \Delta_A(x)).
            VkA (tf.Tensor): The base point matrix.
            kwargs: Additional configuration arguments.
        """
        super(BPN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.VkA = tf.convert_to_tensor(VkA, dtype=tf.float32)

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

        for i, l in enumerate(self.hidden_layers):
            setattr( self, "dense%i"%i, Dense(l, activation='relu'))

        self.output_layer = Dense(out_features, activation=None)

        # Store any additional configuration
        self.config = kwargs

    def call(self, x):
        """
        Forward pass of the BPN model.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        for i, l in enumerate(self.hidden_layers):
            x = getattr(self, "dense%i"%i)(x)
        return self.output_layer(x)

    def layer_size( self ):
        return [self.in_features]+self.hidden_layers+[self.out_features]
