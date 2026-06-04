"""
mlp/
====
Pacote de implementação do Multi-Layer Perceptron do zero.

Uso rápido:
    from mlp.network import MLP
    from mlp.optimizers import SGD

    model = MLP(layer_sizes=[784, 256, 128, 10], activation="relu",
                optimizer=SGD(learning_rate=0.01))
    model.train(X_train, y_train, epochs=20, batch_size=128)
"""

from mlp.network import MLP
from mlp.optimizers import SGD, SGDMomentum
from mlp.activations import relu, softmax
from mlp.losses import cross_entropy_loss, one_hot

__all__ = [
    "MLP",
    "SGD",
    "SGDMomentum",
    "relu",
    "softmax",
    "cross_entropy_loss",
    "one_hot",
]
