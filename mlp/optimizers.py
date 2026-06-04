"""
mlp/optimizers.py
=================
Otimizadores para atualização dos parâmetros do MLP.

O papel do otimizador:
-----------------------
Após o backward pass, temos os gradientes dW e db para cada camada.
O otimizador decide *como* usar esses gradientes para atualizar os pesos.

O mais simples é o SGD (Stochastic Gradient Descent):
    W ← W - lr * dW
    b ← b - lr * db

Mas há versões mais sofisticadas que convergem mais rápido,
como SGD com Momentum e Adam.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Classe base (interface comum para todos os otimizadores)
# ─────────────────────────────────────────────────────────────────────────────

class Optimizer:
    """
    Interface base para otimizadores.
    Todo otimizador deve implementar o método `update`.
    """

    def update(self, params: list, grads: list) -> None:
        """
        Atualiza os parâmetros in-place usando os gradientes.

        Parâmetros
        ----------
        params : list of np.ndarray
            Lista de parâmetros a atualizar [W1, b1, W2, b2, ...].
        grads : list of np.ndarray
            Lista de gradientes correspondentes [dW1, db1, dW2, db2, ...].
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# SGD — Stochastic Gradient Descent
# ─────────────────────────────────────────────────────────────────────────────

class SGD(Optimizer):
    """
    Gradiente Descendente Estocástico (SGD) puro.

    Regra de atualização:
        θ ← θ - lr * ∇_θ L

    Onde θ é qualquer parâmetro (W ou b), lr é o learning rate e
    ∇_θ L é o gradiente da loss em relação a θ.

    Parâmetros
    ----------
    learning_rate : float
        Taxa de aprendizado. Controla o tamanho do passo na
        direção oposta ao gradiente.
        - lr muito alto  → oscila, pode divergir
        - lr muito baixo → converge muito devagar
        Valor típico para MNIST com SGD: 0.01 a 0.1
    """

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, params: list, grads: list) -> None:
        """
        Atualiza cada parâmetro in-place.

        params e grads devem estar alinhados:
            params[0] = W1, grads[0] = dW1
            params[1] = b1, grads[1] = db1
            ... etc.
        """
        for param, grad in zip(params, grads):
            # Atualização in-place: modifica o array original
            param -= self.lr * grad


# ─────────────────────────────────────────────────────────────────────────────
# SGD com Momentum (opcional — melhora convergência)
# ─────────────────────────────────────────────────────────────────────────────

class SGDMomentum(Optimizer):
    """
    SGD com Momentum.

    Motivação:
    ----------
    O SGD puro pode oscilar muito em direções com alta curvatura (dimensões
    "estreitas" da superfície de loss). O Momentum acumula uma média
    exponencial dos gradientes passados, suavizando o percurso.

    Regra de atualização:
        v ← β * v + (1 - β) * ∇_θ L     ← velocidade (média dos gradientes)
        θ ← θ - lr * v

    O parâmetro β (geralmente 0.9) controla o "peso" do histórico.
    Com β=0.9, cada gradiente passado tem influência decrescente:
        peso do gradiente t-k: (0.9)^k

    Parâmetros
    ----------
    learning_rate : float
        Taxa de aprendizado.
    momentum : float
        Fator de momentum (β). Tipicamente 0.9.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.beta = momentum
        self.velocities = None  # inicializado no primeiro update

    def update(self, params: list, grads: list) -> None:
        # Inicializa as velocidades com zeros na primeira chamada
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for v, param, grad in zip(self.velocities, params, grads):
            # Atualiza velocidade (média exponencial dos gradientes)
            v[:] = self.beta * v + (1 - self.beta) * grad
            # Atualiza parâmetro
            param -= self.lr * v


# ─────────────────────────────────────────────────────────────────────────────
# Testes rápidos (executar com: python -m mlp.optimizers)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testando otimizadores ===\n")

    # Simula parâmetros e gradientes simples
    W = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5])
    dW = np.array([0.1, 0.2, 0.3])
    db = np.array([0.05])

    # SGD
    sgd = SGD(learning_rate=0.1)
    W_sgd = W.copy()
    b_sgd = b.copy()
    sgd.update([W_sgd, b_sgd], [dW, db])
    print("SGD — W após update:", W_sgd)
    print("SGD — b após update:", b_sgd)
    # W_esperado = [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
    assert np.allclose(W_sgd, [0.99, 1.98, 2.97]), "SGD errado!"
    print("✓ SGD OK\n")

    # SGD Momentum
    mom = SGDMomentum(learning_rate=0.1, momentum=0.9)
    W_mom = W.copy()
    b_mom = b.copy()
    mom.update([W_mom, b_mom], [dW, db])
    print("Momentum — W após 1º update:", W_mom)
    mom.update([W_mom, b_mom], [dW, db])
    print("Momentum — W após 2º update:", W_mom)
    print("✓ SGD Momentum OK\n")

    print("=== Todos os testes passaram! ===")
