"""
mlp/activations.py
==================
Funções de ativação e suas derivadas.

Por que precisamos das derivadas?
----------------------------------
Durante o backpropagation, precisamos propagar o gradiente da loss
de volta para cada camada. Quando o sinal passa por uma ativação f(z),
o gradiente precisa ser multiplicado por f'(z) (regra da cadeia).

Cada função aqui aceita arrays NumPy de qualquer shape e opera
de forma elementwise — exatamente o que precisamos para mini-batches.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ReLU  (Rectified Linear Unit)
# ─────────────────────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU(z) = max(0, z)

    Intuição: "zera" neurônios com ativação pré-ativação negativa.
    Vantagem chave: gradiente constante (1) para z > 0 → sem vanishing gradient
    para redes de profundidade moderada como a nossa.

    Parâmetros
    ----------
    z : np.ndarray
        Entrada (qualquer shape) — saída linear de uma camada (z = Wx + b).

    Retorna
    -------
    np.ndarray
        Mesma shape que z, com negativos substituídos por 0.
    """
    return np.maximum(0, z)


def relu_backward(z: np.ndarray) -> np.ndarray:
    """
    Derivada da ReLU em relação a z.

    ReLU'(z) = 1  se z > 0
               0  se z ≤ 0

    Usamos isso no backward pass:
        dL/dz = dL/dA * ReLU'(z)   (multiplicação elementwise)

    Parâmetros
    ----------
    z : np.ndarray
        A pré-ativação (ANTES de aplicar ReLU), salva no forward pass.
        Precisamos de z, não de A=ReLU(z), porque a derivada depende
        do sinal de z.

    Retorna
    -------
    np.ndarray
        Máscara booleana (como float): 1.0 onde z > 0, 0.0 caso contrário.
    """
    return (z > 0).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Softmax
# ─────────────────────────────────────────────────────────────────────────────

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax(z)_i = exp(z_i) / sum_j(exp(z_j))

    Transforma um vetor de logits em probabilidades que somam 1.
    Usada SOMENTE na camada de saída — nunca em camadas ocultas.

    Truque de estabilidade numérica (shift by max):
    ------------------------------------------------
    exp(z) pode explodir para z grande. Subtrair o máximo não muda
    a saída (cancela no numerador e denominador), mas evita overflow:
        softmax(z) = softmax(z - max(z))

    Parâmetros
    ----------
    z : np.ndarray
        Shape (n_classes, m) — logits para m exemplos e n_classes classes.

    Retorna
    -------
    np.ndarray
        Mesma shape, cada coluna soma 1.
    """
    # Subtrai o máximo de cada coluna para estabilidade numérica
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# NOTA: Não implementamos a derivada do Softmax de forma isolada.
# No backward pass, a derivada combinada Softmax + Cross-Entropy
# simplifica para:  dL/dz_saída = A_saída - Y_onehot
# Isso está implementado em losses.py.


# ─────────────────────────────────────────────────────────────────────────────
# Sigmoid  (opcional — não usada no MLP principal, mas útil para estudo)
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    σ(z) = 1 / (1 + exp(-z))

    Mapeia qualquer real para (0, 1).
    Problema: para |z| grande, gradiente ≈ 0 → vanishing gradient.
    Por isso preferimos ReLU em camadas ocultas profundas.

    Implementada aqui para estudo e comparação opcional.
    """
    # Clipping para evitar overflow no exp
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_backward(z: np.ndarray) -> np.ndarray:
    """
    σ'(z) = σ(z) * (1 - σ(z))

    Note que a derivada pode ser expressa em função da própria sigmoid.
    Para z ≈ 0: σ'(z) ≈ 0.25 (valor máximo).
    Para |z| grande: σ'(z) ≈ 0 → problema do vanishing gradient.
    """
    s = sigmoid(z)
    return s * (1 - s)


# ─────────────────────────────────────────────────────────────────────────────
# Tanh  (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def tanh(z: np.ndarray) -> np.ndarray:
    """
    tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))

    Saída centrada em 0 (intervalo -1 a 1), o que pode acelerar
    convergência comparado a sigmoid. Ainda sofre de vanishing gradient
    para |z| grande, porém menos severamente.
    """
    return np.tanh(z)


def tanh_backward(z: np.ndarray) -> np.ndarray:
    """
    tanh'(z) = 1 - tanh(z)²

    Derivada máxima em z=0: 1.0.
    Ainda satura para |z| grande.
    """
    return 1.0 - np.tanh(z) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Registro de ativações disponíveis
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATIONS = {
    "relu":    (relu,    relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh":    (tanh,    tanh_backward),
}
"""
Dicionário que mapeia nome (str) → (forward_fn, backward_fn).
Usado em network.py para selecionar a ativação de cada camada
de forma dinâmica, sem if/elif em cascata.

Exemplo de uso:
    forward_fn, backward_fn = ACTIVATIONS["relu"]
    A = forward_fn(z)
    dZ = backward_fn(z) * dA  # elementwise
"""


# ─────────────────────────────────────────────────────────────────────────────
# Testes rápidos (executar com: python -m mlp.activations)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testando funções de ativação ===\n")

    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("Entrada z:", z)
    print()

    # ReLU
    print("ReLU(z)          :", relu(z))
    print("ReLU'(z) (máscara):", relu_backward(z))
    assert np.allclose(relu(z), [0, 0, 0, 1, 2]), "ReLU errada!"
    assert np.allclose(relu_backward(z), [0, 0, 0, 1, 1]), "ReLU' errada!"
    print("✓ ReLU e derivada OK\n")

    # Softmax
    z2 = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
    sm = softmax(z2)
    print("Softmax([1,2,3])  :", sm.flatten())
    print("Soma das probs    :", sm.sum())
    assert abs(sm.sum() - 1.0) < 1e-9, "Softmax não soma 1!"
    print("✓ Softmax OK\n")

    # Sigmoid
    print("Sigmoid(z)        :", sigmoid(z))
    print("Sigmoid'(z)       :", sigmoid_backward(z))
    print("✓ Sigmoid OK\n")

    # Tanh
    print("Tanh(z)           :", tanh(z))
    print("Tanh'(z)          :", tanh_backward(z))
    print("✓ Tanh OK\n")

    print("=== Todos os testes passaram! ===")
