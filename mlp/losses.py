"""
mlp/losses.py
=============
Funções de custo (loss) e utilitários relacionados.

Por que Cross-Entropy + Softmax?
---------------------------------
A Cross-Entropy é a função de custo natural para classificação
multi-classe porque ela mede a "distância" entre a distribuição
predita (softmax) e a distribuição real (one-hot).

A combinação Softmax + Cross-Entropy tem uma propriedade poderosa:
o gradiente em relação à entrada da última camada simplifica para:

    dL/dZ_out = A_out - Y_onehot

Isso é muito mais simples do que calcular as derivadas separadamente.
Veja a prova matemática no comentário da função `softmax_crossentropy_backward`.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# One-Hot Encoding
# ─────────────────────────────────────────────────────────────────────────────

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converte um vetor de rótulos inteiros em uma matriz one-hot.

    Exemplo:
        y = [0, 2, 1], n_classes = 3
        →   [[1, 0, 0],
             [0, 0, 1],
             [0, 1, 0]]   (shape: n_classes × m)

    Por que precisamos disso?
    --------------------------
    A Cross-Entropy espera que o "alvo" seja uma distribuição de
    probabilidade (valores entre 0 e 1 somando 1). O one-hot é a
    distribuição "perfeita": toda a probabilidade em uma classe.

    Parâmetros
    ----------
    y : np.ndarray
        Shape (m,) — vetor de rótulos inteiros (0 a n_classes-1).
    n_classes : int
        Número total de classes.

    Retorna
    -------
    np.ndarray
        Shape (n_classes, m) — cada coluna é um vetor one-hot.
    """
    m = y.shape[0]
    Y = np.zeros((n_classes, m), dtype=float)
    Y[y, np.arange(m)] = 1.0
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(A_out: np.ndarray, Y: np.ndarray) -> float:
    """
    Cross-Entropy Loss (média sobre o mini-batch).

    CE(A, Y) = -1/m * Σ_i Σ_k  Y[k,i] * log(A[k,i])

    Como Y é one-hot, apenas o log da classe correta importa:
    CE = -1/m * Σ_i  log(A[y_i, i])

    Onde y_i é o índice da classe verdadeira do exemplo i.

    Propriedade importante:
    -----------------------
    Quanto mais próxima de 1 a probabilidade prevista para a classe
    correta, mais próximo de 0 é o -log(p). Quando p → 0, o custo
    dispara para +∞ → penaliza fortemente previsões erradas com
    alta confiança.

    Parâmetros
    ----------
    A_out : np.ndarray
        Shape (n_classes, m) — saída do softmax (probabilidades previstas).
    Y : np.ndarray
        Shape (n_classes, m) — alvos em formato one-hot.

    Retorna
    -------
    float
        Loss escalar (média sobre o batch).
    """
    m = A_out.shape[1]

    # Clipping para evitar log(0) = -inf
    # log(1e-15) ≈ -34 — bem ruim, mas não explode
    A_clipped = np.clip(A_out, 1e-15, 1.0)

    # Seleciona apenas o log da classe correta (onde Y=1)
    # Equivalente a: -sum(Y * log(A)) / m
    loss = -np.sum(Y * np.log(A_clipped)) / m

    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Gradiente combinado Softmax + Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

def softmax_crossentropy_backward(A_out: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Gradiente da Cross-Entropy Loss em relação à pré-ativação Z_out
    (ANTES do softmax), ou seja: dL/dZ_out.

    Resultado: dL/dZ_out = (A_out - Y) / m

    ─── Prova rápida ────────────────────────────────────────────────
    Seja:
        Z_out  = saída linear da última camada (logits)
        A_out  = Softmax(Z_out)
        L      = CrossEntropy(A_out, Y)

    Por regra da cadeia:
        dL/dZ_out = dL/dA_out * dA_out/dZ_out

    A derivada de Softmax(z)_i em relação a z_j é a Jacobiana:
        ∂A_i/∂Z_j = A_i * (δ_ij - A_j)   ← resultado bem conhecido

    Quando multiplicamos pela derivada da Cross-Entropy:
        dL/dA_out_i = -Y_i / A_i

    E somamos sobre todos os k (pela regra da cadeia sobre a Jacobiana),
    tudo simplifica para:
        dL/dZ_out = A_out - Y

    Dividimos por m para obter a média do batch.
    ──────────────────────────────────────────────────────────────────

    Parâmetros
    ----------
    A_out : np.ndarray
        Shape (n_classes, m) — saída do softmax.
    Y : np.ndarray
        Shape (n_classes, m) — alvos one-hot.

    Retorna
    -------
    np.ndarray
        Shape (n_classes, m) — gradiente dL/dZ_out.
    """
    m = A_out.shape[1]
    return (A_out - Y) / m


# ─────────────────────────────────────────────────────────────────────────────
# Testes rápidos (executar com: python -m mlp.losses)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testando funções de loss ===\n")

    # Teste one_hot
    y = np.array([0, 2, 1])
    Y = one_hot(y, n_classes=3)
    print("Rótulos y:", y)
    print("One-hot Y:\n", Y)
    assert Y.shape == (3, 3)
    assert Y[0, 0] == 1.0 and Y[2, 1] == 1.0 and Y[1, 2] == 1.0
    print("✓ One-hot OK\n")

    # Teste cross_entropy_loss
    # Caso perfeito: predição = target → loss perto de 0
    A_perfeito = np.array([[0.99, 0.01, 0.01],
                            [0.01, 0.01, 0.97],
                            [0.00, 0.98, 0.02]])
    loss_perfeito = cross_entropy_loss(A_perfeito, Y)
    print(f"Loss (previsão quase perfeita): {loss_perfeito:.4f}  (esperado: perto de 0)")

    # Caso ruim: predição uniforme → loss alta
    A_uniforme = np.ones((3, 3)) / 3
    loss_uniforme = cross_entropy_loss(A_uniforme, Y)
    print(f"Loss (previsão uniforme 1/3):   {loss_uniforme:.4f}  (esperado: ~1.099 = log(3))")
    assert loss_perfeito < loss_uniforme
    print("✓ Cross-Entropy Loss OK\n")

    # Teste do gradiente
    grad = softmax_crossentropy_backward(A_perfeito, Y)
    print("Gradiente dL/dZ_out (previsão quase perfeita):\n", grad.round(4))
    print("Shape do gradiente:", grad.shape)
    print("✓ Gradiente OK\n")

    print("=== Todos os testes passaram! ===")
