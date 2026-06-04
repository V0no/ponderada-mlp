"""
mlp/network.py
==============
Implementação principal do Multi-Layer Perceptron (MLP).

Arquitetura suportada:
-----------------------
    Entrada → [Dense → Ativação] × L camadas ocultas → Dense → Softmax

Representação interna:
-----------------------
    self.W[l] : shape (n_out, n_in)  — pesos da camada l
    self.b[l] : shape (n_out, 1)     — bias da camada l

Mini-batch (convenção de shapes ao longo do forward):
    X    : (n_features, m)   — m exemplos empilhados como colunas
    Z[l] : (n_out_l, m)      — pré-ativação da camada l
    A[l] : (n_out_l, m)      — pós-ativação da camada l
    Y    : (n_classes, m)    — alvos one-hot

Essa convenção (features nas linhas, exemplos nas colunas) é a
mais comum em implementações "from scratch" porque Z = W @ A_prev + b
funciona naturalmente com broadcasting do bias.
"""

import numpy as np
import time

from mlp.activations import ACTIVATIONS, softmax
from mlp.losses import cross_entropy_loss, one_hot, softmax_crossentropy_backward
from mlp.optimizers import SGD, Optimizer


class MLP:
    """
    Multi-Layer Perceptron implementado do zero com NumPy.

    Parâmetros
    ----------
    layer_sizes : list of int
        Número de neurônios em cada camada, incluindo entrada e saída.
        Exemplo: [784, 256, 128, 10] cria:
            - Camada de entrada: 784 neurônios
            - Oculta 1: 256 neurônios
            - Oculta 2: 128 neurônios
            - Saída: 10 neurônios

    activation : str
        Nome da função de ativação das camadas OCULTAS.
        Opções: "relu", "sigmoid", "tanh"
        A camada de saída SEMPRE usa softmax.

    optimizer : Optimizer
        Instância do otimizador a usar. Default: SGD(lr=0.01).

    seed : int or None
        Semente aleatória para reprodutibilidade.
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        optimizer: Optimizer = None,
        seed: int = 42,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes deve ter ao menos 2 elementos (entrada e saída).")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Ativação '{activation}' desconhecida. Use: {list(ACTIVATIONS.keys())}")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1   # número de matrizes de pesos
        self.activation_name = activation
        self.act_fn, self.act_backward = ACTIVATIONS[activation]
        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)

        # Histórico de métricas (preenchido durante o treino)
        self.history = {
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
        }

        # Inicializa os parâmetros
        self._init_params(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Inicialização de parâmetros
    # ─────────────────────────────────────────────────────────────────────────

    def _init_params(self, seed: int) -> None:
        """
        Inicializa pesos com He initialization e bias com zeros.

        He initialization (para ReLU):
            W ~ N(0, sqrt(2 / n_entrada))

        Por que He e não Xavier?
        ------------------------
        Xavier assume que a ativação é linear em torno de 0 (válido para
        tanh). Para ReLU, metade das unidades é zerada — He compensa isso
        com um fator de 2, mantendo a variância dos gradientes estável
        ao longo das camadas.

        Por que bias = 0?
        -----------------
        O bias não causa o problema de simetria (somente os pesos causam).
        Inicializar com 0 é seguro e conveniente.
        """
        if seed is not None:
            np.random.seed(seed)

        self.W = []
        self.b = []

        for l in range(self.n_layers):
            n_in  = self.layer_sizes[l]
            n_out = self.layer_sizes[l + 1]

            # He initialization: escala sqrt(2/n_in)
            W_l = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b_l = np.zeros((n_out, 1))

            self.W.append(W_l)
            self.b.append(b_l)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> tuple:
        """
        Propaga X pela rede e retorna a saída final.

        Salva os valores intermediários (cache) necessários para o
        backward pass: para cada camada l, precisamos de Z[l] (para
        calcular a derivada da ativação) e A[l-1] (para calcular dW[l]).

        Parâmetros
        ----------
        X : np.ndarray
            Shape (n_features, m) — batch de m exemplos.

        Retorna
        -------
        A_out : np.ndarray
            Shape (n_classes, m) — probabilidades preditas (após softmax).
        cache : dict
            Valores intermediários: {"Z": [Z0,...,ZL], "A": [X, A0,...,AL]}
            A[0] = X (entrada), A[l] = ativação da camada l.
        """
        cache = {"Z": [], "A": [X]}  # A[0] = X (entrada)

        A_prev = X

        # Camadas ocultas (l = 0, 1, ..., n_layers - 2)
        for l in range(self.n_layers - 1):
            Z_l = self.W[l] @ A_prev + self.b[l]   # (n_out, m)
            A_l = self.act_fn(Z_l)                  # aplica ReLU (ou outra)

            cache["Z"].append(Z_l)
            cache["A"].append(A_l)
            A_prev = A_l

        # Camada de saída (sem ativação ReLU — softmax vem depois)
        l_out = self.n_layers - 1
        Z_out = self.W[l_out] @ A_prev + self.b[l_out]  # (n_classes, m)
        A_out = softmax(Z_out)                           # probabilidades

        cache["Z"].append(Z_out)
        cache["A"].append(A_out)

        return A_out, cache

    # ─────────────────────────────────────────────────────────────────────────
    # Backward pass (Backpropagation)
    # ─────────────────────────────────────────────────────────────────────────

    def backward(self, A_out: np.ndarray, Y: np.ndarray, cache: dict) -> tuple:
        """
        Calcula os gradientes de L em relação a todos os W e b.

        Utiliza a regra da cadeia recursivamente, da camada de saída
        até a camada de entrada (daí o nome "backward").

        Fluxo do gradiente (exemplo com 3 camadas):
            dZ_out = A_out - Y                   ← gradiente combinado softmax+CE
            dW3 = dZ_out @ A2.T / m
            db3 = mean(dZ_out, axis=1)
            dA2 = W3.T @ dZ_out
            dZ2 = dA2 * relu'(Z2)               ← elementwise
            dW2 = dZ2 @ A1.T / m
            ... e assim por diante.

        Parâmetros
        ----------
        A_out : np.ndarray
            Saída do forward pass (probabilidades).
        Y : np.ndarray
            Alvos one-hot, shape (n_classes, m).
        cache : dict
            Valores salvos durante o forward pass.

        Retorna
        -------
        dW : list of np.ndarray
            Gradientes dos pesos, um por camada.
        db : list of np.ndarray
            Gradientes dos bias, um por camada.
        """
        m = A_out.shape[1]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # ── Gradiente da camada de saída (softmax + cross-entropy) ──────────
        l_out = self.n_layers - 1
        dZ = softmax_crossentropy_backward(A_out, Y)   # (n_classes, m)

        A_prev = cache["A"][l_out]                     # ativação da camada anterior
        dW[l_out] = dZ @ A_prev.T                      # (n_classes, n_prev)
        db[l_out] = np.sum(dZ, axis=1, keepdims=True)  # (n_classes, 1)

        # ── Gradiente das camadas ocultas (de trás para frente) ─────────────
        for l in range(l_out - 1, -1, -1):
            # Propaga gradiente para a camada anterior (em relação a A[l])
            dA = self.W[l + 1].T @ dZ         # (n_out_l, m)

            # Multiplica pela derivada da ativação (regra da cadeia)
            # Precisamos de Z[l] salvo no forward pass
            dZ = dA * self.act_backward(cache["Z"][l])  # elementwise

            # Gradientes de W[l] e b[l]
            A_prev = cache["A"][l]             # A[0] = X para l=0
            dW[l] = dZ @ A_prev.T
            db[l] = np.sum(dZ, axis=1, keepdims=True)

        return dW, db

    # ─────────────────────────────────────────────────────────────────────────
    # Atualização de parâmetros
    # ─────────────────────────────────────────────────────────────────────────

    def _update_params(self, dW: list, db: list) -> None:
        """
        Chama o otimizador para atualizar W e b.

        Intercala W e b em uma lista única:
            params = [W[0], b[0], W[1], b[1], ...]
            grads  = [dW[0], db[0], dW[1], db[1], ...]

        Isso permite que qualquer otimizador (SGD, Momentum, etc.)
        trate todos os parâmetros de forma uniforme.
        """
        params = []
        grads = []
        for l in range(self.n_layers):
            params.extend([self.W[l], self.b[l]])
            grads.extend([dW[l], db[l]])

        self.optimizer.update(params, grads)

    # ─────────────────────────────────────────────────────────────────────────
    # Predição e avaliação
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna a classe predita para cada exemplo em X.

        Parâmetros
        ----------
        X : np.ndarray
            Shape (n_features, m).

        Retorna
        -------
        np.ndarray
            Shape (m,) — índice da classe com maior probabilidade.
        """
        A_out, _ = self.forward(X)
        return np.argmax(A_out, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Calcula a loss e a acurácia sobre um conjunto de dados.

        Parâmetros
        ----------
        X : np.ndarray
            Shape (n_features, m).
        y : np.ndarray
            Shape (m,) — rótulos inteiros.

        Retorna
        -------
        loss : float
        accuracy : float  (0.0 a 1.0)
        """
        A_out, _ = self.forward(X)
        Y = one_hot(y, n_classes=self.layer_sizes[-1])
        loss = cross_entropy_loss(A_out, Y)
        preds = np.argmax(A_out, axis=0)
        accuracy = np.mean(preds == y)
        return loss, accuracy

    # ─────────────────────────────────────────────────────────────────────────
    # Treinamento
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True,
    ) -> dict:
        """
        Treina o MLP usando SGD com mini-batches.

        Em cada época:
            1. Embaralha os dados (shuffle)
            2. Divide em mini-batches
            3. Para cada mini-batch: forward → loss → backward → update
            4. Avalia no conjunto de validação (se fornecido)

        Por que embaralhar a cada época?
        ----------------------------------
        Se os dados fossem apresentados sempre na mesma ordem, o modelo
        poderia memorizar a sequência ao invés de aprender padrões gerais.
        O shuffle garante que cada mini-batch tenha amostras variadas.

        Parâmetros
        ----------
        X_train : np.ndarray, shape (n_features, m_train)
        y_train : np.ndarray, shape (m_train,)
        epochs : int
        batch_size : int
        X_val : np.ndarray, shape (n_features, m_val)  [opcional]
        y_val : np.ndarray, shape (m_val,)              [opcional]
        verbose : bool
            Se True, imprime métricas a cada época.

        Retorna
        -------
        dict
            Histórico de métricas: train_loss, val_loss, train_acc, val_acc.
        """
        m = X_train.shape[1]

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Shuffle ───────────────────────────────────────────────────
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            y_shuffled = y_train[perm]

            epoch_losses = []

            # ── Mini-batches ───────────────────────────────────────────────
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]
                Y_batch = one_hot(y_batch, n_classes=self.layer_sizes[-1])

                # Forward
                A_out, cache = self.forward(X_batch)

                # Loss
                loss = cross_entropy_loss(A_out, Y_batch)
                epoch_losses.append(loss)

                # Backward
                dW, db = self.backward(A_out, Y_batch, cache)

                # Atualização
                self._update_params(dW, db)

            # ── Métricas da época ──────────────────────────────────────────
            train_loss, train_acc = self.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0

            if verbose:
                val_str = ""
                if val_loss is not None:
                    val_str = f"  |  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
                print(
                    f"Época {epoch:3d}/{epochs}"
                    f"  train_loss: {train_loss:.4f}"
                    f"  train_acc: {train_acc:.4f}"
                    f"{val_str}"
                    f"  ({elapsed:.1f}s)"
                )

        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient check numérico (ferramenta de debugging)
    # ─────────────────────────────────────────────────────────────────────────

    def gradient_check(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5,
        n_checks: int = 10,
    ) -> float:
        """
        Verifica se os gradientes analíticos estão corretos comparando-os
        com a aproximação numérica (diferença central).

        Aproximação numérica do gradiente:
            ∂L/∂θ_i ≈ [L(θ_i + ε) - L(θ_i - ε)] / (2ε)

        Se a diferença relativa entre o gradiente analítico e o numérico
        for menor que 1e-5, os gradientes estão corretos.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_features, m)
        y : np.ndarray, shape (m,)
        epsilon : float
            Perturbação para a diferença central.
        n_checks : int
            Quantos parâmetros aleatórios verificar (checar todos é lento).

        Retorna
        -------
        max_diff : float
            Maior diferença relativa encontrada. Deve ser < 1e-5.
        """
        Y = one_hot(y, n_classes=self.layer_sizes[-1])
        A_out, cache = self.forward(X)
        dW_analytic, db_analytic = self.backward(A_out, Y, cache)

        # Coleta todos os parâmetros e gradientes em listas flat
        all_params = []
        all_grads = []
        for l in range(self.n_layers):
            all_params.append(self.W[l])
            all_grads.append(dW_analytic[l])
            all_params.append(self.b[l])
            all_grads.append(db_analytic[l])

        max_diff = 0.0
        np.random.seed(0)

        print(f"\n=== Gradient Check (ε={epsilon}, {n_checks} parâmetros) ===")

        for _ in range(n_checks):
            # Seleciona parâmetro e índice aleatórios
            layer_idx = np.random.randint(0, len(all_params))
            param = all_params[layer_idx]
            grad = all_grads[layer_idx]
            idx = tuple(np.random.randint(0, s) for s in param.shape)

            # Guarda valor original
            original = param[idx]

            # Perturba +ε
            param[idx] = original + epsilon
            A_plus, _ = self.forward(X)
            loss_plus = cross_entropy_loss(A_plus, Y)

            # Perturba -ε
            param[idx] = original - epsilon
            A_minus, _ = self.forward(X)
            loss_minus = cross_entropy_loss(A_minus, Y)

            # Restaura valor original
            param[idx] = original

            # Gradiente numérico
            grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)

            # Gradiente analítico
            grad_analytic = grad[idx]

            # Diferença relativa
            num = abs(grad_analytic - grad_numeric)
            den = abs(grad_analytic) + abs(grad_numeric) + 1e-15
            diff = num / den

            max_diff = max(max_diff, diff)
            status = "✓" if diff < 1e-4 else "✗ ERRO"
            print(
                f"  param[{layer_idx}]{idx} | analítico: {grad_analytic:.6f}"
                f" | numérico: {grad_numeric:.6f}"
                f" | diff_rel: {diff:.2e}  {status}"
            )

        print(f"\nDiferença máxima: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("✓ Gradientes CORRETOS!\n")
        else:
            print("✗ ATENÇÃO: gradientes potencialmente incorretos!\n")

        return max_diff

    def __repr__(self) -> str:
        layers_str = " → ".join(str(n) for n in self.layer_sizes)
        return (
            f"MLP({layers_str})"
            f" | ativação: {self.activation_name}"
            f" | otimizador: {type(self.optimizer).__name__}"
        )
