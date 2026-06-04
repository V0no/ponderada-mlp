"""
mlp/data.py
===========
Utilitário para carregar o MNIST diretamente dos arquivos IDX originais.

Faz o download automático do OpenML ou dos arquivos originais,
sem depender de TensorFlow, Keras ou PyTorch.
"""

import os
import gzip
import struct
import urllib.request
import numpy as np

# URL dos arquivos IDX originais do MNIST
_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: str) -> None:
    """Baixa um arquivo se ainda não existir."""
    if not os.path.exists(dest):
        print(f"  Baixando {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Salvo em {dest}")


def _load_images(path: str) -> np.ndarray:
    """Lê arquivo IDX3 (imagens) e retorna array (n, 784)."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number inválido: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def _load_labels(path: str) -> np.ndarray:
    """Lê arquivo IDX1 (rótulos) e retorna array (n,)."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number inválido: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir: str = "./data") -> tuple:
    """
    Carrega o MNIST, fazendo download automático se necessário.

    Parâmetros
    ----------
    data_dir : str
        Diretório onde os arquivos serão salvos/lidos.

    Retorna
    -------
    (X_train, y_train, X_test, y_test) com:
        X_train : (784, 60000) float64, valores em [0, 1]
        y_train : (60000,)     int
        X_test  : (784, 10000) float64, valores em [0, 1]
        y_test  : (10000,)     int
    """
    os.makedirs(data_dir, exist_ok=True)

    paths = {}
    for key, filename in _FILES.items():
        dest = os.path.join(data_dir, filename)
        _download(_BASE_URL + filename, dest)
        paths[key] = dest

    print("Carregando dados...")
    X_train_raw = _load_images(paths["train_images"])
    y_train     = _load_labels(paths["train_labels"])
    X_test_raw  = _load_images(paths["test_images"])
    y_test      = _load_labels(paths["test_labels"])

    # Normaliza para [0, 1] e transpõe para (784, m)
    X_train = X_train_raw.T.astype(np.float64) / 255.0
    X_test  = X_test_raw.T.astype(np.float64)  / 255.0

    print(f"✓ MNIST carregado: treino {X_train.shape}, teste {X_test.shape}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Treino: {X_train.shape}, rótulos: {y_train.shape}")
    print(f"Teste:  {X_test.shape},  rótulos: {y_test.shape}")
    print(f"Classes: {sorted(set(y_train.tolist()))}")
