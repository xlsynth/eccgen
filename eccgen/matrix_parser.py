# SPDX-License-Identifier: Apache-2.0

import ast
import numpy as np
import os
import re
from typing import Tuple


def _parse_matrix_file(path: str, symbol: str) -> Tuple[int, int, int, np.ndarray]:
    """
    Parse a file with headers:
      Number of data bits (k): <int>
      Number of parity bits (r): <int>
      Number of codeword bits (n): <int>
      <symbol> = [[...], ...]
    symbol is 'G' or 'H'.
    Returns (k, r, n, matrix) with matrix dtype=uint8.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    k = int(re.search(r"Number of data bits \(k\):\s*(\d+)", text).group(1))
    r = int(re.search(r"Number of parity bits \(r\):\s*(\d+)", text).group(1))
    n = int(re.search(r"Number of codeword bits \(n\):\s*(\d+)", text).group(1))

    m_hdr = re.search(rf"\b{symbol}\s*=", text)
    if not m_hdr:
        raise ValueError(f"Could not find '{symbol} =' in {path}")

    start = text.find("[", m_hdr.end())
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Matrix bracket block not found in {path}")

    mat = ast.literal_eval(text[start : end + 1])
    arr = np.array(mat, dtype=np.uint8)

    # Basic checks.
    if not ((arr == 0) | (arr == 1)).all():
        raise ValueError(f"Non-binary entries in {symbol} matrix of {path}")

    expected_rows = k if symbol == "G" else r
    if arr.shape != (expected_rows, n):
        raise ValueError(
            f"{symbol} shape {arr.shape} does not match expected {(expected_rows, n)} for {path}"
        )
    return k, r, n, arr


def parse_g_file(path: str) -> Tuple[int, int, int, np.ndarray]:
    """Parse the G matrix from a file.

    Returns (k, r, n, matrix).
    """
    return _parse_matrix_file(path, "G")


def parse_h_file(path: str) -> Tuple[int, int, int, np.ndarray]:
    """Parse the H matrix from a file.

    Returns (k, r, n, matrix).
    """
    return _parse_matrix_file(path, "H")


def parse_g_and_h_files(
    matrix_dir: str, k_list: list[int]
) -> dict[int, tuple[int, int, np.ndarray, np.ndarray]]:
    """Parse the G and H matrices from files in the given directory.
    For each G, H pair, checks that the dimensions match, but not that
    the matrices are correctly constructed.

    Args:
        matrix_dir: Path to the directory containing the G and H matrices.
        The files must be named "hsiao_G_k<k>.txt" and "hsiao_H_k<k>.txt" for some k.

    Returns:
        dict[int, tuple[int, int, np.ndarray, np.ndarray]]:
            key: k
            value: (r, n, G, H)
                r: Number of parity bits
                n: Number of codeword bits
                G: Generator matrix of shape k x n
                H: Parity-check matrix of shape r x n
        Such that c = m * G and s = H * c^T, where:
            - m is the 1 x k message
            - c is the 1 x n codeword
            - s is the r x 1 syndrome
    """
    codes = {}
    for k in k_list:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        g_path = os.path.join(matrix_dir, f"hsiao_G_k{k}.txt")
        h_path = os.path.join(matrix_dir, f"hsiao_H_k{k}.txt")
        k_g, r_g, n_g, G = parse_g_file(g_path)
        k_h, r_h, n_h, H = parse_h_file(h_path)
        if k_g != k_h:
            raise ValueError(
                f"G and H matrices have different k: {g_path} and {h_path}"
            )
        if r_g != r_h:
            raise ValueError(
                f"G and H matrices have different r: {g_path} and {h_path}"
            )
        if n_g != n_h:
            raise ValueError(
                f"G and H matrices have different n: {g_path} and {h_path}"
            )
        k = k_g
        r = r_g
        n = n_g
        codes[k] = (r, n, G, H)
    return codes
