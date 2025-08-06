# Copyright 2024-2025 The XLSynth and ECCGen Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from eccgen.hsiao_secded import (
    hsiao_secded_code,
    check_construction,
    MAX_K_FOR_OPTIMAL_ALGORITHM,
)
import numpy as np

RTL_SUPPORTED_N_K = [
    (8, 4),
    (13, 8),
    (16, 11),
    (22, 16),
    (32, 26),
    (39, 32),
    (64, 57),
    (72, 64),
    (128, 120),
    (137, 128),
    (256, 247),
    (266, 256),
    (512, 502),
    (523, 512),
    (1024, 1013),
    (1036, 1024),
]

def main():
    parser = argparse.ArgumentParser(description="Error Correction Code Generator")
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["hsiao_secded"],
        required=True,
        help="The error correction code scheme to use (e.g., hsiao_secded)",
    )
    parser.add_argument("--k", type=int, required=True, help="The number of data bits (k)")
    parser.add_argument(
        "--print0",
        action="store_true",
        help="Print 0s in the outputs (otherwise, leave blanks instead).",
    )
    parser.add_argument(
        "--generator-matrix-output",
        "--G",
        type=argparse.FileType("w"),
        required=True,
        help="The output file to write the generator matrix to",
    )
    parser.add_argument(
        "--parity-check-matrix-output",
        "--H",
        type=argparse.FileType("w"),
        required=True,
        help="The output file to write the parity check matrix to",
    )

    args = parser.parse_args()

    if args.scheme == "hsiao_secded":
        r, n, G, H = hsiao_secded_code(args.k)
        check_construction(
            G,
            H,
            # Code distance check is prohibitively expensive for large k
            check_code_distance=(args.k <= MAX_K_FOR_OPTIMAL_ALGORITHM),
            # Row balance check is omitted for non-optimal constructions
            # since they aren't guaranteed to satisfy the property.
            check_row_balance=(args.k <= MAX_K_FOR_OPTIMAL_ALGORITHM),
        )

        file_header = "\n".join(
            [
                f"Number of data bits (k): {args.k}",
                f"Number of parity bits (r): {r}",
                f"Number of codeword bits (n): {n}",
            ]
        )

        # Convert matrices to strings without ellipses
        G_str = np.array2string(
            G, separator=", ", threshold=np.inf, max_line_width=np.inf
        ).replace("0", " " if args.print0 else "0")
        H_str = np.array2string(
            H, separator=", ", threshold=np.inf, max_line_width=np.inf
        ).replace("0", " " if args.print0 else "0")

        args.generator_matrix_output.write(
            file_header + "\nG =\n" + G_str + "\n"
        )
        args.parity_check_matrix_output.write(
            file_header + "\nH =\n" + H_str + "\n"
        )


if __name__ == "__main__":
    main()
