# SPDX-License-Identifier: Apache-2.0

import argparse
from eccgen.hsiao_secded import (
    hsiao_secded_code,
    check_construction,
    MAX_K_FOR_OPTIMAL_ALGORITHM,
)
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Error Correction Code Generator")
    parser.add_argument(
        "--scheme",
        "-s",
        type=str,
        choices=["hsiao_secded"],
        required=True,
        help="The error correction code scheme to use (e.g., hsiao_secded)",
    )
    parser.add_argument(
        "-message-size",
        "-k",
        type=int,
        required=True,
        help="The number of bits in the message (k)",
    )
    parser.add_argument(
        "--print0",
        action="store_true",
        help="Print 0s in the outputs (otherwise, leave blanks instead). Defaults to true.",
    )
    parser.add_argument(
        "--generator-matrix-output",
        "-G",
        type=argparse.FileType("w"),
        required=True,
        help="The output file to write the generator matrix to",
    )
    parser.add_argument(
        "--parity-check-matrix-output",
        "-H",
        type=argparse.FileType("w"),
        required=True,
        help="The output file to write the parity check matrix to",
    )

    args = parser.parse_args()
    k = args.message_size

    if args.scheme == "hsiao_secded":
        r, n, G, H = hsiao_secded_code(k)
        check_construction(
            G,
            H,
            # Code distance check is prohibitively expensive for large k
            check_code_distance=(k <= MAX_K_FOR_OPTIMAL_ALGORITHM),
            # Row balance check is omitted for non-optimal constructions
            # since they aren't guaranteed to satisfy the property.
            check_row_balance=(k <= MAX_K_FOR_OPTIMAL_ALGORITHM),
        )

        file_header = "\n".join(
            [
                f"Number of data bits (k): {k}",
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

        args.generator_matrix_output.write(file_header + "\nG =\n" + G_str + "\n")
        args.parity_check_matrix_output.write(file_header + "\nH =\n" + H_str + "\n")


if __name__ == "__main__":
    main()
