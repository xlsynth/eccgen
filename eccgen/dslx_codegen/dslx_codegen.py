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
import numpy as np
import textwrap
from jinja2 import Template
from eccgen.matrix_parser import parse_g_and_h_files
    
DSLX_SUPPORTED_K = [
    4,
    8,
    11,
    16,
    26,
    32,
    57,
    64,
    120,
    128,
    247,
    256,
    502,
    512,
    1013,
    1024,
]

def check_filename_extension(filename: str, allowed_extensions: tuple[str]) -> str:
    """Checks if a filename has one of the allowed extensions and returns it as-is."""
    if not filename.endswith(allowed_extensions):
        raise argparse.ArgumentTypeError(
            f"File '{filename}' must have one of the extensions: {allowed_extensions}"
        )
    return filename


def x_jinja2_file(filename: str) -> str:
    return check_filename_extension(filename, (".x.jinja2"))


def txt_file(filename: str) -> str:
    return check_filename_extension(filename, (".txt"))


def render_encoder_jinja2_template(
    codes, template_file: str, output_file: str
) -> None:
    with open(template_file, "r") as template_file:
        template = Template(template_file.read())
        mapping = {}
        for k in codes:
            r, n, G, _H = codes[k]
            mapping[f"secded_enc_{n}_{k}"] = G_to_x(G, r)
        rendered = template.render(mapping)
        rendered += "\n"
        output_file.write(rendered)


def render_decoder_jinja2_template(
    codes, template_file: str, output_file: str
) -> None:
    with open(template_file, "r") as template_file:
        template = Template(template_file.read())
        mapping = {}
        for k in codes:
            _r, n, _G, H = codes[k]
            mapping[f"secded_dec_syndrome_{n}_{k}"] = syndrome_to_x(H)
            mapping[f"secded_dec_H_{n}_{k}"] = H_to_x(H)
        rendered = template.render(mapping)
        rendered += "\n"
        output_file.write(rendered)


def G_to_x(G: np.ndarray, r: int) -> str:
    """Generate DSLX code for the given generator matrix G."""

    def G_col_to_x(col: np.ndarray, col_idx: int) -> str:
        """Generate a DSLX assignment for a single column of the generator matrix G."""
        xors = []
        nonzero_indices = np.nonzero(col)[0]
        for i in nonzero_indices:
            xors.append(f"m[{i}+:u1]")
        return f"            let parity_{col_idx} = " + " ^ ".join(xors) + ";"

    let_parity_bit = []
    # Since we know G is in systematic form, we can just assign the message bits to the codeword bits.
    # We don't need to codegen that part.
    k = G.shape[0]
    for i in range(r):
        let_parity_bit.append(G_col_to_x(G[:, k + i], i))
    expr = []
    for i in range(r):
        expr.append(f"parity_{r - i - 1}")
    expr = " ++ ".join(expr)
    expr = "(" + expr + ") as uN[MAX_PARITY_WIDTH]"
    expr = textwrap.indent(expr, " " * 12)
    return "\n".join(let_parity_bit + [expr])


def syndrome_to_x(H: np.ndarray) -> str:
    """Generate DSLX code for the syndrome of the given parity-check matrix H."""

    def syndrome_bit_to_x(row: np.ndarray, row_idx: int) -> str:
        """Generate a DSLX assignment for a single bit of the syndrome."""
        xors = []
        nonzero_indices = np.nonzero(row)[0]
        for i in nonzero_indices:
            xors.append(f"cw[{i}+:u1]")
        return f"            let syndrome_{row_idx} = " + " ^ ".join(xors) + ";"

    lets = []
    r = H.shape[0]
    for i in range(r):
        # Reverse row index (r - i - 1)  because row 0 is actually the MSb of the syndrome
        lets.append(syndrome_bit_to_x(H[i, :], r - i - 1))
    expr = []
    for i in range(r):
        expr.append(f"syndrome_{r - i - 1}")
    expr = " ++ ".join(expr)
    expr = "(" + expr + ") as uN[MAX_PARITY_WIDTH]"
    expr = "let syndrome = " + expr + ";"
    expr = textwrap.indent(expr, " " * 12)
    return "\n".join(lets + [expr])


def H_to_x(H: np.ndarray) -> str:
    """Generate DSLX code for the columns of the given parity-check matrix H."""

    def H_col_to_x(col: np.ndarray, col_idx: int) -> str:
        """Generate a DSLX assignment for a single column of the parity-check matrix H."""
        r = col.shape[0]
        return (
            f"            let parity_check_matrix_{col_idx} = u{r}:0b"
            + "".join(col.astype(str))
            + " as uN[MAX_PARITY_WIDTH];"
        )

    lets = []
    n = H.shape[1]
    for i in range(n):
        lets.append(H_col_to_x(H[:, i], i))
    expr = []
    for i in range(n):
        expr.append(f"parity_check_matrix_{n - i - 1}")
    expr = " ++ ".join(expr)
    expr = f"let parity_check_matrix = {expr};"
    expr = textwrap.indent(expr, " " * 12)
    return "\n".join(lets + [expr])


def main():
    parser = argparse.ArgumentParser(description="DSLX ECC Codegen")
    parser.add_argument(
        "--matrix-dir",
        type=str,
        required=True,
        help="The directory containing the generator and parity-check matrices.",
    )
    parser.add_argument(
        "--encoder-template",
        type=x_jinja2_file,
        required=True,
        help="The input file containing the Jinja2 DSLX code template for the encoder.",
    )
    parser.add_argument(
        "--encoder-output",
        type=argparse.FileType("w"),
        required=True,
        help="Dump the encoder implementation for all supported codes to the provided output file.",
    )
    parser.add_argument(
        "--decoder-template",
        type=x_jinja2_file,
        required=True,
        help="The input file containing the Jinja2 DSLX code template for the decoder.",
    )
    parser.add_argument(
        "--decoder-output",
        type=argparse.FileType("w"),
        required=True,
        help="Dump the decoder implementation for all supported codes to the provided output file.",
    )

    args = parser.parse_args()
    codes = parse_g_and_h_files(args.matrix_dir, DSLX_SUPPORTED_K)

    render_encoder_jinja2_template(
        codes, args.encoder_template, args.encoder_output
    )
    render_decoder_jinja2_template(
        codes,
        args.decoder_template,
        args.decoder_output,
    )


if __name__ == "__main__":
    main()


