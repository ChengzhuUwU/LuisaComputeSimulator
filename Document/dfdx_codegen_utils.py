from __future__ import annotations

import re
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import sympy as sp
from sympy.printing.c import ccode
from sympy.printing.pycode import pycode
from sympy.utilities.iterables import numbered_symbols


def _to_cpp_expr(expr: sp.Expr, mapping: Dict[str, str]) -> str:
    s = ccode(sp.sympify(expr))
    for src, dst in mapping.items():
        s = re.sub(rf"\b{re.escape(src)}\b", dst, s)
    return s


def _matrix_entries(mat: sp.Matrix) -> List[sp.Expr]:
    return [sp.sympify(mat[r, c]) for r in range(mat.rows) for c in range(mat.cols)]


def _should_use_intermediate_vars(
    entries: Sequence[sp.Expr],
    cse_subs: Sequence[Tuple[sp.Symbol, sp.Expr]],
    reduced_entries: Sequence[sp.Expr],
    policy: Union[str, bool],
) -> bool:
    if isinstance(policy, bool):
        return policy

    raw_ops = sum(int(sp.count_ops(e, visual=False)) for e in entries)
    cse_ops = sum(int(sp.count_ops(e, visual=False)) for _, e in cse_subs)
    cse_ops += sum(int(sp.count_ops(e, visual=False)) for e in reduced_entries)
    saved_ops = raw_ops - cse_ops
    return len(cse_subs) > 0 and saved_ops >= 2


def print_matrix_expression_cpp(
    mat: sp.Matrix,
    *,
    function_name: str,
    result_type: str,
    input_type: str,
    input_name: str = "Input",
    scalar_type: str = "float",
    extract_lines: Optional[Sequence[str]] = None,
    mapping: Optional[Dict[str, str]] = None,
    emit_zero_entries: bool = False,
    use_intermediate_vars: Union[str, bool] = "auto",
    section_title: Optional[str] = None,
    result_var_name: str = "result",
    temp_prefix: str = "val_",
) -> None:
    local_mapping = mapping if mapping is not None else {}
    entries = _matrix_entries(mat)
    cse_subs, reduced = sp.cse(
        entries,
        symbols=numbered_symbols(temp_prefix, start=1),
        optimizations="basic",
    )
    use_tmp = _should_use_intermediate_vars(entries, cse_subs, reduced, use_intermediate_vars)

    if section_title:
        print(f"\n=== {section_title} ===")
    print(f"inline {result_type} {function_name}(const {input_type}& {input_name})")
    print("{")
    if extract_lines:
        for line in extract_lines:
            print(line)
        print("")

    print(f"    {result_type} {result_var_name} = {result_type}::zero();")

    if use_tmp:
        for sym, expr in cse_subs:
            expr_cpp = _to_cpp_expr(expr, local_mapping)
            print(f"    const {scalar_type} {sym} = {expr_cpp};")

    active_entries = reduced if use_tmp else entries
    idx = 0
    for r in range(mat.rows):
        for c in range(mat.cols):
            value = sp.sympify(active_entries[idx])
            idx += 1
            if (not emit_zero_entries) and value == 0:
                continue
            expr_cpp = _to_cpp_expr(value, local_mapping)
            print(f"    set_matrix_scalar({result_var_name}, {r}, {c}, {expr_cpp});")

    print(f"    return {result_var_name};")
    print("}")


def print_matrix_expression_python(
    mat: sp.Matrix,
    *,
    function_name: str,
    arg_names: Sequence[str],
    scalar_type: str = "float",
) -> None:
    py_args = ", ".join(f"{name}: {scalar_type}" for name in arg_names)
    print("=== Python Function ===")
    print(f"def {function_name}({py_args}):")
    print(f"    return {pycode(mat)}")


def _default_mapping(dm_symbol_names: Iterable[str]) -> Dict[str, str]:
    names = list(dm_symbol_names)
    if len(names) == 4:
        return {
            names[0]: "d0",
            names[1]: "d1",
            names[2]: "d2",
            names[3]: "d3",
        }
    if len(names) == 9:
        short = ["m", "n", "o", "p", "q", "r", "s", "t", "u"]
        return {k: v for k, v in zip(names, short)}
    return {name: f"d{i}" for i, name in enumerate(names)}


LayoutType = Literal["RowMajor", "ColumnMajor", "Both"]


def _normalize_layout(layout: str) -> LayoutType:
    text = layout.strip().lower()
    if text in ("rowmajor", "row_major", "row"):
        return "RowMajor"
    if text in ("columnmajor", "column_major", "column", "col"):
        return "ColumnMajor"
    if text in ("both", "all"):
        return "Both"
    raise ValueError("layout must be one of: RowMajor, ColumnMajor, Both")


def print_dfdx_codegen(
    dfdx: sp.Matrix,
    dm_symbol_names: Iterable[str],
    scalar_type: str = "float",
    emit_zero_entries: bool = False,
    layout: LayoutType = "Both",
    mapping: Optional[Dict[str, str]] = None,
    rowmajor_name: str = "get_dFdx_RowMajor",
    columnmajor_name: str = "get_dFdx_ColumnMajor",
) -> None:
    dm_symbol_names = list(dm_symbol_names)
    selected_layout = _normalize_layout(layout)
    local_mapping = mapping if mapping is not None else _default_mapping(dm_symbol_names)
    print_matrix_expression_python(
        dfdx,
        function_name="build_dFdx",
        arg_names=dm_symbol_names,
        scalar_type=scalar_type,
    )

    dm_dim = int(round(len(dm_symbol_names) ** 0.5))
    dm_type = f"luisa::float{dm_dim}x{dm_dim}"
    extract_lines: List[str] = []
    if len(dm_symbol_names) == 4:
        extract_lines = [
            f"    const {scalar_type} d0 = InverseDm[0][0];",
            f"    const {scalar_type} d1 = InverseDm[0][1];",
            f"    const {scalar_type} d2 = InverseDm[1][0];",
            f"    const {scalar_type} d3 = InverseDm[1][1];",
        ]
    elif len(dm_symbol_names) == 9:
        extract_lines = [
            f"    const {scalar_type} m = InverseDm[0][0];",
            f"    const {scalar_type} n = InverseDm[1][0];",
            f"    const {scalar_type} o = InverseDm[2][0];",
            f"    const {scalar_type} p = InverseDm[0][1];",
            f"    const {scalar_type} q = InverseDm[1][1];",
            f"    const {scalar_type} r = InverseDm[2][1];",
            f"    const {scalar_type} s = InverseDm[0][2];",
            f"    const {scalar_type} t = InverseDm[1][2];",
            f"    const {scalar_type} u = InverseDm[2][2];",
        ]
    else:
        for i, alias in enumerate(local_mapping.values()):
            extract_lines.append(f"    const {scalar_type} {alias} = InverseDm[{i}][0];")

    row_mat = dfdx
    col_mat = dfdx.T

    row_result_type = f"LargeMatrix<{row_mat.rows}, {row_mat.cols}>"
    col_result_type = f"LargeMatrix<{col_mat.rows}, {col_mat.cols}>"
    in_type = dm_type if len(dm_symbol_names) in (4, 9) else "auto"

    if selected_layout in ("RowMajor", "Both"):
        print_matrix_expression_cpp(
            row_mat,
            function_name=rowmajor_name,
            result_type=row_result_type,
            input_type=in_type,
            input_name="InverseDm",
            scalar_type=scalar_type,
            extract_lines=extract_lines,
            mapping=local_mapping,
            emit_zero_entries=emit_zero_entries,
            use_intermediate_vars="auto",
            section_title="C++ RowMajor Interface (style like get_dFdx)",
            result_var_name="result_rm",
        )

    if selected_layout in ("ColumnMajor", "Both"):
        print_matrix_expression_cpp(
            col_mat,
            function_name=columnmajor_name,
            result_type=col_result_type,
            input_type=in_type,
            input_name="InverseDm",
            scalar_type=scalar_type,
            extract_lines=extract_lines,
            mapping=local_mapping,
            emit_zero_entries=emit_zero_entries,
            use_intermediate_vars="auto",
            section_title="C++ ColumnMajor Interface",
            result_var_name="result_cm",
        )
