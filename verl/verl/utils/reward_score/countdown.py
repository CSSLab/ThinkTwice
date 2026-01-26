from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import re
from typing import Any

_SOLUTION_CLIP_CHARS = 4000
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _get_field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except Exception:
        return getattr(obj, key, None)


def _coerce_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in list(values)]


def _extract_answer(solution_str: str) -> str | None:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    matches = _ANSWER_RE.findall(solution_str)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_expr(expr: str) -> str:
    expr = expr.strip()
    expr = expr.strip("`")
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("\n", " ").strip()
    expr = expr.rstrip(" .;")
    return expr


def _const_to_fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("bool constant not allowed")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        frac = Fraction(str(value))
        if frac.denominator != 1:
            raise ValueError("non-integer float constant not allowed")
        return frac
    raise ValueError(f"unsupported constant type: {type(value)}")


def _eval_expr(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        frac = _const_to_fraction(node.value)
        return frac, [abs(int(frac.numerator))]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val, used = _eval_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return -val, used
        return val, used

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left_val, left_used = _eval_expr(node.left)
        right_val, right_used = _eval_expr(node.right)

        if isinstance(node.op, ast.Add):
            return left_val + right_val, left_used + right_used
        if isinstance(node.op, ast.Sub):
            return left_val - right_val, left_used + right_used
        if isinstance(node.op, ast.Mult):
            return left_val * right_val, left_used + right_used
        if isinstance(node.op, ast.Div):
            if right_val == 0:
                raise ZeroDivisionError("division by zero")
            return left_val / right_val, left_used + right_used

    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _try_parse(expr: str) -> tuple[Fraction, list[int]] | None:
    expr = _normalize_expr(expr)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    try:
        return _eval_expr(tree)
    except Exception:
        return None


def compute_score(solution_str: str, ground_truth: Any, extra_info: dict | None = None, **_: Any) -> dict[str, Any]:
    """Countdown reward: 0/1 score from strict equation parsing."""
    _ = extra_info  # reserved for future use
    allowed_numbers = _coerce_int_list(_get_field(ground_truth, "numbers"))
    target_raw = _get_field(ground_truth, "target")
    if target_raw is None:
        return {"score": 0.0, "acc": False}
    target = int(target_raw)

    answer = _extract_answer(solution_str)
    if answer is None:
        return {"score": 0.0, "acc": False}

    answer_expr = _normalize_expr(answer)
    if not answer_expr:
        return {"score": 0.0, "acc": False}

    numbers_ok: bool | None = None
    numbers_counter = Counter(allowed_numbers)
    parsed = _try_parse(answer_expr)
    chosen_value: Fraction | None = None
    used_numbers: list[int] | None = None
    if parsed is not None:
        chosen_value, used_numbers = parsed
        numbers_ok = Counter(used_numbers) == numbers_counter
    else:
        if allowed_numbers:
            digits = [abs(int(x)) for x in re.findall(r"-?\d+", answer_expr)]
            numbers_ok = Counter(digits) == numbers_counter

    answer_present = answer is not None
    exact_match = bool(chosen_value is not None and chosen_value == Fraction(target, 1))
    score = 1.0 if answer_present and (numbers_ok is not False) and exact_match else 0.0
    return {"score": float(score), "acc": bool(score)}


__all__ = ["compute_score"]
