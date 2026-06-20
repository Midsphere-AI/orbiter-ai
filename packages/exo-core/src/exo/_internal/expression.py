"""Safe AST-based expression evaluator for workflow conditions.

Evaluates user-provided expressions in a restricted sandbox that blocks
dangerous operations while supporting comparisons, boolean, arithmetic, and string ops.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from exo.types import ExoError

MAX_EXPRESSION_LENGTH = 500
MAX_AST_DEPTH = 10
MAX_COLLECTION_SIZE = 1000


class ExpressionError(ExoError):
    """Raised when an expression is invalid or violates safety constraints."""


_JS_REPLACEMENTS = [("&&", " and "), ("||", " or "), ("===", "=="), ("!==", "!=")]
_LITERAL_MAP = {"true": "True", "false": "False", "null": "None", "undefined": "None"}
_SAFE_NAMES = frozenset(
    {
        "True",
        "False",
        "None",
        "int",
        "float",
        "str",
        "bool",
        "len",
        "abs",
        "min",
        "max",
        "round",
        "list",
        "dict",
        "tuple",
        "set",
        "isinstance",
        "type",
    }
)
_BLOCKED_ATTRS = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__func__",
        "__self__",
        "__dict__",
        "__init__",
        "__new__",
        "__del__",
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
    }
)
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_UNA = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}


def _normalise(expr: str) -> str:
    for old, new in _JS_REPLACEMENTS:
        expr = expr.replace(old, new)
    return " ".join(_LITERAL_MAP.get(t, t) for t in expr.split())


def _check_depth(node: ast.AST, depth: int = 0) -> None:
    if depth > MAX_AST_DEPTH:
        raise ExpressionError(f"Expression exceeds maximum depth of {MAX_AST_DEPTH}")
    for child in ast.iter_child_nodes(node):
        _check_depth(child, depth + 1)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _validate(node: ast.AST) -> None:
    for n in ast.walk(node):
        if isinstance(n, ast.Import | ast.ImportFrom):
            raise ExpressionError("Import statements are not allowed")
        if isinstance(n, ast.Attribute) and (_is_dunder(n.attr) or n.attr in _BLOCKED_ATTRS):
            raise ExpressionError(f"Dunder attribute access is not allowed: {n.attr}")
        if isinstance(n, ast.Lambda):
            raise ExpressionError("Lambda expressions are not allowed")
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id not in _SAFE_NAMES:
                raise ExpressionError(f"Function call not allowed: {f.id}")
            elif isinstance(f, ast.Attribute) and _is_dunder(f.attr):
                raise ExpressionError(f"Dunder method call is not allowed: {f.attr}")
            elif not isinstance(f, ast.Name | ast.Attribute):
                raise ExpressionError("Complex function calls are not allowed")
        if isinstance(n, ast.Starred):
            raise ExpressionError("Starred expressions are not allowed")
        if isinstance(n, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            raise ExpressionError("Comprehensions are not allowed")
        if isinstance(n, ast.NamedExpr):
            raise ExpressionError("Named expressions are not allowed")
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Store) and _is_dunder(n.id):
            raise ExpressionError(f"Dunder name access is not allowed: {n.id}")


def _check_size(col: Any) -> Any:
    if isinstance(col, (list, dict, tuple, set)) and len(col) > MAX_COLLECTION_SIZE:
        raise ExpressionError(f"Collection exceeds maximum size of {MAX_COLLECTION_SIZE}")
    return col


def _ev(node: ast.AST, v: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _ev(node.body, v)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in v:
            return v[node.id]
        if node.id in _SAFE_NAMES:
            return (
                __builtins__[node.id]
                if isinstance(__builtins__, dict)
                else getattr(__builtins__, node.id)
            )  # type: ignore[index]
        raise ExpressionError(f"Undefined variable: {node.id}")
    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        result: Any = bool(is_and)
        for val in node.values:
            result = _ev(val, v)
            if (is_and and not result) or (not is_and and result):
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _ev(node.left, v)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _ev(comp, v)
            fn = _CMP.get(type(op))
            if fn is None:
                raise ExpressionError(f"Unsupported comparison: {type(op).__name__}")
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BinOp):
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"Unsupported binary op: {type(node.op).__name__}")
        return fn(_ev(node.left, v), _ev(node.right, v))
    if isinstance(node, ast.UnaryOp):
        fn = _UNA.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"Unsupported unary op: {type(node.op).__name__}")
        return fn(_ev(node.operand, v))
    if isinstance(node, ast.IfExp):
        return _ev(node.body, v) if _ev(node.test, v) else _ev(node.orelse, v)
    if isinstance(node, ast.Subscript):
        return _ev(node.value, v)[_ev(node.slice, v)]
    if isinstance(node, ast.Attribute):
        return getattr(_ev(node.value, v), node.attr)
    if isinstance(node, ast.Call):
        func = _ev(node.func, v)
        args = [_ev(a, v) for a in node.args]
        kw = {k.arg: _ev(k.value, v) for k in node.keywords if k.arg is not None}
        return _check_size(func(*args, **kw))
    if isinstance(node, ast.List):
        return _check_size([_ev(e, v) for e in node.elts])
    if isinstance(node, ast.Tuple):
        return _check_size(tuple(_ev(e, v) for e in node.elts))
    if isinstance(node, ast.Dict):
        ks = [_ev(k, v) for k in node.keys if k is not None]
        vs = [_ev(val, v) for val in node.values]
        return _check_size(dict(zip(ks, vs, strict=True)))
    if isinstance(node, ast.Set):
        return _check_size({_ev(e, v) for e in node.elts})
    if isinstance(node, ast.JoinedStr):
        parts = []
        for val in node.values:
            parts.append(
                str(_ev(val.value, v)) if isinstance(val, ast.FormattedValue) else str(_ev(val, v))
            )
        return "".join(parts)
    raise ExpressionError(f"Unsupported expression type: {type(node).__name__}")


def evaluate_expression(expr: str, variables: dict[str, Any] | None = None) -> Any:
    """Evaluate a restricted expression string safely.

    Args:
        expr: The expression string to evaluate (max 500 chars).
        variables: Mapping of variable names to their values.

    Returns:
        The result of evaluating the expression.

    Raises:
        ExpressionError: If the expression is invalid, too long, too
            deeply nested, or uses a blocked construct.
    """
    if variables is None:
        variables = {}
    expr = expr.strip()
    if not expr:
        raise ExpressionError("Empty expression")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH}")
    expr = _normalise(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression syntax: {exc.msg}") from exc
    _check_depth(tree)
    _validate(tree)
    return _ev(tree, variables)
