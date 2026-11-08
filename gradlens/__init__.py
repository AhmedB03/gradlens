"""GradLens: incremental source-span gradients with a pluggable grammar."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "EvalError", "Plugin", "Registry", "Node", "Tree", "Parser", "Session",
    "Tape", "Var", "gradcheck", "load_builtin", "load_plugin", "main",
]


class EvalError(Exception):
    """Raised for evaluation, differentiation or grammar errors."""


# ---------------------------------------------------------------------------
# Plugins and registry
# ---------------------------------------------------------------------------

@dataclass
class FunctionSpec:
    name: str
    arity: int
    forward: Callable[..., float]
    partials: Callable[..., Tuple[float, ...]]


@dataclass
class OperatorSpec:
    symbol: str
    precedence: int
    right_assoc: bool
    forward: Callable[[float, float], float]
    partials: Callable[[float, float, float], Tuple[float, float]]


@dataclass
class Plugin:
    name: str
    functions: Dict[str, FunctionSpec] = field(default_factory=dict)
    operators: Dict[str, OperatorSpec] = field(default_factory=dict)
    depends: Tuple[str, ...] = ()

    def function(self, name, arity, forward, partials):
        self.functions[name] = FunctionSpec(name, arity, forward, partials)
        return self

    def operator(self, symbol, precedence, forward, partials, right_assoc=False):
        self.operators[symbol] = OperatorSpec(symbol, precedence, right_assoc, forward, partials)
        return self


class Registry:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self._functions: Dict[str, FunctionSpec] = {}
        self._operators: Dict[str, OperatorSpec] = {}

    def _rebuild(self):
        self._functions = {}
        self._operators = {}
        for p in self.plugins.values():
            self._functions.update(p.functions)
            self._operators.update(p.operators)

    def register(self, plugin: Plugin):
        if plugin.name in self.plugins:
            raise EvalError(f"plugin already loaded: {plugin.name}")
        for d in plugin.depends:
            if d not in self.plugins:
                raise EvalError(f"missing dependency {d} for {plugin.name}")
        self.plugins[plugin.name] = plugin
        self._rebuild()

    def unregister(self, name: str):
        if name not in self.plugins:
            raise EvalError(f"plugin not loaded: {name}")
        for p in self.plugins.values():
            if name in p.depends:
                raise EvalError(f"{p.name} depends on {name}")
        del self.plugins[name]
        self._rebuild()

    def function(self, name) -> Optional[FunctionSpec]:
        return self._functions.get(name)

    def operators(self) -> Dict[str, OperatorSpec]:
        return self._operators

    def operator(self, symbol) -> Optional[OperatorSpec]:
        return self._operators.get(symbol)


def _math_plugin() -> Plugin:
    p = Plugin("math")
    p.function("sin", 1, lambda a: math.sin(a), lambda a: (math.cos(a),))
    p.function("cos", 1, lambda a: math.cos(a), lambda a: (-math.sin(a),))
    p.function("exp", 1, lambda a: math.exp(a), lambda a: (math.exp(a),))
    p.function("log", 1, lambda a: math.log(a), lambda a: (1.0 / a,))
    p.function("sqrt", 1, lambda a: math.sqrt(a), lambda a: (0.5 / math.sqrt(a),))
    p.function("tanh", 1, lambda a: math.tanh(a), lambda a: (1.0 - math.tanh(a) ** 2,))
    p.function("atan2", 2, lambda a, b: math.atan2(a, b),
               lambda a, b: (b / (a * a + b * b), -a / (a * a + b * b)))
    return p


def _activation_plugin() -> Plugin:
    p = Plugin("activation")
    p.function("relu", 1, lambda a: max(0.0, a), lambda a: (1.0 if a > 0 else 0.0,))

    def sig(a):
        return 1.0 / (1.0 + math.exp(-a))

    p.function("sigmoid", 1, sig, lambda a: (sig(a) * (1 - sig(a)),))
    p.function("softplus", 1, lambda a: math.log1p(math.exp(a)), lambda a: (sig(a),))
    p.function("max", 2, lambda a, b: max(a, b),
               lambda a, b: (1.0, 0.0) if a >= b else (0.0, 1.0))
    return p


def _core_plugin() -> Plugin:
    p = Plugin("core")
    p.operator("+", 10, lambda a, b: a + b, lambda a, b, h: (1.0, 1.0))
    p.operator("-", 10, lambda a, b: a - b, lambda a, b, h: (1.0, -1.0))
    p.operator("*", 20, lambda a, b: a * b, lambda a, b, h: (b, a))
    p.operator("/", 20, lambda a, b: a / b, lambda a, b, h: (1.0 / b, -a / (b * b)))
    p.operator("^", 40, lambda a, b: a ** b,
               lambda a, b, h: (b * a ** (b - 1), h * math.log(a) if a > 0 else 0.0),
               right_assoc=True)
    return p


BUILTINS = {"math": _math_plugin, "activation": _activation_plugin, "core": _core_plugin}


def load_builtin(registry: Registry, name: str):
    if name not in BUILTINS:
        raise EvalError(f"unknown builtin plugin {name}")
    registry.register(BUILTINS[name]())


def load_plugin(registry: Registry, spec: str) -> str:
    """Load a builtin by name or a .py file defining make_plugin() -> Plugin."""
    if spec in BUILTINS:
        load_builtin(registry, spec)
        return spec
    if not os.path.isfile(spec):
        raise EvalError(f"plugin not found: {spec}")
    mod_name = "gradlens_plugin_" + re.sub(r"\W", "_", os.path.abspath(spec))
    mspec = importlib.util.spec_from_file_location(mod_name, spec)
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    if not hasattr(mod, "make_plugin"):
        raise EvalError(f"{spec} must define make_plugin()")
    plugin = mod.make_plugin()
    if not isinstance(plugin, Plugin):
        raise EvalError(f"{spec}: make_plugin() must return Plugin")
    registry.register(plugin)
    return plugin.name


# ---------------------------------------------------------------------------
# Lexer (operators come from the registry, so plugins change tokenization)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str    # 'num', 'id', 'op', 'punct', 'bad', 'eof'
    text: str
    start: int   # absolute offset within the line text
    trivia: str


_NUM_RE = re.compile(r"\d+(\.\d+)?([eE][+-]?\d+)?")
_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def lex_line(text: str, ops: List[str]) -> List[Token]:
    ops_sorted = sorted(ops, key=len, reverse=True)
    toks: List[Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "#":
            break
        m = _NUM_RE.match(text, i)
        if m:
            toks.append(Token("num", m.group(0), i, ""))
            i = m.end()
            continue
        m = _ID_RE.match(text, i)
        if m:
            toks.append(Token("id", m.group(0), i, ""))
            i = m.end()
            continue
        hit = None
        for op in ops_sorted:
            if op and text.startswith(op, i):
                hit = op
                break
        if hit is not None:
            toks.append(Token("op", hit, i, ""))
            i += len(hit)
            continue
        if c in "()=,":
            toks.append(Token("punct", c, i, ""))
            i += 1
            continue
        toks.append(Token("bad", c, i, ""))
        i += 1
    toks.append(Token("eof", "", n, ""))
    return toks


# ---------------------------------------------------------------------------
# Lossless green tree with hash-consing
# ---------------------------------------------------------------------------

class Node:
    """Immutable green node storing only its width (text length)."""
    __slots__ = ("kind", "text", "children", "width", "id")

    def __init__(self, kind: str, text: str, children: Tuple["Node", ...], nid: int):
        self.kind = kind
        self.text = text
        self.children = children
        self.width = len(text)
        self.id = nid

    def __repr__(self):
        return f"Node({self.kind!r}, {self.text!r})"


class Interner:
    def __init__(self):
        self._table: Dict[tuple, Node] = {}
        self._next = 0

    def make(self, kind: str, text: str, children: Tuple[Node, ...] = ()) -> Node:
        key = (kind, text, tuple(c.id for c in children))
        node = self._table.get(key)
        if node is None:
            node = Node(kind, text, children, self._next)
            self._next += 1
            self._table[key] = node
        return node


class Tree:
    def __init__(self, root: Node, source: str):
        self.root = root
        self.text = root.text
        self.width = root.width
        self.source = source


@dataclass
class ErrorInfo:
    line: int
    message: str


# ---------------------------------------------------------------------------
# Expression AST and Pratt parser
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    kind: str                     # 'num', 'var', 'bin', 'neg', 'call'
    value: str = ""
    op: Optional[OperatorSpec] = None
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None
    args: List["Expr"] = field(default_factory=list)
    start: int = 0                # absolute offset within the line text
    end: int = 0


class _ExprParser:
    def __init__(self, toks: List[Token], registry: Registry):
        self.toks = toks
        self.pos = 0
        self.reg = registry

    def peek(self) -> Token:
        return self.toks[self.pos]

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if t.kind != "eof":
            self.pos += 1
        return t

    def is_punct(self, t: Token, s: str) -> bool:
        return t.kind == "punct" and t.text == s

    def parse(self, min_bp: int = 0) -> Expr:
        left = self.prefix()
        while True:
            t = self.peek()
            if t.kind != "op":
                break
            spec = self.reg.operator(t.text)
            if spec is None:
                raise EvalError(f"unknown operator {t.text}")
            if spec.precedence < min_bp:
                break
            self.advance()
            next_bp = spec.precedence if spec.right_assoc else spec.precedence + 1
            right = self.parse(next_bp)
            left = Expr("bin", op=spec, left=left, right=right,
                        start=left.start, end=right.end)
        return left

    def prefix(self) -> Expr:
        t = self.advance()
        if t.kind == "op" and t.text == "-":
            operand = self.parse(30)
            return Expr("neg", left=operand, start=t.start, end=operand.end)
        if t.kind == "num":
            return Expr("num", value=t.text, start=t.start, end=t.start + len(t.text))
        if t.kind == "id":
            if self.is_punct(self.peek(), "("):
                self.advance()
                args: List[Expr] = []
                if not self.is_punct(self.peek(), ")"):
                    while True:
                        args.append(self.parse(0))
                        nt = self.advance()
                        if self.is_punct(nt, ")"):
                            break
                        if not self.is_punct(nt, ","):
                            raise EvalError("expected , or )")
                else:
                    self.advance()
                close = self.toks[self.pos - 1]
                return Expr("call", value=t.text, args=args,
                            start=t.start, end=close.start + 1)
            return Expr("var", value=t.text, start=t.start, end=t.start + len(t.text))
        if self.is_punct(t, "("):
            e = self.parse(0)
            c = self.advance()
            if not self.is_punct(c, ")"):
                raise EvalError("expected )")
            return e
        raise EvalError(f"unexpected token {t.text!r}")


# ---------------------------------------------------------------------------
# Line parsing: "name(params) = body"
# ---------------------------------------------------------------------------

@dataclass
class LineDef:
    name: str
    params: List[str]
    body: Expr


def parse_line_text(line: str, registry: Registry):
    """Parse one line without its newline.
    Returns (LineDef or None, error message or None)."""
    toks = lex_line(line, list(registry.operators().keys()))
    if toks[0].kind == "eof":
        return None, None
    bad = next((t for t in toks if t.kind == "bad"), None)
    if bad is not None:
        return None, f"unexpected character {bad.text!r}"
    i = 0

    def expect(kind, text=None):
        nonlocal i
        t = toks[i]
        if t.kind != kind or (text is not None and t.text != text):
            return None
        i += 1
        return t

    name_tok = expect("id")
    if name_tok is None:
        return None, "expected definition name"
    if expect("punct", "(") is None:
        return None, "expected ("
    params: List[str] = []
    if not (toks[i].kind == "punct" and toks[i].text == ")"):
        while True:
            p = expect("id")
            if p is None:
                return None, "expected parameter name"
            params.append(p.text)
            if toks[i].kind == "punct" and toks[i].text == ",":
                i += 1
                continue
            break
    if expect("punct", ")") is None:
        return None, "expected )"
    if expect("punct", "=") is None:
        return None, "expected ="
    if toks[i].kind == "eof":
        return None, "empty body"
    try:
        ep = _ExprParser(toks[i:], registry)
        body = ep.parse(0)
        if ep.peek().kind != "eof":
            return None, f"unexpected token {ep.peek().text!r}"
    except EvalError as e:
        return None, str(e)
    return LineDef(name_tok.text, params, body), None


# ---------------------------------------------------------------------------
# Parser with per-line reuse
# ---------------------------------------------------------------------------

@dataclass
class LineRecord:
    node: Node
    defn: Optional[LineDef]
    error: Optional[str]
    raw: str


class Parser:
    def __init__(self, interner: Optional[Interner] = None):
        self.interner = interner or Interner()
        self.line_cache: Dict[str, LineRecord] = {}

    def parse_line(self, raw: str, registry: Registry) -> LineRecord:
        if raw in self.line_cache:
            return self.line_cache[raw]
        body = raw[:-1] if raw.endswith("\n") else raw
        defn, error = parse_line_text(body, registry)
        node = self.interner.make("line", raw)
        rec = LineRecord(node, defn, error, raw)
        self.line_cache[raw] = rec
        return rec


def split_lines(source: str) -> List[str]:
    return source.splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Tape and reverse-mode AD
# ---------------------------------------------------------------------------

class Var:
    __slots__ = ("value", "parents", "adjoint")

    def __init__(self, value: float, parents=()):
        self.value = value
        self.parents = list(parents)  # (Var, local partial)
        self.adjoint = 0.0


class Tape:
    def __init__(self):
        self.nodes: List[Var] = []

    def var(self, value: float, parents=()) -> Var:
        v = Var(value, parents)
        self.nodes.append(v)
        return v

    def backward(self, out: Var):
        for v in self.nodes:
            v.adjoint = 0.0
        out.adjoint = 1.0
        for v in reversed(self.nodes):
            for p, partial in v.parents:
                p.adjoint += v.adjoint * partial


class Evaluator:
    def __init__(self, registry: Registry, defs: Dict[str, LineDef],
                 body_offsets: Dict[str, int]):
        self.registry = registry
        self.defs = defs
        self.body_offsets = body_offsets

    def differentiate(self, name: str, env: Dict[str, float]):
        if name not in self.defs:
            raise EvalError(f"undefined definition {name}")
        d = self.defs[name]
        tape = Tape()
        arg_vars: Dict[str, Var] = {}
        for p in d.params:
            if p not in env:
                raise EvalError(f"missing value for {p}")
            arg_vars[p] = tape.var(float(env[p]))
        attr: List[Tuple[Var, int, int, str]] = []
        out = self._eval(d.body, arg_vars, tape, name, (name,), attr)
        tape.backward(out)
        grad = {p: arg_vars[p].adjoint for p in d.params}
        occ = [(s, e, txt, v.adjoint) for v, s, e, txt in attr]
        return out.value, grad, occ

    def _eval(self, e: Expr, env: Dict[str, Var], tape: Tape, owner: str,
              stack: Tuple[str, ...], attr: list) -> Var:
        base = self.body_offsets[owner]
        if e.kind == "num":
            v = tape.var(float(e.value))
            attr.append((v, base + e.start, base + e.end, e.value))
            return v
        if e.kind == "var":
            if e.value not in env:
                raise EvalError(f"unbound variable {e.value} in {owner}")
            src = env[e.value]
            leaf = tape.var(src.value, parents=[(src, 1.0)])
            attr.append((leaf, base + e.start, base + e.end, e.value))
            return leaf
        if e.kind == "neg":
            a = self._eval(e.left, env, tape, owner, stack, attr)
            return tape.var(-a.value, parents=[(a, -1.0)])
        if e.kind == "bin":
            a = self._eval(e.left, env, tape, owner, stack, attr)
            b = self._eval(e.right, env, tape, owner, stack, attr)
            val = e.op.forward(a.value, b.value)
            pa, pb = e.op.partials(a.value, b.value, val)
            return tape.var(val, parents=[(a, pa), (b, pb)])
        if e.kind == "call":
            args = [self._eval(x, env, tape, owner, stack, attr) for x in e.args]
            spec = self.registry.function(e.value)
            if spec is not None:
                if len(args) != spec.arity:
                    raise EvalError(f"{e.value} expects {spec.arity} args")
                vals = [a.value for a in args]
                val = spec.forward(*vals)
                parts = spec.partials(*vals)
                return tape.var(val, parents=list(zip(args, parts)))
            if e.value in self.defs:
                if e.value in stack:
                    raise EvalError(f"recursion detected in {e.value}")
                callee = self.defs[e.value]
                if len(callee.params) != len(args):
                    raise EvalError(f"{e.value} expects {len(callee.params)} args")
                sub_env = dict(zip(callee.params, args))
                return self._eval(callee.body, sub_env, tape, e.value,
                                  stack + (e.value,), attr)
            raise EvalError(f"unknown function {e.value}")
        raise EvalError(f"cannot evaluate {e.kind}")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, source: str = "", registry: Optional[Registry] = None):
        self.registry = registry or Registry()
        if not self.registry.plugins:
            load_builtin(self.registry, "core")
        self.parser = Parser()
        self.source = ""
        self.lines: List[str] = []
        self.recs: List[LineRecord] = []
        self.last_stats = {"lines_reparsed": 0, "lines_reused": 0}
        self._full_parse(source)

    def _full_parse(self, source: str):
        self.source = source
        self.lines = split_lines(source)
        self.recs = [self.parser.parse_line(l, self.registry) for l in self.lines]
        self.last_stats = {"lines_reparsed": len(self.lines), "lines_reused": 0}

    @property
    def tree(self) -> Tree:
        root = self.parser.interner.make("file", "".join(self.lines),
                                         tuple(r.node for r in self.recs))
        return Tree(root, self.source)

    def _apply_source(self, new_source: str):
        new_lines = split_lines(new_source)
        old = self.lines
        pre = 0
        while pre < len(old) and pre < len(new_lines) and old[pre] == new_lines[pre]:
            pre += 1
        suf = 0
        while (suf < len(old) - pre and suf < len(new_lines) - pre
               and old[len(old) - 1 - suf] == new_lines[len(new_lines) - 1 - suf]):
            suf += 1
        mid = new_lines[pre:len(new_lines) - suf]
        mid_recs = [self.parser.parse_line(l, self.registry) for l in mid]
        self.recs = self.recs[:pre] + mid_recs + self.recs[len(old) - suf:]
        self.lines = new_lines
        self.source = new_source
        self.last_stats = {"lines_reparsed": len(mid), "lines_reused": pre + suf}

    def edit(self, start: int, end: int, text: str):
        if not (0 <= start <= end <= len(self.source)):
            raise EvalError("edit out of range")
        self._apply_source(self.source[:start] + text + self.source[end:])

    def reparse(self, source: str):
        self._apply_source(source)

    def diagnostics(self) -> List[ErrorInfo]:
        return [ErrorInfo(i, r.error) for i, r in enumerate(self.recs) if r.error is not None]

    def load_plugin(self, spec: str) -> str:
        name = load_plugin(self.registry, spec)
        self._regrammar()
        return name

    def unload_plugin(self, name: str):
        self.registry.unregister(name)
        self._regrammar()

    def _regrammar(self):
        # Grammar changed: drop line caches and do a full reparse,
        # keeping the interner so node identities stay stable.
        self.parser = Parser(self.parser.interner)
        self._full_parse(self.source)

    def _defs(self) -> Dict[str, LineDef]:
        out: Dict[str, LineDef] = {}
        for r in self.recs:
            if r.defn is not None:
                out[r.defn.name] = r.defn
        return out

    def _body_offsets(self) -> Dict[str, int]:
        offs: Dict[str, int] = {}
        pos = 0
        for r in self.recs:
            if r.defn is not None:
                offs[r.defn.name] = pos
            pos += len(r.raw)
        return offs

    def grad(self, name: str, env: Dict[str, float]):
        defs = self._defs()
        if name not in defs:
            raise EvalError(f"undefined definition {name}")
        ev = Evaluator(self.registry, defs, self._body_offsets())
        value, grad, occ = ev.differentiate(name, env)
        return value, grad, occ

    def stats(self) -> dict:
        return dict(self.last_stats)


# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------

def gradcheck(registry: Registry, eps: float = 1e-6, tol: float = 1e-4) -> List[str]:
    """Compare primitive partials with central differences; return failing names."""
    failures: List[str] = []
    one_d = [0.3, 0.7, 1.9]
    one_d_pos = [0.5, 1.2, 2.6]
    two_d = [(0.4, 1.3), (1.5, 0.8), (-0.6, 2.1)]
    for plugin in registry.plugins.values():
        for name, spec in plugin.functions.items():
            ok = True
            if spec.arity == 1:
                pts = [[x] for x in (one_d_pos if name in ("log", "sqrt") else one_d)]
            else:
                pts = [list(p) for p in two_d]
            for args in pts:
                try:
                    analytic = spec.partials(*args)
                    for i in range(spec.arity):
                        plus = list(args); minus = list(args)
                        plus[i] += eps; minus[i] -= eps
                        num = (spec.forward(*plus) - spec.forward(*minus)) / (2 * eps)
                        if abs(num - analytic[i]) > tol * max(1.0, abs(num)):
                            ok = False
                except (ValueError, ZeroDivisionError, OverflowError):
                    ok = False
            if not ok:
                failures.append(name)
        for sym, op in plugin.operators.items():
            ok = True
            for a, b in two_d:
                try:
                    val = op.forward(a, b)
                    pa, pb = op.partials(a, b, val)
                    na = (op.forward(a + eps, b) - op.forward(a - eps, b)) / (2 * eps)
                    nb = (op.forward(a, b + eps) - op.forward(a, b - eps)) / (2 * eps)
                    if abs(na - pa) > tol * max(1.0, abs(na)) or \
                       abs(nb - pb) > tol * max(1.0, abs(nb)):
                        ok = False
                except (ValueError, ZeroDivisionError, OverflowError):
                    ok = False
            if not ok:
                failures.append(sym)
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_env(items: List[str]) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for it in items:
        if "=" not in it:
            raise EvalError(f"bad assignment {it}")
        k, v = it.split("=", 1)
        env[k.strip()] = float(v)
    return env


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gradlens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grad")
    g.add_argument("path")
    g.add_argument("name")
    g.add_argument("assignments", nargs="*")
    g.add_argument("--plugin", action="append", default=[])
    g.add_argument("--edit", action="append", default=[])
    args = ap.parse_args(argv)

    with open(args.path, "r", encoding="utf-8") as fh:
        source = fh.read()
    sess = Session(source)
    for p in args.plugin:
        sess.load_plugin(p)
    for e in args.edit:
        s, en, t = e.split(":", 2)
        sess.edit(int(s), int(en), t)
    value, grad, occ = sess.grad(args.name, _parse_env(args.assignments))
    out = {
        "value": value,
        "grad": grad,
        "attribution": [{"start": s, "end": e, "text": t, "adjoint": a}
                        for s, e, t, a in occ],
        "stats": sess.stats(),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0
