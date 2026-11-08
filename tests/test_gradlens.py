import pytest

from gradlens import (
    EvalError, Parser, Plugin, Registry, Session, gradcheck, load_builtin, main,
)


def fresh_session(src=""):
    return Session(src)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_lossless_tree():
    src = "f(x) = x*x + 3 # hi\n\ng(a,b) = a/b\n"
    s = fresh_session(src)
    assert s.tree.text == src
    assert s.tree.width == len(src)


def test_gradient_and_attribution_sin():
    s = fresh_session("f(x, y) = x*y + sin(x)\n")
    s.load_plugin("math")
    value, grad, _ = s.grad("f", {"x": 0.0, "y": 2.0})
    assert value == 0.0
    assert grad["x"] == pytest.approx(3.0)
    assert grad["y"] == pytest.approx(0.0)


def test_attribution_exact_spans():
    s = fresh_session("f(x) = x*x + 3*x\n")
    value, grad, attr = s.grad("f", {"x": 2.0})
    assert value == 10.0
    assert grad == {"x": 7.0}
    assert attr == [(7, 8, "x", 2.0), (9, 10, "x", 2.0),
                    (13, 14, "3", 2.0), (15, 16, "x", 3.0)]


def test_incremental_reuse_identity():
    src = "a(x) = x+1\nb(x) = x*2\nc(x) = x^2\n"
    s = fresh_session(src)
    old = [r.node for r in s.recs]
    s.edit(20, 21, "3")
    assert s.stats()["lines_reparsed"] == 1
    assert s.stats()["lines_reused"] == 2
    assert s.recs[0].node is old[0]
    assert s.recs[2].node is old[2]
    assert s.recs[1].raw == "b(x) = x*3\n"
    fresh = Parser()
    full = [fresh.parse_line(l, s.registry).raw for l in s.lines]
    assert s.tree.root.text == "".join(full)
    assert s.tree.root.text == s.source


def test_newline_deletion_merges_lines():
    s = fresh_session("f(x) = x +\n1\n")
    diags = [d.line for d in s.diagnostics()]
    assert diags == [0, 1]
    s.edit(10, 11, "")
    assert s.source == "f(x) = x +1\n"
    assert s.diagnostics() == []
    assert s.stats()["lines_reparsed"] == 1
    value, grad, _ = s.grad("f", {"x": 5.0})
    assert value == 6.0
    assert grad == {"x": 1.0}


def hyp_plugin():
    import math
    p = Plugin("hyp")
    p.operator("<>", 30,
               lambda a, b: math.sqrt(a * a + b * b),
               lambda a, b, h: (a / h, b / h))
    return p


def test_plugin_operator_changes_syntax_and_ad():
    s = fresh_session("h(a,b) = a <> b\n")
    assert len(s.diagnostics()) == 1
    s.registry.register(hyp_plugin())
    s.reload_grammar = None
    s._regrammar()
    assert s.diagnostics() == []
    value, grad, _ = s.grad("h", {"a": 3.0, "b": 4.0})
    assert value == pytest.approx(5.0)
    assert grad["a"] == pytest.approx(0.6)
    assert grad["b"] == pytest.approx(0.8)
    s.unload_plugin("hyp")
    assert len(s.diagnostics()) == 1
    with pytest.raises(EvalError):
        s.grad("h", {"a": 3.0, "b": 4.0})


def test_plugin_precedence_drives_parse():
    import math
    s = fresh_session("k(a,b) = 1 + a <> b * 2\n")
    p = Plugin("hyp30")
    p.operator("<>", 30, lambda a, b: math.sqrt(a * a + b * b),
               lambda a, b, h: (a / h, b / h))
    s.registry.register(p)
    s._regrammar()
    value, _, _ = s.grad("k", {"a": 3.0, "b": 4.0})
    assert value == pytest.approx(11.0)
    s.unload_plugin("hyp30")
    import math as _m
    p2 = Plugin("hyp15")
    p2.operator("<>", 15, lambda a, b: _m.sqrt(a * a + b * b),
                lambda a, b, h: (a / h, b / h))
    s.registry.register(p2)
    s._regrammar()
    value, _, _ = s.grad("k", {"a": 3.0, "b": 4.0})
    assert value == pytest.approx(1 + 8.5440037453, abs=1e-6)


def test_unary_minus_and_right_assoc_power():
    s = fresh_session("p(x) = -x^2\nr(x) = x^2^2\n")
    value, grad, _ = s.grad("p", {"x": 3.0})
    assert value == -9.0
    assert grad == {"x": -6.0}
    value, grad, _ = s.grad("r", {"x": 2.0})
    assert value == 16.0
    assert grad == {"x": 32.0}


def test_user_function_calls():
    s = fresh_session("sq(x) = x*x\nf(a,b) = sq(a) + 3*sq(b)\n")
    value, grad, attr = s.grad("f", {"a": 2.0, "b": 1.0})
    assert value == 7.0
    assert grad == {"a": 4.0, "b": 6.0}
    assert any(t == "a" and a == 4.0 for _, _, t, a in attr)


def test_recursion_detected():
    s = fresh_session("f(x) = f(x)\n")
    with pytest.raises(EvalError):
        s.grad("f", {"x": 1.0})


def test_edit_then_differentiate():
    s = fresh_session("f(x) = x*x\ng(x) = x+1\n")
    value, grad, _ = s.grad("f", {"x": 3.0})
    assert grad == {"x": 6.0}
    g_before = s.recs[1].node
    s.edit(7, 10, "x*x*x")
    value, grad, _ = s.grad("f", {"x": 3.0})
    assert value == 27.0
    assert grad == {"x": 27.0}
    assert s.recs[1].node is g_before


def test_spans_shift_without_reparsing_definition():
    s = fresh_session("f(x) = x*x\n")
    f_node = s.recs[0].node
    s.edit(0, 0, "# header\n")
    assert s.stats()["lines_reparsed"] == 1
    assert s.stats()["lines_reused"] == 1
    assert s.recs[1].node is f_node
    _, _, attr = s.grad("f", {"x": 3.0})
    assert attr == [(16, 17, "x", 3.0), (18, 19, "x", 3.0)]


def test_gradcheck_builtins_pass():
    reg = Registry()
    load_builtin(reg, "math")
    load_builtin(reg, "activation")
    assert gradcheck(reg) == []


def test_gradcheck_detects_bad_rule():
    reg = Registry()
    p = Plugin("bad")
    p.function("sq", 1, lambda x: x * x, lambda x: (3 * x,))
    reg.register(p)
    assert "sq" in gradcheck(reg)


def test_cli_grad(tmp_path, capsys):
    path = write(tmp_path, "m.gl", "f(x,y) = x*y\n")
    rc = main(["grad", path, "f", "x=2", "y=3"])
    assert rc == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == 6.0
    assert out["grad"] == {"x": 3.0, "y": 2.0}
    assert len(out["attribution"]) == 2


def test_cli_with_edit_and_plugin(tmp_path, capsys):
    path = write(tmp_path, "m.gl", "f(x) = x*x\n")
    rc = main(["grad", path, "f", "x=3", "--plugin", "math", "--edit", "0:0:# c\n"])
    assert rc == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == 9.0
    assert out["stats"]["lines_reused"] == 1
