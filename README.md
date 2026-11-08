# GradLens: Incremental Source-Span Gradients with Pluggable Grammar

**Created by Ahmed ([AhmedB03](https://github.com/AhmedB03))** with [Uncharted](https://github.com/AhmedB03/uncharted).

>  I built and ran Uncharted, a system that
> finds technique combinations that make sense but that no existing project combines. A language model
> (Claude Sonnet 5.5) designed this project from one such combination, and another (Claude Haiku 5.5) wrote
> the code and iterated until its own test suite passed. I chose it for publication;

GradLens keeps a text file of small math definitions incrementally parsed as you edit it, and reverse-mode differentiates the syntax tree itself, so every variable and literal occurrence in the source gets its own ∂output/∂occurrence value at its current text offset. It is for people building live sensitivity or what-if tooling, or editor integrations, for numeric DSLs.

**Why this is unusual:** AD tools differentiate traced values and discard source positions, while incremental parsers never compute numbers. Here a plugin supplies a token, its precedence and its derivative rule as one unit, and the AD tape is built directly over a lossless, width-based, interned syntax tree.

**Closest existing tool:** JAX (jax.grad) for source-level differentiation, with tree-sitter's incremental reparse as the editor-side counterpart. The two are never connected, so the gradient cannot be tied to source spans.

## The combination

| Technique | Normally found in |
| --- | --- |
| automatic differentiation | [FluxML/Flux.jl](https://github.com/FluxML/Flux.jl), [JuliaAI/MLJ.jl](https://github.com/JuliaAI/MLJ.jl), [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) |
| modular plugin architecture for extensibility | [AppFlowy-IO/AppFlowy](https://github.com/AppFlowy-IO/AppFlowy), [AstroNvim/AstroNvim](https://github.com/AstroNvim/AstroNvim), [DevToys-app/DevToys](https://github.com/DevToys-app/DevToys) |
| incremental syntax parsing with tree-sitter | [AstroNvim/AstroNvim](https://github.com/AstroNvim/AstroNvim), [DeusData/codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp), [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) |

Checked against about 3,000 popular GitHub repos: none of them combines all three.

## Features

* Lossless green-tree incremental parser: tokens carry leading trivia, nodes store only widths, and nodes are hash-consed (interned) by (kind, text, child ids). An edit relexes and reparses only the damaged lines, extending the damage when a newline is deleted. Untouched line nodes are reused by identity, absolute offsets are computed during traversal, and the result is the identical object a full reparse would produce.
* Reverse-mode automatic differentiation over the syntax tree. Every evaluated node becomes a tape Var with (parent, local partial) edges, and a reversed sweep accumulates adjoints. Each variable or literal occurrence in the differentiated definition becomes its own tape node, so the output lists (start, end, text, adjoint) per source span. User-defined function calls are differentiated through, with recursion detected.
* Modular plugin registry in which a plugin registers functions (arity, forward, partials) and infix operators (symbol, precedence, associativity, forward, partials). Registered operator symbols drive the lexer's longest-match table and the Pratt parser's binding powers, so a plugin changes syntax and differentiation together. Plugins declare dependencies, can be loaded from built-in names or .py files via importlib, and can be unloaded. Any grammar change triggers a full reparse.
* Plugin gradcheck: each registered primitive's partials are compared with central finite differences at fixed sample points. Wrong derivative rules are reported by name.
* Edit-driven CLI and library: a Session applies (start, end, text) edits and reports lines_reparsed and lines_reused. It exposes diagnostics for malformed lines (kept as error nodes, so one bad line never breaks the others) and returns JSON with value, parameter gradients, per-span attributions and parse stats.

## Usage

Requires Python 3.11+, standard library only.

```
python -m gradlens grad examples/model.gl f x=0 y=2 --plugin math
# examples/model.gl:  f(x, y) = x*y + sin(x)
# -> value 0.0, grad {x: 3.0, y: 0.0}, plus each occurrence of x and y in the source with its own adjoint
```

Run the tests:

```
pip install pytest && python -m pytest -q
```

## Build record

* 16 tests passing, 749 lines of code
* Reached passing tests after 2 build iteration(s)
* Published 2026-11-08

If this turns out to be useful, a star tells the experiment that this kind of combination works.
