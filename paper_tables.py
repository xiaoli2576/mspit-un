"""
paper_tables.py -- Verify and generate the worked-example tables of the paper.

For the 10-node worked example (Appendix A) this script:

  1. Runs the DP (solve_mspit_unu with return_tables=True) and extracts every
     table cell g(chi, eps, k) and f(T^{a:q}_v, eps, k) with its witness set.
  2. Independently verifies EVERY cell by brute-force enumeration over all
     feasible upgrade sets of the corresponding substructure (the brute force
     shares no code with the DP recursion).
  3. Verifies that each DP witness set actually achieves the cell value by
     direct simulation on the original tree.
  4. Writes machine-readable JSON to  results/example_tables.json.
  5. Emits LaTeX sources for the two appendix tables (tab-g, tab-f) to
     ../revision/generated-tables.tex.

Branch ordering: the CD orderings are the ones produced by tree.py
(ascending node index), i.e.
    CD(v1) = (v2, v6, v7),  CD(v2) = (v3, v4),  CD(v7) = (v8, v10).
The manuscript must state this ordering explicitly next to the tables.

Exit code 0 iff every cell passes both verifications.
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree import Tree
from mspit_un import solve_mspit_unu, _build_g_tables, NEG_INF
from example import build_example1

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TEX_OUT = os.path.join(HERE, "..", "revision", "generated-tables.tex")

K_EXAMPLE = 1


def vname(i):
    return f"v{i + 1}"


def vset_names(S):
    return "{" + ",".join(vname(v) for v in sorted(S)) + "}"


# ---------------------------------------------------------------------------
# Independent simulation: path lengths under an upgrade set
# ---------------------------------------------------------------------------

def upgraded_len(tree, top, leaf, S):
    """Length of the path top -> leaf when every node in S is upgraded.
    Edge (p, c) counts u if p in S else w."""
    d = 0.0
    cur = leaf
    while cur != top:
        p, w, uw = tree.parent[cur]
        d += uw if p in S else w
        cur = p
    return d


def piece_nodes(tree, v, branch_tips):
    """Node set of T^{...}_v = union over the given chain tips of
    (chain nodes from v to tip) + (full subtree below tip)."""
    nodes = {v}
    for tip in branch_tips:
        cur = tip
        while cur != v:
            nodes.add(cur)
            cur = tree.parent[cur][0]
        stack = [tip]
        while stack:
            u = stack.pop()
            nodes.add(u)
            for (c, _, _) in tree.children[u]:
                stack.append(c)
    return nodes


def piece_value(tree, v, nodes, S):
    """min over leaves in the piece of the upgraded v->leaf length."""
    leaves = [t for t in nodes if tree.is_leaf[t]]
    return min(upgraded_len(tree, v, t, S) for t in leaves)


def brute_f(tree, v, nodes, eps, k):
    """Brute-force f(T^{piece}_v, eps, k): max over S subset of non-leaf
    piece nodes, |S| <= k, [v in S] == eps."""
    cand = [u for u in nodes if not tree.is_leaf[u] and u != v]
    best = NEG_INF
    best_S = None
    for r in range(0, k + 1 - eps):
        for combo in itertools.combinations(cand, r):
            S = set(combo) | ({v} if eps else set())
            val = piece_value(tree, v, nodes, S)
            if val > best:
                best, best_S = val, S
    return best, best_S


def brute_g(tree, chain, eps, k):
    """Brute-force g(chi, eps, k): chain-edge sum when exactly k of
    {CA} + interior degree-2 nodes are upgraded, [CA in S] == eps,
    maximised over the choice of interior nodes."""
    ca = chain.ca
    interior = [e[0] for e in chain.edges[1:]]  # owners of non-CA edges
    best = NEG_INF
    best_S = None
    for combo in itertools.combinations(interior, k - eps):
        S = set(combo) | ({ca} if eps else set())
        val = sum((uw if p in S else w) for (p, c, w, uw) in chain.edges)
        if val > best:
            best, best_S = val, S
    return best, best_S


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def nonleaf_count(tree, nodes, v):
    return sum(1 for u in nodes if not tree.is_leaf[u])


def collect_cells(tree, K):
    """Run the DP and enumerate all g- and f-cells of the worked example."""
    opt, S_opt, tb = solve_mspit_unu(tree, K, return_tables=True)

    g_rows, f_rows = [], []
    failures = []

    # ---- g rows: grouped by branching node (v1 first, then bottom-up CDs) --
    order_branching = [tree.root] + sorted(
        (v for v in tree.branching if v != tree.root))
    # full-range g table (independent of K): recompute with K = n
    g_val_full, g_nds_full = _build_g_tables(tree, tree.n)

    for vb in order_branching:
        for tip in tree.CD[vb]:
            chain = tree.chains[tip]
            beta = chain.beta
            cells = [(0, 0)]
            cells += [(0, k) for k in range(1, beta)]        # eps=0: k<=beta-1
            cells += [(1, k) for k in range(1, beta + 1)]    # eps=1: k<=beta
            for (eps, k) in cells:
                gv = g_val_full[tip][eps].get(k, NEG_INF)
                gn = set(g_nds_full[tip][eps].get(k, frozenset()))
                bv, _ = brute_g(tree, chain, eps, k)
                ok_val = abs(gv - bv) < 1e-9
                # witness check: simulate chain sum under DP witness
                wit_val = sum((uw if p in gn else w)
                              for (p, c, w, uw) in chain.edges)
                ok_wit = abs(wit_val - gv) < 1e-9 and \
                    ((chain.ca in gn) == bool(eps)) and len(gn) == k
                if not (ok_val and ok_wit):
                    failures.append(
                        f"g(chi_{vname(tip)},{eps},{k}): DP={gv} brute={bv} "
                        f"witness={vset_names(gn)} wit_val={wit_val}")
                g_rows.append({
                    "chain": vname(tip), "eps": eps, "k": k,
                    "g": gv, "V": sorted(vname(x) for x in gn),
                    "ok": ok_val and ok_wit,
                })

    # ---- f rows: per branching node (bottom-up), pieces T^{q:q} then T^{1:q}
    dp_order = sorted(tree.branching, key=lambda v: tree.layer[v],
                      reverse=True)
    for vb in dp_order:
        p = len(tree.CD[vb])
        pieces = [("qq", q) for q in range(1, p + 1)] + \
                 [("1q", q) for q in range(2, p + 1)]
        for (kind, q) in pieces:
            if kind == "qq":
                tips = [tree.CD[vb][q - 1]]
                label = f"{q}:{q}"
                table, ntable = tb["fqq"], tb["fqq_N"]
                key = lambda e, k: (vb, q, e, k)
            else:
                tips = tree.CD[vb][:q]
                label = f"1:{q}"
                table, ntable = tb["f1q"], tb["f1q_N"]
                key = lambda e, k: (vb, q, e, k)
            nodes = piece_nodes(tree, vb, tips)
            B = nonleaf_count(tree, nodes, vb)
            kmax = min(B, K)
            cells = [(0, k) for k in range(0, kmax + 1)] + \
                    [(1, k) for k in range(1, kmax + 1)]

            def cell(eps, k):
                """Definition 7/8 use at-most-k semantics: if the DP stores
                no exact-k entry (the piece cannot absorb the full budget at
                this eps), the value is that of the largest feasible smaller
                budget."""
                best_v, best_n = NEG_INF, set()
                kk = k
                while kk >= eps:
                    fv = table.get(key(eps, kk), NEG_INF)
                    if fv != NEG_INF:
                        if best_v == NEG_INF:
                            best_v, best_n = fv, set(
                                ntable.get(key(eps, kk), frozenset()))
                        elif abs(fv - best_v) < 1e-9:
                            # same value with smaller budget: prefer the
                            # smaller witness for readability
                            best_n = set(ntable.get(key(eps, kk), frozenset()))
                    kk -= 1
                return best_v, best_n

            # F values for star marking
            Fvals = {}
            for k in range(0, kmax + 1):
                Fvals[k] = max(cell(e, k)[0] for e in (0, 1) if e <= k)
            for (eps, k) in cells:
                fv, fn = cell(eps, k)
                bv, bS = brute_f(tree, vb, nodes, eps, k)
                ok_val = (fv != NEG_INF) and abs(fv - bv) < 1e-9
                # witness: |fn|<=k, eps-consistent, achieves fv
                ok_wit = (len(fn) <= k and ((vb in fn) == bool(eps)) and
                          fn <= {u for u in nodes if not tree.is_leaf[u]} and
                          abs(piece_value(tree, vb, nodes, fn) - fv) < 1e-9)
                if not (ok_val and ok_wit):
                    failures.append(
                        f"f(T^{{{label}}}_{vname(vb)},{eps},{k}): DP={fv} "
                        f"brute={bv} witness={vset_names(fn)}")
                f_rows.append({
                    "node": vname(vb), "piece": label, "eps": eps, "k": k,
                    "f": fv, "V": sorted(vname(x) for x in fn),
                    "star": abs(fv - Fvals[k]) < 1e-9,
                    "ok": ok_val and ok_wit,
                })

    return opt, S_opt, g_rows, f_rows, failures


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def _fmt_num(x):
    return f"{int(x)}" if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _fmt_set(names):
    if not names:
        return r"$\emptyset$"
    inner = ",".join(n.replace("v", "v_{") + "}" for n in names)
    return "$\\{" + inner + "\\}$"


def emit_latex(g_rows, f_rows):
    lines = []
    lines.append("% ------------------------------------------------------")
    lines.append("% AUTO-GENERATED by code/paper_tables.py -- do not edit.")
    lines.append("% Every value below is verified cell-by-cell against an")
    lines.append("% independent brute-force enumeration (see the script).")
    lines.append("% CD orderings: CD(v1)=(v2,v6,v7), CD(v2)=(v3,v4),")
    lines.append("%               CD(v7)=(v8,v10).")
    lines.append("% ------------------------------------------------------")
    lines.append("")
    # ---- tab-g ----
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Chain $g$-values for Example~\ref{ex-worked}.}")
    lines.append(r"\label{tab-g}")
    lines.append(r"\begin{tabular}{llrrl}")
    lines.append(r"\toprule")
    lines.append(r"$\chi$ & $\varepsilon$ & $k$ & $g$ & $V(\chi,\varepsilon,k)$ \\")
    lines.append(r"\midrule")
    prev = None
    for r in g_rows:
        name = r"$\chi_{v_{" + r["chain"][1:] + r"}}$" if prev != r["chain"] else ""
        prev = r["chain"]
        lines.append(f"{name:<18} & {r['eps']} & {r['k']} & "
                     f"{_fmt_num(r['g'])} & {_fmt_set(r['V'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    # ---- tab-f ----
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Subtree $f$-values for Example~\ref{ex-worked}. "
                 r"Entries marked $*$ attain "
                 r"$F(T^{\cdot\,:\,\cdot}_v,k)=\max_{\varepsilon}"
                 r"f(T^{\cdot\,:\,\cdot}_v,\varepsilon,k)$. "
                 r"Table entries give the at-most-$k$ optimum and a witness "
                 r"of cardinality at most $k$; a witness of strictly smaller "
                 r"cardinality (e.g.\ $\emptyset$) indicates that additional "
                 r"upgrades would not improve the objective at that state.}")
    lines.append(r"\label{tab-f}")
    lines.append(r"\begin{tabular}{llrrl}")
    lines.append(r"\toprule")
    lines.append(r"$T^{a:q}_v$ & $\varepsilon$ & $k$ & $f$ & $V(T)$ \\")
    lines.append(r"\midrule")
    prev = None
    for r in f_rows:
        tag = (r["node"], r["piece"])
        name = (r"$T^{" + r["piece"] + r"}_{v_{" + r["node"][1:] + r"}}$"
                if prev != tag else "")
        prev = tag
        star = "$*$" if r["star"] else "   "
        lines.append(f"{name:<20} & {r['eps']} & {r['k']} & "
                     f"{_fmt_num(r['f'])}{star} & {_fmt_set(r['V'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    tree = build_example1()
    opt, S_opt, g_rows, f_rows, failures = collect_cells(tree, K_EXAMPLE)

    print(f"g-cells verified: {sum(r['ok'] for r in g_rows)}/{len(g_rows)}")
    print(f"f-cells verified: {sum(r['ok'] for r in f_rows)}/{len(f_rows)}")
    print(f"Final: F(T_v1,{K_EXAMPLE}) = {_fmt_num(opt)}, "
          f"S = {vset_names(S_opt)}")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print("  " + msg)
        print("\nFAIL")
        return 1

    ok_final = abs(opt - 13) < 1e-9 and S_opt == {0}
    print("Final-value check vs hand computation (13, {v1}): "
          + ("PASS" if ok_final else "FAIL"))
    if not ok_final:
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "example": "Appendix worked example, K=1",
        "cd_orderings": {"v1": ["v2", "v6", "v7"], "v2": ["v3", "v4"],
                         "v7": ["v8", "v10"]},
        "final": {"F": opt, "S": sorted(vname(v) for v in S_opt)},
        "g_rows": g_rows, "f_rows": f_rows,
    }
    jpath = os.path.join(RESULTS_DIR, "example_tables.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"JSON written: {jpath}")

    tex = emit_latex(g_rows, f_rows)
    with open(TEX_OUT, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"LaTeX written: {os.path.normpath(TEX_OUT)}")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
