"""
bench_edge.py -- edge-model comparison, budget doubling for MCSPIT-UNu, and
brute-force validation (v1.2.0 manuscript, Section 5).

1. Validation on small random trees (n <= 12), every budget K:
     node model:  Algorithm 1 == brute force
     edge model:  Lei et al. DP == Algorithm 1 on the
                  subdivided tree of Proposition 2 == brute force
     MCSPIT-UNu:  doubling == one-run == brute-force K*
2. Timing of the unit-Hamming edge problem: the O(NK^2) DP of Lei et al.
   (Optim. Lett. 17:453-469, 2023, eqs. (4)-(5), every split k1+k2=k scanned)
   and Algorithm 1 on the tree of Proposition 2.
3. Timing of MCSPIT-UNu by doubling vs the one-run solution, on the
   random-recursive instances of the T2 experiment.

Writes results/edge-bench-20261017.json and prints a log.
"""

import json
import math
import os
import platform
import random
import sys
import time
from itertools import combinations

from tree import Tree
from brute import brute_force
from mspit_un import solve_mspit_unu
from mcspit_un import solve_mcspit_unu_oneshot, solve_mcspit_unu_doubling
from experiments import random_tree, comb_tree, path_tree, layered_tree

NEG_INF = float('-inf')
CAP = 90.0
OUT = os.path.join(os.path.dirname(__file__), 'results', 'edge-bench-20261017.json')


def log(*a):
    print(*a, flush=True)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def best_of(reps, fn, *a, **kw):
    """Minimum wall time over `reps` runs (the result of the last run)."""
    best = math.inf
    for _ in range(reps):
        dt, out = timed(fn, *a, **kw)
        best = min(best, dt)
    return best, out


def postorder(tree):
    order, stack = [], [tree.root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(c for c, _, _ in tree.children[v])
    return order[::-1]


def lei_edge_dp(tree, K):
    """Max-min DP of Lei et al. for the unit-Hamming edge problem.

    Branch table:  b(k) = max{w(e) + F_c(k), u(e) + F_c(k-1)}       (eq. 5)
    Merge:         f(k) = max_{k1+k2=k} min{f_prev(k1), b(k2)}       (eq. 4)
    Every split with k <= K is scanned (O(NK^2), as in the paper).
    """
    F = {}
    for v in postorder(tree):
        ch = tree.children[v]
        if not ch:
            F[v] = [0.0] * (K + 1)
            continue
        acc = None
        for c, w, u in ch:
            Fc = F.pop(c)
            br = [0.0] * (K + 1)
            for k in range(K + 1):
                a = w + Fc[k]
                if k >= 1:
                    b = u + Fc[k - 1]
                    if b > a:
                        a = b
                br[k] = a
            if acc is None:
                acc = br
                continue
            res = [NEG_INF] * (K + 1)
            for k in range(K + 1):
                best = NEG_INF
                for k1 in range(k + 1):
                    a1 = acc[k1]
                    b2 = br[k - k1]
                    val = a1 if a1 < b2 else b2
                    if val > best:
                        best = val
                res[k] = best
            acc = res
        F[v] = acc
    return F[tree.root][K]


def subdivide(tree):
    """Proposition 2: put a node z_e on every edge e=(p,c), with
    w'(p,z_e)=u'(p,z_e)=w(e), w'(z_e,c)=0 and u'(z_e,c)=u(e)-w(e)."""
    n = tree.n
    m = n + (n - 1)
    adj = [[] for _ in range(m)]
    z = n
    for p in range(n):
        for c, w, u in tree.children[p]:
            adj[p].append((z, w, w)); adj[z].append((p, w, w))
            adj[z].append((c, 0, u - w)); adj[c].append((z, 0, u - w))
            z += 1
    return Tree(n=m, adj=adj, root=tree.root)


def edge_brute(tree, K):
    edges = [(p, c) for p in range(tree.n) for c, _, _ in tree.children[p]]
    wt = {(p, c): (w, u) for p in range(tree.n) for c, w, u in tree.children[p]}
    leaves = [v for v in range(tree.n) if tree.is_leaf[v]]
    best = NEG_INF
    for k in range(min(K, len(edges)) + 1):
        for F in combinations(edges, k):
            Fs = set(F)
            val = math.inf
            for t in leaves:
                d, cur = 0, t
                while tree.parent[cur] is not None:
                    p = tree.parent[cur][0]
                    w, u = wt[(p, cur)]
                    d += u if (p, cur) in Fs else w
                    cur = p
                val = min(val, d)
            best = max(best, val)
    return best


def validate(n_trees=300, seed=7):
    rng = random.Random(seed)
    cnt = {'trees': 0, 'node_cases': 0, 'edge_cases': 0, 'mc_cases': 0}
    for i in range(n_trees):
        n = rng.randint(2, 12)
        t = random_tree(n, seed=rng.randrange(10 ** 9))
        ts = subdivide(t)
        node_vals = []
        for K in range(n + 1):
            b, _ = brute_force(t, K)
            a, _ = solve_mspit_unu(t, K)
            assert abs(a - b) < 1e-9, ('node', i, K, a, b)
            node_vals.append(b)
            cnt['node_cases'] += 1
            eb = edge_brute(t, K)
            e1 = lei_edge_dp(t, K)
            e3, _ = solve_mspit_unu(ts, K)
            assert max(abs(x - eb) for x in (e1, e3)) < 1e-9, ('edge', i, K, eb, e1, e3)
            cnt['edge_cases'] += 1
        lo, hi = node_vals[0], node_vals[-1]
        for D in (lo, (lo + hi) / 2, hi, hi + 1, rng.uniform(lo, hi)):
            exp = next((k for k, v in enumerate(node_vals) if v >= D - 1e-12), None)
            k1 = solve_mcspit_unu_oneshot(t, D)[0]
            k2 = solve_mcspit_unu_doubling(t, D)[0]
            assert k1 == exp and k2 == exp, ('mc', i, D, exp, k1, k2)
            cnt['mc_cases'] += 1
        cnt['trees'] += 1
    return cnt


def edge_timing():
    inst = [('Random', random_tree(1000, seed=42 + 1000 * 1000)),
            ('Comb', comb_tree(1000))]
    rows = []
    for name, t in inst:
        ts = subdivide(t)
        dead = False
        for K in (10, 30, 100, 300):
            r = {'family': name, 'n': t.n, 'K': K}
            if not dead:
                r['lei'], v1 = best_of(3, lei_edge_dp, t, K)
                dead = r['lei'] > CAP / 3
            else:
                v1 = None
            r['alg1'], (v3, _) = best_of(3, solve_mspit_unu, ts, K)
            assert v1 is None or abs(v1 - v3) < 1e-9
            r['val'] = v3
            rows.append(r)
            log(f"edge {name} K={K}: lei {r.get('lei')} alg1 {r['alg1']:.4f}")
    return rows


def mc_timing():
    rows = []
    for n in (50, 100, 200, 500, 1000, 2000):
        ts_d, ts_o, ks = [], [], []
        for r in range(3):
            t = random_tree(n, seed=42 + 1000 * n + r)
            Fn, _ = solve_mspit_unu(t, t.n)
            D = 0.7 * Fn
            dt_d, out_d = timed(solve_mcspit_unu_doubling, t, D)
            dt_o, out_o = timed(solve_mcspit_unu_oneshot, t, D)
            assert out_d[0] == out_o[0]
            ts_d.append(dt_d); ts_o.append(dt_o); ks.append(out_d[0])
        rows.append({'n': n, 'Kstar': ks,
                     'doubling': {'avg': sum(ts_d) / 3, 'min': min(ts_d), 'max': max(ts_d)},
                     'oneshot': {'avg': sum(ts_o) / 3, 'min': min(ts_o), 'max': max(ts_o)}})
        log(f"mc n={n} K*={ks} doubling {sum(ts_d)/3:.4f} oneshot {sum(ts_o)/3:.4f}")
    return rows


def t1_timing():
    rows = []
    for n in (50, 100, 200, 500, 1000, 2000):
        K = math.ceil(0.3 * n)
        ts = []
        for r in range(5):
            t = random_tree(n, seed=42 + 1000 * n + r)
            dt, _ = timed(solve_mspit_unu, t, K)
            ts.append(dt)
        rows.append({'n': n, 'K': K, 'avg': sum(ts) / 5, 'min': min(ts), 'max': max(ts)})
        log(f"T1 n={n}: avg {sum(ts)/5:.4f}")
    return rows


def structured_timing():
    out = {}
    for name, fac in (('Path', path_tree), ('Comb', comb_tree),
                      ('Layered 4-ary', lambda n: layered_tree(n, 4))):
        out[name] = []
        for n in (100, 200, 500, 1000, 2000):
            t = fac(n)
            dt, _ = best_of(3, solve_mspit_unu, t, math.ceil(0.3 * n))
            out[name].append({'n': n, 't': dt})
            log(f"S {name} n={n}: {dt:.4f}")
    return out


def main():
    res = {'meta': {'date': '2026-10-17', 'python': sys.version,
                    'platform': platform.platform(), 'processor': platform.processor()}}
    res['validation'] = validate()
    log('validation', res['validation'])
    res['T1'] = t1_timing()
    res['mc'] = mc_timing()
    res['structured'] = structured_timing()
    res['edge'] = edge_timing()
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    log('written', OUT)


if __name__ == '__main__':
    main()
