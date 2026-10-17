"""
bench_onk.py -- old (v1) vs paper-loop (kk, O(nK^2)) vs size-capped pairs
(pairs, O(nK)) benchmark, 2026-10-16.  Writes results/onk-bench-20261016.json.

  T1  : random-recursive trees, n in {50,...,2000}, K=ceil(0.3n), 5 trials,
        seed 42+1000n+r  (paper Table tab-experiments, Alg. 1 row)
  T2  : MCSPIT bisection, D=0.7 F(T,n), 3 trials (paper Table, Alg. 2 row)
        + one-shot MCSPIT (root profile from one run at K=n)
  S   : structured families path/comb/star/layered-4 (+ comb-rev: same comb
        with the leaf listed before the spine child in CD(v)), K=ceil(0.3n)
  KS  : K-scaling, fixed n, K in {0.05n,0.1n,0.3n,0.6n,n}
A method is skipped for the rest of a sweep once one run exceeds CAP seconds.
"""
import json
import math
import os
import platform
import sys
import time

import mspit_un
from mspit_un import solve_mspit_unu
from mcspit_un import solve_mcspit_unu, solve_mcspit_unu_oneshot
from experiments import (random_tree, path_tree, comb_tree, star_tree,
                         layered_tree, _empty_adj, _add_deterministic_edge)
from tree import Tree

CAP = 90.0
METHODS = ('v1', 'kk', 'pairs')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   'onk-bench-20261016.json')


def comb_rev_tree(n):
    """Comb (spine + one leaf per spine node); spine node s = 2i, its leaf
    s+1 < next spine node s+2, so CD(s) lists the leaf first (q=1) and the
    large spine branch second (q=2): the worst ordering for the 'kk' loop."""
    adj = _empty_adj(n)
    idx = 1
    for s in range(0, n, 2):
        if s + 1 < n:
            _add_deterministic_edge(adj, s, s + 1, idx); idx += 1
        if s + 2 < n:
            _add_deterministic_edge(adj, s, s + 2, idx); idx += 1
    return Tree(n=n, adj=adj, root=0)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return time.perf_counter() - t0, r


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den


def log(*a):
    print(*a, flush=True)


def main():
    res = {'meta': {'date': '2026-10-16', 'python': sys.version,
                    'platform': platform.platform(),
                    'processor': platform.processor(), 'cap_s': CAP}}
    sizes = [50, 100, 200, 500, 1000, 2000]

    # ---------------- T1 ----------------
    T1 = {m: [] for m in METHODS}
    dead = set()
    for n in sizes:
        K = math.ceil(0.3 * n)
        for m in METHODS:
            if m in dead:
                continue
            ts = []; vals = []
            for r in range(5):
                t = random_tree(n, seed=42 + 1000 * n + r)
                dt, (val, S) = timed(solve_mspit_unu, t, K, method=m)
                ts.append(dt); vals.append(val)
            T1[m].append({'n': n, 'K': K, 'avg': sum(ts) / 5, 'min': min(ts),
                          'max': max(ts), 'vals': vals})
            log(f"T1 n={n} {m}: avg {sum(ts)/5:.4f}s")
            if max(ts) > CAP:
                dead.add(m)
    res['T1'] = T1
    # value agreement
    for i in range(len(T1['pairs'])):
        for m in ('v1', 'kk'):
            if i < len(T1[m]):
                assert all(abs(a - b) < 1e-9 for a, b in
                           zip(T1[m][i]['vals'], T1['pairs'][i]['vals']))

    # ---------------- T2 (MCSPIT) ----------------
    T2 = {m: [] for m in METHODS + ('oneshot',)}
    dead = set()
    for n in sizes:
        trees = []
        for r in range(3):
            t = random_tree(n, seed=42 + 1000 * n + r)
            Fn, _ = solve_mspit_unu(t, t.n, method='pairs')
            trees.append((t, 0.7 * Fn))
        for m in METHODS + ('oneshot',):
            if m in dead:
                continue
            ts = []; ks = []
            for (t, D) in trees:
                if m == 'oneshot':
                    dt, out = timed(solve_mcspit_unu_oneshot, t, D)
                else:
                    mspit_un.DEFAULT_METHOD = m
                    dt, out = timed(solve_mcspit_unu, t, D)
                    mspit_un.DEFAULT_METHOD = 'pairs'
                ts.append(dt); ks.append(out[0])
            T2[m].append({'n': n, 'avg': sum(ts) / 3, 'min': min(ts),
                          'max': max(ts), 'Kstar': ks})
            log(f"T2 n={n} {m}: avg {sum(ts)/3:.4f}s K*={ks}")
            if max(ts) > CAP:
                dead.add(m)
    res['T2'] = T2

    # ---------------- Structured ----------------
    fams = [('Path', path_tree), ('Comb', comb_tree), ('Comb-rev', comb_rev_tree),
            ('Star', star_tree), ('Layered 4-ary', lambda n: layered_tree(n, 4))]
    S = {}
    for name, fac in fams:
        S[name] = {m: [] for m in METHODS}
        dead = set()
        for n in [100, 200, 500, 1000, 2000]:
            K = math.ceil(0.3 * n)
            t = fac(n)
            vals = {}
            for m in METHODS:
                if m in dead:
                    continue
                dt, (val, Sset) = timed(solve_mspit_unu, t, K, method=m)
                vals[m] = val
                S[name][m].append({'n': n, 'K': K, 't': dt, 'val': val})
                log(f"S {name} n={n} {m}: {dt:.4f}s")
                if dt > CAP / 3:
                    dead.add(m)
            assert len({round(v, 9) for v in vals.values()}) == 1, vals
    res['structured'] = S

    # ---------------- K scaling ----------------
    KS = {}
    for label, fac, n in [('Comb-rev n=1000', comb_rev_tree, 1000),
                          ('Comb n=1000', comb_tree, 1000),
                          ('Random n=2000 (seed 42+1000n)',
                           lambda n: random_tree(n, seed=42 + 1000 * n), 2000)]:
        t = fac(n)
        KS[label] = {m: [] for m in METHODS}
        dead = set()
        for frac in [0.05, 0.1, 0.3, 0.6, 1.0]:
            K = max(1, math.ceil(frac * n))
            for m in METHODS:
                if m in dead:
                    continue
                dt, (val, _) = timed(solve_mspit_unu, t, K, method=m)
                KS[label][m].append({'K': K, 't': dt, 'val': val})
                log(f"KS {label} K={K} {m}: {dt:.4f}s")
                if dt > CAP / 3:
                    dead.add(m)
    res['Kscaling'] = KS

    # ---------------- slopes ----------------
    sl = {}
    for m in METHODS:
        rows = [r for r in T1[m] if r['n'] >= 200]
        sl[f'T1 {m} (n>=200)'] = slope([r['n'] for r in rows], [r['avg'] for r in rows])
    for m in METHODS + ('oneshot',):
        rows = [r for r in T2[m] if r['n'] >= 200]
        sl[f'T2 {m} (n>=200)'] = slope([r['n'] for r in rows], [r['avg'] for r in rows])
    for name in S:
        for m in METHODS:
            rows = S[name][m]
            sl[f'S {name} {m}'] = slope([r['n'] for r in rows], [r['t'] for r in rows])
    for label in KS:
        for m in METHODS:
            rows = KS[label][m]
            sl[f'KS {label} {m} (vs K)'] = slope([r['K'] for r in rows], [r['t'] for r in rows])
    res['slopes'] = sl
    for k, v in sl.items():
        log(f"slope {k}: {v if v is None else round(v, 2)}")
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    log("written", OUT)


if __name__ == '__main__':
    main()
