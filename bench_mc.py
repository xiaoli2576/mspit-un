"""
bench_mc.py -- MCSPIT-UNu timing with a controlled threshold (revision of
Section 5, 2026-10-17).

The threshold is D = F(T,0) + alpha (F(T,n) - F(T,0)), alpha in (0, 1], so
every instance needs at least one upgrade and alpha sets how much of the
attainable increase is demanded.  Three families:
  Random       random-recursive tree of Section 5 (seed 42 + 1000 n + r)
  Binary-rand  complete binary tree, w ~ U{1..20}, u = w + U{0..10}
  Binary-unit  complete binary tree, w = 1, u = 2; at alpha = 1 every
               non-leaf node must be upgraded, so K* = (n - 1) / 2
For each instance the doubling solution (Theorem 14) and the one-run
solution (K = n) are timed (median of REPS runs, including the witness run),
and the returned set S is checked on the original tree: |S| = K*, the
recomputed shortest root-leaf distance equals the DP value and is >= D, and
the DP value at K* - 1 is < D.

Writes results/mc-bench-20261017.json and prints a log.
"""

import json
import os
import platform
import random
import statistics
import sys
import time

from tree import Tree
from experiments import random_tree
from mspit_un import root_profile
from mcspit_un import solve_mcspit_unu_oneshot, solve_mcspit_unu_doubling

REPS = 5
ALPHAS = (0.25, 0.5, 0.75, 1.0)
SIZES = (63, 127, 255, 511, 1023, 2047)
OUT = os.path.join(os.path.dirname(__file__), 'results', 'mc-bench-20261017.json')


def log(*a):
    print(*a, flush=True)


def median_time(reps, fn, *a):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*a)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts), out


def binary_tree(n, seed=None, unit=False):
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        p = (i - 1) // 2
        if unit:
            w, u = 1, 2
        else:
            w = rng.randint(1, 20)
            u = w + rng.randint(0, 10)
        adj[i].append((p, w, u))
        adj[p].append((i, w, u))
    return Tree(n=n, adj=adj, root=0)


def evaluate(tree, S):
    """Shortest root-leaf distance on the original tree when every edge
    leaving a node of S is raised to u."""
    best, stack = float('inf'), [(tree.root, 0)]
    while stack:
        v, d = stack.pop()
        if tree.is_leaf[v]:
            best = min(best, d)
            continue
        for c, w, u in tree.children[v]:
            stack.append((c, d + (u if v in S else w)))
    return best


FAMILIES = (
    ('Random', lambda n, r: random_tree(n, seed=42 + 1000 * n + r), 3),
    ('Binary-rand', lambda n, r: binary_tree(n, seed=42 + 1000 * n + r), 3),
    ('Binary-unit', lambda n, r: binary_tree(n, unit=True), 1),
)


def main():
    rows = []
    for name, make, trials in FAMILIES:
        for n in SIZES:
            for r in range(trials):
                t = make(n, r)
                prof = root_profile(t, t.n)
                F0, Fn = prof[0], prof[-1]
                for alpha in ALPHAS:
                    D = F0 + alpha * (Fn - F0)
                    td, td_lo, td_hi, (kd, vd, Sd) = median_time(REPS, solve_mcspit_unu_doubling, t, D)
                    to, to_lo, to_hi, (ko, vo, So) = median_time(REPS, solve_mcspit_unu_oneshot, t, D)
                    assert kd == ko and abs(vd - vo) < 1e-9, (name, n, r, alpha, kd, ko)
                    S = set(Sd)
                    assert len(S) <= kd and all(not t.is_leaf[v] for v in S), (name, n, r, alpha)
                    ev = evaluate(t, S)
                    assert abs(ev - vd) < 1e-9 and ev >= D - 1e-9, (name, n, r, alpha, ev, vd, D)
                    assert kd == 0 or prof[kd - 1] < D - 1e-12, (name, n, r, alpha)
                    rows.append({'family': name, 'n': n, 'trial': r, 'alpha': alpha,
                                 'F0': F0, 'Fn': Fn, 'D': D, 'Kstar': kd, 'set_size': len(S),
                                 'doubling': {'median': td, 'min': td_lo, 'max': td_hi},
                                 'oneshot': {'median': to, 'min': to_lo, 'max': to_hi}})
                    log(f"{name} n={n} r={r} a={alpha}: K*={kd} doubling {td:.4f} oneshot {to:.4f}")
    res = {'meta': {'date': '2026-10-17', 'reps': REPS, 'timing': 'median wall time, witness run included',
                    'python': sys.version, 'platform': platform.platform(),
                    'processor': platform.processor()},
           'rows': rows}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    log('written', OUT)


if __name__ == '__main__':
    main()
