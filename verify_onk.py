"""
verify_onk.py -- correctness check for the O(nK) implementation (2026-10-16).

  1. >=1000 random small trees (random-recursive + subdivided "chainy" trees,
     random weights incl. Dw=0 ties, K uniform in [0, n+1]):
     methods v1 / kk / pairs must all equal brute force, and every returned
     set S must satisfy |S|<=K and min_path_length(S) == value.
  2. Cell-by-cell agreement of the DP tables (fqq, f1q) of 'pairs' and 'kk'
     against v1 on 200 medium trees (n in [12, 60]).
Usage: python verify_onk.py [n_small_trials]
"""
import random
import sys

from tree import Tree
from brute import brute_force, min_path_length
from mspit_un import solve_mspit_unu, NEG_INF


def rand_tree(n, rng, chainy):
    if chainy:
        # random recursive skeleton, then subdivide edges to create chains
        m = max(2, n // 2)
        par = [None] + [rng.randint(0, i - 1) for i in range(1, m)]
        edges = [(par[i], i) for i in range(1, m)]
        nxt = m
        out = []
        for (a, b) in edges:
            cur = a
            while nxt < n and rng.random() < 0.5:
                out.append((cur, nxt)); cur = nxt; nxt += 1
            out.append((cur, b))
        while nxt < n:                      # leftover nodes as leaves
            out.append((rng.randint(0, nxt - 1), nxt)); nxt += 1
        edges = out
    else:
        edges = [(rng.randint(0, i - 1), i) for i in range(1, n)]
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        w = rng.randint(1, 9)
        uw = w + rng.choice([0, 0, 1, 2, 3, 5, 8])
        adj[a].append((b, w, uw)); adj[b].append((a, w, uw))
    return Tree(n=n, adj=adj, root=0)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    rng = random.Random(20261016)
    fails = {m: 0 for m in ('v1', 'kk', 'pairs')}
    setfails = {m: 0 for m in ('v1', 'kk', 'pairs')}
    for t in range(trials):
        n = rng.randint(2, 12)
        tr = rand_tree(n, rng, chainy=(t % 2 == 1))
        K = rng.randint(0, n + 1)
        bf, _ = brute_force(tr, min(K, n))
        for m in fails:
            val, S = solve_mspit_unu(tr, K, method=m)
            if abs(val - bf) > 1e-9:
                fails[m] += 1
                if fails[m] <= 3:
                    print(f"VALUE FAIL {m} t={t} n={n} K={K} dp={val} bf={bf}")
            if len(S) > max(K, 0) or abs(min_path_length(tr, S) - val) > 1e-9:
                setfails[m] += 1
                if setfails[m] <= 3:
                    print(f"SET FAIL {m} t={t} n={n} K={K} S={S}")
    print(f"[brute] trials={trials}  value-fails={fails}  set-fails={setfails}")

    # ---- cell-by-cell tables vs v1 ----
    cell_fail = 0; cells = 0
    for t in range(200):
        n = rng.randint(12, 60)
        tr = rand_tree(n, rng, chainy=(t % 2 == 1))
        K = rng.randint(1, n)
        o1, _, T1 = solve_mspit_unu(tr, K, return_tables=True, method='v1')
        for m in ('kk', 'pairs'):
            o2, S2, T2 = solve_mspit_unu(tr, K, return_tables=True, method=m)
            if abs(o1 - o2) > 1e-9:
                cell_fail += 1
            for name in ('fqq', 'f1q'):
                keys = set(T1[name]) | set(T2[name])
                for key in keys:
                    a = T1[name].get(key, NEG_INF); b = T2[name].get(key, NEG_INF)
                    cells += 1
                    if not (a == b or abs(a - b) < 1e-9):
                        cell_fail += 1
                        if cell_fail <= 5:
                            print(f"CELL {m} {name}{key}: v1={a} new={b}")
                    # witness check for new tables
                    if b != NEG_INF and name == 'f1q':
                        Sx = T2[name + '_N'][key]
                        if len(Sx) > key[3] or ((key[0] in Sx) != bool(key[2])):
                            cell_fail += 1
    print(f"[tables] trees=200 cells compared={cells} mismatches={cell_fail}")
    ok = not any(fails.values()) and not any(setfails.values()) and cell_fail == 0
    print("ALL OK" if ok else "FAILURES PRESENT")


if __name__ == '__main__':
    main()
