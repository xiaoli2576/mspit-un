"""
experiments.py  --  Random tree generator and timing harness for MSPIT-UNu.

Generates random trees of various sizes and measures the running time of
Algorithm 1 (MSPIT-UNu) and Algorithm 2 (MCSPIT-UNu).

Usage:
    python experiments.py

Output: tables with n, K/D, avg time over repetitions, and min/max time.
The default configuration matches the paper table:
  - MSPIT-UNu: n in {50,100,200,500,1000,2000}, 5 repetitions
  - MCSPIT-UNu: n in {50,100,200,500,1000,2000}, 3 repetitions
"""

import json
import os
import platform
import random
import sys
import time
from datetime import datetime

from tree import Tree
from mspit_un import solve_mspit_unu
from mcspit_un import solve_mcspit_unu


def random_tree(n, max_w=20, max_delta=10, seed=None):
    """
    Generate a random rooted tree with n nodes (root = 0).

    Each non-root node i picks a parent uniformly from {0, ..., i-1},
    giving a random recursive tree (Cayley model).

    Edge weights are assigned:
      w(e)  ~ Uniform(1, max_w)
      u(e)  = w(e) + Uniform(0, max_delta)
    """
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        w = rng.randint(1, max_w)
        uw = w + rng.randint(0, max_delta)
        adj[i].append((parent, w, uw))
        adj[parent].append((i, w, uw))
    return Tree(n=n, adj=adj, root=0)


def _empty_adj(n):
    return [[] for _ in range(n)]


def _add_deterministic_edge(adj, a, b, idx):
    """Deterministic weights used by the structured stress tests."""
    w = 1 + (idx % 20)
    uw = w + 1 + ((3 * idx) % 10)
    adj[a].append((b, w, uw))
    adj[b].append((a, w, uw))


def path_tree(n):
    """Rooted path on n nodes."""
    adj = _empty_adj(n)
    for i in range(n - 1):
        _add_deterministic_edge(adj, i, i + 1, i + 1)
    return Tree(n=n, adj=adj, root=0)


def star_tree(n):
    """Star with root at the center."""
    adj = _empty_adj(n)
    for i in range(1, n):
        _add_deterministic_edge(adj, 0, i, i)
    return Tree(n=n, adj=adj, root=0)


def comb_tree(n):
    """A long spine with one leaf attached to each spine node when possible."""
    adj = _empty_adj(n)
    spine_len = (n - 1) // 2
    idx = 1
    for i in range(spine_len):
        _add_deterministic_edge(adj, i, i + 1, idx)
        idx += 1
    next_node = spine_len + 1
    for i in range(1, spine_len + 1):
        if next_node < n:
            _add_deterministic_edge(adj, i, next_node, idx)
            idx += 1
            next_node += 1
    while next_node < n:
        _add_deterministic_edge(adj, spine_len, next_node, idx)
        idx += 1
        next_node += 1
    return Tree(n=n, adj=adj, root=0)


def layered_tree(n, branching=4):
    """Truncated complete branching-ary tree with root 0."""
    adj = _empty_adj(n)
    for child in range(1, n):
        parent = (child - 1) // branching
        _add_deterministic_edge(adj, parent, child, child)
    return Tree(n=n, adj=adj, root=0)


def time_mspit(n_list, K_fraction=0.3, reps=5, seed_base=42):
    """
    Benchmark solve_mspit_unu on trees of various sizes.

    Parameters
    ----------
    n_list     : list of tree sizes to benchmark
    K_fraction : budget K as a fraction of n
    reps       : number of random trees per size
    seed_base  : base seed for reproducibility

    Returns
    -------
    results : list of (n, K, avg_time, min_time, max_time)
    """
    results = []
    for n in n_list:
        K = max(1, int(n * K_fraction))
        times = []
        for rep in range(reps):
            t = random_tree(n, seed=seed_base + n * 1000 + rep)
            t0 = time.perf_counter()
            val, S = solve_mspit_unu(t, K)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        results.append((n, K, avg, min(times), max(times)))
    return results


def time_mcspit(n_list, D_fraction=0.7, reps=5, seed_base=42):
    """
    Benchmark solve_mcspit_unu on trees of various sizes.

    Parameters
    ----------
    n_list     : list of tree sizes to benchmark
    D_fraction : D as a fraction of the max achievable path length
    reps       : number of random trees per size
    seed_base  : base seed for reproducibility

    Returns
    -------
    results : list of (n, D, avg_time, min_time, max_time)
    """
    results = []
    for n in n_list:
        times = []
        for rep in range(reps):
            t = random_tree(n, seed=seed_base + n * 1000 + rep)
            # Use a feasible threshold tied to the all-upgraded optimum
            # shortest root-leaf distance.
            max_shortest_val, _ = solve_mspit_unu(t, t.n)
            D = max_shortest_val * D_fraction

            t0 = time.perf_counter()
            K_star, val, S = solve_mcspit_unu(t, D)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        results.append((n, D_fraction, avg, min(times), max(times)))
    return results


def time_structured_mspit(n_list, K_fraction=0.3):
    """Benchmark Algorithm 1 on deterministic path/comb/star/layered trees."""
    families = [
        ("Path", path_tree),
        ("Comb", comb_tree),
        ("Star", star_tree),
        ("Layered 4-ary", lambda n: layered_tree(n, branching=4)),
    ]
    results = []
    for family_name, factory in families:
        row = []
        for n in n_list:
            K = max(1, int(n * K_fraction + 0.999999))
            tree = factory(n)
            t0 = time.perf_counter()
            val, S = solve_mspit_unu(tree, K)
            t1 = time.perf_counter()
            row.append((n, K, t1 - t0, val, len(S)))
        results.append((family_name, row))
    return results


def print_results_table(results, header):
    """Print timing results as a formatted table."""
    print("\n" + "="*70)
    print(header)
    print("="*70)
    print(f"{'n':>8}  {'K/D':>8}  {'avg(s)':>10}  {'min(s)':>10}  {'max(s)':>10}")
    print("-"*70)
    for row in results:
        n, kd, avg, mn, mx = row
        if isinstance(kd, float) and kd < 1:
            kd_str = f"D={kd:.0%}"
        else:
            kd_str = f"K={kd}"
        print(f"{n:>8}  {kd_str:>8}  {avg:>10.4f}  {mn:>10.4f}  {mx:>10.4f}")
    print()


def print_structured_table(results):
    """Print structured-family stress-test timings."""
    sizes = [n for n, *_ in results[0][1]]
    print("\n" + "="*70)
    print("Structured MSPIT-UNu stress tests")
    print("="*70)
    print(f"{'family':>14}  " + "  ".join(f"n={n:>4}" for n in sizes))
    print("-"*70)
    for family, row in results:
        times = "  ".join(f"{dt:>8.4f}" for _, _, dt, _, _ in row)
        print(f"{family:>14}  {times}")
    print()


def main():
    print("MSPIT-UNu / MCSPIT-UNu Performance Experiments")
    print("="*70)
    print(f"platform: {platform.platform()}, "
          f"python: {platform.python_version()}, "
          f"cpu: {platform.processor()}")
    print(f"started: {datetime.now().isoformat(timespec='seconds')}")

    # Tree sizes to benchmark
    sizes_mspit = [50, 100, 200, 500, 1000, 2000]
    sizes_mcspit = [50, 100, 200, 500, 1000, 2000]

    print("\nBenchmark 1: MSPIT-UNu (Algorithm 1), K = 30% of n")
    r1 = time_mspit(sizes_mspit, K_fraction=0.3, reps=5)
    print_results_table(r1, "MSPIT-UNu timings")

    print("\nBenchmark 2: MCSPIT-UNu (Algorithm 2), D = 70% of all-upgraded optimum")
    r2 = time_mcspit(sizes_mcspit, D_fraction=0.7, reps=3)
    print_results_table(r2, "MCSPIT-UNu timings")

    print("\nBenchmark 3: Structured stress tests for MSPIT-UNu, K = 30% of n")
    r3 = time_structured_mspit([100, 200, 500], K_fraction=0.3)
    print_structured_table(r3)

    # ---- archive raw results as JSON (canonical source for paper tables) --
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "config": {
            "mspit": {"sizes": sizes_mspit, "K_fraction": 0.3, "reps": 5,
                      "seed": "42 + 1000*n + rep"},
            "mcspit": {"sizes": sizes_mcspit, "D_fraction": 0.7, "reps": 3,
                       "seed": "42 + 1000*n + rep"},
            "structured": {"sizes": [100, 200, 500], "K_fraction": 0.3,
                           "weights": "w=1+(j mod 20), u=w+1+(3j mod 10)"},
        },
        "mspit": [{"n": n, "K": K, "avg": avg, "min": mn, "max": mx}
                  for (n, K, avg, mn, mx) in r1],
        "mcspit": [{"n": n, "D_fraction": d, "avg": avg, "min": mn, "max": mx}
                   for (n, d, avg, mn, mx) in r2],
        "structured": [{"family": fam,
                        "rows": [{"n": n, "K": K, "time": dt, "value": val,
                                  "S_size": s}
                                 for (n, K, dt, val, s) in row]}
                       for (fam, row) in r3],
    }
    jpath = os.path.join(results_dir, f"experiments-{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nJSON archived: {jpath}")


if __name__ == "__main__":
    main()
