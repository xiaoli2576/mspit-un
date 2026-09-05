"""
test_basic.py  --  Unit tests for MSPIT-UNu / MCSPIT-UNu implementation.

Tests cover:
  1. g-function values on hand-checked chains
  2. Small tree correctness (path, star, bamboo)
  3. Edge cases: K=0, K=n, single leaf, path graph
  4. Brute-force cross-checks on random trees
  5. MCSPIT-UNu correctness

Run with:
    python -m pytest test_basic.py -v
  or
    python test_basic.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(__file__))

from tree import Tree
from mspit_un import solve_mspit_unu, _build_g_tables
from mcspit_un import solve_mcspit_unu
from brute import brute_force


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_path_tree(weights, upgraded_weights):
    """
    Build a path v1 -> v2 -> ... -> v_{n} where n = len(weights)+1.
    weights[i] = w(e_{i+1}), upgraded_weights[i] = u(e_{i+1}).
    Root = node 0, single leaf = node n-1.
    """
    n = len(weights) + 1
    adj = [[] for _ in range(n)]
    for i in range(n - 1):
        w = weights[i]
        uw = upgraded_weights[i]
        adj[i].append((i + 1, w, uw))
        adj[i + 1].append((i, w, uw))
    return Tree(n=n, adj=adj, root=0)


def make_star_tree(n, w_list, uw_list):
    """
    Build a star with root 0 and n-1 leaves.
    w_list[i] = weight of edge (0, i+1).
    """
    assert len(w_list) == n - 1 and len(uw_list) == n - 1
    adj = [[] for _ in range(n)]
    for i in range(n - 1):
        adj[0].append((i + 1, w_list[i], uw_list[i]))
        adj[i + 1].append((0, w_list[i], uw_list[i]))
    return Tree(n=n, adj=adj, root=0)


def random_tree(n, seed=None):
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        w = rng.randint(1, 10)
        uw = rng.randint(w, w + 10)
        adj[i].append((parent, w, uw))
        adj[parent].append((i, w, uw))
    return Tree(n=n, adj=adj, root=0)


# ---------------------------------------------------------------------------
# Tests: g-function
# ---------------------------------------------------------------------------

def test_g_single_edge_chain():
    """g for a beta=1 chain (one edge): only eps=0,k=0 and eps=1,k=1."""
    # Single-leaf path: v1 -> v2 (leaf), w=3, u=8
    t = make_path_tree([3], [8])
    g_val, g_nds = _build_g_tables(t, K=1)
    vh = 1  # only chain: chi_v2 (tip=1, CA=0)
    assert vh in g_val, "Chain for v2 must exist"
    # eps=0, k=0
    assert abs(g_val[vh][0][0] - 3) < 1e-9, f"g(0,0) expected 3, got {g_val[vh][0][0]}"
    assert g_nds[vh][0][0] == frozenset(), "g_nds(0,0) must be empty"
    # eps=1, k=1
    assert abs(g_val[vh][1][1] - 8) < 1e-9, f"g(1,1) expected 8, got {g_val[vh][1][1]}"
    assert g_nds[vh][1][1] == frozenset({0}), f"g_nds(1,1) must be {{CA=0}}"
    # eps=0, k=1 must NOT exist (beta=1)
    assert 1 not in g_val[vh][0], "g(0,1) must not exist for beta=1 chain"
    print("PASS: test_g_single_edge_chain")


def test_g_two_edge_chain_ca_lower_dw():
    """
    For a 2-edge chain where CA's edge has LOWER Dw than the intermediate edge.
    v1 -> v2 -> v3 (leaf).
    e1=(v1->v2): w=1, u=6, Dw=5
    e2=(v2->v3): w=1, u=11, Dw=10

    Chain: CA=v1, intermediate=v2, tip=v3.
    After Lemma 1 fix: CA's edge is FIXED at delta[0]=5;
    intermediate edge delta[1]=10.

    g(eps=0, k=0) = 2
    g(eps=0, k=1) = 2 + 10 = 12   (upgrade v2 = intermediate)
    g(eps=1, k=1) = 2 + 5  = 7    (upgrade v1 = CA)
    g(eps=1, k=2) = 2 + 5 + 10 = 17
    """
    # 3-node path: v1->v2->v3
    adj = [[] for _ in range(3)]
    adj[0].append((1, 1, 6)); adj[1].append((0, 1, 6))
    adj[1].append((2, 1, 11)); adj[2].append((1, 1, 11))
    t = Tree(n=3, adj=adj, root=0)

    K = 2
    g_val, g_nds = _build_g_tables(t, K)
    vh = 2  # chi_v3

    assert abs(g_val[vh][0][0] - 2) < 1e-9, f"g(0,0) expected 2, got {g_val[vh][0][0]}"
    assert abs(g_val[vh][0][1] - 12) < 1e-9, f"g(0,1) expected 12, got {g_val[vh][0][1]}"
    assert abs(g_val[vh][1][1] - 7) < 1e-9, f"g(1,1) expected 7, got {g_val[vh][1][1]}"
    assert abs(g_val[vh][1][2] - 17) < 1e-9, f"g(1,2) expected 17, got {g_val[vh][1][2]}"
    print("PASS: test_g_two_edge_chain_ca_lower_dw")


def test_g_two_edge_chain_ca_higher_dw():
    """
    For a 2-edge chain where CA's edge has HIGHER Dw.
    v1 -> v2 -> v3 (leaf).
    e1=(v1->v2): w=1, u=11, Dw=10
    e2=(v2->v3): w=1, u=6,  Dw=5

    delta[0] = 10 (CA's edge), delta[1] = 5 (intermediate's edge).

    g(eps=0, k=0) = 2
    g(eps=0, k=1) = 2 + 5 = 7
    g(eps=1, k=1) = 2 + 10 = 12
    g(eps=1, k=2) = 2 + 10 + 5 = 17
    """
    adj = [[] for _ in range(3)]
    adj[0].append((1, 1, 11)); adj[1].append((0, 1, 11))
    adj[1].append((2, 1, 6)); adj[2].append((1, 1, 6))
    t = Tree(n=3, adj=adj, root=0)

    K = 2
    g_val, g_nds = _build_g_tables(t, K)
    vh = 2  # chi_v3

    assert abs(g_val[vh][0][0] - 2) < 1e-9
    assert abs(g_val[vh][0][1] - 7) < 1e-9, f"g(0,1) expected 7, got {g_val[vh][0][1]}"
    assert abs(g_val[vh][1][1] - 12) < 1e-9, f"g(1,1) expected 12, got {g_val[vh][1][1]}"
    assert abs(g_val[vh][1][2] - 17) < 1e-9
    print("PASS: test_g_two_edge_chain_ca_higher_dw")


# ---------------------------------------------------------------------------
# Tests: small trees
# ---------------------------------------------------------------------------

def test_path_tree_k0():
    """Path tree, K=0: no upgrades, answer = sum of w values."""
    t = make_path_tree([2, 3, 5], [5, 7, 9])
    val, S = solve_mspit_unu(t, 0)
    assert abs(val - 10) < 1e-9, f"Expected 10 (2+3+5), got {val}"
    assert S == set()
    print("PASS: test_path_tree_k0")


def test_path_tree_k1():
    """Path tree with 3 edges: K=1, upgrade the edge with largest Dw."""
    # v1 -> v2 -> v3 -> v4 (leaf)
    # e1: w=1, u=10 (Dw=9); e2: w=2, u=5 (Dw=3); e3: w=3, u=6 (Dw=3)
    t = make_path_tree([1, 2, 3], [10, 5, 6])
    val, S = solve_mspit_unu(t, 1)
    bf_val, _ = brute_force(t, 1)
    assert abs(val - bf_val) < 1e-9, f"DP={val}, BF={bf_val}"
    print("PASS: test_path_tree_k1")


def test_star_tree_k1():
    """
    Star tree: root + 3 leaves, all upgrades at root affect ALL edges.
    With parent model: upgrading root upgrades all edges from root.
    K=1: upgrade root -> all leaf paths get u weights.
    """
    t = make_star_tree(4, [3, 5, 2], [7, 9, 8])
    val, S = solve_mspit_unu(t, 1)
    bf_val, _ = brute_force(t, 1)
    assert abs(val - bf_val) < 1e-9, f"DP={val}, BF={bf_val}"
    # With root upgraded: min path = min(7, 9, 8) = 7
    # Without upgrade: min path = min(3, 5, 2) = 2
    # Check K=1 gives 7
    assert abs(val - 7) < 1e-9, f"Expected 7 (min upgraded weight), got {val}"
    print("PASS: test_star_tree_k1")


def test_k_equals_n():
    """K=n: all nodes upgraded, should give max possible min path."""
    t = random_tree(8, seed=12345)
    val_n, S_n = solve_mspit_unu(t, t.n)
    bf_val, _ = brute_force(t, t.n)
    assert abs(val_n - bf_val) < 1e-9, f"K=n: DP={val_n}, BF={bf_val}"
    print("PASS: test_k_equals_n")


def test_k_greater_than_n():
    """K > n should give same result as K = n (can't upgrade more nodes than exist)."""
    t = random_tree(6, seed=54321)
    val_n, S_n = solve_mspit_unu(t, t.n)
    val_2n, S_2n = solve_mspit_unu(t, 2 * t.n)
    assert abs(val_n - val_2n) < 1e-9, f"K=n: {val_n}, K=2n: {val_2n} should be equal"
    print("PASS: test_k_greater_than_n")


def test_example1_k1():
    """Example 1 from paper: F(T_{v1}, 1) = 13, S = {v1}."""
    n = 10
    adj = [[] for _ in range(n)]
    def add_edge(u, v, w, uw=10):
        adj[u].append((v, w, uw))
        adj[v].append((u, w, uw))
    add_edge(0, 1, 6); add_edge(1, 2, 7); add_edge(1, 3, 4)
    add_edge(0, 4, 1); add_edge(4, 5, 8); add_edge(0, 6, 4)
    add_edge(6, 7, 3); add_edge(6, 8, 4); add_edge(8, 9, 5)
    t = Tree(n=n, adj=adj, root=0)

    val, S = solve_mspit_unu(t, 1)
    assert abs(val - 13) < 1e-9, f"Expected 13, got {val}"
    assert S == {0}, f"Expected {{v1}}, got {S}"
    print("PASS: test_example1_k1")


def test_example1_all_k():
    """Example 1 vs brute force for K=0..9."""
    n = 10
    adj = [[] for _ in range(n)]
    def add_edge(u, v, w, uw=10):
        adj[u].append((v, w, uw))
        adj[v].append((u, w, uw))
    add_edge(0, 1, 6); add_edge(1, 2, 7); add_edge(1, 3, 4)
    add_edge(0, 4, 1); add_edge(4, 5, 8); add_edge(0, 6, 4)
    add_edge(6, 7, 3); add_edge(6, 8, 4); add_edge(8, 9, 5)
    t = Tree(n=n, adj=adj, root=0)

    for K in range(0, 10):
        dp_val, _ = solve_mspit_unu(t, K)
        bf_val, _ = brute_force(t, K)
        assert abs(dp_val - bf_val) < 1e-9, f"K={K}: DP={dp_val}, BF={bf_val}"
    print("PASS: test_example1_all_k")


# ---------------------------------------------------------------------------
# Tests: brute-force cross-checks on random trees
# ---------------------------------------------------------------------------

def test_random_trees_vs_brute_force():
    """Cross-check DP against brute force on 100 random trees."""
    rng = random.Random(777)
    failed = 0
    for trial in range(100):
        n = rng.randint(3, 11)
        seed = trial * 31 + 5
        t = random_tree(n, seed=seed)
        K = rng.randint(0, n)
        dp_val, _ = solve_mspit_unu(t, K)
        bf_val, _ = brute_force(t, K)
        if abs(dp_val - bf_val) > 1e-9:
            failed += 1
            print(f"  FAIL trial={trial}: n={n}, K={K}, DP={dp_val}, BF={bf_val}")
    assert failed == 0, f"{failed} failures in random tree test"
    print("PASS: test_random_trees_vs_brute_force (100 trials)")


# ---------------------------------------------------------------------------
# Tests: MCSPIT-UNu
# ---------------------------------------------------------------------------

def test_mcspit_feasible():
    """MCSPIT: verify K_star is the minimum feasible budget for given D."""
    t = random_tree(8, seed=99)
    # Find the max possible min path (all edges upgraded)
    leaves = [v for v in range(t.n) if t.is_leaf[v]]
    max_val_all, _ = solve_mspit_unu(t, t.n)
    # Set D to halfway between original and max
    val0, _ = solve_mspit_unu(t, 0)
    D = (val0 + max_val_all) / 2

    K_star, opt_val, S = solve_mcspit_unu(t, D)
    assert opt_val >= D - 1e-9, f"MCSPIT result {opt_val} < D={D}"
    if K_star > 0:
        # K_star - 1 should be infeasible
        val_less, _ = solve_mspit_unu(t, K_star - 1)
        assert val_less < D - 1e-9, f"K_star-1={K_star-1} gives {val_less} >= D={D}"
    print("PASS: test_mcspit_feasible")


def test_mcspit_exact_match_plateau():
    """MCSPIT: an exact match on a plateau must return its minimum budget."""
    t = make_star_tree(4, [1, 1, 1], [2, 2, 2])
    K_star, opt_val, S = solve_mcspit_unu(t, 2)
    assert K_star == 1, f"Expected K_star=1, got {K_star}"
    assert opt_val == 2, f"Expected opt_val=2, got {opt_val}"
    assert S == {0}, f"Expected root upgrade only, got {S}"
    print("PASS: test_mcspit_exact_match_plateau")


def test_mcspit_d_zero():
    """MCSPIT: D=0 should require 0 upgrades."""
    t = random_tree(7, seed=77)
    K_star, opt_val, S = solve_mcspit_unu(t, 0)
    assert K_star == 0, f"Expected K_star=0 for D=0, got {K_star}"
    print("PASS: test_mcspit_d_zero")


def test_mcspit_infeasible():
    """MCSPIT: return infeasible if D exceeds the all-upgraded optimum."""
    t = make_path_tree([1], [2])
    K_star, opt_val, S = solve_mcspit_unu(t, 100)
    assert K_star is None, f"Expected infeasible K_star=None, got {K_star}"
    assert opt_val is None, f"Expected infeasible opt_val=None, got {opt_val}"
    assert S is None, f"Expected infeasible S=None, got {S}"
    print("PASS: test_mcspit_infeasible")


def test_mcspit_vs_mspit():
    """MCSPIT and MSPIT are consistent: F(T, K_star) >= D for all K >= K_star."""
    t = random_tree(9, seed=123)
    D = 10.0  # arbitrary target
    K_star, opt_val, S = solve_mcspit_unu(t, D)

    # verify all K >= K_star give val >= D
    for K in range(max(0, K_star - 1), K_star + 3):
        val, _ = solve_mspit_unu(t, K)
        if K >= K_star:
            assert val >= D - 1e-9, f"K={K} >= K_star={K_star} but val={val} < D={D}"
        else:
            assert val < D - 1e-9 or K_star == 0, \
                f"K={K} < K_star={K_star} but val={val} >= D={D}"
    print("PASS: test_mcspit_vs_mspit")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_g_single_edge_chain,
        test_g_two_edge_chain_ca_lower_dw,
        test_g_two_edge_chain_ca_higher_dw,
        test_path_tree_k0,
        test_path_tree_k1,
        test_star_tree_k1,
        test_k_equals_n,
        test_k_greater_than_n,
        test_example1_k1,
        test_example1_all_k,
        test_random_trees_vs_brute_force,
        test_mcspit_feasible,
        test_mcspit_exact_match_plateau,
        test_mcspit_d_zero,
        test_mcspit_infeasible,
        test_mcspit_vs_mspit,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed}/{passed + failed} tests passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
