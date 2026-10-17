"""
mcspit_un.py  —  MCSPIT-UNu solvers: budget doubling (Algorithm 2, Theorem 14) one DP run at K=n, and binary search.

Problem: Given tree T=(V,E,w) rooted at v1, upgraded weights u(e)>=w(e),
         and lower bound D on min root-leaf distance, find the MINIMUM
         number of nodes to upgrade so that min_{leaf t} d(root,t) >= D.

Earlier binary-search solver (solve_mcspit_unu), kept for reference:
  1. Check feasibility at K=n. If F(T,n) < D, return infeasible.
  2. Check the early-exit case K=0.
  3. Binary search over K in [0, n].
  4. Return the minimum feasible budget k2 and a witness set.

Time complexity: O(n^3 log n).
"""

from mspit_un import solve_mspit_unu
from tree import Tree


def _shortest_root_leaf_upgraded(tree: Tree):
    """F(T, n): shortest root-leaf distance with every edge at its upgraded
    weight u.  O(n) iterative DFS over tree.children."""
    best = float('inf')
    stack = [(tree.root, 0)]
    while stack:
        v, d = stack.pop()
        if not tree.children[v]:
            best = min(best, d)
            continue
        for (c, w, u) in tree.children[v]:
            stack.append((c, d + u))
    return best


def solve_mcspit_unu(tree: Tree, D: float):
    """
    Solve MCSPIT-UNu: minimise |S| subject to min root-leaf distance >= D.

    Returns
    -------
    (K_star, opt_value, S)
      K_star    : minimum number of upgrades needed (or None if infeasible)
      opt_value : F(T, K_star) when feasible, else None
      S         : set of node indices to upgrade when feasible, else None
    """
    n = tree.n

    # Feasibility check: even upgrading every non-leaf node cannot meet D.
    val_n, S_n = solve_mspit_unu(tree, n)
    if val_n < D:
        return None, None, None

    # Edge case: already feasible with 0 upgrades?
    val0, S0 = solve_mspit_unu(tree, 0)
    if val0 >= D:
        return 0, val0, S0

    k1, k2 = 0, n

    while k2 != k1 + 1:
        k = (k1 + k2 + 1) // 2   # ceiling((k1+k2)/2)
        val, S = solve_mspit_unu(tree, k)

        if val < D:
            k1 = k
        else:
            # A feasible budget, including an exact match, may not be minimum.
            k2 = k

    # Final call with k2 (line 18-19)
    val, S = solve_mspit_unu(tree, k2)
    return k2, val, S


def solve_mcspit_unu_oneshot(tree: Tree, D: float, method='pairs'):
    """MCSPIT-UNu from ONE DP run at K=n (root profile F(T,k), all k), plus
    one run at K* to recover a witness set.  Same output as solve_mcspit_unu.
    Infeasibility is rejected by an O(n) check first.
    Time O(n^2) (+ O(n K*)) with the O(nK) DP, vs O(n^2 log n) for bisection."""
    from mspit_un import root_profile
    if _shortest_root_leaf_upgraded(tree) < D:
        return None, None, None
    prof = root_profile(tree, tree.n, method=method)
    k_star = next(k for k, val in enumerate(prof) if val >= D)
    val, S = solve_mspit_unu(tree, k_star, method=method)
    return k_star, val, S


def solve_mcspit_unu_doubling(tree: Tree, D: float, method='pairs'):
    """MCSPIT-UNu by budget doubling: run the DP with K = 1, 2, 4, ... (capped
    at n) and stop at the first run whose root profile reaches D.  Same output
    as solve_mcspit_unu.  Time and space O(n max{K*,1}) when feasible; an
    infeasible instance is rejected by an O(n) check before any DP run."""
    from mspit_un import root_profile
    n = tree.n
    # O(n) infeasibility check: D is reachable iff F(T, n) >= D.
    if _shortest_root_leaf_upgraded(tree) < D:
        return None, None, None
    K = 1
    while True:
        prof = root_profile(tree, K, method=method)
        hit = next((k for k, val in enumerate(prof) if val >= D), None)
        if hit is not None:
            val, S = solve_mspit_unu(tree, hit, method=method)
            return hit, val, S
        # The profile saturates once K exceeds the number of useful upgrades.
        if K >= n or len(prof) <= K:
            return None, None, None
        K = min(2 * K, n)
