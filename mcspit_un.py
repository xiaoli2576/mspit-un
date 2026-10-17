"""
mcspit_un.py  —  Algorithm 2: Binary search for MCSPIT-UNu.

Problem: Given tree T=(V,E,w) rooted at v1, upgraded weights u(e)>=w(e),
         and lower bound D on min root-leaf distance, find the MINIMUM
         number of nodes to upgrade so that min_{leaf t} d(root,t) >= D.

Algorithm 2 from the paper:
  1. Check feasibility at K=n. If F(T,n) < D, return infeasible.
  2. Check the early-exit case K=0.
  3. Binary search over K in [0, n].
  4. Return the minimum feasible budget k2 and a witness set.

Time complexity: O(n^3 log n).
"""

from mspit_un import solve_mspit_unu
from tree import Tree


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
    Time O(n^2) (+ O(n K*)) with the O(nK) DP, vs O(n^2 log n) for bisection."""
    from mspit_un import root_profile
    prof = root_profile(tree, tree.n, method=method)
    if prof[-1] < D:
        return None, None, None
    k_star = next(k for k, val in enumerate(prof) if val >= D)
    val, S = solve_mspit_unu(tree, k_star, method=method)
    return k_star, val, S
