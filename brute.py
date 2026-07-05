"""
brute.py  --  Brute-force verifier for MSPIT-UNu.

Enumerates all subsets S subset V with |S| <= K, computes the min root-leaf
distance under the upgraded weights, and returns the maximum.

PHYSICAL MODEL: Edge e=(v_parent, v_child) gets weight u(e) if and only if
v_parent in S (the parent node is upgraded).  This matches the paper's
"forward node" definition where u(e) = weight after upgrading the forward
(parent) node of edge e.

Only practical for small trees (n <= 15).
"""

from itertools import combinations
from tree import Tree


def compute_path_length(tree: Tree, leaf: int, S: set) -> float:
    """
    Compute the root-to-leaf path length for the given leaf under
    the upgrade set S.

    Edge (parent -> child) gets u(e) iff parent in S.
    """
    total = 0.0
    cur = leaf
    while tree.parent[cur] is not None:
        p, w, uw = tree.parent[cur]
        # Edge is (p -> cur). Gets upgraded iff parent p in S.
        if p in S:
            total += uw
        else:
            total += w
        cur = p
    return total


def brute_force(tree: Tree, K: int):
    """
    Brute-force solve MSPIT-UNu.

    Returns
    -------
    (opt_value, S)
      opt_value : float
      S         : set of node indices to upgrade
    """
    leaves = [v for v in range(tree.n) if tree.is_leaf[v]]
    nodes = list(range(tree.n))

    best_val = float('-inf')
    best_S = set()

    for k in range(0, K + 1):
        for S_tuple in combinations(nodes, k):
            S = set(S_tuple)
            min_dist = min(compute_path_length(tree, leaf, S)
                           for leaf in leaves)
            if min_dist > best_val:
                best_val = min_dist
                best_S = S

    return best_val, best_S


def min_path_length(tree: Tree, S: set) -> float:
    """Return min root-leaf distance for a given upgrade set S."""
    leaves = [v for v in range(tree.n) if tree.is_leaf[v]]
    return min(compute_path_length(tree, leaf, S) for leaf in leaves)


if __name__ == "__main__":
    # Quick self-test
    n = 3
    adj = [[] for _ in range(n)]
    adj[0].append((1, 2, 5))
    adj[1].append((0, 2, 5))
    adj[1].append((2, 3, 7))
    adj[2].append((1, 3, 7))

    t = Tree(n=n, adj=adj, root=0)

    val, S = brute_force(t, 1)
    print(f"Simple path tree K=1: val={val}, S={S}")
    # Upgrading v1 (0): e1=(v1->v2) -> u=5. Path = 5+3=8.
    # Upgrading v2 (1): e2=(v2->v3) -> u=7. Path = 2+7=9. (parent model: v2 upgrades e2)
    # Upgrading v3 (2): v3 is leaf, child of v2. v3 upgrades edges FROM v3 to children.
    #                   v3 has no children => no edges upgraded. path = 2+3=5.
    # Best: upgrade v2 -> path = 2+7=9.
    assert abs(val - 9) < 1e-9, f"Expected 9, got {val}"
    assert S == {1}, f"Expected {{1}}, got {S}"
    print("PASS")
