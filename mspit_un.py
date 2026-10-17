"""
mspit_un.py  --  Algorithm 1: Dynamic Programming for MSPIT-UNu.

Problem: Tree T=(V,E,w) rooted at v1, upgraded weights u(e)>=w(e),
         budget K.  Choose S subset V with |S|<=K to maximize
           min_{leaf t}  d(v1, t)  under the upgraded weight vector.

Upgrade rule: if v in S, ALL outgoing edges of v get weight u(e).

--------------------------------------------------------------------
KEY MODEL (Lemma 1 / abstract chain formulation)
--------------------------------------------------------------------
Chain chi_{v_h} = path from CA(v_h)=v to v_h.  Let the nodes in
original order be [v=v_{i_0}, ..., v_h=v_{i_beta}] with edges
e_{i_1}..e_{i_beta}.  After sorting edges by Dw=u-w DESCENDING
(Lemma 1 justifies this without loss of generality), the formula is:

  g(chi, eps=0, k) = w_sum + sum(delta[1:k+1])   for 1<=k<=beta,
                     (skip delta[0]: CA not upgraded, no benefit from top edge)
                     valid only when beta >= 2  AND  1 <= k <= beta-1
  g(chi, eps=0, 0) = w_sum
  g(chi, eps=1, k) = w_sum + sum(delta[0:k])      for 1<=k<=beta

  delta = sorted-descending list of Dw values, indexed 0..beta-1.

  For beta=1: eps=0,k=0 and eps=1,k=1 are the only valid cases.
  For beta>=2: eps=0 is also valid for k=1..beta-1 (NOT k=beta,
               because that would require upgrading ALL chain nodes
               including CA, which is the eps=1 full case).

  Node sets (for reconstruction):
    V(chi, 0, k) = {v_{i_1}, ..., v_{i_k}}          = original chain
                                                       positions 1..k
    V(chi, 1, k) = {v_{i_0}, v_{i_1}, ..., v_{i_{k-1}}} = {CA} + pos 1..k-1

--------------------------------------------------------------------
BUDGET COUNTING  (key subtlety)
--------------------------------------------------------------------
  |V(T^{q:q}_v)| - |Y cap V(T^{q:q}_v)|
     = non-leaf nodes in T^{q:q}_v INCLUDING the CA v.
     = 1 (CA) + (beta-1 intermediates) + (1 if tip non-leaf) + sub(v_hq)

  In f(T^{q:q}_v, eps, k):
     k1 + k2 = k, eps<=k1<=min(beta, K), 0<=k2<=min(sub(v_hq), K)
     Note: max k1 = min(beta, K).  The CA is represented by eps (if eps=1
     the CA IS counted in k1, but the chain formula g handles it correctly
     by using delta[0..k1-1]).

  In f(T^{1:q}_v, eps, k):
     k1 + k2 - eps = k, eps<=k1<=min(Nqq, K), eps<=k2<=min(N1qm1, K)
     where Nqq = |V(T^{q:q}_v)| - |Y cap ...| (includes CA)
     and   N1qm1 = |V(T^{1:(q-1)}_v)| - |Y cap ...| (includes CA)
     The -eps avoids double-counting the CA.
"""

from tree import Tree

NEG_INF = float('-inf')


# --------------------------------------------------------------------------
# Helper: count upgradeable (non-leaf) nodes
# --------------------------------------------------------------------------

def _nl_subtree(tree, v_h):
    """
    Total non-leaf nodes in T_{v_h} (the FULL subtree at v_h),
    INCLUDING v_h itself.  Returns 0 if v_h is a leaf.
    """
    if tree.is_leaf[v_h]:
        return 0
    count = 1  # v_h itself
    for vhq in tree.CD[v_h]:
        chain = tree.chains[vhq]
        count += (chain.beta - 1)          # intermediate nodes (non-leaf)
        if not tree.is_leaf[vhq]:
            count += _nl_subtree(tree, vhq)  # sub-branching-tree
        # if vhq is a leaf, it contributes 0 to non-leaf count
    return count


def _Nqq(tree, v, q):
    """
    |V(T^{q:q}_v)| - |Y cap V(T^{q:q}_v)| = non-leaf nodes in T^{q:q}_v
    INCLUDING the CA v.
    """
    v_hq = tree.CD[v][q - 1]
    chain = tree.chains[v_hq]
    beta = chain.beta
    # CA v: 1 (non-leaf, since v in V* u {v1})
    # intermediates: beta - 1
    # tip v_hq: 1 if non-leaf, else 0
    # subtree below v_hq: nl_subtree(v_hq) - 1 if non-leaf (subtract v_hq itself)
    ca = 1
    intermediates = beta - 1
    tip = 0 if tree.is_leaf[v_hq] else 1
    sub = 0 if tree.is_leaf[v_hq] else (_nl_subtree(tree, v_hq) - 1)
    return ca + intermediates + tip + sub


def _N1q(tree, v, q):
    """
    |V(T^{1:q}_v)| - |Y cap V(T^{1:q}_v)| = non-leaf nodes in T^{1:q}_v
    INCLUDING the CA v (counted once, shared across all q branches).
    """
    total = 1  # CA v, counted once
    for i in range(1, q + 1):
        v_hi = tree.CD[v][i - 1]
        chain = tree.chains[v_hi]
        beta = chain.beta
        total += beta - 1                        # intermediates
        total += 0 if tree.is_leaf[v_hi] else 1  # tip
        if not tree.is_leaf[v_hi]:
            total += _nl_subtree(tree, v_hi) - 1  # sub-tree below v_hi
    return total


def _nl_sub_only(tree, v_h):
    """Non-leaf nodes in T_{v_h}, EXCLUDING v_h itself. Used for k2 in fqq."""
    return max(0, _nl_subtree(tree, v_h) - 1)


# --------------------------------------------------------------------------
# Build g tables
# --------------------------------------------------------------------------

def _chain_upgrade_nodes(chain):
    """
    Return the ABSTRACT upgrade node sequence for the chain, in the order
    that corresponds to delta[0], delta[1], ..., delta[beta-1].

    After Lemma 1 reordering (CA's edge fixed at position 0, intermediates
    sorted by Dw desc at positions 1..beta-1):
      - Position 0 owner = CA = edges[0][0]
      - Position j owner = edges[j][0]  for j = 1..beta-1

    The tip (v_h) is NOT an upgrade node in the g formula; it is handled
    separately by F_sub.
    """
    # edges[j][0] is the parent node of the j-th edge in the (re)ordered chain.
    # Position j in the abstract model is "owned" by edges[j][0].
    return [e[0] for e in chain.edges]


def _build_g_tables(tree, K):
    """
    Compute g(chi_{v_h}, eps, k) for all chains.

    After Lemma 1 reordering (see _build_chain in tree.py):
      chain.edges[0]   = CA's outgoing edge (FIXED at position 0)
      chain.edges[1..] = intermediate edges sorted by Dw desc

      delta[0] = Dw(CA's edge)           <- CA's contribution when upgraded
      delta[1..] = Dw of intermediates sorted desc

    eps=0, k=0:            g = w_sum
    eps=1, k in [1..beta]: g = w_sum + sum(delta[0:k])   (CA takes delta[0])
    eps=0, k in [1..beta-1]: g = w_sum + sum(delta[1:k+1])  (skip CA's delta[0])
      (beta=1 has no valid eps=0, k>=1 case)

    Node sets for upgrade tracking:
      upgrade_nodes[j] = edges[j][0] = owner of position j in abstract chain
      V(chi, 1, k) = {upgrade_nodes[0..k-1]} = {CA} + top k-1 intermediates
      V(chi, 0, k) = {upgrade_nodes[1..k]}   = top k intermediates
    """
    g_val = {}   # g_val[v_h][eps][k] -> float
    g_nds = {}   # g_nds[v_h][eps][k] -> frozenset

    for v_h, chain in tree.chains.items():
        beta = chain.beta
        max_k = min(beta, K)
        upg = _chain_upgrade_nodes(chain)
        # upg[j] = owner of abstract position j (the node to upgrade for delta[j])
        # upg[0] = CA, upg[1..beta-1] = intermediates in sorted order

        g_val[v_h] = {0: {}, 1: {}}
        g_nds[v_h] = {0: {}, 1: {}}

        # eps=0, k=0
        g_val[v_h][0][0] = chain.w_sum
        g_nds[v_h][0][0] = frozenset()

        for k in range(1, max_k + 1):
            # eps=1: CA is upgraded (contributes delta[0]), then top k-1 intermediates
            # g = w_sum + sum(delta[0:k])
            g_val[v_h][1][k] = chain.w_sum + sum(chain.delta[0:k])
            # V(chi,1,k) = {upg[0], upg[1], ..., upg[k-1]} = {CA} + top k-1 intermediates
            g_nds[v_h][1][k] = frozenset(upg[j] for j in range(0, k))

            # eps=0: CA NOT upgraded; take top k intermediates (delta[1..k])
            # only valid for beta >= 2 and k <= beta-1
            if beta >= 2 and k <= beta - 1:
                g_val[v_h][0][k] = chain.w_sum + sum(chain.delta[1:k + 1])
                # V(chi,0,k) = {upg[1], ..., upg[k]} = top k intermediates
                g_nds[v_h][0][k] = frozenset(upg[j] for j in range(1, k + 1))

    return g_val, g_nds


# --------------------------------------------------------------------------
# Main DP solver
# --------------------------------------------------------------------------

def _solve_mspit_unu_v1(tree: Tree, K: int, return_tables=False):
    """
    ORIGINAL (v1) implementation, kept verbatim for comparison
    (method='v1' in solve_mspit_unu).  Recomputes subtree sizes
    recursively inside the loops and stores frozensets per DP cell.

    Solve MSPIT-UNu.  Returns (optimal_value, S), or
    (optimal_value, S, tables) when return_tables=True, where tables
    exposes the internal DP dictionaries for verification / paper-table
    generation:
      tables = {'g_val','g_nds','fqq','fqq_N','Fqq','Fqq_N',
                'f1q','f1q_N','F1q','F1q_N'}
    """
    if K <= 0:
        leaves = [v for v in range(tree.n) if tree.is_leaf[v]]
        min_d = float('inf')
        for leaf in leaves:
            d = 0.0
            cur = leaf
            while tree.parent[cur] is not None:
                p, w, uw = tree.parent[cur]
                d += w
                cur = p
            min_d = min(min_d, d)
        if return_tables:
            return min_d, set(), {}
        return min_d, set()

    g_val, g_nds = _build_g_tables(tree, K)

    # Process branching nodes bottom-up (deepest layer first)
    branching_sorted = sorted(tree.branching,
                              key=lambda v: tree.layer[v], reverse=True)

    fqq = {};  Fqq = {}     # (v,q,eps,k) and (v,q,k)
    f1q = {};  F1q = {}
    fqq_N = {}; Fqq_N = {}
    f1q_N = {}; F1q_N = {}

    def F_sub(v_h, k):
        """F(T_{v_h}, k) -- value at sub-branching-tree v_h.
        Clamp k to total upgradeable INCLUDING v_h itself."""
        if tree.is_leaf[v_h]:
            return 0.0  # leaves have no upgradeable nodes
        p = len(tree.CD[v_h])
        if p == 0:
            return 0.0
        # Total upgradeable in T_{v_h} INCLUDING v_h itself
        n_total = _nl_subtree(tree, v_h)
        k_eff = min(k, n_total)
        return F1q.get((v_h, p, k_eff), NEG_INF)

    def N_sub(v_h, k):
        if tree.is_leaf[v_h]:
            return frozenset()
        p = len(tree.CD[v_h])
        if p == 0:
            return frozenset()
        n_total = _nl_subtree(tree, v_h)
        k_eff = min(k, n_total)
        return F1q_N.get((v_h, p, k_eff), frozenset())

    for v in branching_sorted:
        p = len(tree.CD[v])
        if p == 0:
            continue

        for q in range(1, p + 1):
            v_hq = tree.CD[v][q - 1]
            chain = tree.chains[v_hq]
            beta = chain.beta
            max_k_chain = min(beta, K)   # max k1 on the chain g side

            nqq = _Nqq(tree, v, q)      # includes CA
            n1q = _N1q(tree, v, q)      # includes CA (once)
            max_k_T1q = min(n1q, K)

            nsub = _nl_subtree(tree, v_hq)  # sub-tree budget for tip v_hq (incl v_hq itself)

            # -------- f(T^{q:q}_v, eps, k) and F(T^{q:q}_v, k) --------
            # k1 + k2 = k, eps <= k1 <= min(beta, K), 0 <= k2 <= min(nsub, K)
            # nsub = nl_subtree(v_hq) includes v_hq itself (can be upgraded as CA of its branches)
            for k in range(0, max_k_T1q + 1):
                for eps in (0, 1):
                    if k == 0 and eps == 1:
                        continue

                    if k == 0:
                        gv = g_val[v_hq][0].get(0, NEG_INF)
                        fv = F_sub(v_hq, 0)
                        val = (gv + fv) if gv != NEG_INF and fv != NEG_INF else NEG_INF
                        nd = (g_nds[v_hq][0].get(0, frozenset()) |
                              N_sub(v_hq, 0))
                        fqq[(v, q, 0, 0)] = val
                        fqq_N[(v, q, 0, 0)] = nd
                        Fqq[(v, q, 0)] = val
                        Fqq_N[(v, q, 0)] = nd
                        continue

                    # k >= 1
                    best = NEG_INF
                    best_nd = frozenset()
                    for k1 in range(eps, min(max_k_chain, k) + 1):
                        k2 = k - k1
                        if k2 < 0 or k2 > min(nsub, K):
                            continue
                        gv = g_val[v_hq][eps].get(k1, NEG_INF)
                        if gv == NEG_INF:
                            continue
                        fv = F_sub(v_hq, k2)
                        if fv == NEG_INF:
                            continue
                        comb = gv + fv
                        if comb > best:
                            best = comb
                            best_nd = (g_nds[v_hq][eps].get(k1, frozenset()) |
                                       N_sub(v_hq, k2))
                    fqq[(v, q, eps, k)] = best
                    fqq_N[(v, q, eps, k)] = best_nd

                if k > 0:
                    v0 = fqq.get((v, q, 0, k), NEG_INF)
                    v1 = fqq.get((v, q, 1, k), NEG_INF)
                    if v1 >= v0:
                        Fqq[(v, q, k)] = v1
                        Fqq_N[(v, q, k)] = fqq_N.get((v, q, 1, k), frozenset())
                    else:
                        Fqq[(v, q, k)] = v0
                        Fqq_N[(v, q, k)] = fqq_N.get((v, q, 0, k), frozenset())

            # -------- f(T^{1:q}_v, eps, k) and F(T^{1:q}_v, k) --------
            for k in range(0, max_k_T1q + 1):
                for eps in (0, 1):
                    if k == 0 and eps == 1:
                        continue

                    # q=1: T^{1:1} = T^{1:1}_v, same as fqq
                    if q == 1:
                        val = fqq.get((v, 1, eps, k), NEG_INF)
                        nd  = fqq_N.get((v, 1, eps, k), frozenset())
                        f1q[(v, 1, eps, k)] = val
                        f1q_N[(v, 1, eps, k)] = nd
                        if k == 0:
                            F1q[(v, 1, 0)] = val
                            F1q_N[(v, 1, 0)] = nd
                        continue

                    # q >= 2, k=0
                    if k == 0:
                        vqq = fqq.get((v, q, 0, 0), NEG_INF)
                        v1qm1 = f1q.get((v, q - 1, 0, 0), NEG_INF)
                        val = min(vqq, v1qm1)
                        f1q[(v, q, 0, 0)] = val
                        f1q_N[(v, q, 0, 0)] = frozenset()
                        F1q[(v, q, 0)] = val
                        F1q_N[(v, q, 0)] = frozenset()
                        continue

                    # q >= 2, k >= 1
                    # k1 + k2 - eps = k  =>  k2 = k - k1 + eps
                    # eps <= k1 <= min(Nqq, K)
                    # eps <= k2 <= min(N1(q-1), K)
                    n1qm1 = _N1q(tree, v, q - 1)  # includes CA

                    best = NEG_INF
                    best_nd = frozenset()

                    for k1 in range(eps, min(nqq, K) + 1):
                        k2 = k - k1 + eps
                        if k2 < eps or k2 > min(n1qm1, K) or k2 < 0:
                            continue

                        fval_qq = fqq.get((v, q, eps, k1), NEG_INF)
                        if fval_qq == NEG_INF:
                            continue

                        # f(T^{1:q-1}_v, eps, k2)
                        if k2 == 0:
                            fval_1qm1 = f1q.get((v, q - 1, 0, 0), NEG_INF)
                            nd_1qm1 = f1q_N.get((v, q - 1, 0, 0), frozenset())
                        else:
                            fval_1qm1 = f1q.get((v, q - 1, eps, k2), NEG_INF)
                            nd_1qm1 = f1q_N.get((v, q - 1, eps, k2), frozenset())

                        if fval_1qm1 == NEG_INF:
                            continue

                        # Objective: min over branches (the shortest root-leaf
                        # path across the whole subtree is the min across branches)
                        comb = min(fval_qq, fval_1qm1)
                        if comb > best:
                            best = comb
                            nd_qq = fqq_N.get((v, q, eps, k1), frozenset())
                            best_nd = nd_qq | nd_1qm1

                    f1q[(v, q, eps, k)] = best
                    f1q_N[(v, q, eps, k)] = best_nd

                if k > 0:
                    v0 = f1q.get((v, q, 0, k), NEG_INF)
                    v1 = f1q.get((v, q, 1, k), NEG_INF)
                    if v1 >= v0:
                        F1q[(v, q, k)] = v1
                        F1q_N[(v, q, k)] = f1q_N.get((v, q, 1, k), frozenset())
                    else:
                        F1q[(v, q, k)] = v0
                        F1q_N[(v, q, k)] = f1q_N.get((v, q, 0, k), frozenset())

    # Extract answer
    root = tree.root
    p_root = len(tree.CD[root])
    if p_root == 0:
        return 0.0, set()

    actual_K = min(K, _N1q(tree, root, p_root))
    opt = F1q.get((root, p_root, actual_K), NEG_INF)
    S = set(F1q_N.get((root, p_root, actual_K), frozenset()))

    if opt == NEG_INF:
        opt = F1q.get((root, p_root, 0), NEG_INF)
        S = set()

    if return_tables:
        tables = {'g_val': g_val, 'g_nds': g_nds,
                  'fqq': fqq, 'fqq_N': fqq_N, 'Fqq': Fqq, 'Fqq_N': Fqq_N,
                  'f1q': f1q, 'f1q_N': f1q_N, 'F1q': F1q, 'F1q_N': F1q_N}
        return opt, S, tables

    return opt, S


# ==========================================================================
# v2 implementation (2026-10-16): O(nK) time via size-capped pair loops.
#
#   method='pairs' : every recursion enumerates only FEASIBLE PAIRS
#                    (k1,k2), each index capped by min(#non-leaf nodes of its
#                    part, K), and writes into result[k1+k2(-eps)].
#                    Merge cost per (v,q) = O(min(B_q,K) * min(B_{1:q-1},K));
#                    total O(nK) by the capped tree-knapsack charging lemma
#                    (see revision/complexity-onk-20261016.md).
#   method='kk'    : same data structures, but the merge uses the paper's
#                    literal loop "for k, for k1 in [eps, min(B_q,K)]", i.e.
#                    O(min(B_{1:q},K) * min(B_q,K)) per (v,q) -> O(nK^2).
#                    Isolates the algorithmic gain from implementation gains.
#   method='v1'    : the original implementation (_solve_mspit_unu_v1).
#
# Both v2 methods precompute subtree sizes once (O(n)), build g by prefix
# sums (O(sum beta)), and store an O(1) back-pointer per DP cell; the optimal
# set is recovered by one top-down traceback (O(n)).  Values are exact-k,
# identical to v1 cell by cell (ties may select a different optimal set).
# ==========================================================================

DEFAULT_METHOD = 'pairs'


def solve_mspit_unu(tree: Tree, K: int, return_tables=False, method=None):
    """Solve MSPIT-UNu. Returns (optimal_value, S) or (opt, S, tables).

    method in {'pairs' (default, O(nK)), 'kk' (O(nK^2)), 'v1' (original)}.
    """
    if method is None:
        method = DEFAULT_METHOD
    if method == 'v1':
        return _solve_mspit_unu_v1(tree, K, return_tables)
    if method not in ('pairs', 'kk'):
        raise ValueError(f"unknown method {method!r}")
    return _solve_v2(tree, K, return_tables, pairs=(method == 'pairs'))


def _solve_v2(tree, K, return_tables, pairs, profile=False):
    INF = NEG_INF
    Kc = max(0, K)
    is_leaf = tree.is_leaf
    root = tree.root

    if len(tree.CD[root]) == 0:          # single-node tree
        if profile:
            return [0.0]
        if return_tables:
            return 0.0, set(), {}
        return 0.0, set()

    order = sorted(tree.branching, key=lambda v: tree.layer[v], reverse=True)
    nl = {}      # nl[v]  = # non-leaf nodes of T_v incl. v  (branching v)
    Ftab = {}    # Ftab[v] = (F list, r0, r1) of T_v = T^{1:p}_v
    st = {}      # st[(v,q)] = per-branch tables and back-pointers

    for v in order:
        CDv = tree.CD[v]
        if not CDv:
            continue
        Bprev = 0
        r0 = r1 = None
        for q, tip in enumerate(CDv, 1):
            ch = tree.chains[tip]
            beta = ch.beta
            d = ch.delta
            ws = ch.w_sum
            # ---- g(chi, eps, k) by prefix sums (Theorem chain) ----
            m0 = min(beta - 1, Kc)
            m1 = min(beta, Kc)
            g0 = [ws]
            s = ws
            for j in range(1, m0 + 1):
                s += d[j]
                g0.append(s)
            g1 = [INF]
            s = ws
            for j in range(m1):
                s += d[j]
                g1.append(s)
            if is_leaf[tip]:
                subF = (0.0,)
                nsub = 0
            else:
                subF = Ftab[tip][0]
                nsub = nl[tip]
            Bq = beta + nsub                 # non-leaf nodes of T^{q:q}_v incl v
            LQ = min(Bq, Kc) + 1
            LS = len(subF)
            f0 = [INF] * LQ; f1 = [INF] * LQ
            a0 = [-1] * LQ;  a1 = [-1] * LQ
            # ---- single branch: f(T^{q:q}_v, eps, k) ----
            for g, f, a, eps in ((g0, f0, a0, 0), (g1, f1, a1, 1)):
                Lg = len(g)
                if pairs:
                    for k1 in range(eps, Lg):
                        gv = g[k1]
                        top = LQ - k1
                        if top > LS:
                            top = LS
                        for k2 in range(top):
                            val = gv + subF[k2]
                            kk = k1 + k2
                            if val > f[kk]:
                                f[kk] = val
                                a[kk] = k1
                else:
                    for kk in range(eps, LQ):
                        hi = min(Lg - 1, kk)
                        for k1 in range(eps, hi + 1):
                            k2 = kk - k1
                            if k2 < LS:
                                val = g[k1] + subF[k2]
                                if val > f[kk]:
                                    f[kk] = val
                                    a[kk] = k1
            # ---- merge: f(T^{1:q}_v, eps, k), k1+k2-eps=k ----
            if q == 1:
                n0, n1, b0, b1 = f0, f1, None, None
                Bnew = Bq
            else:
                Bnew = Bprev + Bq - 1
                LR = min(Bnew, Kc) + 1
                n0 = [INF] * LR; n1 = [INF] * LR
                b0 = [-1] * LR;  b1 = [-1] * LR
                for A, Bt, R, ar, eps in ((f0, r0, n0, b0, 0),
                                          (f1, r1, n1, b1, 1)):
                    LA = len(A)
                    LB = len(Bt)
                    if pairs:
                        for k1 in range(eps, LA):
                            x = A[k1]
                            if x == INF:
                                continue
                            hi = LR - k1 + eps
                            if hi > LB:
                                hi = LB
                            base = k1 - eps
                            for k2 in range(eps, hi):
                                y = Bt[k2]
                                val = x if x < y else y
                                kk = base + k2
                                if val > R[kk]:
                                    R[kk] = val
                                    ar[kk] = k1
                    else:
                        for kk in range(eps, LR):
                            for k1 in range(eps, LA):
                                k2 = kk - k1 + eps
                                if k2 < eps or k2 >= LB:
                                    continue
                                x = A[k1]
                                y = Bt[k2]
                                val = x if x < y else y
                                if val > R[kk]:
                                    R[kk] = val
                                    ar[kk] = k1
            st[(v, q)] = (tip, ch, f0, f1, a0, a1, n0, n1, b0, b1)
            r0, r1 = n0, n1
            Bprev = Bnew
        nl[v] = Bprev
        Fv = [r1[k] if r1[k] >= r0[k] else r0[k] for k in range(len(r0))]
        Ftab[v] = (Fv, r0, r1)

    def best_eps(v, k):
        _, r0_, r1_ = Ftab[v]
        return 1 if r1_[k] >= r0_[k] else 0

    def trace(items):
        """items: list of (kind, v, q, eps, k); kind in {'1q','qq'}.
        Returns the upgrade set realising those cells (O(size) per call)."""
        S = set()
        stack = list(items)
        while stack:
            kind, v, q, eps, k = stack.pop()
            if kind == '1q':
                while q >= 2:
                    rec = st[(v, q)]
                    k1 = (rec[9] if eps else rec[8])[k]
                    stack.append(('qq', v, q, eps, k1))
                    k = k - k1 + eps
                    q -= 1
            tip, ch, f0_, f1_, a0_, a1_ = st[(v, q)][:6]
            k1 = (a1_ if eps else a0_)[k]
            k2 = k - k1
            if eps:
                for j in range(k1):
                    S.add(ch.edges[j][0])
            else:
                for j in range(1, k1 + 1):
                    S.add(ch.edges[j][0])
            if not is_leaf[tip]:
                stack.append(('1q', tip, len(tree.CD[tip]),
                              best_eps(tip, k2), k2))
        return S

    p_root = len(tree.CD[root])
    Froot = Ftab[root][0]
    kr = len(Froot) - 1                     # = min(K, nl[root])
    opt = Froot[kr]
    if profile:          # F(T_{v1}, k) for every k = 0..min(K, B): one run
        return list(Froot)
    S = trace([('1q', root, p_root, best_eps(root, kr), kr)])

    if not return_tables:
        return opt, S

    g_val, g_nds = _build_g_tables(tree, Kc)
    fqq = {}; Fqq = {}; f1q = {}; F1q = {}
    fqq_N = {}; Fqq_N = {}; f1q_N = {}; F1q_N = {}
    for (v, q), rec in st.items():
        _, _, f0_, f1_, _, _, n0_, n1_, _, _ = rec
        for kind, t0, t1, tab, tabN, Ft, FtN in (
                ('qq', f0_, f1_, fqq, fqq_N, Fqq, Fqq_N),
                ('1q', n0_, n1_, f1q, f1q_N, F1q, F1q_N)):
            for k in range(len(t0)):
                for eps, t in ((0, t0), (1, t1)):
                    if k < eps:
                        continue
                    tab[(v, q, eps, k)] = t[k]
                    tabN[(v, q, eps, k)] = (
                        frozenset(trace([(kind, v, q, eps, k)]))
                        if t[k] != INF else frozenset())
                e = 1 if t1[k] >= t0[k] else 0
                Ft[(v, q, k)] = (t1 if e else t0)[k]
                FtN[(v, q, k)] = tabN.get((v, q, e, k), frozenset())
    tables = {'g_val': g_val, 'g_nds': g_nds,
              'fqq': fqq, 'fqq_N': fqq_N, 'Fqq': Fqq, 'Fqq_N': Fqq_N,
              'f1q': f1q, 'f1q_N': f1q_N, 'F1q': F1q, 'F1q_N': F1q_N}
    return opt, S, tables


def root_profile(tree: Tree, K: int, method='pairs'):
    """Return [F(T_{v1}, k) for k = 0..min(K, B)] from ONE DP run.
    Valid because cell k of every table depends only on cells <= k, so the
    run with budget K contains the runs with every smaller budget."""
    return _solve_v2(tree, K, False, pairs=(method == 'pairs'), profile=True)
