"""
weighted_node_costs.py -- prototypes for MSPIT-UN with NON-UNIT node costs.

Self-contained (does not import or modify the unit-cost code).  Model:
  rooted tree, children[v] = [(child, w, u)], u >= w >= 0 (integers),
  node cost c[v] > 0 (integers).  Upgrading v sets every edge v->child to u.
  Leaves are never worth upgrading (no outgoing edge).

Weighted sum-Hamming:
  MAX   : max_{S: c(S)<=B}  min_leaf d_S(root, leaf)
  MINC  : min_{S: d_S >= T} c(S)
Node-l1 (Section 2 of the note): x_v in [0,1], edge e in A(v) gets w+Delta*x_v,
  cost sum c_v x_v  -> an LP (solved with scipy for checking only).

Algorithms here:
  dp_max_budget    : F_v(b), O(nB) via monotone (max,min)-convolution
  dp_min_cost      : G_v(t), O(n * T) value-indexed DP (integer weights)
  fptas_max        : (1-eps)-approx for MAX, budget respected exactly
  fptas_mincost    : (1+eps)-approx for MINC, distance target respected exactly
  brute_*          : enumeration over subsets of non-leaf nodes
Run:  python weighted_node_costs.py
"""
import itertools
import math
import random

INF = float('inf')


# ---------------------------------------------------------------- instances
def random_instance(n, rng, cmax=6, wmax=6, dmax=8, homogeneous=False):
    children = [[] for _ in range(n)]
    for v in range(1, n):
        p = rng.randrange(v)
        children[p].append(v)
    ch = [[] for _ in range(n)]
    for v in range(n):
        dv = rng.randint(0, dmax)
        for c in children[v]:
            w = rng.randint(0, wmax)
            d = dv if homogeneous else rng.randint(0, dmax)
            ch[v].append((c, w, w + d))
    cost = [rng.randint(1, cmax) for _ in range(n)]
    return ch, cost


def internal(ch):
    return [v for v in range(len(ch)) if ch[v]]


def value(ch, S, v=0):
    """min distance from v to a leaf below v under upgrade set S."""
    if not ch[v]:
        return 0
    return min((u if v in S else w) + value(ch, S, c) for c, w, u in ch[v])


def height(ch, v=0):
    return 0 if not ch[v] else 1 + max(height(ch, c) for c, _, _ in ch[v])


# ---------------------------------------------------------------- brute force
def brute_max(ch, cost, B):
    best, bestS = -INF, None
    I = internal(ch)
    for r in range(len(I) + 1):
        for S in itertools.combinations(I, r):
            if sum(cost[v] for v in S) <= B:
                val = value(ch, set(S))
                if val > best:
                    best, bestS = val, set(S)
    return best, bestS


def brute_mincost(ch, cost, T):
    best = INF
    I = internal(ch)
    for r in range(len(I) + 1):
        for S in itertools.combinations(I, r):
            cs = sum(cost[v] for v in S)
            if cs < best and value(ch, set(S)) >= T:
                best = cs
    return best


# ---------------------------------------------------------------- budget DP
def maxmin_conv(h1, h2):
    """H(b) = max_{b1+b2<=b} min(h1(b1), h2(b2)) for nondecreasing arrays of
    equal length L.  O(L) via the inverse (min-budget) functions:
    H(b) = max{t : g1(t)+g2(t) <= b}, g_i(t) = min{b : h_i(b) >= t}."""
    L = len(h1)
    vals = sorted(set(h1) | set(h2))
    H = [-INF] * L
    # pointers: g1(t), g2(t) nondecreasing in t
    p1 = p2 = 0
    for t in vals:  # increasing
        while p1 < L and h1[p1] < t:
            p1 += 1
        while p2 < L and h2[p2] < t:
            p2 += 1
        if p1 >= L or p2 >= L:
            break
        b = p1 + p2
        if b < L:
            H[b] = max(H[b], t)
    for b in range(1, L):  # budget "<= b"
        H[b] = max(H[b], H[b - 1])
    return H


def dp_max_budget(ch, cost, B):
    """Return F_root as a list of length B+1 (F[b] = OPT with budget b)."""
    def F(v):
        if not ch[v]:
            return [0] * (B + 1)
        sub = [(F(c), w, u) for c, w, u in ch[v]]
        # case v not upgraded
        acc0 = None
        acc1 = None
        for f, w, u in sub:
            a0 = [x + w for x in f]
            a1 = [x + u for x in f]
            acc0 = a0 if acc0 is None else maxmin_conv(acc0, a0)
            acc1 = a1 if acc1 is None else maxmin_conv(acc1, a1)
        out = []
        for b in range(B + 1):
            best = acc0[b]
            if b >= cost[v]:
                best = max(best, acc1[b - cost[v]])
            out.append(best)
        return out
    return F(0)


# ---------------------------------------------------------------- value DP
def dp_min_cost_table(ch, cost, Tmax):
    """G[t] for t=0..Tmax: min cost so that every root-leaf distance >= t."""
    def G(v):
        if not ch[v]:
            return [0] + [INF] * Tmax
        g0 = [0] * (Tmax + 1)
        g1 = [cost[v]] * (Tmax + 1)
        for c, w, u in ch[v]:
            gc = G(c)
            for t in range(Tmax + 1):
                g0[t] += gc[t - w] if t - w > 0 else 0
                g1[t] += gc[t - u] if t - u > 0 else 0
        return [min(a, b) for a, b in zip(g0, g1)]
    return G(0)


# ---------------------------------------------------------------- FPTAS (max)
def _threshold_min_cost(ch, cost, X):
    """min cost of S s.t. every root-leaf path has an edge of weight >= X."""
    def C(v):
        if not ch[v]:
            return INF
        s0 = sum(0 if w >= X else C(c) for c, w, u in ch[v])
        s1 = cost[v] + sum(0 if u >= X else C(c) for c, w, u in ch[v])
        return min(s0, s1)
    return C(0)


def fptas_max(ch, cost, B, eps):
    """Return a set S with c(S)<=B and value(S) >= (1-eps) OPT."""
    cands = sorted({w for v in range(len(ch)) for _, w, _ in ch[v]} |
                   {u for v in range(len(ch)) for _, _, u in ch[v]})
    Xs = [X for X in cands if _threshold_min_cost(ch, cost, X) <= B]
    Xstar = max(Xs)
    if Xstar <= 0:
        return set()
    h = height(ch)
    mu = eps * Xstar / h
    U = int(math.ceil(h * h / eps)) + 1      # OPT' <= OPT/mu <= h^2/eps
    # value DP on rounded weights with reconstruction
    rw = [[(c, int(w // mu), int(u // mu)) for c, w, u in ch[v]] for v in range(len(ch))]
    memo = {}

    def G(v):
        if v in memo:
            return memo[v]
        if not rw[v]:
            res = ([0] + [INF] * U, None)
        else:
            g0 = [0] * (U + 1)
            g1 = [cost[v]] * (U + 1)
            for c, w, u in rw[v]:
                gc = G(c)[0]
                for t in range(U + 1):
                    g0[t] += gc[t - w] if t - w > 0 else 0
                    g1[t] += gc[t - u] if t - u > 0 else 0
            choice = [1 if g1[t] < g0[t] else 0 for t in range(U + 1)]
            res = ([min(a, b) for a, b in zip(g0, g1)], choice)
        memo[v] = res
        return res

    groot = G(0)[0]
    tstar = max(t for t in range(U + 1) if groot[t] <= B)
    S = set()

    def rec(v, t):
        if not rw[v] or t <= 0:
            return
        up = G(v)[1][t]
        if up:
            S.add(v)
        for c, w, u in rw[v]:
            rec(c, t - (u if up else w))
    rec(0, tstar)
    return S


# ---------------------------------------------------------------- FPTAS (min cost)
def fptas_mincost(ch, cost, T, eps):
    """Return S with value(S) >= T and c(S) <= (1+eps) OPT (None if infeasible)."""
    n = len(ch)
    I = internal(ch)
    if value(ch, set(I)) < T:
        return None
    if value(ch, set()) >= T:
        return set()
    best, bestS = INF, None
    for g in sorted({cost[v] for v in I}):
        allowed = [v for v in I if cost[v] <= g]
        mu = eps * g / n
        cr = {v: int(math.ceil(cost[v] / mu)) for v in allowed}
        Bmax = sum(cr.values())
        # budget DP over rounded costs, naive (max,min) merge with argmax record
        memo = {}

        def F(v):
            if v in memo:
                return memo[v]
            if not ch[v]:
                memo[v] = ([0] * (Bmax + 1), None)
                return memo[v]
            def combine(delta_sel):
                acc, splits = None, []
                for c, w, u in ch[v]:
                    f = [x + delta_sel(w, u) for x in F(c)[0]]
                    if acc is None:
                        acc = f
                        splits.append(None)
                        continue
                    new, arg = [], []
                    for b in range(Bmax + 1):
                        bv, ba = -INF, 0
                        for b1 in range(b + 1):
                            val = min(acc[b1], f[b - b1])
                            if val > bv:
                                bv, ba = val, b1
                        new.append(bv)
                        arg.append(ba)
                    acc = new
                    splits.append(arg)
                return acc, splits
            a0, s0 = combine(lambda w, u: w)
            if v in cr:
                a1, s1 = combine(lambda w, u: u)
            else:
                a1, s1 = [-INF] * (Bmax + 1), None
            out, ch_ = [], []
            for b in range(Bmax + 1):
                x0 = a0[b]
                x1 = a1[b - cr[v]] if (v in cr and b >= cr[v]) else -INF
                out.append(max(x0, x1))
                ch_.append(1 if x1 > x0 else 0)
            memo[v] = (out, (ch_, s0, s1))
            return memo[v]

        Froot = F(0)[0]
        bs = [b for b in range(Bmax + 1) if Froot[b] >= T]
        if not bs:
            continue
        S = set()

        def rec(v, b):
            if not ch[v]:
                return
            chs, s0, s1 = F(v)[1]
            up = chs[b]
            if up:
                S.add(v)
                b -= cr[v]
            splits = s1 if up else s0
            kids = ch[v]
            # unwind left-to-right merges from the last child backwards
            budgets = [0] * len(kids)
            rem = b
            for i in range(len(kids) - 1, 0, -1):
                b1 = splits[i][rem]
                budgets[i] = rem - b1
                rem = b1
            budgets[0] = rem
            for (c, _, _), bc in zip(kids, budgets):
                rec(c, bc)
        rec(0, bs[0])
        cs = sum(cost[v] for v in S)
        assert value(ch, S) >= T
        if cs < best:
            best, bestS = cs, S
    return bestS


# ---------------------------------------------------------------- node-l1 LP
def l1_lp_max(ch, cost, B):
    from scipy.optimize import linprog
    I = internal(ch)
    idx = {v: i for i, v in enumerate(I)}
    m = len(I)
    rows, rhs = [], []

    def walk(v, coeffs, base):
        if not ch[v]:
            row = [0.0] * (m + 1)
            for j, a in coeffs.items():
                row[j] -= a
            row[m] = 1.0  # z - sum Delta x <= w(P)
            rows.append(row)
            rhs.append(base)
            return
        for c, w, u in ch[v]:
            cc = dict(coeffs)
            cc[idx[v]] = cc.get(idx[v], 0) + (u - w)
            walk(c, cc, base + w)
    walk(0, {}, 0.0)
    rows.append([float(cost[v]) for v in I] + [0.0])
    rhs.append(float(B))
    obj = [0.0] * m + [-1.0]
    res = linprog(obj, A_ub=rows, b_ub=rhs,
                  bounds=[(0, 1)] * m + [(None, None)], method='highs')
    return -res.fun, res.x


def l1_split_edge_lp_max(ch, cost, B):
    """Homogeneous case: split v -> v' with interior edge (0, delta_v),
    cost per unit c_v/delta_v (ZhangQ20 edge-l1 model), solved as an LP."""
    from scipy.optimize import linprog
    I = [v for v in internal(ch) if ch[v][0][2] - ch[v][0][1] > 0]
    idx = {v: i for i, v in enumerate(I)}
    m = len(I)
    rows, rhs = [], []

    def walk(v, cols, base):
        if not ch[v]:
            row = [0.0] * (m + 1)
            for j in cols:
                row[j] = -1.0
            row[m] = 1.0
            rows.append(row)
            rhs.append(base)
            return
        for c, w, u in ch[v]:
            walk(c, cols + ([idx[v]] if v in idx else []), base + w)
    walk(0, [], 0.0)
    # y_v in [0, delta_v], cost (c_v/delta_v) y_v
    rows.append([cost[v] / (ch[v][0][2] - ch[v][0][1]) for v in I] + [0.0])
    rhs.append(float(B))
    res = linprog([0.0] * m + [-1.0], A_ub=rows, b_ub=rhs,
                  bounds=[(0, ch[v][0][2] - ch[v][0][1]) for v in I] + [(None, None)],
                  method='highs')
    return -res.fun


# ---------------------------------------------------------------- tests
def main():
    rng = random.Random(20261016)
    stats = dict(max_dp=0, max_conv_naive=0, mincost=0, fptas_max=0,
                 fptas_min=0, bottleneck=0, l1_homog=0, l1_vs_int=0, trials=0)
    N = 400
    for trial in range(N):
        n = rng.randint(2, 11)
        ch, cost = random_instance(n, rng)
        I = internal(ch)
        totc = sum(cost[v] for v in I)
        B = rng.randint(0, totc + 1)
        # 1. budget DP vs brute
        opt, _ = brute_max(ch, cost, B)
        F = dp_max_budget(ch, cost, B)
        stats['max_dp'] += (F[B] == opt)
        # 1b. maxmin_conv vs naive on random monotone arrays
        L = rng.randint(1, 8)
        h1 = sorted(rng.randint(0, 9) for _ in range(L))
        h2 = sorted(rng.randint(0, 9) for _ in range(L))
        naive = [max(min(h1[i], h2[j]) for i in range(L) for j in range(L) if i + j <= b)
                 for b in range(L)]
        stats['max_conv_naive'] += (naive == maxmin_conv(h1, h2))
        # 2. min-cost DP vs brute
        Tmax = value(ch, set(I)) + 2
        G = dp_min_cost_table(ch, cost, Tmax)
        T = rng.randint(0, Tmax)
        stats['mincost'] += (G[T] == brute_mincost(ch, cost, T))
        # 3. FPTAS (max)
        eps = rng.choice([0.5, 0.25, 0.1])
        S = fptas_max(ch, cost, B, eps)
        ok = sum(cost[v] for v in S) <= B and value(ch, S) >= (1 - eps) * opt - 1e-9
        stats['fptas_max'] += ok
        # 4. FPTAS (min cost)
        bc = brute_mincost(ch, cost, T)
        S2 = fptas_mincost(ch, cost, T, eps) if n <= 9 else None
        if n <= 9:
            ok2 = (S2 is None and bc == INF) or (
                S2 is not None and value(ch, S2) >= T and
                sum(cost[v] for v in S2) <= (1 + eps) * bc + 1e-9)
        else:
            ok2 = True
        stats['fptas_min'] += ok2
        # 5. bottleneck: max c(v) over S <= B  ->  S = {v: c(v) <= B}
        Sb = {v for v in I if cost[v] <= B}
        best_b = max(value(ch, set(S)) for r in range(len(I) + 1)
                     for S in itertools.combinations(I, r)
                     if all(cost[v] <= B for v in S))
        stats['bottleneck'] += (value(ch, Sb) == best_b)
        # 6. l1: LP optimum >= integer (Hamming) optimum; homogeneous split equality
        lpv, _ = l1_lp_max(ch, cost, B)
        stats['l1_vs_int'] += (lpv >= opt - 1e-7)
        chh, costh = random_instance(n, rng, homogeneous=True)
        Bh = rng.randint(0, sum(costh) + 1)
        a, _ = l1_lp_max(chh, costh, Bh)
        b = l1_split_edge_lp_max(chh, costh, Bh)
        stats['l1_homog'] += abs(a - b) < 1e-6
        stats['trials'] += 1
    print(stats)


if __name__ == '__main__':
    main()
