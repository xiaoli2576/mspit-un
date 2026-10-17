# Implementation Notes

This file documents modelling subtleties and design decisions made when
implementing the algorithms, together with the consistency checks that tie
the code to the manuscript.

## 1. Worked-Example Weights

The worked example uses the weight vector w(e2..e10) = (6, 7, 4, 1, 8, 4,
3, 4, 5) with u(e) = 10 for every edge (an earlier draft contained a
transcription error in this vector; the values here are the ones consistent
with the g-table, e.g. g(chi_v3, 0, 0) = w(e3) = 7 and
g(chi_v6, 1, 1) = u(e5) + w(e6) = 18, which forces w(e5) = 1, w(e6) = 8).
`paper_tables.py` verifies every table cell of the worked example (Online Resource 1)
against an independent brute-force enumeration.

## 2. Lemma 1 Reordering Scope

The paper states we can reorder edges with the same layer number on the same
path by Dw descending. However, the CA node is NOT subject to this reordering:

- The CA node (v in V* U {v1}) is fixed at position 0 on the chain.
- The CA's OUTGOING edge is FIXED (it connects CA to the first intermediate).
- Only the INTERMEDIATE nodes (degree-2 nodes between CA and tip) can be
  freely reordered.

**Implementation**: In `_build_chain` (tree.py), edges[0] = CA's outgoing
edge (FIXED), edges[1..] = intermediate edges sorted by Dw desc.

**Consequence**: g(eps=1, k=1) = w_sum + delta[0] where delta[0] = Dw of the
CA's actual outgoing edge -- NOT necessarily the maximum Dw on the chain.
The abstract "CA takes the best slot" formula is only valid after the CA's
edge is correctly placed at position 0.

**Example of incorrect vs correct**:
  Chain v1->v2->v5 with Dw(v1->v2)=5, Dw(v2->v5)=10:
  - WRONG (sort all): delta=[10,5], g(eps=1,k=1)=2+10=12 (overestimates!)
  - CORRECT: delta=[5,10], g(eps=1,k=1)=2+5=7

## 3. Physical Upgrade Model

The problem definition uses A(v) = "set of edges adjacent to v" but the model
is "edge (parent, child) gets u(e) iff parent in S" (forward node model).

This means:
  - Upgrading leaf v upgrades ONLY edges from v to v's children.
    Since leaves have no children, upgrading a leaf has NO effect.
  - Upgrading an internal node v upgrades all edges from v to its children.

The brute-force verifier (`brute.py`) correctly implements this parent-only
model.

## 4. Budget Counting (Non-Leaf Nodes)

The budget equation for f(T^{q:q}_v, eps, k):
  k1 + k2 = k
where:
  k1 = upgrades on chain chi_{v_hq} (0 <= k1 <= min(beta, K))
  k2 = upgrades on subtree T_{v_hq} (0 <= k2 <= nl_subtree(v_hq))

IMPORTANT: nl_subtree(v_hq) INCLUDES v_hq itself. This is because v_hq can
be upgraded as the CA (critical ancestor) of its own sub-branches.

If v_hq has children (is a branching node), F_sub(v_hq, k2) with k2=1 means
"upgrade v_hq as the CA of its own branches." This is valid and necessary for
the algorithm to find optimal solutions where a branching node is upgraded.

## 5. MCSPIT-UNu Binary Search Initialization

The binary search initialises k1=0, not k1=1: if F(T, 1) >= D the search
must still be able to return K* = 1, so the lower bookend has to start at
an infeasible budget. (The manuscript states this in a remark next to the
algorithm.)

## 6. CD Ordering and the f-Table

The final optimum F(T_{v1}, K) does not depend on the ordering chosen for
CD(v) (the manuscript proves this), but intermediate f-values do. The
implementation and the manuscript tables both use ascending node-index
order, e.g. CD(v1) = (v2, v6, v7) in the worked example; `paper_tables.py`
generates the manuscript tables under exactly this ordering.

## 7. Chain Tip Node Not Upgraded Via g-Function

In the g-function, the "k nodes upgraded" are positions 0..k-1 of the
abstract chain, corresponding to the CA and up to (beta-1) intermediates.
The TIP node (v_h) is NEVER counted in g -- it is handled separately via
F_sub(v_hq, k2) which treats v_hq as a potential CA of its own sub-branches.

This separation is critical: the chain g-function handles the path FROM the
CA DOWN TO (but not including) the tip, and the tip's sub-tree budget is
handled independently.
