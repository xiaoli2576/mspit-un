# MSPIT-UNu / MCSPIT-UNu Implementation

Python implementation of two algorithms from the paper on network interdiction
on trees with node upgrades (unit cost).

## Problem Description

Given a rooted tree T=(V,E,w) where each edge has an original weight w(e) and
an upgraded weight u(e) >= w(e):

**MSPIT-UNu**: Choose at most K nodes to upgrade (S, |S|<=K) to maximise the
minimum root-to-leaf distance. An edge (parent, child) gets weight u(e) if the
parent is in S.

**MCSPIT-UNu**: Find the minimum number of nodes to upgrade so that the minimum
root-to-leaf distance is at least D.

## Files

| File | Description |
|------|-------------|
| `tree.py` | Tree data structure: BFS, Layer/CD/CA computation, chain construction |
| `mspit_un.py` | Algorithm 1 for MSPIT-UNu: O(nK) DP with size-capped merge (default); the earlier loops are kept as `method='v1'` / `'kk'` |
| `mcspit_un.py` | MCSPIT-UNu: one DP run at K=n (`solve_mcspit_unu_oneshot`, O(n^2)) and the earlier binary search (`solve_mcspit_unu`) |
| `brute.py` | Brute-force verifier for small trees (enumerates all subsets) |
| `example.py` | Reproduces the worked example from the paper (10-node tree) |
| `paper_tables.py` | Verifies EVERY appendix table cell against an independent brute force; emits the LaTeX tables |
| `experiments.py` | Timing harness (random recursive trees + structured stress families); archives raw JSON to `results/` |
| `gen_latex_tables.py` | Generates the manuscript experiment tables from an archived results JSON |
| `test_basic.py` | Unit tests and brute-force cross-checks |
| `verify_onk.py` | Checks the O(nK) DP against brute force and cell by cell against the earlier DP |
| `bench_onk.py` | Timing runs for the v1.1.0 manuscript (writes `results/onk-bench-20261016.json`) |
| `results/` | Raw timing archives (JSON + log); canonical source of every number in the paper |

## Quick Start

```python
from tree import Tree
from mspit_un import solve_mspit_unu
from mcspit_un import solve_mcspit_unu

# Build a tree: adjacency list adj[u] = [(v, w, u_weight), ...]
n = 5
adj = [[] for _ in range(n)]
def add_edge(u, v, w, uw):
    adj[u].append((v, w, uw))
    adj[v].append((u, w, uw))

add_edge(0, 1, 2, 8)   # v1-v2: w=2, u=8
add_edge(0, 2, 3, 7)   # v1-v3: w=3, u=7
add_edge(1, 3, 4, 9)   # v2-v4: w=4, u=9
add_edge(1, 4, 5, 10)  # v2-v5: w=5, u=10

t = Tree(n=n, adj=adj, root=0)

# MSPIT-UNu: maximise min root-leaf distance with K=2 upgrades
K = 2
opt_val, S = solve_mspit_unu(t, K)
print(f"F(T, K={K}) = {opt_val}")
print(f"Upgrade nodes: {sorted(v+1 for v in S)}")  # 1-indexed

# MCSPIT-UNu: minimum upgrades to achieve distance >= D
D = 10.0
K_star, val, S = solve_mcspit_unu(t, D)
print(f"Minimum upgrades for D={D}: K* = {K_star}, achieved = {val}")
```

## Running the Example

```
python example.py
```

Reproduces the worked example (F(T_{v1}, 1) = 13, S = {v1}), cross-checks
it by brute force, and verifies every cell of the paper's two appendix
tables against an independent enumeration (via `paper_tables.py`).

To regenerate the LaTeX sources of the appendix tables:

```
python paper_tables.py
```

## Running Tests

```
python test_basic.py
# or with pytest:
python -m pytest test_basic.py -v
```

Includes a 1000-instance randomized brute-force cross-check (n <= 14).

## Running Experiments

```
python experiments.py
```

Benchmarks both algorithms on random recursive trees (n = 50..2000,
seed = 42 + 1000n + trial) and on deterministic structured families
(path / comb / star / layered 4-ary). Raw results are archived to
`results/experiments-<timestamp>.json`; the manuscript tables are then
generated with

```
python gen_latex_tables.py results/experiments-<timestamp>.json
```

so every number in the paper's experiment section is traceable to an
archived JSON.

## Algorithm Details

### Tree Structure

- Nodes are 0-indexed internally (paper uses 1-indexed v1..vn).
- `Layer(v1) = 1`; Layer increases by 1 at branching nodes (degree > 2).
- `V*` = branching nodes (degree > 2, not root).
- `CD(v)` = critical descendants of v.
- `CA(v)` = critical ancestor of v.
- `chain chi_v` = path from CA(v) to v.

### Physical Upgrade Model

Edge (parent, child) gets weight u(e) if and only if `parent in S`.
This is the "forward node" model from the paper.

### g-Function

For chain chi_v_h with CA node v and beta edges:
- edges[0] = CA's outgoing edge (fixed at position 0).
- edges[1..] = intermediate edges sorted by Dw desc (Lemma 1).

```
delta[0] = Dw(CA's edge)
delta[1..beta-1] = Dw of intermediates, sorted descending

g(chi, eps=0, k) = w_sum + sum(delta[1:k+1])   (CA not upgraded)
g(chi, eps=1, k) = w_sum + sum(delta[0:k])      (CA upgraded)
```

### Complexity

- MSPIT-UNu (Algorithm 1): O(nK) time and space, hence O(n^2)
- MCSPIT-UNu: O(n^2) by one run with K = n

The timing numbers of the current manuscript come from `python bench_onk.py`
and are archived in `results/onk-bench-20261016.json`; `python verify_onk.py`
reproduces the correctness checks.

## Implementation Notes

See `NOTES.md` for modelling subtleties (chain reordering scope, budget
counting, CD ordering) and the consistency checks that tie the code to the
manuscript.
