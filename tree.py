r"""
tree.py  -- Tree data structure for MSPIT-UNu / MCSPIT-UNu.

Terminology follows the paper exactly:
  Layer(v1) = 1
  Layer(v)  = Layer(father(v))     if deg(v) <= 2
            = Layer(father(v)) + 1 if deg(v) >  2
  V*        = {v in V \ {v1}  |  deg(v) > 2}
  CD(v_bar) : critical descendants of v_bar in V* U {v1}
  CA(v)     : critical ancestor of v
  chain chi_v : path from CA(v) to v  (a sequence of degree-2 nodes between
              two "branching" nodes)
  Left q-subtree T^{1:q}_v : union of first q critical-child subtrees
"""

from collections import deque


class Tree:
    """
    Rooted edge-weighted tree.

    Parameters
    ----------
    n        : number of nodes (labelled 0..n-1 internally; v1 == node 0)
    adj      : adjacency list  adj[u] = list of (v, w_uv, u_uv)
               where w_uv  = original edge weight
                     u_uv  = upgraded edge weight  (>= w_uv)
               The edge is directed *away from the root* in terms of
               parent–child relationships; adj stores undirected edges.
    root     : index of root (default 0)
    """

    def __init__(self, n, adj, root=0):
        self.n = n
        self.adj = adj          # adj[u] = [(v, w, u_weight), ...]
        self.root = root

        # Populated by build()
        self.parent = [None] * n       # parent[v] = (p, w, u_weight) or None
        self.children = [[] for _ in range(n)]  # children[v] = [(c, w, u_weight)]
        self.degree = [0] * n
        self.layer = [0] * n
        self.is_leaf = [False] * n
        self.Vstar = []                # nodes with deg > 2, excluding root
        self.branching = []            # V* ∪ {root}
        self.CD = {}                   # CD[v] = [list of critical descendants]
        self.CA = {}                   # CA[v] = critical ancestor of v
        self.chains = {}               # chains[v] = Chain object for v in V*\{v1} ∪ Y

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        """BFS from root: compute parent/children, degrees, layers, CD, chains."""
        root = self.root
        n = self.n

        # --- Step 1: orient tree (BFS to find parent/child relationships) ---
        visited = [False] * n
        visited[root] = True
        queue = deque([root])
        bfs_order = []

        while queue:
            u = queue.popleft()
            bfs_order.append(u)
            for (v, w, uw) in self.adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.parent[v] = (u, w, uw)
                    self.children[u].append((v, w, uw))
                    queue.append(v)

        # --- Step 2: degrees ---
        for u in range(n):
            self.degree[u] = len(self.adj[u])

        # Root's degree in the rooted sense = number of children (no parent)
        # But the paper uses the undirected degree for the Layer formula.
        # deg(root) = len(adj[root]) in the undirected tree.

        # --- Step 3: layers (BFS order guarantees parents are done first) ---
        self.layer[root] = 1
        for u in bfs_order[1:]:
            p = self.parent[u][0]
            if self.degree[u] <= 2:
                self.layer[u] = self.layer[p]
            else:
                self.layer[u] = self.layer[p] + 1

        # --- Step 4: leaves and V* ---
        for u in range(n):
            if u != root and len(self.children[u]) == 0:
                self.is_leaf[u] = True

        self.Vstar = [u for u in range(n)
                      if u != root and self.degree[u] > 2]
        self.branching = [root] + self.Vstar  # V* ∪ {v1}

        # --- Step 5: CD and CA ---
        # For each v̄ in V* ∪ {v1}, CD(v̄) consists of nodes v that are:
        #   (a) in V* \ {v1} with Layer(v) = Layer(v̄) + 1, or
        #   (b) in Y (leaf) with Layer(v) = Layer(v̄)
        # AND v̄ lies on the path from v1 to v.

        for vbar in self.branching:
            self.CD[vbar] = []

        # Walk from each non-root node upward to find its critical ancestor.
        # v ∈ V* \ {v1}  → CA(v) is the nearest ancestor u in V*∪{v1} with
        #                    Layer(u) = Layer(v) - 1
        # v ∈ Y          → CA(v) is the nearest ancestor u in V*∪{v1} with
        #                    Layer(u) = Layer(v)
        branching_set = set(self.branching)

        for v in range(n):
            if v == root:
                continue
            if self.is_leaf[v] or v in branching_set:
                # Walk up to find critical ancestor
                ca = self._find_ca(v)
                self.CA[v] = ca
                if ca is not None:
                    self.CD[ca].append(v)

        # --- Step 6: For V*∪{v1}, order CD children so that within each
        #             "layer group" on a chain, Δw is sorted descending.
        #             (Lemma 1 guarantees we can do this without loss of
        #             generality.)  The reordering is done inside chain
        #             construction below.

        # --- Step 7: Build chains ---
        # χ_v = path from CA(v) to v, including both endpoints.
        # For v in V*\{v1} ∪ Y.
        for v in range(n):
            if v == root:
                continue
            if self.is_leaf[v] or v in branching_set:
                self.chains[v] = self._build_chain(v)

    # ------------------------------------------------------------------
    def _find_ca(self, v):
        """
        Walk up the tree from v to find the critical ancestor of v.

        Rules (Definition 3 in the paper):
          - if v in V* (not v1): CA(v) is the nearest ancestor u_bar with
                                Layer(u_bar) = Layer(v) - 1
          - if v in Y:          CA(v) is the nearest ancestor u_bar with
                                Layer(u_bar) = Layer(v)
        In both cases u_bar must be in V* U {v1}.
        """
        root = self.root
        branching_set = set(self.branching)
        is_vstar = (v in branching_set and v != root)
        is_leaf = self.is_leaf[v]

        target_layer = self.layer[v] - 1 if is_vstar else self.layer[v]

        cur = self.parent[v][0]
        while cur is not None:
            if cur in branching_set and self.layer[cur] == target_layer:
                return cur
            if cur == root:
                break
            cur = self.parent[cur][0]
        # If we reach root without matching, root is the CA
        return root

    # ------------------------------------------------------------------
    def _build_chain(self, v):
        """
        Build the chain χ_v = path from CA(v) to v.

        The chain is returned as a Chain object whose edges are in order
        from CA(v) downward toward v, with Δw sorted descending within
        the chain (Lemma 1 reordering).
        """
        ca = self.CA[v]
        # Collect path from v up to ca (exclusive of ca node's edge above it)
        path_nodes = []   # nodes from v up to ca (inclusive)
        cur = v
        while cur != ca:
            path_nodes.append(cur)
            cur = self.parent[cur][0]
        path_nodes.append(ca)
        path_nodes.reverse()  # now: ca, ..., v

        # Edges in chain: (parent_node, child_node, w, u_weight)
        edges = []
        for i in range(len(path_nodes) - 1):
            par = path_nodes[i]
            child = path_nodes[i + 1]
            # find edge weight
            for (nb, w, uw) in self.adj[par]:
                if nb == child:
                    edges.append((par, child, w, uw))
                    break

        # Lemma 1: sort INTERMEDIATE edges by Dw descending.
        # The CA's outgoing edge (edges[0]) is FIXED at position 0:
        # upgrading the CA always activates its own outgoing edge.
        # Edges 1..beta-1 are intermediate edges owned by degree-2 nodes,
        # which can be freely reordered by Lemma 1 (swap which intermediate
        # node is adjacent to CA), so sort only those by Dw desc.
        if len(edges) > 1:
            ca_edge = edges[0]          # fixed: CA's outgoing edge
            int_edges = edges[1:]       # intermediate edges (degree-2 nodes)
            int_edges.sort(key=lambda e: (e[3] - e[2]), reverse=True)
            edges = [ca_edge] + int_edges

        return Chain(ca=ca, tip=v, edges=edges)

    # ------------------------------------------------------------------
    def internal_count(self, node_set):
        """
        Count internal (non-leaf) nodes in a node set.
        This is |V(substructure)| - |Y ∩ V(substructure)|.
        """
        return sum(1 for v in node_set if not self.is_leaf[v])

    # ------------------------------------------------------------------
    def __repr__(self):
        lines = [f"Tree(n={self.n}, root={self.root})"]
        lines.append(f"  Layers: {self.layer}")
        lines.append(f"  V*:     {self.Vstar}")
        lines.append(f"  CD:     {self.CD}")
        return "\n".join(lines)


class Chain:
    """
    Represents a chain chi_v in the tree.

    Attributes
    ----------
    ca    : critical ancestor (top of chain)
    tip   : bottom of chain (v in V*\\{v1} or v in Y)
    edges : list of (parent, child, w, u_weight).
            edges[0] = CA's outgoing edge (fixed).
            edges[1..] = intermediate edges sorted by Dw desc (Lemma 1).
    beta  : len(edges)  (number of edges = beta in the paper)
    w_sum : sum of original weights
    delta : list of Dw values: [Dw(CA's edge), sorted Dw of intermediates]
    """

    def __init__(self, ca, tip, edges):
        self.ca = ca
        self.tip = tip
        self.edges = edges          # sorted by Δw desc
        self.beta = len(edges)
        self.w_sum = sum(e[2] for e in edges)
        self.delta = [e[3] - e[2] for e in edges]  # Δw, sorted desc

    def __repr__(self):
        return (f"Chain(ca={self.ca}, tip={self.tip}, β={self.beta}, "
                f"w_sum={self.w_sum}, δ={self.delta})")
