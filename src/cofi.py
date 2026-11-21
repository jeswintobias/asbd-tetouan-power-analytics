"""
ASBD Hackathon — COFI-tree mining + closed itemsets + closed (non-redundant) rules
Student: Jeswin Tobias J | Roll No: CS23I1013

COFI (Co-Occurrence Frequent-Item trees). Steps:
  1. Build an FP-tree (items in descending global support, header table + node links).
  2. For each frequent item i, least frequent first, build ONE small Co-Occurrence
     Frequent-Item tree from the FP-tree paths above i (only items more frequent than i
     that are locally frequent with i). Nodes carry (count, participation).
  3. Mine it: for each node, branch count = count − participation; every sub-pattern of
     the branch (∪ {i}) gets that count; then participation += branch count along the path.
  4. Discard the COFI-tree → only one COFI-tree is in memory at a time.
Closed itemsets and Min-Max rule bases (non-redundant exact + approximate rules).
"""
from collections import Counter, defaultdict
from itertools import combinations
import time

ENUMERATED = [0]          # sub-patterns enumerated from COFI branches (instrumentation only)


class _Node:
    __slots__ = ('item', 'count', 'part', 'parent', 'children', 'link')

    def __init__(self, item, parent):
        self.item, self.count, self.part, self.parent = item, 0, 0, parent
        self.children, self.link = {}, None


def _insert(root, path, count, header, tails):
    node = root
    for it in path:
        child = node.children.get(it)
        if child is None:
            child = node.children[it] = _Node(it, node)
            if it in tails:
                tails[it].link = child
            else:
                header[it] = child
            tails[it] = child
        child.count += count
        node = child


def _chain(header, it):
    n = header.get(it)
    while n is not None:
        yield n
        n = n.link


def mine_item(i, paths, min_cnt, rank):
    """Build and mine the COFI-tree of item i.

    paths: [(items more frequent than i in one transaction/FP-path, count), ...]
    Returns ({frozenset pattern containing i: count}, COFI-tree size). Used by the
    single-machine COFI and, unchanged, by the PySpark version (one task per item).
    """
    local = defaultdict(int)
    for anc, c in paths:
        for j in anc:
            local[j] += c
    order = sorted((j for j in local if local[j] >= min_cnt), key=lambda j: (-local[j], rank[j]))
    if not order:
        return {}, 0
    pos = {j: k for k, j in enumerate(order)}
    croot, ch, ct = _Node(i, None), {}, {}
    for anc, c in paths:                                   # build the COFI-tree of i
        p = sorted((j for j in anc if j in pos), key=pos.get)
        if p:
            _insert(croot, p, c, ch, ct)
    size = sum(1 for j in order for _ in _chain(ch, j))

    # mine: deepest (least locally frequent) items first, so descendants are done before ancestors
    cand = defaultdict(int)
    for j in reversed(order):
        for m in _chain(ch, j):
            bc = m.count - m.part
            if bc <= 0:
                continue
            branch, q = [], m
            while q is not croot:
                branch.append(q.item); q = q.parent
            ENUMERATED[0] += (1 << len(branch)) - 1      # 2^len - 1 sub-patterns of this branch
            for r in range(1, len(branch) + 1):
                for sub in combinations(branch, r):
                    cand[sub] += bc
            q = m
            while q is not croot:
                q.part += bc; q = q.parent
    out = {frozenset(sub) | {i}: c for sub, c in cand.items() if c >= min_cnt}
    return out, size                                       # tree goes out of scope -> discarded


def cofi(transactions, min_sup):
    """All frequent itemsets via COFI-trees. Returns ({frozenset: count}, stats)."""
    t0 = time.perf_counter()
    ENUMERATED[0] = 0
    # one scan: merge identical transactions (52 k rows -> ~4 k distinct); everything after works on these
    W = Counter(tuple(sorted(t)) for t in transactions)
    N = len(transactions)
    min_cnt = min_sup * N
    sup = defaultdict(int)                                 # global supports, from the merged transactions
    for t, w in W.items():
        for it in t:
            sup[it] += w
    F1 = sorted((i for i in sup if sup[i] >= min_cnt), key=lambda i: (-sup[i], i))
    rank = {it: r for r, it in enumerate(F1)}

    # FP-tree: frequent items of each distinct transaction, in descending support
    root, header, tails = _Node(None, None), {}, {}
    merged = defaultdict(int)
    for t, w in W.items():
        merged[tuple(sorted((i for i in t if i in rank), key=rank.get))] += w
    for path, c in merged.items():
        _insert(root, path, c, header, tails)
    fp_nodes = sum(1 for it in F1 for _ in _chain(header, it))

    freq = {frozenset([i]): sup[i] for i in F1}
    cofi_sizes = {}
    for i in reversed(F1):                                 # least frequent item first
        paths = []
        for n in _chain(header, i):                        # FP-tree paths above every node of i
            anc, p = [], n.parent
            while p.item is not None:
                anc.append(p.item); p = p.parent
            paths.append((anc, n.count))
        pats, cofi_sizes[i] = mine_item(i, paths, min_cnt, rank)
        freq.update(pats)                                  # COFI-tree of i is discarded inside mine_item
    stats = {'time_s': time.perf_counter() - t0, 'frequent': len(freq), 'fp_tree_nodes': fp_nodes,
             'subpatterns_enumerated': ENUMERATED[0],
             'max_cofi_nodes': max(cofi_sizes.values()), 'total_cofi_nodes': sum(cofi_sizes.values())}
    return freq, stats, cofi_sizes


def closed_and_maximal(freq):
    """Closed: no immediate superset with equal support. Maximal: no frequent immediate superset."""
    closed, maximal = set(freq), set(freq)
    for Y, cy in freq.items():
        if len(Y) < 2:
            continue
        for a in Y:
            X = Y - {a}
            maximal.discard(X)
            if freq.get(X) == cy:
                closed.discard(X)
    return {X: freq[X] for X in closed}, {X: freq[X] for X in maximal}


def closure(X, closed, freq):
    """Smallest closed superset of X with the same support (Galois closure f(X) = i(t(X)))."""
    best = X
    for C, c in closed.items():
        if c == freq[X] and X <= C and len(C) > len(best):
            best = C
    return best


def minimal_generators(freq, N):
    """X is a minimal generator if every immediate subset has strictly higher support."""
    gens = []
    for X, c in freq.items():
        subs = [N] if len(X) == 1 else [freq[X - {a}] for a in X]
        if all(s > c for s in subs):
            gens.append(X)
    return gens


def minmax_rules(freq, closed, N, min_conf):
    """Min-Max bases — non-redundant rules from closed itemsets.
    Exact:       G -> C \\ G   (G minimal generator of closed C, confidence = 100 %)
    Approximate: G -> Z \\ G   (Z closed, Z ⊃ closure(G), confidence = sup(Z)/sup(G) >= min_conf)"""
    gens = minimal_generators(freq, N)
    rows = []
    for G in gens:
        C = closure(G, closed, freq)
        if C != G:
            rows.append({'antecedent': G, 'consequent': C - G, 'support': freq[C] / N, 'confidence': 1.0, 'type': 'exact'})
        for Z, cz in closed.items():
            if C < Z:
                conf = cz / freq[G]
                if conf >= min_conf:
                    rows.append({'antecedent': G, 'consequent': Z - G, 'support': cz / N, 'confidence': conf, 'type': 'approximate'})
    return rows, gens
