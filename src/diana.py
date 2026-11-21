"""
ASBD Hackathon — DIANA (DIvisive ANAlysis) clustering, written from scratch
Student: Jeswin Tobias J | Roll No: CS23I1013

DIANA (DIvisive ANAlysis), divisive hierarchical clustering:
1. Start with all objects in one cluster.
2. Pick the cluster with the largest diameter (max pairwise distance).
3. Splinter: the object with the largest mean distance to the rest starts a new group.
   Repeatedly move the object whose (mean distance to the old group − mean distance to the
   splinter group) is largest and > 0. Stop when no object prefers the splinter group.
4. Repeat until every object is a singleton. Split height = diameter of the cluster split.
"""
import heapq

import numpy as np


def _split(D, idx, log=None, names=None):
    """Split cluster `idx` (array of indices) into (splinter, rest) by the DIANA rule."""
    sub = D[np.ix_(idx, idx)]
    m = len(idx)
    in_spl = np.zeros(m, bool)
    first = int(np.argmax(sub.sum(1) / (m - 1)))          # largest mean dissimilarity
    in_spl[first] = True
    if log is not None:
        log.append(f'  start splinter with {names[idx[first]]} (mean dist {sub[first].sum() / (m - 1):.3f})')
    while in_spl.sum() < m - 1:
        rest = ~in_spl
        n_rest = rest.sum()
        a_rest = (sub[:, rest].sum(1) - 0) / np.maximum(n_rest - 1, 1)   # mean dist to the other old-group members
        a_spl = sub[:, in_spl].mean(1)                                  # mean dist to splinter group
        diff = np.where(rest, a_rest - a_spl, -np.inf)
        j = int(np.argmax(diff))
        if diff[j] <= 0:
            break
        in_spl[j] = True
        if log is not None:
            log.append(f'  move {names[idx[j]]}: mean to old group {a_rest[j]:.3f} − to splinter {a_spl[j]:.3f} = {diff[j]:.3f} > 0')
    return idx[in_spl], idx[~in_spl]


def diana(D, trace=False, names=None):
    """Full DIANA hierarchy on a distance matrix D (n x n).

    Returns dict with:
      splits  – list of (height, parent_members, splinter, rest) in the order performed (largest diameter first)
      linkage – SciPy-format linkage matrix (for dendrogram / cophenetic correlation)
      dc      – divisive coefficient
      log     – text trace (if trace=True)
    """
    n = len(D)
    names = names or [str(i) for i in range(n)]
    log = [] if trace else None
    diam = lambda ix: float(D[np.ix_(ix, ix)].max()) if len(ix) > 1 else 0.0

    root = np.arange(n)
    heap = [(-diam(root), 0, root)]                         # max-heap on diameter
    counter, splits, children = 1, [], {}
    node_members, node_height = {0: root}, {0: diam(root)}
    single_at = np.zeros(n)                                 # diameter of the cluster an object last left
    while heap:
        neg_d, nid, idx = heapq.heappop(heap)
        if len(idx) < 2:
            continue
        if log is not None:
            log.append(f'Split cluster {{{", ".join(names[i] for i in idx)}}} (diameter {-neg_d:.3f})')
        a, b = _split(D, idx, log, names)
        splits.append((-neg_d, idx, a, b))
        kids = []
        for part in (a, b):
            cid, counter = counter, counter + 1
            node_members[cid], node_height[cid] = part, diam(part)
            kids.append(cid)
            if len(part) == 1:
                single_at[part[0]] = -neg_d
            else:
                heapq.heappush(heap, (-node_height[cid], cid, part))
        children[nid] = kids
        if log is not None:
            log.append(f'  -> {{{", ".join(names[i] for i in a)}}} | {{{", ".join(names[i] for i in b)}}}')

    # SciPy linkage via post-order traversal (children always created before parents)
    Z, sid = [], {}
    def build(nid):
        if nid not in children:
            return int(node_members[nid][0])
        l, r = (build(c) for c in children[nid])
        Z.append([l, r, node_height[nid], len(node_members[nid])])
        return n + len(Z) - 1
    build(0)

    dc = float(np.mean(1 - single_at / node_height[0]))
    return {'splits': splits, 'linkage': np.array(Z, float), 'dc': dc, 'log': log}


def cut(result, n, k):
    """Labels 0..k-1 after the first k-1 splits (= cutting the tree at the k largest diameters)."""
    labels = np.zeros(n, int)
    clusters = [np.arange(n)]
    for _, parent, a, b in result['splits'][:k - 1]:
        clusters = [c for c in clusters if not np.array_equal(np.sort(c), np.sort(parent))] + [a, b]
    clusters.sort(key=lambda c: c.min())                   # stable numbering by first day
    for lab, c in enumerate(clusters):
        labels[c] = lab
    return labels
