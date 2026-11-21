"""
ASBD Hackathon — Association-rule helpers (DIC + rule generation)
Student: Jeswin Tobias J | Roll No: CS23I1013

DIC (Dynamic Itemset Counting) written from scratch.
Itemset states:  DC = dashed circle (counting, below minsup)
                 DS = dashed square (counting, already >= minsup)
                 SC = solid circle  (counted through all data, infrequent)
                 SS = solid square  (counted through all data, frequent)
"""
from itertools import combinations
import time

import numpy as np
import pandas as pd


def encode(transactions):
    """List of item lists -> (bool matrix N x I, item names)."""
    items = sorted({i for t in transactions for i in t})
    pos = {it: k for k, it in enumerate(items)}
    X = np.zeros((len(transactions), len(items)), dtype=bool)
    for r, t in enumerate(transactions):
        X[r, [pos[i] for i in t]] = True
    return X, items


def _count(block, itemset):
    """Transactions in `block` that contain every item of `itemset`."""
    return int(block[:, list(itemset)].all(axis=1).sum())


def dic(X, min_sup, M, trace=False, item_names=None, counter=None, shape=None):
    """Dynamic Itemset Counting.

    X: bool matrix (N transactions x I items); min_sup: fraction; M: transactions per station.
    counter: optional f(pos, end, itemsets) -> counts for rows [pos, end) — lets a distributed
             engine (PySpark) do the counting while this function keeps the DIC state machine.
    Returns (frequent {itemset: count}, checkpoint log DataFrame, stats dict).
    """
    N, I = shape if counter is not None else X.shape
    min_cnt = int(np.ceil(min_sup * N))
    # every 1-itemset starts as a dashed circle; the empty set is an implicit solid square
    state = {(i,): {'count': 0, 'seen': 0, 'st': 'DC'} for i in range(I)}
    pos, read, log, t0, max_active = 0, 0, [], time.perf_counter(), 0
    name = (lambda s: '{' + ','.join(item_names[i] for i in s) + '}') if item_names else str

    while any(v['st'][0] == 'D' for v in state.values()):
        end = min(pos + M, N)                         # a station every M rows (and at end of file)
        dashed = [s for s, v in state.items() if v['st'][0] == 'D']
        max_active = max(max_active, len(dashed))
        if counter is None:
            block = X[pos:end]
            counts = [_count(block, s) for s in dashed]
        else:
            counts = counter(pos, end, dashed)
        for s, c in zip(dashed, counts):              # only dashed itemsets ride the "train"
            state[s]['count'] += int(c)
            state[s]['seen'] += end - pos
        read += end - pos
        pos = end % N                                 # wrap around to the start of the file

        # --- station: update states ---
        for s in dashed:
            v = state[s]
            if v['st'] == 'DC' and v['count'] >= min_cnt:
                v['st'] = 'DS'                        # count can only grow -> already frequent
            if v['seen'] >= N:                        # one full lap -> stop counting
                v['st'] = 'SS' if v['count'] >= min_cnt else 'SC'

        # --- start counting supersets whose immediate subsets are all squares ---
        squares = {s for s, v in state.items() if v['st'] in ('DS', 'SS')}
        by_len = {}
        for s in squares:
            by_len.setdefault(len(s), []).append(s)
        new = []
        for k, group in by_len.items():
            group.sort()
            for a, b in combinations(group, 2):       # Apriori-style join on a shared (k-1)-prefix
                if a[:-1] != b[:-1]:
                    continue
                cand = a + (b[-1],) if a[-1] < b[-1] else b + (a[-1],)
                if cand in state:
                    continue
                if all(sub in squares for sub in combinations(cand, k)):
                    state[cand] = {'count': 0, 'seen': 0, 'st': 'DC'}
                    new.append(cand)

        row = {'station': len(log) + 1, 'read': read, 'passes': read / N, 'new_DC': len(new)}
        row.update({st: sum(v['st'] == st for v in state.values()) for st in ('DC', 'DS', 'SS', 'SC')})
        if trace:
            row['states'] = {name(s): v['st'] for s, v in sorted(state.items(), key=lambda kv: (len(kv[0]), kv[0]))}
        log.append(row)

    freq = {s: v['count'] for s, v in state.items() if v['st'] == 'SS'}
    stats = {'time_s': time.perf_counter() - t0, 'passes': read / N, 'stations': len(log),
             'itemsets_counted': len(state), 'frequent': len(freq), 'max_active_counters': max_active}
    return freq, pd.DataFrame(log), stats


def apriori_counts(X, min_sup):
    """Level-wise Apriori on the same counting engine (baseline for passes / time)."""
    N, I = X.shape
    min_cnt = int(np.ceil(min_sup * N))
    t0, passes, freq, counted, max_level = time.perf_counter(), 0, {}, 0, 0
    level = [(i,) for i in range(I)]
    while level:
        passes += 1                                    # one full scan per level
        counted += len(level)
        max_level = max(max_level, len(level))
        counts = {s: _count(X, s) for s in level}
        Lk = sorted(s for s, c in counts.items() if c >= min_cnt)
        freq.update({s: counts[s] for s in Lk})
        Lset, nxt = set(Lk), []
        for a, b in combinations(Lk, 2):
            if a[:-1] == b[:-1]:
                cand = a + (b[-1],)
                if all(sub in Lset for sub in combinations(cand, len(a))):
                    nxt.append(cand)
        level = nxt
    return freq, {'time_s': time.perf_counter() - t0, 'passes': passes, 'itemsets_counted': counted,
                  'frequent': len(freq), 'max_active_counters': max_level}


def generate_rules(freq, N, items, min_conf):
    """All rules A -> C from frequent itemsets with confidence >= min_conf."""
    sup = {frozenset(s): c / N for s, c in freq.items()}
    rows = []
    for s, c in freq.items():
        if len(s) < 2:
            continue
        S = frozenset(s)
        for r in range(1, len(s)):
            for A in combinations(s, r):
                A = frozenset(A); C = S - A
                conf = sup[S] / sup[A]
                if conf >= min_conf:
                    rows.append({'antecedent': A, 'consequent': C, 'support': sup[S], 'confidence': conf,
                                 'lift': conf / sup[C], 'leverage': sup[S] - sup[A] * sup[C]})
    rules = pd.DataFrame(rows)
    named = lambda fs: ', '.join(sorted(items[i] for i in fs))
    rules['A'] = rules.antecedent.map(named)
    rules['C'] = rules.consequent.map(named)
    return rules


def drop_redundant(rules, tol=1e-9):
    """Remove A -> C when some A' ⊂ A gives A' -> C with confidence >= that of A -> C (A adds nothing)."""
    keep = []
    by_c = {c: g for c, g in rules.groupby(rules.consequent.map(lambda s: tuple(sorted(s))))}
    for c, g in by_c.items():
        recs = list(g[['antecedent', 'confidence']].itertuples())
        for r in recs:
            red = any(o.antecedent < r.antecedent and o.confidence >= r.confidence - tol for o in recs)
            if not red:
                keep.append(r.Index)
    return rules.loc[sorted(keep)]
