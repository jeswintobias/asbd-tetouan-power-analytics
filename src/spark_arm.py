"""
ASBD Hackathon — DIC and COFI on PySpark
Student: Jeswin Tobias J | Roll No: CS23I1013

DIC  : the driver keeps the DIC state machine (src/arm.dic); each station's block of M
       transactions is spread over P partitions and counted in parallel (sc.runJob on just
       that block's partitions). Transactions are 64-bit item masks: X contains S <=> x & s == s.
COFI : every frequent item's COFI-tree is independent, so each becomes one Spark task.
       spark_cofi       : combiner first (each partition merges its identical transactions), then
                          only the ~4 k weighted transactions are projected -> one task per item -> mine_item.
       spark_cofi_naive : first version (every record projected before merging), kept for the ablation.
"""
import time
from collections import Counter
from operator import add

import numpy as np

from src.arm import dic
from src.cofi import mine_item


# ----------------------------------------------------------------------------- DIC
def masks_rdd(sc, X, M, P):
    """Cache transactions as numpy uint64 masks: block b = rows [bM, (b+1)M) in partitions b*P .. b*P+P-1."""
    bits = (X.astype(np.uint64) << np.arange(X.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64)
    N = len(bits)
    n_blocks = -(-N // M)
    keyed = sc.parallelize(list(enumerate(bits.tolist())), max(sc.defaultParallelism, n_blocks)) \
              .map(lambda kv: ((kv[0] // M) * P + kv[0] % P, kv[1]))
    rdd = keyed.partitionBy(n_blocks * P, lambda k: k) \
               .mapPartitions(lambda it: [np.fromiter((v for _, v in it), dtype=np.uint64)]) \
               .cache()
    rdd.count()                                            # materialise the cache
    return rdd, n_blocks


def spark_dic(sc, rdd, N, I, min_sup, M, P):
    """DIC with distributed block counting. Same output as src.arm.dic."""
    def counter(pos, end, itemsets):
        b = pos // M
        masks = np.array([sum(1 << i for i in s) for s in itemsets], dtype=np.uint64)

        def count_part(it):
            for arr in it:
                yield np.array([np.count_nonzero((arr & m) == m) for m in masks], dtype=np.int64)
        parts = sc.runJob(rdd, count_part, partitions=list(range(b * P, b * P + P)))
        return np.sum(parts, axis=0) if parts else np.zeros(len(itemsets), np.int64)
    return dic(None, min_sup, M, counter=counter, shape=(N, I))


# ----------------------------------------------------------------------------- COFI
def spark_cofi_naive(sc, tx_rdd, min_sup):
    """First version: every record is projected into FP-paths before merging. tx_rdd: RDD of item-id lists. Returns ({frozenset: count}, stats)."""
    t0 = time.perf_counter()
    N = tx_rdd.count()
    min_cnt = min_sup * N
    # scan 1: global supports
    sup = dict(tx_rdd.flatMap(lambda t: [(i, 1) for i in t]).reduceByKey(add).collect())
    F1 = sorted((i for i in sup if sup[i] >= min_cnt), key=lambda i: (-sup[i], i))
    rank = {it: r for r, it in enumerate(F1)}
    b_rank = sc.broadcast(rank)

    # scan 2: for every item i in a transaction, emit the items more frequent than i (its FP-path)
    def project(t):
        r = b_rank.value
        s = sorted((i for i in t if i in r), key=r.get)
        return [((i, tuple(s[:k])), 1) for k, i in enumerate(s)]

    def mine(kv):
        i, paths = kv
        pats, _ = mine_item(i, [(list(p), c) for p, c in paths], min_cnt, b_rank.value)
        return list(pats.items())

    freq = dict(tx_rdd.flatMap(project).reduceByKey(add)                  # merge identical paths
                .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
                .groupByKey(numPartitions=len(F1))                         # one COFI-tree per item
                .flatMap(mine).collect())
    freq.update({frozenset([i]): sup[i] for i in F1})
    return freq, {'time_s': time.perf_counter() - t0, 'frequent': len(freq)}


def spark_cofi(sc, tx_rdd, min_sup):
    """COFI with a map-side combiner and one task per item. Returns ({frozenset: count}, stats).

    job 1 : each partition merges its identical transactions (combiner), reduceByKey merges across
            partitions -> ~4 k (transaction, count) pairs. N and item supports follow on the driver.
    job 2 : each weighted transaction emits its FP-path per item -> groupByKey -> mine_item.
    """
    t0 = time.perf_counter()
    W = tx_rdd.mapPartitions(lambda it: Counter(tuple(sorted(t)) for t in it).items()) \
              .reduceByKey(add).collect()
    N = sum(w for _, w in W)
    min_cnt = min_sup * N
    sup = Counter()
    for t, w in W:
        for i in t:
            sup[i] += w
    F1 = sorted((i for i in sup if sup[i] >= min_cnt), key=lambda i: (-sup[i], i))
    b_rank = sc.broadcast({it: r for r, it in enumerate(F1)})

    def project(tw):                                       # (i, (items more frequent than i, count))
        t, w = tw
        r = b_rank.value
        s = sorted((i for i in t if i in r), key=r.get)
        return [(i, (s[:k], w)) for k, i in enumerate(s)]

    def mine(kv):
        i, paths = kv
        pats, _ = mine_item(i, list(paths), min_cnt, b_rank.value)
        return list(pats.items())

    freq = dict(sc.parallelize(W, sc.defaultParallelism).flatMap(project)
                .groupByKey(numPartitions=len(F1)).flatMap(mine).collect())
    freq.update({frozenset([i]): sup[i] for i in F1})
    return freq, {'time_s': time.perf_counter() - t0, 'frequent': len(freq), 'distinct_transactions': len(W)}
