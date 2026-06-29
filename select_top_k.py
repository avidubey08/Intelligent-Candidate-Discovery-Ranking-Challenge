"""
select_top_k.py — Stage E: O(N log K) Heap-based Top-K Selection
Heap key: (score, -cid_int) in a min-heap.
  heap[0] = worst item = lowest score, or on tie: highest numeric cid (evictee).
Tie-break matches validate_submission.py: higher score wins; on tie, lower cid wins.
"""
import heapq


def _cid_int(cid):
    try: return int(str(cid).split("_")[1])
    except: return 0


def select_top_k(scored_iter, k=100):
    """
    Select top-K scored candidates from a stream.
    Args:
      scored_iter: iterable of dicts with keys: candidate_id, score, features,
                   score_result, suspicion_result
      k: number of top candidates (default 100)
    Returns:
      list of k dicts sorted by (score DESC, candidate_id ASC)
    """
    heap=[]
    for item in scored_iter:
        score=float(item.get("score",0.0) or 0.0)
        cid=item.get("candidate_id","")
        # Key: (score, -cid_int) in a min-heap
        # heap[0] = min key = item with lowest score; on tie, highest cid_num = evictee
        key=(score,-_cid_int(cid))
        if len(heap)<k:
            heapq.heappush(heap,(key,item))
        elif key>heap[0][0]:
            heapq.heapreplace(heap,(key,item))
    results=[item for (_,item) in heap]
    results.sort(key=lambda x:(-float(x.get("score",0.0) or 0.0),_cid_int(x.get("candidate_id",""))))
    return results
