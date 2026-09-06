"""
Flip statistics table (Table 5 style) for ethnicity and gender perturbations.

Metrics:
  # Flips      : total prediction changes (orig != pert)
  Bad flips    : orig correct -> pert incorrect
  Disjoint (%) : fraction of bad flips where Jaccard(orig_ev, pert_ev) == 0
  Avg.J (bad)  : mean Jaccard similarity of retrieved docs among bad-flip cases

Usage:
    python analyze_flip_statistics.py
"""

import json
from collections import defaultdict

BASE = "results/perturb_test/"

ETHNICITY_FILES = {
    "MedCPT (k=1)":  [BASE + "medcpt_k1_ethnicity_test.json"],
    "MedRAG (k=32)": [BASE + "medrag_ethnicity_p%d.json" % i for i in (1, 2, 3)],
    "RAG2 (clf)":    [BASE + "rag2_ethnicity_p%d.json"   % i for i in (1, 2, 3)],
    "RAG2+Ours (v2)":  [BASE + "rag2_wring_v2_eth_p%d.json" % i for i in (1, 2, 3)],
    "RAG2+Ours (v3)":  [BASE + "rag2_wring_v3_eth_p%d.json" % i for i in (1, 2, 3)],
    "RAG2+Ours":  [BASE + "rag2_wring_v4_eth_p%d.json" % i for i in (1, 2, 3)],
}

GENDER_FILES = {
    "MedCPT (k=1)":  [BASE + "medcpt_k1_gender_test.json"],
    "MedRAG (k=32)": [BASE + "medrag_gender_k32_part1.json",
                      BASE + "medrag_gender_k32_part2.json"],
    "RAG2 (clf)":    [BASE + "rag2_gender_test.json"],
    "RAG2+Ours (v3)":  [BASE + "rag2_wring_v3_gender.json"],
    "RAG2+Ours":  [BASE + "rag2_wring_v4_gender.json"],
}

SYSTEMS = list(ETHNICITY_FILES.keys())


def load(paths):
    results = []
    for p in paths:
        try:
            d = json.load(open(p))
            results.extend(d.get("results", []))
        except FileNotFoundError:
            print(f"  [MISSING] {p}")
    return results


def doc_set(r):
    return set(str(e)[:80] for e in (r.get("top_evidence") or []))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def flip_stats(results):
    """Returns dict with flip stats and retrieval consistency, or None."""
    if not results:
        return None

    groups = defaultdict(list)
    for r in results:
        gid = r.get("_group_id")
        if gid is not None:
            groups[gid].append(r)

    n_flips = bad_flips = good_flips = total_pairs = 0
    jaccard_bad, jaccard_good, jaccard_all = [], [], []

    for items in groups.values():
        orig = next((x for x in items if not x.get("is_perturbed", False)), None)
        perts = [x for x in items if x.get("is_perturbed", False)]
        if not orig:
            continue
        orig_ev = doc_set(orig)
        for p in perts:
            total_pairs += 1
            j = jaccard(orig_ev, doc_set(p))
            jaccard_all.append(j)
            if orig["predicted_answer"] != p["predicted_answer"]:
                n_flips += 1
                if orig["correct"] and not p["correct"]:
                    bad_flips += 1
                    jaccard_bad.append(j)
                elif not orig["correct"] and p["correct"]:
                    good_flips += 1
                    jaccard_good.append(j)

    if total_pairs == 0:
        return None

    return {
        "n_flips":    n_flips,
        "bad_flips":  bad_flips,
        "good_flips": good_flips,
        "flip_ratio": bad_flips / good_flips if good_flips > 0 else float("inf"),
        "ev_sim_bad":  sum(jaccard_bad)  / len(jaccard_bad)  if jaccard_bad  else None,
        "ev_sim_good": sum(jaccard_good) / len(jaccard_good) if jaccard_good else None,
        "retrieval_consistency": sum(jaccard_all) / len(jaccard_all),
    }


def fmt_f(v):
    return f"{v:.3f}" if v is not None else "  -  "

def fmt_stats(s):
    if s is None:
        return "  -     -     -    -    -     -  "
    ratio = f"{s['flip_ratio']:.2f}" if s["flip_ratio"] != float("inf") else " inf"
    return (f"{fmt_f(s['ev_sim_bad']):>6} {fmt_f(s['ev_sim_good']):>6} "
            f"{s['n_flips']:>5} {ratio:>5} "
            f"{fmt_f(s['retrieval_consistency']):>6}")


def main():
    col_hdr = " EvSim✓→✗ EvSim✗→✓ Flips Ratio  RetrvC"
    header1 = f"{'System':<18} | {'Ethnicity':^40} | {'Gender':^40}"
    header2 = f"{'':18} |{col_hdr} |{col_hdr}"
    sep = "-" * (18 + 2 + 41 + 2 + 41)

    print(header1)
    print(header2)
    print(sep)

    for name in SYSTEMS:
        eth_files = [f for f in ETHNICITY_FILES.get(name, []) if __import__('os').path.exists(f)]
        gen_files  = [f for f in GENDER_FILES.get(name, [])    if __import__('os').path.exists(f)]
        e = flip_stats(load(eth_files)) if eth_files else None
        g = flip_stats(load(gen_files))  if gen_files  else None
        print(f"{name:<18} | {fmt_stats(e)} | {fmt_stats(g)}")

    print()
    print("Definitions:")
    print("  EvSim ✓→✗ : Jaccard(orig_ev, pert_ev) when orig correct → pert wrong")
    print("  EvSim ✗→✓ : Jaccard(orig_ev, pert_ev) when orig wrong  → pert correct")
    print("  RetrvC     : overall Retrieval Consistency (Jaccard, all pairs)")


if __name__ == "__main__":
    main()
