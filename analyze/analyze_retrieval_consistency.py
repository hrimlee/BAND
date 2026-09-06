"""
Retrieval Consistency Analysis:
Find cases where ours retrieves consistently (high Jaccard orig↔pert)
but RAG²/MedCPT does not — showing our debiasing effect on retrieval.

Usage:
    python analyze_retrieval_consistency.py --split gender
    python analyze_retrieval_consistency.py --split ethnicity --source_jsonl perturb_dataset/phrases_ethnicity_test_perturbed.jsonl
    python analyze_retrieval_consistency.py --split gender --ours_min_jaccard 0.5 --rag2_max_jaccard 0.3 --top_n 5
"""

import json
import argparse
from collections import defaultdict


def load_results(path, source_jsonl=None):
    """Load results and optionally join swapped_ethnicity from source JSONL."""
    data = json.load(open(path))
    results = data.get("results", data) if isinstance(data, dict) else data

    if source_jsonl:
        src = [json.loads(l) for l in open(source_jsonl)]
        q_to_meta = {}
        for s in src:
            key = s["question"][:80]
            q_to_meta[key] = {k: s[k] for k in
                              ("swapped_ethnicity", "original_ethnicity",
                               "swapped_gender", "original_gender",
                               "is_perturbed", "_group_id")
                              if k in s}
        for r in results:
            key = r["question"][:80]
            if key in q_to_meta:
                for k, v in q_to_meta[key].items():
                    if k not in r or r[k] is None:
                        r[k] = v

    groups = defaultdict(list)
    for r in results:
        gid = r.get("_group_id", r.get("id"))
        groups[gid].append(r)
    return groups


def doc_set(r):
    return set(str(e)[:80] for e in (r.get("top_evidence") or []))


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def get_docs(r, n=3):
    return [str(e)[:130] for e in (r.get("top_evidence") or [])[:n]]


def find_cases(ours_g, compare_g, compare_name,
               demo_key,         # "swapped_gender" or "swapped_ethnicity"
               orig_demo_key,    # "original_gender" or "original_ethnicity"
               ours_min_j=0.5, cmp_max_j=0.3,
               require_ours_correct=True,
               require_cmp_flip=False):
    """
    Find groups where:
      - Ours jaccard(orig↔pert) >= ours_min_j
      - compare jaccard(orig↔pert) <= cmp_max_j
      - (optionally) Ours correct on both
      - (optionally) compare system flips on pert
    """
    cases = []

    for gid in ours_g:
        if gid not in compare_g:
            continue

        o_rs = ours_g[gid]
        c_rs = compare_g[gid]

        wo = next((r for r in o_rs if not r.get("is_perturbed")), None)
        co = next((r for r in c_rs if not r.get("is_perturbed")), None)
        if not wo or not co:
            continue

        w_perts = [r for r in o_rs if r.get("is_perturbed")]
        c_perts = [r for r in c_rs if r.get("is_perturbed")]

        for wp in w_perts:
            eth = wp.get(demo_key)
            # Match by demographic label
            cp = next((r for r in c_perts if r.get(demo_key) == eth), None)
            if cp is None and c_perts:
                cp = c_perts[0]
            if not cp:
                continue

            wj = jaccard(doc_set(wo), doc_set(wp))
            cj = jaccard(doc_set(co), doc_set(cp))

            if wj < ours_min_j or cj > cmp_max_j:
                continue
            if require_ours_correct and not (wo["correct"] and wp["correct"]):
                continue
            if require_cmp_flip and not (co["correct"] and not cp["correct"]):
                continue

            cases.append({
                "gid":          gid,
                "demographic":  eth,
                "orig_demo":    wo.get(orig_demo_key),
                "gold":         wo["gold_answer"],
                "ours_jaccard": wj,
                "cmp_jaccard":   cj,
                "wo": wo, "wp": wp,
                "co": co, "cp": cp,
                "cmp_name": compare_name,
            })

    cases.sort(key=lambda x: x["ours_jaccard"] - x["cmp_jaccard"], reverse=True)
    return cases


def print_case(c, demo_key):
    wo, wp = c["wo"], c["wp"]
    co, cp = c["co"], c["cp"]
    cmp = c["cmp_name"]

    print(f"\n{'='*72}")
    print(f"GROUP {c['gid']} | gold={c['gold']} | {c['orig_demo']} → {c['demographic']}")
    print(f"  Jaccard:  Ours={c['ours_jaccard']:.3f}   {cmp}={c['cmp_jaccard']:.3f}")
    print(f"  Ours  orig={wo['predicted_answer']} {'✅' if wo['correct'] else '❌'}  "
          f"pert={wp['predicted_answer']} {'✅' if wp['correct'] else '❌'}")
    print(f"  {cmp:<6} orig={co['predicted_answer']} {'✅' if co['correct'] else '❌'}  "
          f"pert={cp['predicted_answer']} {'✅' if cp['correct'] else '❌'}")

    print(f"\n[Original question ({c['orig_demo']})]")
    print(f"  {wo['question'][:220]}")
    print(f"\n[Perturbed question ({c['demographic']})]")
    print(f"  {wp['question'][:220]}")

    print(f"\n[Ours docs - original]")
    for d in get_docs(wo): print(f"  • {d}")
    print(f"[Ours docs - perturbed ({c['demographic']})]")
    for d in get_docs(wp): print(f"  • {d}")

    print(f"\n[{cmp} docs - original]")
    for d in get_docs(co): print(f"  • {d}")
    print(f"[{cmp} docs - perturbed ({c['demographic']})]")
    for d in get_docs(cp): print(f"  • {d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["gender", "ethnicity"], default="gender")
    parser.add_argument("--source_jsonl", default=None,
                        help="Source JSONL to join swapped_ethnicity (for ethnicity split)")
    parser.add_argument("--ours_min_jaccard", type=float, default=0.4)
    parser.add_argument("--rag2_max_jaccard",  type=float, default=0.3)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--require_ours_correct", action="store_true", default=True)
    parser.add_argument("--require_cmp_flip", action="store_true", default=False,
                        help="Only show cases where compare system flips orig✅→pert❌")
    args = parser.parse_args()

    base = "results/perturb_test"

    if args.split == "gender":
        demo_key      = "swapped_gender"
        orig_demo_key = "original_gender"
        files = {
            "Ours":  f"{base}/gender_test_results.json",
            "RAG²":   f"{base}/rag2_gender_test.json",
            "MedCPT": f"{base}/medcpt_k1_gender_test.json",
        }
        source_jsonl = args.source_jsonl
    else:
        demo_key      = "swapped_ethnicity"
        orig_demo_key = "original_ethnicity"
        files = {
            "Ours":  f"{base}/ethnicity_test_results_small.json",
            "RAG²":   f"{base}/rag2_ethnicity_test_small.json",
            "MedCPT": f"{base}/medcpt_k1_ethnicity_test.json",
        }
        source_jsonl = args.source_jsonl or "perturb_dataset/phrases_ethnicity_test_perturbed.jsonl"

    ours_g = load_results(files["Ours"], source_jsonl if args.split == "ethnicity" else None)

    for cmp_name in ["RAG²", "MedCPT"]:
        if cmp_name not in files:
            continue
        import os
        if not os.path.exists(files[cmp_name]):
            print(f"[skip] {cmp_name}: file not found")
            continue

        cmp_g = load_results(files[cmp_name],
                             source_jsonl if args.split == "ethnicity" else None)

        cases = find_cases(
            ours_g, cmp_g, cmp_name,
            demo_key, orig_demo_key,
            ours_min_j=args.ours_min_jaccard,
            cmp_max_j=args.rag2_max_jaccard,
            require_ours_correct=args.require_ours_correct,
            require_cmp_flip=args.require_cmp_flip,
        )

        print(f"\n{'#'*72}")
        print(f"# Ours vs {cmp_name}  |  split={args.split}")
        print(f"# Found {len(cases)} cases  "
              f"(Ours Jaccard≥{args.ours_min_jaccard}, {cmp_name} Jaccard≤{args.rag2_max_jaccard})")
        print(f"{'#'*72}")

        for c in cases[:args.top_n]:
            print_case(c, demo_key)

    print()


if __name__ == "__main__":
    main()
