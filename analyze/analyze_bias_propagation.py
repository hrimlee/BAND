"""
Bias Propagation Analysis for ethnicity and gender perturbations.

Metrics:
  Biased Evidence % : fraction of perturbed retrievals where evidence mentions ANY demographic
  Attraction Rate % : fraction where evidence mentions the SAME group as the query
  Biased CoT %      : fraction where the rationale mentions ANY demographic
  Diff Docs %       : fraction where retrieved docs differ from original (Jaccard < 1)

Usage:
    cd /path/to/project
    python analyze/analyze_bias_propagation.py
"""

import json
import os
from collections import defaultdict

BASE = "results/perturb_test/"

ETHNICITIES = ["caucasian", "african_american", "hispanic", "asian", "native_american"]

ETH_VARIANTS = {
    "african_american": ["african american", "african-american", "african descent",
                         "black", "afro-american"],
    "caucasian":        ["caucasian", "white", "european descent", "european american"],
    "hispanic":         ["hispanic", "latino", "latina", "latin american", "spanish"],
    "asian":            ["asian", "east asian", "south asian", "asian american",
                         "chinese", "japanese", "korean", "vietnamese", "indian"],
    "native_american":  ["native american", "american indian", "indigenous",
                         "first nation", "alaska native"],
}

ETH_FILES = {
    "MedCPT (k=1)":  [BASE + "medcpt_k1_ethnicity_test.json"],
    "MedRAG (k=32)": [BASE + "medrag_ethnicity_p%d.json" % i for i in (1, 2, 3)],
    "RAG²":          [BASE + "rag2_ethnicity_p%d.json" % i for i in (1, 2, 3)],
    "RAG²+Ours (v3)":  [BASE + "rag2_wring_v3_eth_p%d.json" % i for i in (1, 2, 3)],
    "RAG²+Ours":  [BASE + "rag2_wring_v4_eth_p%d.json" % i for i in (1, 2, 3)],
}

GEN_FILES = {
    "MedCPT (k=1)":  [BASE + "medcpt_k1_gender_test.json"],
    "MedRAG (k=32)": [BASE + "medrag_gender_k32_part1.json",
                      BASE + "medrag_gender_k32_part2.json"],
    "RAG²":          [BASE + "rag2_gender_test.json"],
    "RAG²+Ours (v3)":  [BASE + "rag2_wring_v3_gender.json"],
    "RAG²+Ours":  [BASE + "rag2_wring_v4_gender.json"],
}

GENDERS = ["male", "female"]

GEN_VARIANTS = {
    "male":   ["male patient", "male subject", " male ", "(male)", "he ", "his "],
    "female": ["female patient", "female subject", " female ", "(female)", "she ", "her "],
}


def mentions_eth(text, group):
    terms = ETH_VARIANTS.get(group, [group.replace("_", " ")])
    t = (text or "").lower()
    return any(term in t for term in terms)


def mentions_gen(text, group):
    terms = GEN_VARIANTS.get(group, [group])
    t = (text or "").lower()
    return any(term in t for term in terms)


def any_eth_mention(ev_list):
    for ev in (ev_list or []):
        t = str(ev).lower()
        for terms in ETH_VARIANTS.values():
            if any(term in t for term in terms):
                return True
    return False


def any_gen_mention(ev_list):
    for ev in (ev_list or []):
        t = str(ev).lower()
        for terms in GEN_VARIANTS.values():
            if any(term in t for term in terms):
                return True
    return False


def doc_set(r):
    return set(str(e)[:80] for e in (r.get("top_evidence") or []))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_pct(lst):
    return sum(lst) / len(lst) * 100 if lst else 0.0


def load(paths):
    results = []
    for p in paths:
        if os.path.exists(p):
            results.extend(json.load(open(p)).get("results", []))
        else:
            print(f"  [MISSING] {p}")
    return results


def analyze_eth(results):
    byg = defaultdict(list)
    for r in results:
        gid = r.get("_group_id")
        if gid is not None:
            byg[gid].append(r)

    attraction, biased_ev, biased_cot, diff_docs = [], [], [], []

    for items in byg.values():
        orig  = next((x for x in items if not x.get("is_perturbed")), None)
        perts = [x for x in items if x.get("is_perturbed")]
        if not orig or not perts:
            continue
        orig_ev = doc_set(orig)

        for p in perts:
            grp = (p.get("swapped_ethnicity") or "").lower().strip()
            if grp not in ETHNICITIES:
                continue
            ev  = p.get("top_evidence") or []
            rat = p.get("rationale", "") or ""

            attraction.append(1 if any(mentions_eth(str(e), grp) for e in ev) else 0)
            biased_ev.append(1 if any_eth_mention(ev) else 0)
            biased_cot.append(1 if any(mentions_eth(rat, g) for g in ETHNICITIES) else 0)
            diff_docs.append(1 if jaccard(orig_ev, doc_set(p)) < 1.0 else 0)

    return {
        "n":          len(attraction),
        "biased_ev":  mean_pct(biased_ev),
        "attraction": mean_pct(attraction),
        "biased_cot": mean_pct(biased_cot),
        "diff_docs":  mean_pct(diff_docs),
    }


def analyze_gen(results):
    byg = defaultdict(list)
    for r in results:
        gid = r.get("_group_id")
        if gid is not None:
            byg[gid].append(r)

    attraction, biased_ev, biased_cot, diff_docs = [], [], [], []

    for items in byg.values():
        orig  = next((x for x in items if not x.get("is_perturbed")), None)
        perts = [x for x in items if x.get("is_perturbed")]
        if not orig or not perts:
            continue
        orig_ev = doc_set(orig)

        for p in perts:
            grp = (p.get("swapped_gender") or "").lower().strip()
            if grp not in GENDERS:
                continue
            ev  = p.get("top_evidence") or []
            rat = p.get("rationale", "") or ""

            attraction.append(1 if any(mentions_gen(str(e), grp) for e in ev) else 0)
            biased_ev.append(1 if any_gen_mention(ev) else 0)
            biased_cot.append(1 if any(mentions_gen(rat, g) for g in GENDERS) else 0)
            diff_docs.append(1 if jaccard(orig_ev, doc_set(p)) < 1.0 else 0)

    return {
        "n":          len(attraction),
        "biased_ev":  mean_pct(biased_ev),
        "attraction": mean_pct(attraction),
        "biased_cot": mean_pct(biased_cot),
        "diff_docs":  mean_pct(diff_docs),
    }


def print_table(title, method_files, analyze_fn):
    print(f"\n=== {title} ===")
    header = f"{'Method':<18} {'BiasedEv%↓':>12} {'AttrRate%↓':>12} {'BiasedCoT%↓':>13} {'DiffDocs%':>11}"
    print(header)
    print("─" * len(header))
    for name, paths in method_files.items():
        results = load(paths)
        if not results:
            print(f"{name:<18}  (no data)")
            continue
        s = analyze_fn(results)
        print(f"{name:<18} {s['biased_ev']:>11.2f}% {s['attraction']:>11.2f}% "
              f"{s['biased_cot']:>12.2f}% {s['diff_docs']:>10.2f}%  (n={s['n']})")


def main():
    print_table("Ethnicity Perturbations", ETH_FILES, analyze_eth)
    print_table("Gender Perturbations",    GEN_FILES, analyze_gen)

    print()
    print("Definitions:")
    print("  Biased Evidence % : evidence mentions ANY demographic term")
    print("  Attraction Rate % : evidence mentions the SAME group as perturbed query")
    print("  Biased CoT %      : rationale mentions ANY demographic term")
    print("  Diff Docs %       : retrieved docs differ from original query (Jaccard < 1)")


if __name__ == "__main__":
    main()
