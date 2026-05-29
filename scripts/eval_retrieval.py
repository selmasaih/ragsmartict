"""Retrieval evaluation harness.

Runs a small labelled question set through the real retrieval pipeline (vector
+ BM25 + cross-encoder rerank) and reports hit@k and Mean Reciprocal Rank
(MRR) based on whether the expected source document appears in the results.
Generation (LLM) is skipped — this measures retrieval only.

Usage:  python -m scripts.eval_retrieval
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import query

# Each case: question + one or more substrings expected in a retrieved
# filename (case-insensitive, any match counts). Optional "topic" filters.
EVAL = [
    {"q": "Qu'est-ce que la transformée de Fourier ?", "expect": ["support_cours", "fourier", "tiv"]},
    {"q": "Explique la méthode du simplexe en programmation linéaire", "expect": ["simplexe", "optimisation", "lineaire", "linéaire"]},
    {"q": "Qu'est-ce que le beamforming et le MIMO massif ?", "expect": ["mimo", "beamforming"]},
    {"q": "Décris le modèle relationnel d'une base de données", "expect": ["relationnel", "bd_inpt", "bdr"]},
    {"q": "Comment fonctionne la modulation OFDM ?", "expect": ["ofdm", "large", "wbt"]},
    {"q": "Qu'est-ce que la transformée de Laplace et Z ?", "expect": ["fn_ch1", "filtre", "laplace"]},
    {"q": "Explique le protocole TCP/IP et l'interconnexion", "expect": ["tcp", "ip", "ospf", "bgp"]},
    {"q": "Qu'est-ce que la virtualisation et le cloud ?", "expect": ["support", "cloud", "chap", "chp"]},
    {"q": "Décris l'architecture d'un microcontrôleur PIC18", "expect": ["pic18", "architecture"]},
    {"q": "Qu'est-ce qu'une chaîne de blocs (blockchain) ?", "expect": ["chapitre", "chapitre 2", "chapitre 3"]},
    {"q": "Explique le théorème de Bayes et l'algorithme kNN", "expect": ["bayes", "knn", "classification"]},
    {"q": "Qu'est-ce que le spectrum sensing en radio cognitive ?", "expect": ["spectrum", "cognitive", "sharing"]},
    {"q": "Comment configurer un VLAN sur un switch Cisco ?", "expect": ["vlan", "packet-tracer", "switch", "ccna"]},
    {"q": "Qu'est-ce que la normalisation des bases de données ?", "expect": ["normalisation", "bd_inpt", "bdr"]},
    {"q": "Explique les antennes et la propagation des ondes", "expect": ["antenne", "antennes", "ondes"]},
    {"q": "Qu'est-ce que la régression linéaire en machine learning ?", "expect": ["regression", "linear", "régression"]},
    {"q": "Décris le réseau cœur 5G", "expect": ["coeur", "core", "5g", "nfv", "sdn"]},
    {"q": "Qu'est-ce que la quantification et la compression de source ?", "expect": ["compression", "quantification", "codage"]},
]

TOP_K = 5


def _matches(names, expects):
    return next((i + 1 for i, n in enumerate(names) if any(e in n for e in expects)), 0)


def main(min_hit: float = 0.0):
    # Lazy-loads embedding model, reranker and collection on first retrieval;
    # no LLM needed since we only evaluate retrieval.
    collection, err = query._ensure_collection_ready()
    if err:
        print("Collection not ready:", err["error"])
        return 1

    hits = 0
    mrr_total = 0.0
    print(f"\n{'HIT':<4} {'RANK':<5} QUESTION")
    print("-" * 70)
    for case in EVAL:
        _, sources, _, _ = query._retrieve(collection, case["q"], case.get("topic"))
        names = [(s.get("filename") or "").lower() for s in sources[:TOP_K]]
        rank = _matches(names, case["expect"])
        if rank:
            hits += 1
            mrr_total += 1.0 / rank
        mark = "OK" if rank else "--"
        print(f"{mark:<4} {rank or '-':<5} {case['q'][:58]}")

    n = len(EVAL)
    hit_rate = hits / n
    print("-" * 70)
    print(f"hit@{TOP_K}: {hits}/{n} = {hit_rate:.1%}    MRR: {mrr_total / n:.3f}")
    if min_hit and hit_rate < min_hit:
        print(f"FAIL: hit@{TOP_K} {hit_rate:.1%} < seuil {min_hit:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue la qualité du retrieval (hit@k, MRR).")
    parser.add_argument("--min-hit", type=float, default=0.0,
                        help="Seuil hit@k (0-1) ; code de sortie 1 si en-dessous.")
    args = parser.parse_args()
    sys.exit(main(min_hit=args.min_hit))
