"""Retrieval evaluation harness.

Runs a small labelled question set through the real retrieval pipeline (vector
+ BM25 + cross-encoder rerank) and reports hit@k and Mean Reciprocal Rank
(MRR) based on whether the expected source document appears in the results.
Generation (LLM) is skipped — this measures retrieval only.

Usage:  python -m scripts.eval_retrieval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import query

# Each case: question + a substring expected to appear in a retrieved filename
# (case-insensitive). Optional "topic" restricts the search to one domain.
EVAL = [
    {"q": "Qu'est-ce que la transformée de Fourier ?", "expect": "support_cours"},
    {"q": "Explique la méthode du simplexe en programmation linéaire", "expect": "simplexe"},
    {"q": "Qu'est-ce que le beamforming et le MIMO massif ?", "expect": "mimo"},
    {"q": "Décris le modèle relationnel d'une base de données", "expect": "modèle_relationnel"},
    {"q": "Comment fonctionne la modulation OFDM ?", "expect": "ofdm"},
    {"q": "Qu'est-ce que la transformée de Laplace et Z ?", "expect": "fn_ch1"},
    {"q": "Explique le protocole TCP/IP et l'interconnexion", "expect": "tcp"},
    {"q": "Qu'est-ce que la virtualisation et le cloud ?", "expect": "support"},
    {"q": "Décris l'architecture d'un microcontrôleur PIC18", "expect": "pic18"},
    {"q": "Qu'est-ce qu'une chaîne de blocs (blockchain) ?", "expect": "chapitre"},
    {"q": "Explique le théorème de Bayes et l'algorithme kNN", "expect": "bayes"},
    {"q": "Qu'est-ce que le spectrum sensing en radio cognitive ?", "expect": "spectrum"},
    {"q": "Comment configurer un VLAN sur un switch Cisco ?", "expect": "vlan"},
    {"q": "Qu'est-ce que la normalisation des bases de données ?", "expect": "normalisation"},
    {"q": "Explique les antennes et la propagation des ondes", "expect": "antenne"},
    {"q": "Qu'est-ce que la régression linéaire en machine learning ?", "expect": "linear_regression"},
    {"q": "Décris le réseau cœur 5G", "expect": "coeur"},
    {"q": "Qu'est-ce que la quantification et la compression de source ?", "expect": "compression"},
]

TOP_K = 5


def main():
    # Lazy-loads embedding model, reranker and collection on first retrieval;
    # no LLM needed since we only evaluate retrieval.
    collection, err = query._ensure_collection_ready()
    if err:
        print("Collection not ready:", err["error"])
        return

    hits = 0
    mrr_total = 0.0
    print(f"\n{'HIT':<4} {'RANK':<5} QUESTION")
    print("-" * 70)
    for case in EVAL:
        _, sources, _, _ = query._retrieve(collection, case["q"], case.get("topic"))
        names = [(s.get("filename") or "").lower() for s in sources[:TOP_K]]
        rank = next((i + 1 for i, n in enumerate(names) if case["expect"] in n), 0)
        if rank:
            hits += 1
            mrr_total += 1.0 / rank
        mark = "OK" if rank else "--"
        print(f"{mark:<4} {rank or '-':<5} {case['q'][:58]}")

    n = len(EVAL)
    print("-" * 70)
    print(f"hit@{TOP_K}: {hits}/{n} = {hits / n:.1%}    MRR: {mrr_total / n:.3f}")


if __name__ == "__main__":
    main()
