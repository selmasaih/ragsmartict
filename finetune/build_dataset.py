"""Generate a styled instruction dataset for the LoRA style fine-tune.

For each seed question, this runs the real RAG retrieval to build a grounded
context, then asks the configured LLM to draft an answer in the target style.
Each example is written as a chat record (system/user/assistant) to JSONL,
ready for Unsloth / TRL / Axolotl.

The point is NOT to teach facts (RAG already supplies those at inference) but
to teach the *style*: structure, scientific tone, [n] citations, and the
honest "Information non trouvée" behavior.

Usage:
  python -m finetune.build_dataset --out finetune/style_dataset.jsonl --limit 60
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import query
from src.config import LLM_PROVIDER

# Seed questions spanning the 8 domains. Extend freely — more variety = better
# style coverage. Facts come from retrieval, so these only need to be plausible.
SEED_QUESTIONS = [
    "Qu'est-ce que la transformée de Fourier ?",
    "Explique le théorème d'échantillonnage de Shannon.",
    "Qu'est-ce que la modulation OFDM et à quoi sert-elle ?",
    "Décris le principe du MIMO massif.",
    "Comment fonctionne la méthode du simplexe ?",
    "Qu'est-ce que la normalisation d'une base de données ?",
    "Explique le protocole OSPF.",
    "Qu'est-ce que la virtualisation et le cloud computing ?",
    "Décris l'architecture d'un microcontrôleur.",
    "Qu'est-ce qu'une blockchain ?",
    "Explique la régression linéaire en machine learning.",
    "Qu'est-ce que le spectrum sensing en radio cognitive ?",
    "Comment configurer un VLAN ?",
    "Qu'est-ce que la transformée de Laplace ?",
    "Explique le concept de réseau cœur 5G.",
    "Qu'est-ce que la quantification d'un signal ?",
    "Décris le théorème de Bayes.",
    "Qu'est-ce que le beamforming ?",
    "Explique le multiplexage temporel.",
    "Qu'est-ce qu'un filtre passe-bas ?",
]


def main():
    ap = argparse.ArgumentParser(description="Build a styled instruction dataset.")
    ap.add_argument("--out", default="finetune/style_dataset.jsonl")
    ap.add_argument("--limit", type=int, default=len(SEED_QUESTIONS))
    args = ap.parse_args()

    collection, err = query._ensure_collection_ready()
    if err:
        print("Collection not ready:", err["error"])
        return 1

    system_prompt = query._build_system_prompt("fr")
    written = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for q in SEED_QUESTIONS[: args.limit]:
            context, sources, chunks, _ = query._retrieve(collection, q)
            if not chunks:
                continue
            user_msg = query._build_user_message(context, [], q)
            try:
                if LLM_PROVIDER == "gemini":
                    answer = query._call_gemini(user_msg, system_prompt=system_prompt)
                else:
                    answer = query._call_ollama(user_msg, system_prompt=system_prompt)
            except Exception as e:
                print(f"  skip '{q[:40]}': {e}")
                continue
            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": answer},
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{written}] {q[:50]}")

    print(f"\nWrote {written} examples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
