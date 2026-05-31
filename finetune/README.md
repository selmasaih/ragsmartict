# Style specialization — RAG + light style fine-tune

RAG already supplies the **facts** (your indexed notes). This folder specializes
the **answer style** — structure, scientific tone, `[n]` citations, honest
"Information non trouvée" — in two complementary ways.

## A. Modelfile (no GPU, works now)

Bakes a persona + tuned decoding parameters + few-shot exemplars into a custom
Ollama model. This is the pragmatic "style tune" and runs on your current setup.

```bash
ollama create inpt-smart-ict -f finetune/Modelfile.inpt
```
Then in `.env`:
```
OLLAMA_MODEL="inpt-smart-ict"
```
Restart the backend. The app also injects a worked example into the prompt at
inference (`STYLE_FEWSHOT=true`, on by default) — disable with
`STYLE_FEWSHOT=false`.

> Swap the `FROM` line in `Modelfile.inpt` to use a stronger base you've pulled
> (e.g. `qwen2.5:7b`, `llama3.1:8b`) for a bigger quality jump.

## B. LoRA fine-tune (GPU / Colab, optional)

A real weight-level style tune. Three steps:

1. **Build the dataset** (locally, with the backend's models available):
   ```bash
   python -m finetune.build_dataset --out finetune/style_dataset.jsonl --limit 60
   ```
   For each seed question it runs real retrieval and asks the LLM to draft a
   styled, sourced answer → chat-format JSONL.

2. **Train on Colab** (GPU): open `finetune/train_lora.ipynb`, upload
   `style_dataset.jsonl`, run all cells. It LoRA-tunes a 3B instruct model with
   Unsloth and exports a `.gguf`.

3. **Serve via Ollama**: download the `.gguf`, create a Modelfile starting with
   `FROM ./model-unsloth.Q4_K_M.gguf` (reuse the SYSTEM/PARAMETER blocks from
   `Modelfile.inpt`), then `ollama create inpt-smart-ict-ft -f Modelfile.ft` and
   set `OLLAMA_MODEL="inpt-smart-ict-ft"`.

## Which should I use?

- **Just Modelfile (A):** recommended for this project. ~90% of the benefit, zero
  training cost, instantly updatable.
- **Add LoRA (B):** only if you want the style baked into the weights (lower
  per-request prompt overhead, more consistent tone). Needs a GPU and is worth it
  mainly with a larger seed dataset (100s of examples).
