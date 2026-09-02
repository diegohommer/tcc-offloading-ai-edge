#!/usr/bin/env python3
"""Coleta a camada fog numa GPU da Modal, no hardware que mediu sua energia.

POR QUE MODAL E POR QUE L4
--------------------------
O ponto de energia do fog (Gemma-2-9B FP16, 1.75 J/token no regime CNN/DM) vem
do Bench360, medido em L4 / A30 / A10. A Modal oferece L4 e A10 -- entao rodar
aqui nao e so mais rapido que no CPU local, e a unica opcao que faz a confianca
ser coletada na MESMA CLASSE DE HARDWARE em que a energia foi medida.

Isso fecha, para o fog, o descasamento que a secao 7.4 da proposta documenta:
a precisao (FP16) e o hardware (L4) passam a bater dos dois lados. Sobra apenas
o par user/ONU, onde Q4_K_M contra AWQ/GPTQ e irredutivel sem possuir um
Snapdragon e um Jetson.

FIDELIDADE COM AS OUTRAS CAMADAS
--------------------------------
Tudo aqui replica o que o run_leaderboard_matrix.py faz localmente:

  - o prompt e o `full_prompt` do parquet, enviado VERBATIM (nunca reconstruido)
  - geracao gulosa, temperatura 0, ate 256 tokens
  - parada em "Question:" ou "\\n\\n", aplicada na geracao e nao por corte
    posterior -- a confianca e calculada sobre o vetor de logprobs retornado, e
    um token emitido apos a parada contaminaria o exp(min logprob)
  - confianca = exp(media dos logprobs), a definicao do RecServe, identica a
    confidence_from_logprobs() do ollama_layer.py
  - o vetor completo de logprobs e os tokens sao gravados, entao qualquer outra
    definicao de confianca (Chow-Quantile em qualquer alpha) e recomputavel
    offline sem reexecutar nada
  - batch=1, como as camadas user e onu rodaram via Ollama

O registro de saida tem exatamente os mesmos campos que write_record() grava,
para o sweep_energy_policy.py ler sem alteracao.

MODELO GATED
------------
google/gemma-2-9b-it exige aceitar a licenca no Hugging Face. Aceite em
huggingface.co/google/gemma-2-9b-it (instantaneo, gratuito) e crie um secret:

    modal secret create huggingface HF_TOKEN=hf_...

USO
---
    modal run src/scripts/modal_fog.py --limit 200

Grava results/traces/fog_modal.jsonl localmente. Depois basta concatenar ao
trace principal -- os indices sao os mesmos do parquet, entao o pareamento com
as outras camadas e exato.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

MODEL = "google/gemma-2-9b-it"
GPU = "L4"                 # o mesmo hardware do Bench360; A10 tambem serve
MAX_NEW_TOKENS = 256
STOP = ["Question:", "\n\n"]

app = modal.App("tcc-fog")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1")
)

# O modelo tem ~18,5 GB em FP16. Sem volume, cada execucao baixaria tudo de novo.
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def gerar(prompts: list[dict]) -> list[dict]:
    """Roda o Gemma-2-9B FP16 sobre os prompts, devolvendo logprobs completos."""
    import math
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/cache"

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir="/cache")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda", cache_dir="/cache"
    )
    model.eval()

    out = []
    for n, p in enumerate(prompts, 1):
        t0 = time.perf_counter()
        enc = tok(p["full_prompt"], return_tensors="pt").to("cuda")
        n_prompt = int(enc.input_ids.shape[1])

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                 # guloso: a confianca precisa ser reproduzivel
                temperature=None, top_p=None, top_k=None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )

        seq = gen.sequences[0][n_prompt:]
        # transition_scores da o logprob do token efetivamente escolhido, ja
        # normalizado -- e o analogo exato do campo logprobs do Ollama.
        scores = model.compute_transition_scores(
            gen.sequences, gen.scores, normalize_logits=True
        )[0]

        toks, lps = [], []
        for tid, lp in zip(seq.tolist(), scores.tolist()):
            if lp != lp:                          # NaN em posicao de padding
                continue
            toks.append(tok.decode([tid]))
            lps.append(float(lp))

        texto = tok.decode(seq, skip_special_tokens=True)

        # Parada aplicada ao texto E ao vetor de logprobs, em conjunto: cortar so
        # o texto deixaria tokens pos-parada dentro da confianca.
        corte = len(texto)
        for s in STOP:
            i = texto.find(s)
            if i != -1:
                corte = min(corte, i)
        if corte < len(texto):
            texto = texto[:corte]
            mantidos = len(tok(texto, add_special_tokens=False).input_ids)
            toks, lps = toks[:mantidos], lps[:mantidos]

        conf = math.exp(sum(lps) / len(lps)) if lps else 0.0

        out.append({
            "index": p["index"],
            "generated_text": texto,
            "logprobs": lps,
            "tokens": toks,
            "confidence": conf,
            "tokens_prompt": n_prompt,
            "tokens_gen": len(lps),
            "latency_s": time.perf_counter() - t0,
        })
        if n % 20 == 0:
            print(f"  {n}/{len(prompts)}")
    return out


@app.local_entrypoint()
def main(limit: int = 200, parquet: str = "", out: str = ""):
    """Le o parquet local, manda os prompts para a GPU, grava os registros."""
    raiz = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(raiz / "src"))
    from tasks.gsm8k import is_correct, load_gsm8k  # noqa: E402

    import pyarrow.parquet as pq

    pq_path = Path(parquet) if parquet else raiz / "results/leaderboard/cloud.parquet"
    out_path = Path(out) if out else raiz / "results/traces/fog_modal.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = pq.read_table(pq_path).to_pydict()
    prompts = [{"index": i, "full_prompt": d["full_prompt"][i]} for i in range(limit)]
    perguntas = {i: d["example"][i] for i in range(limit)}

    golds = {}
    for item in load_gsm8k(split="test", limit=None):
        golds[item.question.strip()] = (item.reference_answer, item.difficulty_steps)

    print(f"enviando {len(prompts)} prompts para uma {GPU}...")
    resultados = gerar.remote(prompts)

    n_ok = 0
    with open(out_path, "w") as f:
        for r in resultados:
            q = perguntas[r["index"]]
            gold, steps = golds.get(q.strip(), (None, None))
            correct = is_correct(r["generated_text"], gold) if gold else False
            n_ok += int(correct)
            f.write(json.dumps({
                "dataset": "gsm8k", "split": "test", "index": r["index"],
                "tier": "fog", "model": MODEL,
                "question": q,
                "reference_answer": gold,
                "difficulty_steps": steps,
                "generated_text": r["generated_text"],
                "logprobs": r["logprobs"],
                "tokens": r["tokens"],
                "confidence": r["confidence"],
                "correct": correct,
                "tokens_prompt": r["tokens_prompt"],
                "tokens_gen": r["tokens_gen"],
                "latency_s": r["latency_s"],
                "prompt_source": "leaderboard_v1_harness_gsm8k_5shot",
                "hardware": f"modal_{GPU}_fp16",
            }) + "\n")

    print(f"\n{len(resultados)} registros em {out_path}")
    print(f"acuracia do fog: {n_ok / len(resultados):.4f}")
    print(f"\nconcatene ao trace principal com:\n"
          f"  cat {out_path} >> results/traces/lb_pilot.raw.jsonl")
