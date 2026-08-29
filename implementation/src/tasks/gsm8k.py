"""GSM8K generation task: prompt template, answer scoring, difficulty stratification.

Why this task replaces SST-2 for the generative cascade
--------------------------------------------------------
Every J/token value in config/layer_energy.yaml is DECODE energy -- joules per
OUTPUT token. The SST-2 classification harness generates zero output tokens, so
it exercises none of what those numbers measure (see the smoke-test caveat in
compute_energy_report.py). GSM8K asks for chain-of-thought reasoning, which
produces ~100-300 output tokens per query -- the decode-dominated regime the
energy tables were actually measured in (Caravaca et al. use 300 output tokens;
Fadel Argerich et al. use up to 100).

It also satisfies the requirement the escalation mechanism depends on: a real
spread of difficulty. RecServe's beta-quantile threshold is calibrated from the
distribution of recent confidence scores, so a workload of uniform difficulty
collapses the policy into "escalate everything" or "escalate nothing". GSM8K
problems range from one arithmetic step to eight or more, and that variation is
*measurable* -- see difficulty_steps() -- so routing behaviour can be reported
against true difficulty rather than only in aggregate.

Scoring is exact-match on the final integer, so no LLM judge is needed (unlike
MT-Bench) and no BLEU approximation is involved (unlike the WMT sets RecServe
used for its Seq2Seq experiments).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# One identical prompt for every tier. This is a deliberate methodological
# choice, not an oversight: tuning the prompt per tier would confound "capability
# gradient" with "prompt fit", and any handicap a shared prompt imposes on the
# smaller tiers is conservative -- it makes escalation look more necessary, never
# less, so it cannot inflate the policy's apparent benefit.
#
# Kept deliberately short: prompt tokens are billed by the same energy model as
# generated ones, so a verbose preamble would inflate every tier's cost equally
# and dilute the differences the experiment is trying to measure.
PROMPT_TEMPLATE = """Solve the problem. Reason step by step, then give the final numeric answer on its own last line in exactly this form:
#### <number>

Problem: {question}"""

# GSM8K reference answers end with "#### 42"; models are asked to match that form.
_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
# Fallback: last standalone number anywhere in the output, for models that ignore
# the format instruction (smaller tiers do this more often -- which is itself a
# capability signal, so answer-repair is kept minimal rather than masking it).
_FALLBACK_NUM_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


@dataclass
class GSM8KItem:
    question: str
    reference_answer: str      # gold final number, normalized
    reference_solution: str    # full worked solution, kept for difficulty scoring
    difficulty_steps: int      # number of calculator steps in the gold solution


def _normalize_number(raw: str | None) -> str | None:
    """Canonical form so '1,000', '1000', and '1000.0' all compare equal."""
    if raw is None:
        return None
    cleaned = raw.replace(",", "").strip().rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # GSM8K answers are integers; collapse integral floats so 42 != 42.0 never bites.
    return str(int(value)) if value == int(value) else str(value)


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_answer(generated_text: str) -> str | None:
    """Pull the model's final numeric answer out of its generation."""
    matches = _ANSWER_RE.findall(generated_text)
    if matches:
        return _normalize_number(matches[-1])
    fallback = _FALLBACK_NUM_RE.findall(generated_text)
    return _normalize_number(fallback[-1]) if fallback else None


def is_correct(generated_text: str, reference_answer: str) -> bool:
    predicted = extract_answer(generated_text)
    return predicted is not None and predicted == reference_answer


def difficulty_steps(reference_solution: str) -> int:
    """Number of calculator annotations (<<...>>) in GSM8K's gold solution.

    GSM8K marks each arithmetic step inline, e.g. '<<5*3=15>>'. Counting them
    gives an objective difficulty measure supplied by the dataset itself -- not a
    proxy invented here -- which lets routing be reported against true difficulty
    ("did the hard problems actually escalate?") rather than only in aggregate.
    """
    return reference_solution.count("<<")


def parse_reference(raw_answer: str) -> tuple[str | None, int]:
    """Split a GSM8K 'answer' field into (final number, difficulty steps)."""
    match = _ANSWER_RE.search(raw_answer)
    final = _normalize_number(match.group(1)) if match else None
    return final, difficulty_steps(raw_answer)


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[GSM8KItem]:
    """Load GSM8K from the Hugging Face hub (config 'main')."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    items: list[GSM8KItem] = []
    for row in dataset:
        final, steps = parse_reference(row["answer"])
        if final is None:
            continue  # malformed gold answer; skip rather than score against None
        items.append(GSM8KItem(
            question=row["question"],
            reference_answer=final,
            reference_solution=row["answer"],
            difficulty_steps=steps,
        ))
    return items
