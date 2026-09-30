"""The GSM8K task: prompt template, answer scoring and difficulty.

GSM8K asks for step-by-step reasoning, so every tier generates a few hundred tokens (the
decode-dominated regime the energy figures describe). Its problems range from one to eight
or more arithmetic steps, the spread of difficulty RecServe's threshold needs, and scoring
is exact match on the final number, so no judge model is needed.
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
# pylint: disable-next=line-too-long
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
    """One GSM8K test question with its gold answer and difficulty.

    Attributes:
        question: The problem statement.
        reference_answer: The gold final number, normalized.
        reference_solution: The full worked solution, kept for difficulty scoring.
        difficulty_steps: The number of calculator steps in the gold solution.
    """

    question: str
    reference_answer: str
    reference_solution: str
    difficulty_steps: int


def _normalize_number(raw: str | None) -> str | None:
    """Return a number's canonical form, so '1,000', '1000' and '1000.0' all compare equal.

    Args:
        raw: The number as written, or None.

    Returns:
        The canonical form, or None when raw is not a number.
    """
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
    """Return the zero-shot prompt every tier receives for a question (before its chat template).

    Args:
        question: The problem statement.
    """
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_answer(generated_text: str) -> str | None:
    """Return the model's final numeric answer from its generation.

    Args:
        generated_text: The model's answer.

    Returns:
        The number after "####", else the last number in the text, normalized; or None.
    """
    matches = _ANSWER_RE.findall(generated_text)
    if matches:
        return _normalize_number(matches[-1])
    fallback = _FALLBACK_NUM_RE.findall(generated_text)
    return _normalize_number(fallback[-1]) if fallback else None


def is_correct(generated_text: str, reference_answer: str) -> bool:
    """Return whether the final number in the model's answer equals the gold answer.

    Args:
        generated_text: The model's answer.
        reference_answer: The gold final number, normalized.
    """
    predicted = extract_answer(generated_text)
    return predicted is not None and predicted == reference_answer


def difficulty_steps(reference_solution: str) -> int:
    """Return the number of calculator annotations (<<...>>) in GSM8K's gold solution.

    GSM8K marks each arithmetic step inline (e.g. '<<5*3=15>>'), so counting them gives
    a difficulty measure supplied by the dataset itself.

    Args:
        reference_solution: The gold worked solution.
    """
    return reference_solution.count("<<")


def parse_reference(raw_answer: str) -> tuple[str | None, int]:
    """Split a GSM8K 'answer' field into (final number, difficulty steps).

    Args:
        raw_answer: The dataset's answer field (worked solution ending in "#### n").
    """
    match = _ANSWER_RE.search(raw_answer)
    final = _normalize_number(match.group(1)) if match else None
    return final, difficulty_steps(raw_answer)


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[GSM8KItem]:
    """Load GSM8K from the Hugging Face hub (config 'main').

    Args:
        split: The dataset split.
        limit: How many questions to keep (None: all).

    Returns:
        The questions whose gold answer parses.
    """
    # imported here so scoring and prompts work without the datasets package
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    items: list[GSM8KItem] = []
    for row in dataset:
        final, steps = parse_reference(row["answer"])
        if final is None:
            continue  # malformed gold answer; skip rather than score against None
        items.append(
            GSM8KItem(
                question=row["question"],
                reference_answer=final,
                reference_solution=row["answer"],
                difficulty_steps=steps,
            )
        )
    return items
