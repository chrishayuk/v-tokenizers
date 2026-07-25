"""Serializable structural pre-tokenizers for the selective TOK-2 grid."""

from tokenizers import Regex, pre_tokenizers


PRE_TOKENIZATIONS = ("whitespace_split", "digit_isolating", "code_aware")

# The code-aware expression isolates camel/Pascal-case components, acronym
# runs, snake-case separators, and punctuation/operator runs. Digits are
# isolated separately so C6 remains constant across all three arms.
CODE_COMPONENT = r"[A-Z]+(?=[A-Z][a-z]|\d|_|$)|[A-Z]?[a-z]+|_+|[^\w\s]+"


def build_pre_tokenizer(name: str):
    if name not in PRE_TOKENIZATIONS:
        raise ValueError(f"unknown pre-tokenization {name!r}; expected one of {PRE_TOKENIZATIONS}")

    stages = [pre_tokenizers.Split(Regex(r"\s+"), behavior="isolated")]
    if name in ("digit_isolating", "code_aware"):
        stages.append(pre_tokenizers.Digits(individual_digits=True))
    if name == "code_aware":
        stages.append(pre_tokenizers.Split(Regex(CODE_COMPONENT), behavior="isolated"))
    return pre_tokenizers.Sequence(stages)


def definition(name: str) -> dict:
    return {
        "name": name,
        "whitespace_runs": "isolated and preserved verbatim",
        "individual_digits": name in ("digit_isolating", "code_aware"),
        "code_components": CODE_COMPONENT if name == "code_aware" else None,
        "normalization": "identity",
    }
