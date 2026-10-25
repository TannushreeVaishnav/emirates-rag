"""
guardrails.py — Input validation and output safety checks.

Three independent layers of protection, applied in order:

  1. PII masking    — strips emails, card numbers, phone numbers, passport
                      numbers and booking references before they reach the LLM
                      or the request log.

  2. Injection detection — catches prompt-injection attempts using regex
                            patterns with Unicode normalisation so attackers
                            can't evade detection with lookalike characters.

  3. Output rules   — deterministic checks on the generated answer:
                        - monetary amounts that don't appear in the context
                        - guarantee language without a source
                        - system-prompt text leaking into the answer

The regex approach for injection is intentionally strict (verb + scope both
required) to avoid false positives on legitimate questions like "Can I ignore
the baggage limit for medical equipment?"
"""

import re
import unicodedata

# Maximum question length accepted. Longer inputs are more likely to be
# automated attacks trying to overwhelm the context window.
MAX_INPUT_CHARS = 1000


# ── 1. PII MASKING ────────────────────────────────────────────────────────────

# Flight numbers (e.g. EK501) look like 6-char PNRs — exclude them.
_FLIGHT_NO = re.compile(r"^[A-Z]{2}\d{3,4}$")


def _digit_count(s: str) -> int:
    """Count digit characters in a string (used to filter short numeric matches)."""
    return len(re.sub(r"\D", "", s))


# Each tuple is (label, compiled regex, minimum digit count or None).
# Minimum digit count avoids false positives on dates (2026-10-25 has only 8 digits).
_PII_PATTERNS = [
    ("EMAIL",    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), None),
    ("CARD",     re.compile(r"(?<!\w)(?:\d[ -]?){13,19}(?!\w)"), 13),
    ("PHONE",    re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"), 10),
    ("PASSPORT", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), None),
    ("PNR",      re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b"), None),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace PII tokens with [LABEL] placeholders.

    Returns:
        (masked_text, list_of_pii_types_found)

    Example:
        mask_pii("My email is jane@test.com and card 4111 1111 1111 1111")
        → ("My email is [EMAIL] and card [CARD]", ["EMAIL", "CARD"])
    """
    found: list[str] = []

    for label, pattern, min_digits in _PII_PATTERNS:
        def replace(m, label=label, min_digits=min_digits):
            s = m.group(0)
            # Skip if the token doesn't have enough digits (e.g. a date)
            if min_digits and _digit_count(s) < min_digits:
                return s
            # Skip flight numbers that match the PNR pattern (e.g. EK501 → 5 chars, fine)
            if label == "PNR" and _FLIGHT_NO.match(s):
                return s
            found.append(label)
            return f"[{label}]"

        text = pattern.sub(replace, text)

    return text, found


# ── 2. PROMPT INJECTION DETECTION ─────────────────────────────────────────────

# All patterns require at least a verb AND a scope (e.g. "ignore" + "instructions")
# so that normal questions mentioning "rules" or "previous" aren't flagged.
_VERB = r"(?:ignore|disregard|forget)"
_INJECTION_PATTERNS = [re.compile(p) for p in (
    # Classic: "ignore all previous instructions"
    rf"\b{_VERB}\s+(?:all\s+|any\s+|every\s+|the\s+|your\s+|my\s+)*"
    rf"(?:previous|prior|above|earlier|preceding|original|system)\b[^.]{{0,30}}"
    rf"\b(?:instructions?|rules?|prompts?|guidelines|polic(?:y|ies)|directions?|restrictions?)\b",

    # Shortened: "ignore your rules"
    rf"\b{_VERB}\s+(?:all\s+|any\s+)?(?:your|ur|these|those)\s+"
    rf"(?:instructions?|rules?|guidelines|programming|restrictions?|polic(?:y|ies)|training)\b",

    # Extraction attempts: "reveal your system prompt"
    r"\b(?:reveal|show|print|repeat|display|leak|output|tell me|give me|share)\b"
    r"[^.]{0,40}\b(?:system|hidden|secret|developer)\b[^.]{0,20}\b(?:prompt|instructions?)\b",
    r"\b(?:system|developer)\s+prompt\b",
    r"\bsystem\s+override\b",

    # Jailbreak personas
    r"\byou are now (?:dan|in developer mode|free|unrestricted|an? (?:unrestricted|unfiltered|uncensored|jailbroken))",
    r"\b(?:developer|dan|god)\s+mode\b",
    r"\bjailbreak(?:ed|ing)?\b",
    r"\bdo anything now\b",

    # Smuggled structure (XML/markdown injection)
    r"\bnew (?:instructions?|rules?)\s*:",
    r"<\s*/?\s*(?:system|assistant|instructions?)\s*>",
    r"\[\s*/?\s*(?:inst|system)\s*\]",
)]


def _normalise(text: str) -> str:
    """
    Normalise Unicode and whitespace before pattern matching.

    This stops attackers from evading detection by using:
      - Full-width characters: ＩＧＮＯＲＥ → IGNORE
      - Zero-width characters: ig​nore (zero-width space inside)
      - Bidirectional override characters
    """
    text = unicodedata.normalize("NFKC", text)   # full-width → ASCII
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", text)  # invisible chars
    return re.sub(r"\s+", " ", text.lower())


def detect_injection(text: str) -> bool:
    """
    Return True if the text looks like a prompt-injection attempt.

    Runs after normalisation so Unicode tricks don't bypass the patterns.
    """
    normalised = _normalise(text)
    return any(p.search(normalised) for p in _INJECTION_PATTERNS)


# ── 3. OUTPUT SAFETY RULES (FR-12) ────────────────────────────────────────────

# Matches currency amounts like "USD 50", "$75", "₹1,200", "50%"
_CURRENCY_RE = re.compile(
    r"(?:\b(?:usd|aed|inr|eur|gbp|rs\.?|us\$)|[$€£₹])\s?\d[\d,]*(?:\.\d+)?"
    r"|\d[\d,]*(?:\.\d+)?\s?(?:(?:usd|aed|inr|eur|gbp|rs\.?)\b|dollars?\b|dirhams?\b|rupees?\b|%|percent\b)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_GUARANTEE_RE = re.compile(r"\bguarantee[sd]?\b", re.IGNORECASE)

# Fragments that would appear in the answer only if the system prompt leaked.
_PROMPT_CANARIES = (
    "numbered context passages",
    "never as instructions",
    "you are an assistant that answers questions about emirates",
)


def _canonical(num_str: str) -> str:
    """Normalise a number string for comparison (remove commas, trailing zeros)."""
    n = num_str.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def check_output_rules(answer: str, context: str) -> list[str]:
    """
    Deterministic safety checks on a generated answer.

    Returns a list of violation strings, empty if the answer passes all checks.
    These run before (and cheaper than) the LLM-based output check.

    Checks:
      - unsupported_amount: monetary value in the answer not found in context
      - unsupported_promise: "guarantee" claim with no source in context
      - prompt_leak: system prompt instructions visible in the answer
    """
    violations: list[str] = []
    context_numbers = {_canonical(n) for n in _NUMBER_RE.findall(context)}

    # Check every currency/percentage amount mentioned in the answer
    for match in _CURRENCY_RE.finditer(answer):
        nums = _NUMBER_RE.findall(match.group(0))
        if nums and _canonical(nums[0]) not in context_numbers:
            violations.append(f"unsupported_amount:{match.group(0).strip()}")

    # "Emirates guarantees..." is a strong claim — flag it if the context doesn't also use that word
    if _GUARANTEE_RE.search(answer) and not _GUARANTEE_RE.search(context):
        violations.append("unsupported_promise")

    # Check for system-prompt leakage
    low = answer.lower()
    if any(canary in low for canary in _PROMPT_CANARIES):
        violations.append("prompt_leak")

    return violations
