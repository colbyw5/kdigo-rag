"""Dictionary-guided repair of two common PDF text-extraction artifacts.

Both are generic to PDF extraction, not to any one publisher:

- **Split ligatures**: a typeset "fi"/"fl"/... ligature glyph comes out as a
  separate token -- "bene fi ts", "of fi ce", "glomerular fi ltration".
  Word-initial and mid-word splits look identical ("of fi ce" -> office vs.
  "the fi rst" -> the first), so a regex can't decide; word frequencies can.
- **Missing spaces**: a text run with no word gaps -- "WesuggestthatadultswithhighBP...".
  Rare (~0.5% of chunks here) but it wrecks that chunk's embedding.

Word frequencies come from ``wordsegment``'s unigram table (English web
text). Both repairs only act where the evidence is strong, because a wrong
"fix" inside a clinical recommendation is worse than leaving an artifact.
"""

import math
import re
from functools import lru_cache

# Unicode ligature code points some PDFs emit instead of split letters.
LIGATURE_CHARS = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})

# "X lig Y", or "lig Y" at a word start -- the space before a bare ligature
# isn't consumed, so table cells like "| fi rst |" keep their padding.
SPLIT_LIGATURE_PATTERN = re.compile(r"(?:([A-Za-z]+)[ \t]+|(?<![A-Za-z]))(ffi|ffl|ff|fi|fl)[ \t]+([A-Za-z]+)")

# A run this long with no space is almost never one real word.
MISSING_SPACE_RUN_PATTERN = re.compile(r"[A-Za-z]{25,}")
CASE_RUN_PATTERN = re.compile(r"[A-Z]+|[a-z]+")
# Prose merged without spaces virtually always contains one of these; a long
# medical compound ("ethylenediaminetetraacetic") never does -- so their
# presence is the evidence required before splitting.
FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "be",
        "was",
        "were",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "not",
        "no",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "than",
        "then",
        "which",
        "who",
        "plus",
        "versus",
    ]
)


@lru_cache(maxsize=1)
def _unigrams() -> dict[str, float]:
    import wordsegment

    wordsegment.load()
    return wordsegment.UNIGRAMS


def _known(word: str) -> float:
    """log10 frequency of ``word`` (0 for unknown words, e.g. most drug names)."""
    return math.log10(_unigrams().get(word.lower(), 0) + 1)


def repair_split_ligatures(text: str) -> str:
    """Rejoin ligature fragments: "bene fi ts" -> "benefits", "the fi rst" -> "the first".

    For each "X lig Y", ranks "XligY", "X ligY", "Xlig Y" by their
    least-frequent word, then by how many of their words are known -- so
    "and fi nerenone" -> "and finerenone" and "tubulointerstitial fi brosis"
    -> "tubulointerstitial fibrosis" even though the drug/compound is unknown.
    With no known word at all ("Empagli fl ozin"), joins all three.
    """
    text = text.translate(LIGATURE_CHARS)

    def fix(match: re.Match) -> str:
        left, lig, right = match.group(1) or "", match.group(2), match.group(3)
        candidates = [[left + lig + right], [left, lig + right], [left + lig, right]]
        candidates = [[t for t in c if t] for c in candidates]

        def rank(c: list[str]) -> tuple[float, int]:
            return min(_known(t) for t in c), sum(_known(t) > 0 for t in c)

        best = max(candidates, key=rank)
        if rank(best) == (0, 0):
            best = candidates[0]
        return " ".join(best)

    return SPLIT_LIGATURE_PATTERN.sub(fix, text)


def _first_word_length(token: str) -> int:
    """Length of the first segmented word, if it's a real word (0 otherwise).

    Length, not frequency: "ontroversies" segments to "on" + ..., and "on" is
    very frequent -- but "controversies" is the reading that explains more text.
    """
    import wordsegment

    _unigrams()
    words = wordsegment.segment(token)
    return len(words[0]) if words and _known(words[0]) > 2 else 0


def _case_pieces(run: str) -> list[str]:
    """Split a run at case boundaries, keeping acronyms whole.

    At an UPPER->lower boundary the last capital is ambiguous: it starts the
    next word in "KDIGOControversies" but ends the acronym in "CKDbe". Decide
    by which reading begins with the more plausible word.
    """
    runs = CASE_RUN_PATTERN.findall(run)
    pieces: list[str] = []
    i = 0
    while i < len(runs):
        cur = runs[i]
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if cur.isupper() and nxt is not None and nxt.islower():
            if len(cur) == 1:
                pieces.append(cur + nxt)  # "We" + "suggest..."
            elif _first_word_length(cur[-1] + nxt) > _first_word_length(nxt):
                pieces += [cur[:-1], cur[-1] + nxt]  # KDIGO | Controversies
            else:
                pieces += [cur, nxt]  # CKD | be...
            i += 2
            continue
        pieces.append(cur)
        i += 1
    return pieces


def _segment_preserving_case(token: str) -> str:
    import wordsegment

    _unigrams()  # ensure loaded
    words = wordsegment.segment(token)
    if not FUNCTION_WORDS.intersection(words) or "".join(words) != token.lower():
        return token
    out, i = [], 0
    for w in words:
        out.append(token[i : i + len(w)])
        i += len(w)
    return " ".join(out)


def repair_missing_spaces(text: str) -> str:
    """Re-insert spaces in long no-space runs, keeping original case and acronyms.

    Only runs of 25+ letters are touched; each is split at case boundaries
    (acronyms kept whole), then each lowercase piece is word-segmented only
    if the result contains a function word -- so long single words survive.
    """

    def fix(match: re.Match) -> str:
        pieces = _case_pieces(match.group(0))
        if len(pieces) == 1:
            return _segment_preserving_case(pieces[0])
        return " ".join(p if p.isupper() else _segment_preserving_case(p) for p in pieces)

    return MISSING_SPACE_RUN_PATTERN.sub(fix, text)
