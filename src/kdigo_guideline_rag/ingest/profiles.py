"""Format profiles: a guideline's parsing decisions, saved once and reused.

A profile is the serializable subset of :class:`GuidelineSource` that varies
between guideline formats (recommendation/grade/heading patterns, page
furniture, glyph fixes, which extractor supplies recommendation text). The
onboarding step (``onboard.py``) decides it once for a new document and writes
it to ``profiles/<slug>.json``; ingestion then just loads it -- the decision is
never re-made on later ingests.
"""

import json
import re
from dataclasses import fields
from pathlib import Path

from kdigo_guideline_rag.ingest.sources import (
    DATA_DIR,
    KI_SYMBOL_FONT_MAP,
    KI_TEXT_SUBSTITUTIONS,
    GuidelineSource,
)

PROFILES_DIR = Path(__file__).parent / "profiles"

PATTERN_FIELDS = (
    "recommendation_pattern",
    "grade_pattern",
    "chapter_pattern",
    "ignore_heading_pattern",
    "heading_fallback_pattern",
    "reference_heading_pattern",
)
PLAIN_FIELDS = (
    "recommendation_label",
    "recommendation_extractor",
    "repair_split_ligatures",
    "repair_missing_spaces",
)
# Named glyph maps a profile may reference instead of spelling characters out.
NAMED_CHAR_MAPS = {"KI_SYMBOL_FONT_MAP": KI_SYMBOL_FONT_MAP}
NAMED_SUBSTITUTIONS = {"KI_TEXT_SUBSTITUTIONS": KI_TEXT_SUBSTITUTIONS}


def _pattern_str(pattern: re.Pattern | None) -> str | None:
    """Pattern as a self-describing string: case-insensitivity becomes an inline ``(?i)``."""
    if pattern is None:
        return None
    return ("(?i)" if pattern.flags & re.IGNORECASE else "") + pattern.pattern


def profile_from_source(source: GuidelineSource) -> dict:
    """The profile settings of an existing source (patterns as strings)."""
    profile = {f: _pattern_str(getattr(source, f)) for f in PATTERN_FIELDS}
    profile |= {f: getattr(source, f) for f in PLAIN_FIELDS}
    profile["noise_patterns"] = [_pattern_str(p) for p in source.noise_patterns]
    profile["text_substitutions"] = next(
        (name for name, subs in NAMED_SUBSTITUTIONS.items() if subs == source.text_substitutions),
        [[_pattern_str(p), r] for p, r in source.text_substitutions],
    )
    profile["char_replacements"] = next(
        (name for name, m in NAMED_CHAR_MAPS.items() if m == source.char_replacements),
        source.char_replacements,
    )
    return profile


def sources_from_profile(profile: dict, name: str, pdf_paths: list[Path]) -> list[GuidelineSource]:
    """One :class:`GuidelineSource` per PDF, all sharing ``name`` and the profile's settings.

    Raises ``ValueError`` on an invalid pattern, unknown field, or a
    recommendation pattern without a ``number`` group -- a profile must fail
    at load, not mid-ingest.
    """
    known = (
        set(PATTERN_FIELDS)
        | set(PLAIN_FIELDS)
        | {"noise_patterns", "char_replacements", "text_substitutions"}
    )
    unknown = set(profile) - known
    if unknown:
        raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    kwargs: dict = {}
    for f in PATTERN_FIELDS:
        if f in profile:
            kwargs[f] = re.compile(profile[f]) if profile[f] else None
    rec = kwargs.get("recommendation_pattern")
    if rec is not None and "number" not in rec.groupindex:
        raise ValueError("recommendation_pattern needs a named group 'number'")
    kwargs |= {f: profile[f] for f in PLAIN_FIELDS if f in profile}
    kwargs["noise_patterns"] = [re.compile(p) for p in profile.get("noise_patterns", [])]
    subs = profile.get("text_substitutions", [])
    kwargs["text_substitutions"] = (
        NAMED_SUBSTITUTIONS[subs] if isinstance(subs, str) else tuple((re.compile(p), r) for p, r in subs)
    )
    chars = profile.get("char_replacements", {})
    kwargs["char_replacements"] = NAMED_CHAR_MAPS[chars] if isinstance(chars, str) else dict(chars)
    valid = {f.name for f in fields(GuidelineSource)}
    assert set(kwargs) <= valid, set(kwargs) - valid
    return [GuidelineSource(name=name, pdf_path=path, **kwargs) for path in pdf_paths]


def save_profile(
    slug: str,
    name: str,
    pdf_paths: list[Path],
    profile: dict,
    report: dict,
    profiles_dir: Path = PROFILES_DIR,
) -> Path:
    """Write a decided profile (with the scorecard and rationale behind it).

    Anything in :data:`PROFILES_DIR` is ingested as a live source -- write
    experiments elsewhere.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{slug}.json"
    payload = {
        "name": name,
        "pdf_files": [Path(p).name for p in pdf_paths],
        "profile": profile,
        "decided_by": report,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def load_profile_sources(profiles_dir: Path = PROFILES_DIR) -> list[GuidelineSource]:
    """All saved profiles, expanded to sources (PDFs resolved against ``data/``)."""
    sources: list[GuidelineSource] = []
    for path in sorted(profiles_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        sources += sources_from_profile(
            payload["profile"], payload["name"], [DATA_DIR / f for f in payload["pdf_files"]]
        )
    return sources
