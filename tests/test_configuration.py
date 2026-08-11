"""Unit tests for Configuration.from_runnable_config()."""

from kdigo_guideline_rag.configuration import GUIDELINE_FILTER_MAP, Configuration


def test_defaults_with_no_config():
    cfg = Configuration.from_runnable_config(None)
    assert cfg.guideline_filter == "all"
    assert cfg.clinical_focus == "kidney"


def test_reads_known_fields():
    config = {"configurable": {"guideline_filter": "ckd_only", "clinical_focus": "kidney"}}
    cfg = Configuration.from_runnable_config(config)
    assert cfg.guideline_filter == "ckd_only"


def test_ignores_platform_injected_unknown_keys():
    """LangGraph Platform injects extra configurable keys (e.g. thread_id,
    user_id) that aren't part of Configuration -- constructing the dataclass
    must not blow up on them.
    """
    config = {"configurable": {"thread_id": "abc-123", "user_id": "u1", "guideline_filter": "aki_only"}}
    cfg = Configuration.from_runnable_config(config)
    assert cfg.guideline_filter == "aki_only"


def test_guideline_filter_map_covers_every_non_all_option():
    filter_values = set(Configuration.__dataclass_fields__["guideline_filter"].type.__args__)
    assert filter_values - {"all"} == set(GUIDELINE_FILTER_MAP.keys())
