"""Multi-turn eval: short conversations whose last turn only makes sense in context.

Scored on the final turn (``eval/multiturn.py``):

- **retrieval** -- the final turn's needles must be retrieved (same matching as
  tier 1), which only happens if the follow-up was condensed correctly;
- **context handling** -- the condensed standalone question must mention what
  the follow-up depends on (``must_mention``: each group is any-of), and must
  not drag in context from a topic the user moved away from
  (``must_not_mention``).

Needles are verified against the index like the single-turn set. Splits:
assigned once by ``random.Random(42)``, stratified by ``kind``, one third
held out -- stored explicitly.
"""

from dataclasses import dataclass, field
from typing import Literal

from kdigo_guideline_rag.eval.questions import (
    AKI,
    BP,
    CKD,
    GLOMERULAR,
    Requirement,
    any_of,
    figure,
    phrase,
    rec,
    table,
)


@dataclass
class Conversation:
    """One multi-turn eval case.

    Attributes:
        id: Identifier; also the thread id and memory user id.
        turns: User messages, in order.
        kind: ``"carry"`` (follow-up depends on earlier context), ``"switch"``
            (follow-up changes topic -- earlier context must NOT leak in), or
            ``"out_of_scope"`` (follow-up isn't covered by the guidelines).
        needles: Gold evidence for the final turn; empty for out-of-scope.
        must_mention: Groups of terms; the final standalone question must
            contain at least one term from each group (case-insensitive).
        must_not_mention: Terms the final standalone question must not contain.
        split: ``"dev"`` or ``"heldout"``.
        notes: Freeform context for a reviewer.
    """

    id: str
    turns: list[str]
    kind: Literal["carry", "switch", "out_of_scope"]
    needles: list[Requirement]
    split: Literal["dev", "heldout"]
    must_mention: list[tuple[str, ...]] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    notes: str = ""


CONVERSATIONS: list[Conversation] = [
    Conversation(
        id="conv_patient_sglt2",
        turns=[
            "My patient has CKD G3b with A2 albuminuria and type 2 diabetes.",
            "Should they be on an SGLT2 inhibitor?",
        ],
        kind="carry",
        needles=[rec(CKD, "3.7.1")],
        must_mention=[("diabet", "T2D"), ("G3b", "eGFR", "albuminuria", "A2", "CKD")],
        split="dev",
        notes="Patient details carried within the thread -- the reframed form of the original per-patient memory idea.",
    ),
    Conversation(
        id="conv_bp_children",
        turns=[
            "What blood pressure target is recommended for adults with CKD not on dialysis?",
            "What about in children?",
        ],
        kind="carry",
        needles=[any_of(rec(BP, "5.1"), rec(CKD, "3.4.2"))],
        must_mention=[("child", "pediatric", "paediatric"), ("blood pressure", "BP")],
        split="heldout",
    ),
    Conversation(
        id="conv_topic_switch",
        turns=[
            "Should a patient with G3b A2 CKD be on an SGLT2 inhibitor?",
            "Separately, what defines Stage 2 AKI?",
        ],
        kind="switch",
        needles=[table(AKI, "Table 6 | Severity staging system for AKI")],
        must_mention=[("Stage 2", "AKI")],
        must_not_mention=["SGLT2", "G3b", "A2"],
        split="dev",
        notes="Self-contained new topic: condensing must leave it alone.",
    ),
    Conversation(
        id="conv_aav_maintenance",
        turns=[
            "What induction therapy is recommended for new-onset ANCA-associated vasculitis?",
            "And what maintenance therapy should follow it?",
        ],
        kind="carry",
        needles=[rec(GLOMERULAR, "9.3.2.1")],
        must_mention=[("ANCA", "vasculitis", "AAV")],
        split="dev",
    ),
    Conversation(
        id="conv_igan_three_turns",
        turns=[
            "My patient has IgA nephropathy.",
            "Their eGFR is falling -- what is the first-line management approach?",
            "When would this count as rapidly progressive disease?",
        ],
        kind="carry",
        needles=[
            phrase(
                GLOMERULAR, "Rapidly progressive IgAN is defined as a ≥ 50% decline in eGFR over ≤ 3 months"
            )
        ],
        must_mention=[("IgA", "IgAN")],
        split="heldout",
        notes="Context from turn 1 must survive two hops.",
    ),
    Conversation(
        id="conv_oos_followup",
        turns=[
            "What is the target blood pressure for adults with CKD not on dialysis?",
            "And what tacrolimus trough level should a kidney transplant recipient have?",
        ],
        kind="out_of_scope",
        needles=[],
        must_mention=[("tacrolimus",), ("transplant",)],
        must_not_mention=["120"],
        split="dev",
        notes="Out-of-scope follow-up after an in-scope turn: the answer should decline, not reuse the BP context.",
    ),
    Conversation(
        id="conv_finerenone_potassium",
        turns=[
            "Which patients with type 2 diabetes and CKD should be considered for a nonsteroidal mineralocorticoid receptor antagonist?",
            "How should potassium be monitored after starting it?",
        ],
        kind="carry",
        needles=[figure(CKD, "Figure 26| Serum potassium monitoring during treatment")],
        must_mention=[("finerenone", "mineralocorticoid", "MRA")],
        split="dev",
    ),
    Conversation(
        id="conv_rasi_surgery",
        turns=[
            "Should my patient stop their ACE inhibitor before a CT scan with IV contrast?",
            "What about before major elective surgery?",
        ],
        kind="carry",
        needles=[rec(AKI, "4.10.1")],
        must_mention=[("ACE", "RAS", "renin", "angiotensin"), ("surgery",)],
        split="dev",
    ),
]
