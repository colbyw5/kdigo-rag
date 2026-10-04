"""The eval question set.

Each question names its gold evidence as **needles** -- content anchors
(guideline + recommendation number, a table caption, or a short verbatim
phrase) rather than chunk IDs, so the gold labels survive re-chunking. Every
needle is machine-verified to exist in the indexed corpus before it's
trusted (``python -m kdigo_guideline_rag.eval --verify``) -- gold labels come
from the corpus, never from memory.

``split`` partitions the set: tune on ``"dev"``; look at ``"heldout"`` only
to confirm a decision already made on dev. The 8 original questions are all
dev -- they already informed the chunking decisions (DECISIONS.md
2026-10-04), so they can't be held out. The rest were assigned once by
``random.Random(42)``, stratified by category, one third held out -- not
hand-picked, and stored explicitly so adding questions never reshuffles. Later batches\n(e.g. the e-consult batch, ckd_13-17 + oos_7) are assigned the same way, within the batch.
"""

from dataclasses import dataclass, field
from typing import Literal

CKD = "KDIGO 2024 CKD Evaluation and Management"
AKI = "KDIGO 2026 AKI/AKD (Public Review Draft)"
GLOMERULAR = "KDIGO 2021 Glomerular Diseases"
BP = "KDIGO 2021 Blood Pressure in CKD"

Category = Literal["recommendation", "table", "narrative", "synthesis", "out_of_scope"]


@dataclass(frozen=True)
class Needle:
    """One piece of gold evidence a good retrieval should surface.

    A chunk matches when it comes from ``guideline`` and every field that is
    set matches: ``label_type`` and ``number`` against chunk metadata,
    ``phrase`` as a substring after lowercasing and dropping everything but
    letters and digits (so whitespace/ligature/punctuation noise like
    "bene fi ts" vs "benefits" doesn't matter).

    Attributes:
        guideline: Required ``source_guideline``.
        label_type: e.g. ``"Recommendation"``, ``"Practice Point"``, ``"Table"``.
            Set ``"Table"`` on caption needles so a table-of-contents line
            mentioning the caption doesn't count.
        number: Recommendation/practice point number -- always qualified by
            guideline and label, since numbers collide (CKD Rec 3.7.1 is
            SGLT2i; AKI Rec 3.7.1 is rasburicase).
        phrase: Verbatim text the chunk must contain.
    """

    guideline: str
    label_type: str | None = None
    number: str | None = None
    phrase: str | None = None


@dataclass(frozen=True)
class AnyOf:
    """A requirement satisfied by any one of several needles.

    Only for evidence that *independently answers* the question -- e.g. the
    GFR-category table reprinted in three guidelines -- never to accept
    merely related content (that would stop the eval telling a real miss
    from a hit).
    """

    options: tuple[Needle, ...]


Requirement = Needle | AnyOf


def any_of(*options: Needle) -> AnyOf:
    """Requirement met by any one of ``options``."""
    return AnyOf(tuple(options))


def rec(guideline: str, number: str) -> Needle:
    """Needle for a numbered Recommendation."""
    return Needle(guideline, label_type="Recommendation", number=number)


def pp(guideline: str, number: str) -> Needle:
    """Needle for a numbered Practice Point."""
    return Needle(guideline, label_type="Practice Point", number=number)


def table(guideline: str, caption: str) -> Needle:
    """Needle for a table/figure chunk by its caption."""
    return Needle(guideline, label_type="Table", phrase=caption)


def phrase(guideline: str, text: str) -> Needle:
    """Needle for a verbatim passage in any chunk type."""
    return Needle(guideline, phrase=text)


@dataclass
class EvalQuestion:
    """One eval case.

    Attributes:
        id: Short identifier, also used as the per-question memory user_id.
        question: The question text sent to the graph.
        category: Query type -- ``"recommendation"`` lookup, ``"table"``/
            numeric, ``"narrative"`` rationale, cross-guideline
            ``"synthesis"``, or ``"out_of_scope"`` (the correct behavior is
            to say the guidelines don't cover it).
        needles: Gold evidence requirements (each a :class:`Needle` or
            :class:`AnyOf`); empty for out-of-scope questions.
        split: ``"dev"`` (tune on it) or ``"heldout"`` (confirm only).
        phrasing: ``"guideline"`` wording, or ``"lay"`` -- clinician/patient
            wording that doesn't echo the source text.
        scope: For out-of-scope questions -- ``"far"`` (obviously unrelated)
            or ``"near"`` (sounds like KDIGO; tempting adjacent content).
        baseline_answer: A previously captured answer, for human diffing.
        notes: Freeform context for a human reviewer.
    """

    id: str
    question: str
    category: Category
    needles: list[Requirement]
    split: Literal["dev", "heldout"]
    phrasing: Literal["guideline", "lay"] = "guideline"
    scope: Literal["far", "near"] | None = None
    baseline_answer: str = ""
    notes: str = ""
    expected_guidelines: list[str] = field(init=False)

    def __post_init__(self) -> None:
        options = [o for n in self.needles for o in (n.options if isinstance(n, AnyOf) else (n,))]
        self.expected_guidelines = sorted({o.guideline for o in options})


EVAL_QUESTIONS: list[EvalQuestion] = [
    # ---------------------------------------------------------------- CKD
    EvalQuestion(
        id="ckd_1",
        question="Should a patient with G3b A2 CKD be on an SGLT2 inhibitor?",
        category="recommendation",
        needles=[rec(CKD, "3.7.1"), rec(CKD, "3.7.3")],
        split="dev",
        baseline_answer=(
            "Depends on diabetes status: with T2D, SGLT2i for eGFR >=20 (1A, Rec 3.7.1); without "
            "diabetes, eGFR 20-45 with ACR <200 mg/g (2B, Rec 3.7.3) -- A2 spans 30-300, so ACR "
            "200-300 in non-diabetics isn't explicitly addressed by these passages."
        ),
    ),
    EvalQuestion(
        id="ckd_2",
        question="What eGFR stage is 38 mL/min/1.73m²?",
        category="table",
        needles=[
            any_of(
                table(CKD, "Table 2| GFR categories in CKD"),
                # KDIGO 2012 GFR x albuminuria heat map, reprinted in both; carries the same G3b row.
                table(GLOMERULAR, "G3b Moderately to severely decreased 30-44"),
                table(BP, "G3b Moderately to severely decreased 30-44"),
            )
        ],
        split="dev",
        baseline_answer="G3b, 'moderately to severely decreased', 30-44 mL/min/1.73m2.",
        notes=(
            "Known limitation (README): embeddings don't do numeric interval reasoning; resolves "
            "only if the GFR-category table is retrieved."
        ),
    ),
    EvalQuestion(
        id="ckd_3",
        question=(
            "Does KDIGO recommend statin therapy for adults aged 50 or older with eGFR below 60 "
            "who are not on dialysis?"
        ),
        category="recommendation",
        needles=[rec(CKD, "3.15.1.1")],
        split="heldout",
    ),
    EvalQuestion(
        id="ckd_4",
        question=(
            "Is it worth starting allopurinol in a CKD patient with high uric acid but no gout, "
            "just to protect the kidneys?"
        ),
        category="recommendation",
        needles=[rec(CKD, "3.14.2")],
        split="heldout",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ckd_5",
        question="What dietary protein intake is suggested for adults with CKD G3 to G5?",
        category="recommendation",
        needles=[rec(CKD, "3.3.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="ckd_6",
        question="What actions are suggested for moderate versus severe hyperkalemia in CKD?",
        category="table",
        needles=[table(CKD, "Table 28| Suggested action in the event of moderate and severe hyperkalemia")],
        split="dev",
    ),
    EvalQuestion(
        id="ckd_7",
        question="For a patient with CKD and atrial fibrillation, which class of oral anticoagulant is preferred?",
        category="recommendation",
        needles=[rec(CKD, "3.16.1")],
        split="dev",
    ),
    EvalQuestion(
        id="ckd_8",
        question=(
            "My patient is very muscular, so their creatinine-based eGFR might be misleading -- "
            "is there a better way to estimate their kidney function?"
        ),
        category="recommendation",
        needles=[rec(CKD, "1.2.2.1"), table(CKD, "Table 8| Indications for use of cystatin C")],
        split="heldout",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ckd_9",
        question=(
            "Which patients with type 2 diabetes and CKD should be considered for a nonsteroidal "
            "mineralocorticoid receptor antagonist?"
        ),
        category="recommendation",
        needles=[rec(CKD, "3.8.1")],
        split="heldout",
    ),
    EvalQuestion(
        id="ckd_10",
        question="How should the risk of progression to kidney failure be estimated in people with CKD G3 to G5?",
        category="recommendation",
        needles=[rec(CKD, "2.2.1")],
        split="dev",
    ),
    EvalQuestion(
        id="ckd_11",
        question="My CKD patient has bad gastroenteritis -- which of their usual medications should they pause?",
        category="narrative",
        needles=[phrase(CKD, "Sick day rules have been endorsed as useful guidance")],
        split="dev",
        phrasing="lay",
        notes="Text also notes a paucity of evidence for sick day rules -- a good answer keeps that caveat.",
    ),
    EvalQuestion(
        id="ckd_12",
        question="How often should creatinine be monitored in a CKD patient taking lithium?",
        category="narrative",
        needles=[
            phrase(CKD, "such as lithium or methotrexate, require at least annual monitoring of creatinine")
        ],
        split="heldout",
    ),
    # Modeled on real primary-care e-consult categories (Keely et al. 2018, PMC5788130):
    # "should I refer" 11%, lab interpretation 17%, drug use in CKD 12%, imaging 13-17%.
    EvalQuestion(
        id="ckd_13",
        question="When should I refer a patient with CKD to nephrology?",
        category="recommendation",
        needles=[pp(CKD, "5.1.1")],
        split="dev",
        phrasing="lay",
        notes=(
            "PP 5.1.1 only points to Figure 48, whose criteria (eGFR <30, KRT risk >3-5%/5y, ACR >=300 "
            "with hematuria, ...) are Docling picture children the chunker drops. Upgrade the needle to "
            "the figure text once figures are indexed."
        ),
    ),
    EvalQuestion(
        id="ckd_14",
        question="A routine urine test showed a raised albumin-to-creatinine ratio -- does my patient have CKD?",
        category="recommendation",
        needles=[pp(CKD, "1.1.1.2")],
        split="heldout",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ckd_15",
        question="Which urine test should I use for initial testing of albuminuria?",
        category="recommendation",
        needles=[pp(CKD, "1.3.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="ckd_16",
        question="How should I dose renally cleared medications in a patient with CKD -- which eGFR do I use?",
        category="recommendation",
        needles=[any_of(pp(CKD, "4.2.1"), pp(CKD, "4.2.2"))],
        split="heldout",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ckd_17",
        question="When should planning for dialysis access or a preemptive kidney transplant begin?",
        category="recommendation",
        needles=[pp(CKD, "5.4.3")],
        split="dev",
    ),
    # ---------------------------------------------------------------- AKI
    EvalQuestion(
        id="aki_1",
        question="What defines Stage 2 AKI by KDIGO criteria?",
        category="table",
        needles=[table(AKI, "Table 6 | Severity staging system for AKI")],
        split="dev",
        baseline_answer="SCr 2.0-2.9x baseline (C2), or UO <0.5 mL/kg/h for >12 h (U2); higher stage wins.",
    ),
    EvalQuestion(
        id="aki_2",
        question=(
            "Should buffered crystalloids or 0.9% saline be used for volume expansion in "
            "patients at risk of AKI?"
        ),
        category="recommendation",
        needles=[rec(AKI, "3.1.2")],
        split="dev",
    ),
    EvalQuestion(
        id="aki_3",
        question="Is low-dose dopamine recommended for kidney protection in AKI?",
        category="recommendation",
        needles=[rec(AKI, "3.2.2")],
        split="dev",
    ),
    EvalQuestion(
        id="aki_4",
        question="Should renal replacement therapy be started early or deferred in adults with AKI?",
        category="recommendation",
        needles=[rec(AKI, "5.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="aki_5",
        question="What effluent dose is recommended for adults receiving CRRT?",
        category="recommendation",
        needles=[rec(AKI, "5.3.1")],
        split="heldout",
    ),
    EvalQuestion(
        id="aki_6",
        question="What anticoagulation is preferred to keep the CRRT circuit patent?",
        category="recommendation",
        needles=[rec(AKI, "5.5.1")],
        split="heldout",
    ),
    EvalQuestion(
        id="aki_7",
        question="Should my patient stop their ACE inhibitor before a CT scan with IV contrast?",
        category="recommendation",
        needles=[rec(AKI, "4.9.3.1")],
        split="dev",
        phrasing="lay",
    ),
    EvalQuestion(
        id="aki_8",
        question="What criteria define acute kidney disease (AKD)?",
        category="table",
        needles=[table(AKI, "Table 9 | Criteria for AKD")],
        split="heldout",
    ),
    EvalQuestion(
        id="aki_9",
        question="How should hyperuricemia be prevented in patients at high risk of tumor lysis syndrome?",
        category="recommendation",
        needles=[rec(AKI, "3.7.1")],
        split="heldout",
        notes="Disambiguation: AKI Rec 3.7.1, not CKD Rec 3.7.1 (SGLT2i) or CKD 3.14.x (uric acid in CKD).",
    ),
    EvalQuestion(
        id="aki_10",
        question="What follow-up is recommended after hospital discharge for a patient who had AKI?",
        category="recommendation",
        needles=[rec(AKI, "6.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="aki_11",
        question="When should medications that were stopped during an AKI episode be restarted?",
        category="narrative",
        needles=[phrase(AKI, "The optimal timing to restart medications after AKI/AKD is unknown")],
        split="dev",
        notes="Honest answer includes 'timing is unknown; individualize' -- not a confident fixed rule.",
    ),
    # ---------------------------------------------------------- Glomerular
    EvalQuestion(
        id="glom_1",
        question="What is the first-line treatment approach for IgA nephropathy with declining GFR?",
        category="narrative",
        needles=[phrase(GLOMERULAR, "The primary focus of management should be optimized supportive care")],
        split="dev",
        baseline_answer=(
            "Optimized supportive care first (BP control, maximal RAS inhibition, lifestyle); "
            "immunosuppression is an additional consideration, not first-line."
        ),
    ),
    EvalQuestion(
        id="glom_2",
        question=(
            "How should membranous nephropathy be treated in a patient with at least one risk "
            "factor for disease progression?"
        ),
        category="recommendation",
        needles=[rec(GLOMERULAR, "3.3.1")],
        split="dev",
    ),
    EvalQuestion(
        id="glom_3",
        question="What is the initial treatment for minimal change disease in adults?",
        category="recommendation",
        needles=[
            any_of(rec(GLOMERULAR, "5.3.1"), table(GLOMERULAR, "Figure 45 | Treatment of MCD in adults"))
        ],
        split="dev",
    ),
    EvalQuestion(
        id="glom_4",
        question="What induction therapy is recommended for new-onset ANCA-associated vasculitis?",
        category="recommendation",
        needles=[rec(GLOMERULAR, "9.3.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="glom_5",
        question="How should anti-GBM antibody glomerulonephritis be treated?",
        category="recommendation",
        needles=[rec(GLOMERULAR, "11.2.1")],
        split="heldout",
    ),
    EvalQuestion(
        id="glom_6",
        question="What is the second-line therapy for children with steroid-resistant nephrotic syndrome?",
        category="recommendation",
        needles=[rec(GLOMERULAR, "4.4.1")],
        split="dev",
    ),
    EvalQuestion(
        id="glom_7",
        question="When should antiretroviral therapy be started in a patient with HIV and kidney disease?",
        category="recommendation",
        needles=[rec(GLOMERULAR, "7.2.3.3.1")],
        split="dev",
    ),
    EvalQuestion(
        id="glom_8",
        question="Should glucocorticoids be given to prevent nephritis in IgA vasculitis without kidney involvement?",
        category="recommendation",
        needles=[rec(GLOMERULAR, "2.7.1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="glom_9",
        question="How is rapidly progressive IgA nephropathy defined?",
        category="narrative",
        needles=[
            phrase(
                GLOMERULAR, "Rapidly progressive IgAN is defined as a ≥ 50% decline in eGFR over ≤ 3 months"
            )
        ],
        split="dev",
    ),
    # ------------------------------------------------------------------ BP
    EvalQuestion(
        id="bp_1",
        question="What is the target blood pressure for adults with CKD not on dialysis?",
        category="synthesis",
        needles=[rec(CKD, "3.4.1"), rec(BP, "3.1.1")],
        split="dev",
        baseline_answer="SBP <120 mmHg by standardized office measurement, when tolerated (2B); both guidelines agree.",
    ),
    EvalQuestion(
        id="bp_2",
        question=(
            "How should blood pressure management be tailored for a CKD patient based on their "
            "GFR/albuminuria risk category?"
        ),
        category="narrative",
        needles=[
            phrase(BP, "CKD G4 and G5: For people with a lower GFR"),
            phrase(BP, "and no albuminuria are at lower risk of CKD progression"),
        ],
        split="dev",
        notes=(
            "Known limitation (README): full heat-map cell targets aren't extractable; a good answer "
            "gives the directional principles and says so. Regressed under greedy 250w packing "
            "(DECISIONS 2026-10-04)."
        ),
    ),
    EvalQuestion(
        id="bp_3",
        question="A kidney transplant recipient has high blood pressure -- which drug should I start first?",
        category="recommendation",
        needles=[rec(BP, "4.1")],
        split="dev",
        phrasing="lay",
    ),
    EvalQuestion(
        id="bp_4",
        question="Should standardized office blood pressure measurement be used instead of routine office readings?",
        category="recommendation",
        needles=[rec(BP, "1.1")],
        split="dev",
    ),
    EvalQuestion(
        id="bp_5",
        question="Can an ACE inhibitor be combined with an ARB in a patient with CKD?",
        category="synthesis",
        needles=[rec(BP, "3.3.1"), rec(CKD, "3.6.4")],
        split="heldout",
        notes="Both guidelines carry the same recommendation (1B) -- tests whether both surface.",
    ),
    EvalQuestion(
        id="bp_6",
        question="How much salt should someone with kidney disease and high blood pressure be eating?",
        category="synthesis",
        needles=[rec(BP, "2.1.1"), rec(CKD, "3.3.2.1")],
        split="dev",
        phrasing="lay",
    ),
    EvalQuestion(
        id="bp_7",
        question="What blood pressure target is recommended for children with CKD?",
        category="synthesis",
        needles=[rec(BP, "5.1"), rec(CKD, "3.4.2")],
        split="dev",
    ),
    EvalQuestion(
        id="bp_8",
        question="Is there a diastolic blood pressure target for adults with CKD?",
        category="narrative",
        needles=[phrase(BP, "The Work Group chose not to provide a target for DBP")],
        split="heldout",
    ),
    # ------------------------------------------------- cross-guideline
    EvalQuestion(
        id="aki_ckd_1",
        question=(
            "A patient with CKD develops an acute AKI episode -- how should acute-phase "
            "management differ from routine CKD care?"
        ),
        category="synthesis",
        needles=[rec(AKI, "4.11.1"), Needle(AKI, label_type="Practice Point", number="6.2.3")],
        split="dev",
        notes=(
            "Designed as AKI+CKD; in practice answered from the AKI/AKD guideline alone (it has "
            "its own CKD-transition content). Needles reflect what grounds a good answer."
        ),
    ),
    EvalQuestion(
        id="glom_bp_1",
        question="A patient with lupus nephritis also has hypertension -- how should blood pressure be managed?",
        category="synthesis",
        needles=[rec(GLOMERULAR, "2.3.1"), phrase(GLOMERULAR, "Practice Point 1.5.2")],
        split="dev",
        notes=(
            "Glomerular Ch 1 general BP principles apply by reference; PP 1.5.2 (SBP <120) exists "
            "only inside Figure 8. Under greedy packing it also pulled CKD Rec 3.4.1."
        ),
    ),
    # ------------------------------------------------------ out of scope
    EvalQuestion(
        id="oos_1",
        question="What is the recommended starting dose of sacubitril/valsartan for heart failure with reduced ejection fraction?",
        category="out_of_scope",
        needles=[],
        split="heldout",
        scope="far",
    ),
    EvalQuestion(
        id="oos_2",
        question="What is the first-line antibiotic regimen for community-acquired pneumonia in adults?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="far",
    ),
    EvalQuestion(
        id="oos_3",
        question="What hemoglobin target should be used when treating anemia of CKD with an erythropoiesis-stimulating agent?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="near",
        notes="Covered by the separate KDIGO anemia guideline, not in this corpus (0 hits for ESA terms).",
    ),
    EvalQuestion(
        id="oos_4",
        question="What dialysate sodium concentration is recommended for maintenance hemodialysis?",
        category="out_of_scope",
        needles=[],
        split="heldout",
        scope="near",
        notes="AKI covers acute RRT only; maintenance HD prescription isn't in the corpus.",
    ),
    EvalQuestion(
        id="oos_5",
        question="What tacrolimus trough level should be targeted after kidney transplantation?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="near",
        notes=(
            "Extrapolation trap: Glomerular gives a tacrolimus trough (5-10 ng/ml) for FSGS, not "
            "transplant. Quoting it for transplant = failure."
        ),
    ),
    EvalQuestion(
        id="oos_6",
        question="What dose of tolvaptan is recommended for ADPKD?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="near",
        notes="Tolvaptan/ADPKD mentioned only in a risk-prediction context; no dosing in corpus.",
    ),
    EvalQuestion(
        id="oos_7",
        question="How should a simple kidney cyst found incidentally on ultrasound be followed up?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="near",
        notes="Imaging is 13-17% of real e-consults but outside KDIGO's scope (no simple-cyst/Bosniak content).",
    ),
]
