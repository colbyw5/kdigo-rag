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
hand-picked, and stored explicitly so adding questions never reshuffles. Later batches
(the e-consult batch, ckd_13-17 + oos_7; the ADA batch, ada_*, ada_kdigo_*, oos_8-9)
are assigned the same way, within the batch.
"""

from dataclasses import dataclass, field
from typing import Literal

CKD = "KDIGO 2024 CKD Evaluation and Management"
AKI = "KDIGO 2026 AKI/AKD (Public Review Draft)"
GLOMERULAR = "KDIGO 2021 Glomerular Diseases"
BP = "KDIGO 2021 Blood Pressure in CKD"
ADA = "ADA Standards of Care in Diabetes 2026"

Category = Literal["recommendation", "table", "figure", "narrative", "synthesis", "out_of_scope"]


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


def figure(guideline: str, text: str) -> Needle:
    """Needle for a figure chunk (Docling picture text) by its caption or verbatim content."""
    return Needle(guideline, label_type="Figure", phrase=text)


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
            numeric, ``"figure"`` (answer only in a figure's text),
            ``"narrative"`` rationale, cross-guideline
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
        needles=[figure(CKD, "Figure 48| Circumstances for referral to specialist kidney care services")],
        split="dev",
        phrasing="lay",
        notes=(
            "PP 5.1.1 only says 'see Figure 48'; the criteria (eGFR <30, KRT risk >3-5%/5y, ACR >=300 "
            "with hematuria, ...) live in the figure's text, indexed since 2026-10-06."
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
    # Answerable only from figure text (Docling picture children), indexed 2026-10-06.
    EvalQuestion(
        id="ckd_18",
        question="How should serum potassium be monitored after starting finerenone?",
        category="figure",
        needles=[figure(CKD, "Figure 26| Serum potassium monitoring during treatment")],
        split="heldout",
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
        needles=[
            any_of(rec(GLOMERULAR, "3.3.1"), figure(GLOMERULAR, "Figure 31| Risk-based treatment of MN"))
        ],
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
    EvalQuestion(
        id="glom_10",
        question="How should PLA2R antibody levels guide monitoring after starting treatment for membranous nephropathy?",
        category="recommendation",
        needles=[
            any_of(
                pp(GLOMERULAR, "3.3.4"),
                figure(GLOMERULAR, "Figure 33| Immunologic monitoring in MN after start of therapy"),
            )
        ],
        notes="Drafted as figure-only, but PP 3.3.4 answers it directly -- recategorized.",
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
    EvalQuestion(
        id="bp_9",
        question="How should metformin dosing be adjusted as eGFR declines?",
        category="figure",
        needles=[figure(BP, "Stop metformin; do not initiate metformin")],
        split="dev",
        notes="Uncaptioned figure (Docling detached the caption); only the figure carries the eGFR dosing bands.",
    ),
    # ---------------------------------------------------------------- ADA
    EvalQuestion(
        id="ada_1",
        question="What blood pressure goal does the ADA recommend for adults with diabetes and hypertension?",
        category="recommendation",
        needles=[rec(ADA, "10.4")],
        split="dev",
    ),
    EvalQuestion(
        id="ada_2",
        question="My patient with diabetes has a confirmed blood pressure of 155/95 -- should I start one drug or two?",
        category="recommendation",
        needles=[
            any_of(
                rec(ADA, "10.7"),
                figure(ADA, "Figure 10.2 -Recommendations for the treatment of confirmed hypertension"),
            )
        ],
        split="heldout",
        phrasing="lay",
        notes="Figure 10.2's flowchart carries the same >=150/90 two-drug branch.",
    ),
    EvalQuestion(
        id="ada_3",
        question="How often should people with type 2 diabetes be screened for kidney disease?",
        category="recommendation",
        needles=[rec(ADA, "11.1a")],
        split="heldout",
    ),
    EvalQuestion(
        id="ada_4",
        question="Which glucose-lowering drug class is preferred in type 2 diabetes with an eGFR below 30?",
        category="recommendation",
        needles=[rec(ADA, "9.11")],
        split="dev",
    ),
    EvalQuestion(
        id="ada_5",
        question="Should a 55-year-old with diabetes and no heart disease be on a statin?",
        category="recommendation",
        needles=[rec(ADA, "10.18")],
        split="dev",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ada_6",
        question=(
            "Should I start an ACE inhibitor to protect the kidneys of someone with diabetes whose "
            "blood pressure and urine albumin are normal?"
        ),
        category="recommendation",
        needles=[rec(ADA, "11.6c")],
        split="dev",
        phrasing="lay",
    ),
    EvalQuestion(
        id="ada_7",
        question="When can an SGLT2 inhibitor and finerenone be started at the same time?",
        category="recommendation",
        needles=[rec(ADA, "11.9")],
        split="heldout",
    ),
    EvalQuestion(
        id="ada_8",
        question="Which statin doses count as high-intensity statin therapy?",
        category="table",
        needles=[table(ADA, "Table 10.1-High-intensity and moderate-intensity statin therapy")],
        split="dev",
    ),
    EvalQuestion(
        id="ada_9",
        question="What protein intake is recommended for people with diabetes on dialysis?",
        category="recommendation",
        needles=[rec(ADA, "11.3")],
        split="dev",
    ),
    EvalQuestion(
        id="ada_10",
        question=(
            "In type 2 diabetes with obesity and heart failure with preserved ejection fraction, "
            "what should the treatment plan include?"
        ),
        category="recommendation",
        needles=[any_of(rec(ADA, "9.9a"), rec(ADA, "10.44d"))],
        split="dev",
        notes="ADA states this in both Section 9 (glycemic) and Section 10 (heart failure).",
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
        needles=[
            rec(GLOMERULAR, "2.3.1"),
            any_of(
                phrase(GLOMERULAR, "Practice Point 1.5.2"),
                figure(GLOMERULAR, "Figure 9| Management of hypertension in glomerular disease"),
            ),
        ],
        split="dev",
        notes=(
            "Glomerular Ch 1 general BP principles apply by reference; PP 1.5.2 (SBP <120) exists "
            "only inside Figure 8. Under greedy packing it also pulled CKD Rec 3.4.1."
        ),
    ),
    EvalQuestion(
        id="ada_kdigo_1",
        question="How do the ADA and KDIGO blood pressure targets differ for people with diabetes and CKD?",
        category="synthesis",
        needles=[rec(ADA, "11.5"), rec(BP, "3.1.1")],
        split="dev",
        notes=(
            "A real disagreement: ADA <130/80 (SBP <120 encouraged), KDIGO SBP <120 when "
            "tolerated. A good answer states both, attributed -- not a blend."
        ),
    ),
    EvalQuestion(
        id="ada_kdigo_2",
        question="Do ADA and KDIGO agree on using finerenone in type 2 diabetes with CKD and albuminuria?",
        category="synthesis",
        needles=[any_of(rec(ADA, "11.8"), rec(ADA, "10.42")), rec(CKD, "3.8.1")],
        split="heldout",
        notes="Agree on the drug; eGFR thresholds differ in wording (ADA >=25, KDIGO >25).",
    ),
    EvalQuestion(
        id="ada_kdigo_3",
        question=(
            "What eGFR threshold do ADA and KDIGO give for starting an SGLT2 inhibitor in type 2 "
            "diabetes with CKD, and can it be continued below that?"
        ),
        category="synthesis",
        needles=[rec(ADA, "11.7a"), rec(CKD, "3.7.1"), pp(CKD, "3.7.1")],
        split="dev",
    ),
    EvalQuestion(
        id="ada_kdigo_4",
        question="How much dietary protein should someone with diabetes and CKD stage G3 or worse eat?",
        category="synthesis",
        needles=[rec(ADA, "11.3"), rec(CKD, "3.3.1.1")],
        split="dev",
        notes="Both say 0.8 g/kg/day; a test that agreement is stated only where both say it.",
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
    EvalQuestion(
        id="oos_8",
        question="What A1C level is diagnostic of diabetes?",
        category="out_of_scope",
        needles=[],
        split="dev",
        scope="near",
        notes="ADA Section 2 (diagnosis) isn't indexed -- only Sections 9-11 + methodology.",
    ),
    EvalQuestion(
        id="oos_9",
        question="How often should people with diabetes have a dilated eye examination?",
        category="out_of_scope",
        needles=[],
        split="heldout",
        scope="near",
        notes="ADA Section 12 (retinopathy) isn't indexed; Section 9 mentions eye exams only as a GLP-1 RA safety note.",
    ),
]
