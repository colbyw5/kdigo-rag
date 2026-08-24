"""The eval question set: 2 per guideline, factual + cross-guideline
synthesis + known-limitation cases, per DECISIONS.md's "small eval set"
next step.

``expected_guidelines``/``expected_recommendation_numbers`` are grounded --
captured from an actual pipeline run against the ingested corpus, not
guessed -- so a regression run can check whether retrieval/grading still
surfaces the same real citations, not whether it matches a fabricated one.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalQuestion:
    """One eval case.

    Attributes:
        id: Short identifier, also used as the per-question memory user_id
            so answers stay independent (memory must not bleed between
            unrelated eval questions run in the same pass).
        question: The question text sent to the graph.
        category: ``"factual"`` (single-guideline lookup), ``"synthesis"``
            (spans two guidelines), or ``"limitation"`` (a case from
            README's Known limitations section, kept in the set to
            document the gap with evidence rather than hide it).
        grading: ``"strict"`` -- fail the run if ``expected_guidelines``/
            ``expected_recommendation_numbers`` don't appear among the
            graded citations. ``"loose"`` -- no pass/fail; the new answer
            is printed next to ``baseline_answer`` for a human to judge,
            since there's no single correct citation for synthesis or
            limitation cases.
        expected_guidelines: Guideline names the graded citations should
            include (subset check).
        expected_recommendation_numbers: Specific recommendation/practice
            point numbers expected among the graded citations, when the
            answer is anchored to one (empty for table- or narrative-
            sourced answers).
        baseline_answer: The actual answer captured on the run that
            established this question -- a reference point for a human
            reviewing a loose question's new answer, not a target string
            to match exactly.
        notes: Freeform context for a human reviewer.
    """

    id: str
    question: str
    category: Literal["factual", "synthesis", "limitation"]
    grading: Literal["strict", "loose"]
    expected_guidelines: list[str]
    expected_recommendation_numbers: list[str] = field(default_factory=list)
    baseline_answer: str = ""
    notes: str = ""


CKD = "KDIGO 2024 CKD Evaluation and Management"
AKI = "KDIGO 2026 AKI/AKD (Public Review Draft)"
GLOMERULAR = "KDIGO 2021 Glomerular Diseases"
BP = "KDIGO 2021 Blood Pressure in CKD"


EVAL_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id="ckd_1",
        question="Should a patient with G3b A2 CKD be on an SGLT2 inhibitor?",
        category="factual",
        grading="strict",
        expected_guidelines=[CKD],
        expected_recommendation_numbers=["3.7.1", "3.7.3"],
        baseline_answer=(
            "For a G3b A2 patient, whether SGLT2i is recommended depends on diabetes status. "
            "With T2D: KDIGO recommends treating patients with T2D, CKD, and eGFR >=20 ml/min/1.73m2 "
            "with an SGLT2i (1A, Rec 3.7.1) -- G3b (eGFR 30-44) is well above this threshold, so this "
            "applies regardless of the exact ACR within A2. Without diabetes: the applicable "
            "recommendation is narrower -- treating adults with eGFR 20-45 and urine ACR <200 mg/g "
            "with an SGLT2i (2B, Rec 3.7.3). Since A2 spans ACR 30-300 mg/g, this only clearly applies "
            "if the patient's actual ACR is <200 mg/g; the passages don't address ACR 200-300 mg/g in "
            "non-diabetics explicitly."
        ),
        notes=(
            "Correctly splits diabetic (1A, Rec 3.7.1, eGFR>=20 threshold governs) "
            "vs. non-diabetic (2B, Rec 3.7.3, ACR<200 caveat)."
        ),
    ),
    EvalQuestion(
        id="ckd_2",
        question="What eGFR stage is 38 mL/min/1.73m²?",
        category="limitation",
        grading="loose",
        expected_guidelines=[GLOMERULAR, BP],
        baseline_answer=(
            "An eGFR of 38 mL/min/1.73m2 falls into GFR category G3b, 'moderately to severely "
            "decreased,' corresponding to the range 30-44 mL/min/1.73m2, consistent across the "
            "KDIGO 2021 Glomerular Diseases and 2021 Blood Pressure in CKD guidelines' shared GFR "
            "category table (KDIGO 2012 CKD classification framework). Combined with an albuminuria "
            "category (A1/A2/A3, not given in this question) it would map to a cell in KDIGO's "
            "GFR x albuminuria risk stratification."
        ),
        notes=(
            "Known limitation (README): embeddings don't do numeric interval "
            "reasoning. Resolved correctly in the baseline run via a retrieved "
            "GFR-category table (G3b, 30-44), but this isn't guaranteed -- "
            "depends on the table chunk being retrieved. Loose grading: check "
            "the answer still lands on G3b, not that it uses identical wording."
        ),
    ),
    EvalQuestion(
        id="aki_1",
        question="What defines Stage 2 AKI by KDIGO criteria?",
        category="factual",
        grading="strict",
        expected_guidelines=[AKI],
        baseline_answer=(
            "Stage 2 AKI (KDIGO 2026 AKI/AKD staging table, Chapter 1, Table 6): serum creatinine "
            "2.0-2.9x baseline (C2), OR urine output <0.5 mL/kg/h for >12 hours (U2). Staging uses "
            "whichever criterion yields the higher (more severe) stage. A structural/damage biomarker "
            "criterion (B0/B1) can be layered on but doesn't change the numeric stage by itself."
        ),
        notes="Table-sourced (Chapter 1, staging table), no recommendation number to anchor on.",
    ),
    EvalQuestion(
        id="aki_ckd_1",
        question=(
            "A patient with CKD develops an acute AKI episode — how should acute-phase "
            "management differ from routine CKD care?"
        ),
        category="synthesis",
        grading="loose",
        expected_guidelines=[AKI],
        expected_recommendation_numbers=["4.11.1", "6.2.3"],
        baseline_answer=(
            "AKI-on-CKD acute-phase management diverges from routine CKD care in several ways, per "
            "the AKI/AKD guideline's own chapters: (1) temporary discontinuation of RASi/MRA/diuretics "
            "during illness predisposing to volume depletion, to prevent hypovolemia-mediated AKI "
            "(Rec 4.11.1); (2) discontinuation of all nephrotoxic agents when possible; (3) active "
            "volume/hemodynamic optimization, including advanced hemodynamic monitoring in severe "
            "cases; (4) frequent, short-interval monitoring of creatinine/urine output/biomarkers vs. "
            "routine CKD's periodic monitoring; (5) drug-dosing review and diagnostic workup for the "
            "AKI cause; (6) after recovery, RASi/MRA/diuretics are re-evaluated per the 2024 CKD "
            "guideline's framework (Practice Point 6.2.3), i.e. a structured return to chronic care."
        ),
        notes=(
            "Designed as a cross-guideline (AKI+CKD) question. In the baseline run it "
            "was answered entirely from the AKI/AKD guideline's own chapters (3, 4, 6) -- "
            "no CKD guideline chunk was retrieved. Kept as-is deliberately: this is a "
            "real, honest finding (the AKI/AKD document is self-contained enough to "
            "answer this without cross-document retrieval), not a bug to hide. Genuine "
            "cross-document synthesis is not demonstrated by this question."
        ),
    ),
    EvalQuestion(
        id="glom_1",
        question="What is the first-line treatment approach for IgA nephropathy with declining GFR?",
        category="factual",
        grading="strict",
        expected_guidelines=[GLOMERULAR],
        baseline_answer=(
            "First-line for IgAN with declining GFR is non-immunosuppressive supportive care, not "
            "immunosuppression: rigorous BP control, optimal RAS inhibition, and lifestyle modification "
            "(weight reduction, exercise, smoking cessation, dietary sodium restriction). Controlling "
            "BP and minimizing proteinuria are centrally important; immunosuppression is an additional "
            "consideration to modify disease pathogenesis, not the primary first-line strategy."
        ),
        notes="Narrative-sourced (Chapter 2, IgAN), no recommendation number to anchor on.",
    ),
    EvalQuestion(
        id="glom_bp_1",
        question="A patient with lupus nephritis also has hypertension — how should blood pressure be managed?",
        category="synthesis",
        grading="loose",
        expected_guidelines=[GLOMERULAR],
        expected_recommendation_numbers=["2.3.1"],
        baseline_answer=(
            "The Glomerular Diseases guideline directs BP management for lupus nephritis to follow "
            "Chapter 1's general principles (the same cross-reference pattern used for IgAN, Rec 2.3.1, "
            "and membranous nephropathy): (1) ACEi/ARB titrated to max tolerated dose as first-line for "
            "hypertension+proteinuria, not stopped for a stable creatinine rise up to 30%, but avoided "
            "in abrupt-onset nephrotic syndrome; (2) target SBP <120 mmHg by standardized office "
            "measurement, with 120-130 mmHg practically achievable (Practice Point 1.5.2); (3) frequent "
            "creatinine/potassium monitoring during titration, holding ACEi/ARB and diuretics during "
            "sick days; (4) potassium-binders/potassium-wasting diuretics if hyperkalemia limits RAS "
            "blockade; (5) sodium restriction <2.0 g/d, weight/exercise/smoking measures, and MRA "
            "consideration if proteinuria reduction is inadequate. No lupus-specific BP recommendation "
            "with its own grading exists in the retrieved context -- it's incorporated by reference."
        ),
        notes=(
            "Also designed as cross-guideline (Glomerular+BP). Answered entirely from "
            "the Glomerular Diseases guideline's own Chapter 1 general-principles "
            "section (which the IgAN and membranous chapters both reference by pointer) "
            "-- the separate BP guideline document was never retrieved. Same honest "
            "finding as aki_ckd_1: kept as-is rather than rewritten to force a "
            "cross-document result."
        ),
    ),
    EvalQuestion(
        id="bp_1",
        question="What is the target blood pressure for adults with CKD not on dialysis?",
        category="factual",
        grading="strict",
        expected_guidelines=[CKD, BP],
        expected_recommendation_numbers=["3.4.1"],
        baseline_answer=(
            "Target SBP <120 mmHg by standardized office BP measurement for most adults with CKD not "
            "on dialysis, when tolerated (Rec 3.4.1, 2B). Contrast: kidney transplant recipients target "
            "SBP <130 / DBP <80; pediatric CKD targets MAP <= 50th percentile for age/sex/height rather "
            "than a fixed SBP. Both the 2024 CKD guideline and 2021 BP guideline agree on the <120 "
            "target for non-dialysis adult CKD."
        ),
        notes="SBP <120 mmHg, Rec 3.4.1 (2B) -- the one question that did span two guideline documents.",
    ),
    EvalQuestion(
        id="bp_2",
        question=(
            "How should blood pressure management be tailored for a CKD patient based "
            "on their GFR/albuminuria risk category?"
        ),
        category="limitation",
        grading="loose",
        expected_guidelines=[BP],
        baseline_answer=(
            "Tailoring is driven by GFR category and albuminuria status per the BP guideline's Chapter "
            "3: for lower GFR (G4-G5), there's less certainty about the benefit of a lower BP target and "
            "more concern about harm from intensive lowering vs. higher-GFR patients -- more cautious, "
            "individualized targets as GFR falls. For albuminuria: patients with high BP and no "
            "albuminuria are at lower progression risk, and evidence doesn't show a clear RASi-specific "
            "renoprotective benefit for them -- other antihypertensive classes are equally appropriate, "
            "though some with higher GFR and low harm-risk may still choose RASi for cardiovascular "
            "benefit. The full GFR x albuminuria heat-map cell-by-cell targets are not available in the "
            "retrieved context."
        ),
        notes=(
            "Known limitation (README): the full GFR x albuminuria risk heat map "
            "communicates risk via cell color, not extractable text, so there's no "
            "cell-by-cell answer available. A good answer discusses the directional "
            "principles (albuminuria and low-GFR caveats) and explicitly flags that it "
            "can't give the full heat-map breakdown -- the baseline answer did this "
            "correctly; loose grading should check the limitation is still surfaced "
            "honestly, not swept under a confident-sounding answer."
        ),
    ),
]
