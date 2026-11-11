"""
Decision tree v2 for CAIS method selection.

Implements the revised tree from CAIS_assumptions.pdf (page 21): structural
nodes, criteria diamonds, hard gates, advisories, and END states. This module
only decides *which* method to run (or that the pipeline must stop). It does
not execute methods and does not touch the estimator/executor rosters.

Node verdict policy (confirmed with the team):
  passed=True   -> branch succeeds
  passed=False  -> gate: stop the pipeline; advisory: warn and continue
  passed=None   -> never a silent pass: warn and continue

Checks are injected. Tests pass stub verdicts; production uses
``build_default_checks`` which binds the real functions from
``cais.methods.pre_model_assumption_utils``. Unknown/missing checks become
"not implemented yet" placeholders (can't tell -> warn and continue).

The traversal result (``TreeResult``) records the path, every check verdict,
warnings, and any post-estimation checks to run later, so any run can be
debugged after the fact.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cais.models import AssumptionResult
from cais.config import load_thresholds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leaf identifiers. Implemented leaves use the executor names from
# cais.methods.METHOD_MAPPING verbatim, so the tree result can be executed
# directly. Unimplemented variants keep descriptive names and are flagged in
# LEAF_IMPLEMENTED; they become base-method + options once implemented.
# ---------------------------------------------------------------------------
M_OLS_PRE_TREATMENT = "linear_regression"
M_DIFF_IN_MEANS = "diff_in_means"
M_IV = "instrumental_variable"
M_DID = "difference_in_differences"
M_DID_TIME_WINDOW = "difference_in_differences_time_window"
M_RDD = "regression_discontinuity_design"
M_DONUT_RDD = "donut_rdd"
M_FUZZY_RDD = "fuzzy_rdd"  # reserved: draft variant of RDD, not traversed yet
M_IPW = "propensity_score_weighting"
M_PSM = "propensity_score_matching"
M_TRIMMED_IPW = "trimmed_ipw"
M_FRONTDOOR = "frontdoor_adjustment"
M_GPS = "generalized_propensity_score"

# Leaves reachable by the traversal with their implementation status.
LEAF_IMPLEMENTED: Dict[str, bool] = {
    M_OLS_PRE_TREATMENT: True,
    M_DIFF_IN_MEANS: True,
    M_IV: True,
    M_DID: True,
    M_DID_TIME_WINDOW: True,
    M_RDD: True,
    M_DONUT_RDD: True,
    M_FUZZY_RDD: False,
    M_IPW: True,
    M_PSM: True,
    M_TRIMMED_IPW: True,  # positivity-failure remedy: IPW trimmed to the overlap band
    M_FRONTDOOR: True,  # mediator-based identification for discrete treatment/mediator
    M_GPS: True,
}

# Every check the traversal can ask for. Unknown entries are placeholders.
ALL_CHECK_KEYS: tuple = (
    "sutva",
    "stable_group_composition",
    "baseline_outcome_balance",
    "parallel_trends",
    "no_anticipation",
    "iv_relevance",
    "iv_exclusion",
    "iv_exogeneity",
    "iv_monotonicity",
    "cond_ignorability",
    "positivity",
    "rdd_no_manipulation",
    "rdd_covariate_continuity",
    "rdd_continuity_potential_outcomes",
    "frontdoor_full_mediation",
    "frontdoor_no_tm_confounding",
    "frontdoor_t_blocks_my",
    "frontdoor_positivity",
    "gps_cond_ignorability",
    "gps_positivity",
)

DESCRIPTION_PROMPT = (
    "Please provide a brief description of the dataset (columns, context, background). "
    "Type 'skip' to continue without one."
)

MAX_PROMPTS = 3  # per-node cap on follow-up questions
MAX_TOTAL_ASKS = 3  # global cap per run, shared by every asking node (SUTVA included)
MAX_FRONTDOOR_LEVELS = 10  # discrete treatment/mediator cardinality the frontdoor sum supports

_PERIOD_TOKEN_RE = re.compile(
    r"(?P<kind>placebo|treatment)\s*(?:period[_ ]?start)?\s*[:=]\s*(?P<num>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Encouragement-branch estimand question. The query interpreter will eventually
# parse the answer from the user query itself; until then, the user's literal
# reply is keyword-parsed and anything unclear defaults to the received-
# treatment estimand, as agreed with the team.
ESTIMAND_QUESTION = (
    "Does the analysis target the effect of the assignment/encouragement "
    "(intention-to-treat, ITT) or the effect of actually receiving the treatment? "
    "Reply 'assignment' or 'received'; 'skip' or empty = default: effect of "
    "receiving the treatment."
)
_ITT_RE = re.compile(
    r"\b(itt|intention\W*to\W*treat|assign\w*|encourag\w*|offered|invited|mobiliz\w*)\b",
    re.IGNORECASE,
)
_RECEIVED_RE = re.compile(
    r"\b(receiv\w*|treat\w*|taken|taking|take\W*up|uptake|attend\w*|compli\w*|actual\w*|delivered|consumed|used)\b",
    re.IGNORECASE,
)


def parse_estimand_answer(answer: str) -> Optional[str]:
    """Keyword-parse a user reply to the estimand question (stub parser).

    Returns "itt", "treatment_effect", or None when nothing recognizable was
    said. Assignment words are matched before received-treatment words, because
    phrases like "effect of being assigned the treatment" contain both.
    """
    if _ITT_RE.search(answer):
        return "itt"
    if _RECEIVED_RE.search(answer):
        return "treatment_effect"
    return None


def extract_period_starts(answer: str, current: Dict[str, Any]) -> Dict[str, float]:
    """Parse the no-anticipation question's answer into period starts.

    The user is asked for the placebo/treatment period starts; prose alone
    cannot rewind a statistical check, so the numbers are parsed out. Named
    tokens ('placebo=1986', 'treatment=1988') win; otherwise with two or more
    distinct numbers the smaller is placebo and the larger is the treatment
    start, and a single number fills whichever of the two is missing.

    Returns only assignments for keys that are currently missing; explicit
    existing values are never overridden. Returns {} when nothing can be
    attributed confidently.
    """
    parsed: Dict[str, float] = {}
    for match in _PERIOD_TOKEN_RE.finditer(answer or ""):
        key = "placebo_period_start" if match.group("kind").lower() == "placebo" else "treatment_period_start"
        try:
            parsed[key] = float(match.group("num"))
        except ValueError:
            continue
    if not parsed:
        unique = sorted({float(x) for x in _NUMBER_RE.findall(answer or "")})
        if len(unique) >= 2:
            parsed = {"placebo_period_start": unique[0], "treatment_period_start": unique[-1]}
        elif len(unique) == 1:
            have_placebo = current.get("placebo_period_start") is not None
            have_treatment = current.get("treatment_period_start") is not None
            if not have_placebo and have_treatment:
                parsed = {"placebo_period_start": unique[0]}
            elif not have_treatment and have_placebo:
                parsed = {"treatment_period_start": unique[0]}
            # both missing (or both known) with one number -> ambiguous; skip
    assignments: Dict[str, float] = {}
    for key, value in parsed.items():
        if current.get(key) is None:
            assignments[key] = value
    return assignments


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class TreeStep:
    """One recorded decision along the traversal."""

    node: str
    kind: str  # structural | criteria | gate | advisory | leaf | process | end
    outcome: str  # Yes/No/Pass/Fail/pass/fail/can't tell/selected/...
    detail: str = ""


@dataclass
class TreeResult:
    """Everything the caller needs to execute (or explain why not)."""

    method: Optional[str] = None
    variant: Optional[str] = None
    # Encouragement-branch estimand: "itt" (effect of the assignment) or
    # "treatment_effect" (effect of the received treatment). None elsewhere.
    estimand: Optional[str] = None
    implemented: bool = True
    asks_used: int = 0  # follow-up questions asked to reach this result
    ended: bool = False
    end_reason: Optional[str] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)  # applied gate-threshold overrides
    steps: List[TreeStep] = field(default_factory=list)
    assumptions: Dict[str, AssumptionResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    planned_post_checks: List[str] = field(default_factory=list)
    description: Optional[str] = None
    # Leaf-specific settings the agent needs in order to execute the method
    # (e.g. {"method": "trimmed_ipw", "display_name": ..., "options": {...}}).
    execution: Dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return " -> ".join(f"{s.node}[{s.outcome}]" for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "variant": self.variant,
            "estimand": self.estimand,
            "implemented": self.implemented,
            "asks_used": self.asks_used,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "thresholds": dict(self.thresholds),
            "warnings": list(self.warnings),
            "planned_post_checks": list(self.planned_post_checks),
            "description": self.description,
            "execution": dict(self.execution),
            "path": self.path,
            "steps": [
                {"node": s.node, "kind": s.kind, "outcome": s.outcome, "detail": s.detail}
                for s in self.steps
            ],
            "assumptions": {k: v.model_dump() for k, v in self.assumptions.items()},
        }


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

class _Traversal:
    def __init__(
        self,
        properties: Dict[str, Any],
        checks: Optional[Dict[str, Callable[[], AssumptionResult]]] = None,
        checks_factory: Optional[Callable[[Optional[str]], Dict[str, Callable[[], AssumptionResult]]]] = None,
        description: Optional[str] = None,
        prompt_callback: Optional[Callable[[str], Optional[str]]] = None,
        df: Any = None,
        llm: Any = None,
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.props = dict(properties or {})
        self.df = df
        self.llm = llm
        self.prompt_callback = prompt_callback
        self.description = description or self.props.get("dataset_description")

        # Gate-threshold overrides: caller-provided dict wins, otherwise the
        # user-editable thresholds.json (cais.config.load_thresholds). Only
        # recorded on the result when the default factory actually uses them.
        self.thresholds = dict(thresholds) if thresholds is not None else load_thresholds()
        uses_default_factory = checks is None and checks_factory is None

        if checks is not None:
            self.factory = lambda _desc: checks
        elif checks_factory is not None:
            self.factory = checks_factory
        else:
            self.factory = lambda desc: build_default_checks(
                self.props, self.df, desc, self.llm, thresholds=self.thresholds
            )

        self.checks = self.factory(self.description)
        record_thresholds = self.thresholds if uses_default_factory else {}
        self.result = TreeResult(description=self.description, thresholds=record_thresholds)
        self.asks_used = 0  # follow-up questions asked so far (global budget)

    # -- recording helpers ------------------------------------------------

    def step(self, node: str, kind: str, outcome: str, detail: str = "") -> None:
        self.result.steps.append(TreeStep(node, kind, outcome, detail))
        logger.info("[tree v2] %-55s %-10s %s", node, outcome, detail)

    def end(self, reason: str) -> TreeResult:
        self.result.ended = True
        self.result.method = None
        self.result.end_reason = reason
        self.step("END", "end", "stop", reason)
        return self.result

    def leaf(self, method: str, variant: Optional[str] = None) -> TreeResult:
        excluded = self.props.get("excluded_methods") or []
        if method in excluded:
            reason = (
                f"{method} is excluded for this traversal (e.g. by the validation "
                "backtrack); the tree identifies no alternative method."
            )
            self.result.method = None
            self.result.implemented = False
            self.step(method, "leaf", "excluded", reason)
            return self.end(reason)
        implemented = LEAF_IMPLEMENTED.get(method, True)
        self.result.method = method
        self.result.variant = variant
        self.result.implemented = implemented
        self.step(
            method,
            "leaf",
            "selected" if implemented else "selected (not implemented)",
            detail=f"variant: {variant}" if variant else "",
        )
        if not implemented:
            self.result.warnings.append(f"{method} is not implemented yet; this study cannot be estimated.")
        return self.result

    def _safe_check(self, check_name: str) -> AssumptionResult:
        fn = self.checks.get(check_name)
        if fn is None:
            return AssumptionResult(
                passed=None, reasoning=f"Check '{check_name}' is not implemented yet."
            )
        try:
            res = fn()
        except Exception as exc:  # checks must never crash the traversal
            logger.warning("Check '%s' raised: %s", check_name, exc)
            return AssumptionResult(passed=None, reasoning=f"Check '{check_name}' failed to run: {exc}")
        if not isinstance(res, AssumptionResult):
            return AssumptionResult(
                passed=None, reasoning=f"Check '{check_name}' did not return an AssumptionResult."
            )
        return res

    def checked(self, node: str, check_name: str, kind: str) -> Optional[bool]:
        res = self._safe_check(check_name)
        self.result.assumptions[check_name] = res
        outcome = {True: "pass", False: "fail", None: "can't tell"}[res.passed]
        self.step(node, kind, outcome, res.reasoning)
        return res.passed

    def advisory(self, node: str, check_name: str) -> Optional[bool]:
        passed = self.checked(node, check_name, "advisory")
        if passed is False:
            self.result.warnings.append(f"{node}: failed (advisory).")
        elif passed is None:
            self.result.warnings.append(f"{node}: could not be assessed (advisory).")
        return passed

    def gate(self, node: str, check_name: str, fail_reason: str) -> bool:
        """Run a gate. Returns True when traversal may continue.

        If the check reports that user-side knowledge could settle it
        (``details.missing_info``), the user is asked before the verdict is
        final; statistical failures carry no question and never ask.
        On a decisive failure, an interactive run gets one override consult:
        the user may pass the failed gate, otherwise the pipeline stops.
        Batch runs have nobody to consult and stop, as before.
        """
        passed = self.checked(node, check_name, "gate")
        passed = self._maybe_reask_gate(node, check_name, "gate", passed)
        if passed is False:
            if self._consult_override(node, check_name, fail_reason):
                return True
            self.end(fail_reason)
            return False
        if passed is None:
            self.result.warnings.append(f"{node}: could not be tested; continuing with warning.")
        return True

    # -- follow-up questions (SUTVA-style, budgeted) -----------------------

    def _question_of(self, check_name: str) -> Optional[str]:
        """The check's own question, when it needs information a user could give."""
        res = self.result.assumptions.get(check_name)
        if res is None:
            return None
        return (res.details or {}).get("missing_info")

    def _ask_and_rerun(self, node: str, check_name: str, kind: str) -> Optional[bool]:
        """Ask the user the check's question, re-run the check, record the trace.

        Up to MAX_PROMPTS asks for this node, subject to the shared global
        budget. Stops on a verdict, no answer, or budget exhaustion.
        """
        attempts = 0
        passed = None
        while attempts < MAX_PROMPTS and self.asks_used < MAX_TOTAL_ASKS:
            question = self._question_of(check_name)
            if not question:
                break
            answer = self._prompt(question)
            if not answer:
                self.step(node, "process", "no user answer", f"question: {question}")
                break
            self.asks_used += 1
            attempts += 1
            self.description = (self.description or "") + "\n" + answer
            # Statistical checks need numbers, not prose: for the no-anticipation
            # gate, parse period starts out of the answer (only fills missing
            # values; what was parsed is recorded in the trace).
            if check_name == "no_anticipation":
                parsed = extract_period_starts(answer, self.props)
                if parsed:
                    self.props.update(parsed)
                    self.step(
                        node, "process", "parsed period starts",
                        ", ".join(f"{k}={v:g}" for k, v in sorted(parsed.items())),
                    )
            self.checks = self.factory(self.description)
            self.step(node, "process", f"answered ({self.asks_used}/{MAX_TOTAL_ASKS})", f"question: {question}")
            redone = self._safe_check(check_name)
            self.result.assumptions[check_name] = redone
            outcome = {True: "pass", False: "fail", None: "can't tell"}[redone.passed]
            self.step(node, kind, f"after answer: {outcome}", redone.reasoning)
            passed = redone.passed
            if redone.passed is not None:
                break
        return passed

    def _maybe_reask_gate(self, node: str, check_name: str, kind: str, passed: Optional[bool]) -> Optional[bool]:
        """Gates only: if the check needs user knowledge, ask for it."""
        if passed is not None and passed is not False:
            return passed
        if not self._question_of(check_name):
            return passed  # statistical fails/can't-tells are facts, not questions
        if self.prompt_callback is None:
            self.step(node, "process", "no user available (batch)", "")
            return passed
        if self.asks_used >= MAX_TOTAL_ASKS:
            self.step(node, "process", "ask budget exhausted", "")
            return passed
        retried = self._ask_and_rerun(node, check_name, kind)
        return passed if retried is None else retried

    def _consult_override(self, node: str, check_name: str, reason: str) -> bool:
        """Interactive override consult after a decisive gate failure.

        Only traversal paths that would otherwise stop consult the user;
        prescribed fallback routes (e.g. trimmed IPW on positivity failure)
        keep their drawn-tree semantics and never consult. The consult is a
        decision only: it never consumes the information-ask budget and
        never re-runs the check. Returns True when the user passed the gate.
        """
        if self.prompt_callback is None:
            self.step(node, "process", "no user available to consult", "")
            return False
        res = self.result.assumptions.get(check_name)
        detail = ((res.reasoning if res is not None else "") or "")[:220]
        question = (
            f"{node} FAILED: {reason}\n"
            f"Assessment: {detail}\n"
            f"Override this gate and continue anyway? (y = proceed, n = stop)"
        )
        answer = self._prompt(question)
        accepted = self._accepts_override(answer)
        self.step(
            node, "process", "override consult",
            f"answered: {answer!r} -> {'pass' if accepted else 'stop'}",
        )
        if not accepted:
            return False
        self.result.warnings.append(
            f"{node}: failed, but was passed by explicit user override; "
            "results may be unreliable."
        )
        if res is not None:
            res.details = dict(res.details or {})
            res.details["user_override"] = True
        self.step(node, "process", "user passed the failed gate", reason)
        return True

    @staticmethod
    def _accepts_override(answer: Optional[str]) -> bool:
        if not answer:
            return False
        norm = answer.strip().lower()
        return norm[:1] == "y" or norm in ("1", "proceed", "override")

    def _prompt(self, message: str) -> Optional[str]:
        if self.prompt_callback is None:
            return None
        try:
            answer = self.prompt_callback(message)
        except Exception as exc:
            logger.warning("prompt_callback raised: %s", exc)
            return None
        if not answer or str(answer).strip().lower() in {"skip", "none"}:
            return None
        return str(answer).strip()

    # -- traversal ----------------------------------------------------------

    def run(self) -> TreeResult:
        self._entry()
        if not self.result.ended:
            is_rct = bool(self.props.get("is_rct"))
            self.step("Is this a randomized trial?", "structural", "Yes" if is_rct else "No")
            if is_rct:
                self._rct_branch()
            else:
                self._observational_branch()
        self.result.description = self.description
        self.result.asks_used = self.asks_used
        return self.result

    def _entry(self) -> None:
        # 1. dataset description (prompt loop in the draft; batch mode skips it)
        if self.description:
            self.step("Is there a dataset description provided?", "structural", "Yes")
        else:
            self.step("Is there a dataset description provided?", "structural", "No")
            for _ in range(MAX_PROMPTS):
                answer = self._prompt(DESCRIPTION_PROMPT)
                if answer:
                    self.description = answer
                    self.checks = self.factory(self.description)
                    self.step("Prompt user for dataset description", "process", "provided")
                    break
            if not self.description:
                self.result.warnings.append(
                    "No dataset description provided (batch mode); LLM-argued checks will report can't tell."
                )

        # 2. data structure supported
        supported = self.props.get("is_structure_supported", True)
        self.step("Is the data structure supported?", "structural", "Yes" if supported else "No")
        if not supported:
            self.end("Data structure not supported (e.g. survival data, competing risks).")
            return

        # 3. global SUTVA advisory. Asking follows the same rule as gates:
        #    only when the check itself supplies a question (missing_info);
        #    without an LLM there is nothing that could use an answer.
        passed = self.checked("Advisory: SUTVA", "sutva", "advisory")
        for _ in range(MAX_PROMPTS):
            question = self._question_of("sutva")
            if passed is not None or not question or self.asks_used >= MAX_TOTAL_ASKS:
                break
            answer = self._prompt(question)
            if not answer:
                self.step("Advisory: SUTVA", "process", "no user answer", f"question: {question}")
                break
            self.asks_used += 1
            self.step("Advisory: SUTVA", "process", f"answered ({self.asks_used}/{MAX_TOTAL_ASKS})", f"question: {question}")
            self.description = (self.description or "") + "\n" + answer
            self.checks = self.factory(self.description)
            passed = self.checked("Advisory: SUTVA", "sutva", "advisory")
        if passed is False:
            self.result.warnings.append("SUTVA assessment failed (advisory); possible interference.")
        elif passed is None:
            self.result.warnings.append("SUTVA could not be assessed (advisory).")

    def _rct_branch(self) -> None:
        is_enc = self.props.get("is_encouragement_design")
        if is_enc is None:
            instr = self.props.get("instrument_variable")
            is_enc = bool(instr) and instr != self.props.get("treatment_variable")
        self.step("Is this an encouragement design?", "structural", "Yes" if is_enc else "No")

        if not is_enc:
            has_pre = self.props.get("has_pre_treatment_variables")
            if has_pre is None:
                has_pre = bool(self.props.get("covariates"))
            self.step("Are there valid pre-treatment variables?", "criteria", "Yes" if has_pre else "No")
            self.leaf(M_OLS_PRE_TREATMENT if has_pre else M_DIFF_IN_MEANS)
            return

        # Left branch (team decision, Nov 8): first settle which quantity the
        # run targets, then either report the assignment effect directly or
        # require a strong instrument for the received-treatment effect.
        estimand = self._ask_estimand()
        if estimand == "itt":
            self._select_dim_on_assignment()
            return
        has_candidate = self.props.get("has_valid_instrument")
        if has_candidate is None:
            has_candidate = bool(self.props.get("instrument_variable"))
        self.step("Is there a candidate instrument?", "criteria", "Yes" if has_candidate else "No")
        if not has_candidate:
            self.end(
                "Unsupported: no candidate instrument, so the effect of the "
                "received treatment cannot be identified on encouragement-design "
                "data; the assignment effect (ITT) also cannot be computed "
                "because no assignment column was identified."
            )
            return
        relevance = self.checked("Gate: Relevance (High F-stat)", "iv_relevance", "gate")
        if relevance is True:
            self._iv_advisories()
            self.leaf(M_IV)
            return
        # Weak or untestable first stage: interactive runs may still insist on
        # IV; otherwise report the assignment effect (ITT), which a randomized
        # encouragement design identifies regardless of instrument strength
        # (team decision, Nov 9) instead of ending the pipeline.
        if self._consult_override(
            "Gate: Relevance (High F-stat)", "iv_relevance",
            "Weak or unverifiable instrument (first-stage F < 10 or not computable): "
            "the effect of the received treatment cannot be reported reliably.",
        ):
            self.leaf(M_IV)
            return
        self.step(
            "Weak first stage: report the assignment effect (ITT) instead",
            "process", "fallback to diff-in-means on assignment", "",
        )
        self.result.warnings.append(
            "First-stage F < 10 or not computable (weak instrument): the "
            "received-treatment effect (LATE) is not reliably estimable; "
            "reporting the assignment effect (ITT) instead of ending."
        )
        self._select_dim_on_assignment()

    def _ask_estimand(self) -> str:
        """Settle the encouragement-branch estimand: "itt" or "treatment_effect".

        Order: (1) an explicit ``estimand`` property (the interpreter's future
        integration point); (2) asking the user -- budgeted like the other
        follow-up questions, replies keyword-parsed until the interpreter owns
        this; (3) default "treatment_effect", per the team decision.
        """
        node = "Which estimand: assignment effect or effect of receiving treatment?"
        provided = self.props.get("estimand")
        if provided in ("itt", "treatment_effect"):
            self.result.estimand = provided
            self.step(node, "process", f"provided: {provided}", "")
            return provided
        if self.prompt_callback is None:
            self.result.estimand = "treatment_effect"
            self.step(node, "process", "default treatment_effect (no user available in batch)", "")
            return "treatment_effect"
        question = ESTIMAND_QUESTION
        for _ in range(MAX_PROMPTS):
            if self.asks_used >= MAX_TOTAL_ASKS:
                break
            answer = self._prompt(question)
            if not answer:
                self.result.estimand = "treatment_effect"
                self.step(node, "process", "default treatment_effect (no user answer)", f"question: {question}")
                return "treatment_effect"
            self.asks_used += 1
            parsed = parse_estimand_answer(answer)
            if parsed is not None:
                self.result.estimand = parsed
                self.step(
                    node, "process",
                    f"answered ({self.asks_used}/{MAX_TOTAL_ASKS}): {parsed}",
                    f"answer: {answer!r}",
                )
                return parsed
            self.step(node, "process", f"unclear answer ({self.asks_used}/{MAX_TOTAL_ASKS})", f"answer: {answer!r}")
            question = f"Could not read that ({answer!r}). {ESTIMAND_QUESTION}"
        self.result.estimand = "treatment_effect"
        self.step(node, "process", "default treatment_effect (no clear answer)", "")
        return "treatment_effect"

    def _select_dim_on_assignment(self) -> None:
        """Select diff-in-means on the assignment column (intention-to-treat)."""
        assignment = self.props.get("instrument_variable")
        if isinstance(assignment, (list, tuple)):
            assignment = assignment[0] if assignment else None
        if not assignment:
            assignment = self.props.get("treatment_variable")
        self.leaf(M_DIFF_IN_MEANS, variant="on_assignment")
        self.result.execution = {
            "method": M_DIFF_IN_MEANS,
            "display_name": "Difference-in-Means on Assignment (ITT)",
            "options": {"treatment_variable": assignment},
        }
        self.result.warnings.append(
            "Estimand is the effect of the assignment (ITT); the estimate groups "
            "units by the assignment column, not by the received treatment."
        )

    def _observational_branch(self) -> None:
        treatment_type = self.props.get("treatment_variable_type", "binary")
        non_binary = treatment_type not in (None, "binary")
        self.step("Is treatment binary?", "structural", "No" if non_binary else "Yes")
        if non_binary:
            self._options(non_binary=True)
            return

        has_temporal = bool(self.props.get("has_temporal_structure", False))
        self.step("Is temporal information available?", "structural", "Pass" if has_temporal else "Fail")
        if has_temporal:
            self._did_path()
        else:
            self._running_variable_path()

    def _did_path(self) -> None:
        if not self.gate(
            "Is the group composition stable?",
            "stable_group_composition",
            "Group composition is not stable; pipeline stopped.",
        ):
            return
        self.advisory("Advisory: Baseline outcome balance", "baseline_outcome_balance")
        if not self.gate("Gate: parallel trends", "parallel_trends", "Parallel trends violated."):
            return

        anticipation = self.checked("Gate: no anticipation (placebo)", "no_anticipation", "gate")
        anticipation = self._maybe_reask_gate(
            "Gate: no anticipation (placebo)", "no_anticipation", "gate", anticipation
        )
        window_start = self.props.get("anticipation_window_start")
        if anticipation is False:
            if window_start is not None or self.props.get("anticipation_known_bounded", False):
                self._select_did_time_window()
            elif self._consult_override(
                "Gate: no anticipation (placebo)", "no_anticipation",
                "Anticipation effects present and not bounded.",
            ):
                self.leaf(M_DID)
            else:
                self.end("Anticipation effects present and not bounded; no-anticipation gate failed.")
            return
        if anticipation is None:
            self.result.warnings.append("No-anticipation could not be tested; continuing with warning.")
        if window_start is not None:
            self._select_did_time_window()
        else:
            self.leaf(M_DID)

    def _select_did_time_window(self) -> None:
        """Select DiD after removing the user-supplied anticipation window."""
        window_start = self.props.get("anticipation_window_start")
        treatment_start = self.props.get("treatment_period_start")
        if window_start is None or treatment_start is None:
            self.end(
                "Time-window DiD needs both the anticipation-window start and the "
                "treatment start; one of them is missing."
            )
            return
        self.leaf(M_DID_TIME_WINDOW)
        self.result.execution = {
            "method": M_DID_TIME_WINDOW,
            "display_name": "Time-Window Difference-in-Differences",
            "options": {
                "anticipation_window_start": window_start,
                "treatment_period_start": treatment_start,
            },
        }
        self.result.warnings.append(
            f"Time-window DiD will drop observations with "
            f"{window_start} <= time < {treatment_start} (the user-supplied "
            "anticipation window) before estimation."
        )
        self.step(
            "Time-window DiD",
            "process",
            "anticipation window removed",
            f"drop {window_start} <= time < {treatment_start}",
        )

    def _running_variable_path(self) -> None:
        has_rv = self.props.get("has_running_variable")
        if has_rv is None:
            has_rv = bool(self.props.get("running_variable"))
        self.step(
            "Is there a running variable (group/unit variable)?",
            "structural",
            "Yes" if has_rv else "No",
        )
        if not has_rv:
            self._options(non_binary=False)
            return

        self.advisory(
            "Advisory: Continuity of potential outcomes at the cutoff",
            "rdd_continuity_potential_outcomes",
        )
        no_manipulation = self.checked(
            "Gates: No manipulation / Covariate continuity (McCrary test)",
            "rdd_no_manipulation",
            "gate",
        )
        continuity = self.checked(
            "Gates: No manipulation / Covariate continuity (McCrary test)",
            "rdd_covariate_continuity",
            "gate",
        )
        if no_manipulation is False or continuity is False:
            donut_width = self.props.get("donut_width")
            if donut_width is not None:
                self._select_donut_rdd(donut_width)
            elif self.props.get("local_manipulation", False):
                self.end(
                    "Local manipulation was indicated but no donut width was supplied."
                )
            elif self._consult_override(
                "Gates: No manipulation / Covariate continuity (McCrary test)",
                "rdd_no_manipulation",
                "Manipulation at the cutoff (density or covariate continuity).",
            ):
                self.leaf(M_RDD)
            else:
                self.end("McCrary/density gate failed: manipulation at the cutoff.")
            return
        if no_manipulation is None or continuity is None:
            self.result.warnings.append(
                "RDD gate: at least one component could not be tested; continuing with warning."
            )
        self.leaf(M_RDD)

    # -- identification options (instrument / mediator / backdoor) ----------

    def _select_donut_rdd(self, donut_width: Any) -> None:
        """Select donut RDD and record the user/analysis-supplied band."""
        width = float(donut_width)
        self.leaf(M_DONUT_RDD)
        self.result.execution = {
            "method": M_DONUT_RDD,
            "display_name": "Donut RDD",
            "options": {"donut_width": width},
        }
        self.result.warnings.append(
            f"Density/covariate-continuity gate failed; using donut RDD with a "
            f"{width:g}-unit band around the cutoff. The estimate refers to units "
            "outside that band, not to the cutoff limit."
        )
        self.step(
            "Donut RDD",
            "process",
            "donut width applied",
            f"drop units with |running variable − cutoff| < {width:g}",
        )

    def _options(self, non_binary: bool) -> None:
        if self._try_instrument():
            return
        if self._try_mediator():
            return
        if self._try_backdoor(non_binary):
            return
        self.end(
            "No viable identification strategy: no valid instrument, candidate mediator, "
            "or backdoor adjustment set."
        )

    def _try_instrument(self) -> bool:
        has = self.props.get("has_valid_instrument")
        if has is None:
            has = bool(self.props.get("instrument_variable"))
        self.step("Is there a valid instrument?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        self._iv_advisories()
        f_stat = self.checked("Gate: High F-stat", "iv_relevance", "gate")
        if f_stat is False:
            if self._consult_override("Gate: High F-stat", "iv_relevance", "Weak instrument (F < 10)."):
                self.leaf(M_IV)
                return True
            self.end("Weak instrument (F < 10).")
            return True
        if f_stat is None:
            self.result.warnings.append("Instrument strength could not be tested; continuing with warning.")
        self.leaf(M_IV)
        return True

    def _iv_advisories(self) -> None:
        self.advisory("Advisory: Exclusion restriction", "iv_exclusion")
        self.advisory("Advisory: Exogeneity", "iv_exogeneity")
        self.advisory("Advisory: Monotonicity", "iv_monotonicity")

    def _try_mediator(self) -> bool:
        has = self.props.get("has_candidate_mediator")
        if has is None:
            has = bool(self.props.get("mediator_variable"))
        self.step("Is there a candidate mediator?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        # A mediator column is not enough: the frontdoor plug-in needs a finite
        # treatment/mediator level set with populated cells. Unusable candidates
        # are treated as "no usable mediator" so the traversal continues to the
        # backdoor branch instead of stopping on a data-suitability problem.
        unsupported = self._frontdoor_data_unsupported_reason()
        if unsupported:
            self.step("Frontdoor data suitability", "process", "unsupported", unsupported)
            self.result.warnings.append(
                f"Candidate mediator cannot support frontdoor adjustment ({unsupported}); "
                "continuing to the backdoor branch."
            )
            return False

        self.advisory("Advisory: Full mediation", "frontdoor_full_mediation")
        self.advisory("Advisory: No TM confounding", "frontdoor_no_tm_confounding")
        self.advisory("Advisory: T blocks M->Y confounding", "frontdoor_t_blocks_my")
        positivity = self.checked("Gate: Positivity", "frontdoor_positivity", "gate")
        if positivity is False:
            if self._consult_override(
                "Gate: Positivity", "frontdoor_positivity",
                "Positivity failed for the mediator (frontdoor) strategy.",
            ):
                self._select_frontdoor()
                return True
            self.end("Positivity failed for the mediator (frontdoor) strategy.")
            return True
        if positivity is None:
            self.result.warnings.append("Positivity could not be tested; continuing with warning.")
        self._select_frontdoor()
        return True

    def _frontdoor_data_unsupported_reason(self) -> Optional[str]:
        """Why frontdoor cannot run on this data, or None when it looks usable.

        Uses the dataframe when the caller supplied one; without it (unit tests,
        stub checks) suitability is left to the estimator's own guards.
        """
        if self.df is None or not hasattr(self.df, "columns"):
            return None
        treatment = self.props.get("treatment_variable")
        mediator = self.props.get("mediator_variable")
        if not treatment or not mediator:
            return "mediator or treatment column is unknown"
        if treatment not in self.df.columns or mediator not in self.df.columns:
            return f"column(s) missing from the dataset ({treatment!r}, {mediator!r})"
        if self.props.get("treatment_variable_type") == "continuous":
            return "the treatment is continuous (frontdoor needs discrete treatment levels)"
        data = self.df[[treatment, mediator]].dropna()
        if data.empty:
            return "no complete treatment/mediator observations"
        n_treatment = int(data[treatment].nunique())
        n_mediator = int(data[mediator].nunique())
        if n_treatment < 2:
            return "the treatment has fewer than two observed levels"
        if n_treatment > MAX_FRONTDOOR_LEVELS:
            return (
                f"the treatment has {n_treatment} levels "
                f"(frontdoor supports at most {MAX_FRONTDOOR_LEVELS})"
            )
        if n_mediator > MAX_FRONTDOOR_LEVELS:
            return (
                f"the mediator has {n_mediator} levels "
                f"(frontdoor supports at most {MAX_FRONTDOOR_LEVELS}; continuous mediators "
                "need a different estimator)"
            )
        return None

    def _select_frontdoor(self) -> None:
        """Select frontdoor adjustment and record what the agent needs to run it."""
        self.leaf(M_FRONTDOOR)
        options: Dict[str, Any] = {}
        if self.props.get("mediator_variable"):
            options["mediator_variable"] = self.props["mediator_variable"]
        if self.props.get("treatment_reference_level") is not None:
            options["treatment_reference_level"] = self.props["treatment_reference_level"]
        self.result.execution = {
            "method": M_FRONTDOOR,
            "display_name": "Frontdoor Adjustment",
            "options": options,
        }
        self.result.warnings.append(
            "Frontdoor adjustment identifies the effect through the mediator; its "
            "validity rests on the audit assumptions (full mediation, no T-M confounding, "
            "T blocks M-Y confounding) reported above."
        )


    def _try_backdoor(self, non_binary: bool) -> bool:
        has = self.props.get("has_valid_backdoor_set")
        if has is None:
            has = bool(self.props.get("covariates"))
        self.step("Is there a valid (backdoor) adjustment set?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        ignorability = self.advisory("Advisory: Cond. Ignorability", "cond_ignorability")
        positivity = self.checked("Gate: Positivity", "positivity", "gate")
        if positivity is False:
            if non_binary:
                if self._consult_override(
                    "Gate: Positivity", "positivity",
                    "Positivity failed for the generalized propensity score strategy.",
                ):
                    self._select_gps()
                    return True
                self.end("Positivity failed for the generalized propensity score strategy.")
            else:
                self._select_trimmed_ipw()
            return True
        if positivity is None:
            self.result.warnings.append("Positivity could not be tested; continuing with warning.")

        if non_binary:
            self._select_gps()
            return True

        # binary: covariate balance chooses between IPW and matching
        self.step(
            "Are the covariates in the two groups balanced?",
            "criteria",
            {True: "Yes", False: "No", None: "can't tell"}[ignorability],
        )
        if ignorability is True:
            self.leaf(M_IPW)
            self.result.planned_post_checks.append("balance_after_weighting")
        else:  # False, or untestable -> matching (safer default, flagged)
            if ignorability is None:
                self.result.warnings.append(
                    "Covariate balance could not be assessed; defaulting to matching."
                )
            self.leaf(M_PSM)
            self.result.planned_post_checks.append("balance_after_matching")
        return True

    def _positivity_bounds(self) -> tuple:
        """Propensity-score overlap band used by the trimmed-IPW leaf.

        Mirrors the bounds the positivity gate is configured with
        (``positivity.extreme_ps_bounds``), falling back to (0.1, 0.9).
        """
        raw = ((self.thresholds or {}).get("positivity") or {}).get("extreme_ps_bounds")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                lo, hi = float(raw[0]), float(raw[1])
                if 0.0 <= lo < hi <= 1.0:
                    return lo, hi
            except (TypeError, ValueError):
                pass
            self.result.warnings.append(
                f"Ignoring invalid positivity.extreme_ps_bounds {list(raw)!r}; using [0.1, 0.9]."
            )
        return 0.1, 0.9

    def _select_trimmed_ipw(self) -> None:
        """Choose trimmed IPW as the positivity-failure remedy.

        Records the execution settings (bounds from the positivity config) and
        the estimand caveat so the agent can run the trimmed estimator and the
        explainer can state what the estimate refers to.
        """
        lo, hi = self._positivity_bounds()
        self.leaf(M_TRIMMED_IPW)
        self.result.execution = {
            "method": M_TRIMMED_IPW,
            "display_name": "Trimmed Inverse Probability Weighting",
            "options": {"ps_bounds": [lo, hi]},
        }
        self.result.warnings.append(
            f"Positivity gate failed; using trimmed IPW with propensity-score bounds "
            f"[{lo:g}, {hi:g}]. Units outside the overlap band are dropped, so the "
            "estimate targets the overlap-satisfying subpopulation."
        )
        self.result.planned_post_checks.append("balance_after_weighting")
        self.step(
            "Trimmed IPW",
            "process",
            "ps_bounds applied",
            f"drop units with propensity score outside [{lo:g}, {hi:g}]",
        )

    def _select_gps(self) -> None:
        """Select the generalized propensity score (GPS) and plan its advisory post-checks."""
        self.leaf(M_GPS)
        self.result.planned_post_checks.extend(["balance_after_gps", "gps_specification"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_method_v2(
    properties: Dict[str, Any],
    *,
    df: Any = None,
    description: Optional[str] = None,
    llm: Any = None,
    checks: Optional[Dict[str, Callable[[], AssumptionResult]]] = None,
    checks_factory: Optional[Callable[[Optional[str]], Dict[str, Callable[[], AssumptionResult]]]] = None,
    prompt_callback: Optional[Callable[[str], Optional[str]]] = None,
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TreeResult:
    """Traverse the revised decision tree.

    ``properties`` keys (missing keys fall back to sensible defaults):
      treatment_variable, outcome_variable, treatment_variable_type,
      covariates, instrument_variable, mediator_variable, running_variable,
      has_running_variable, cutoff_value, time_variable, group_variable,
      unit_variable, donut_width, anticipation_window_start,
      treatment_period_start, placebo_period_start, is_rct,
      has_temporal_structure,
      is_encouragement_design, has_pre_treatment_variables,
      has_valid_instrument, has_candidate_mediator, has_valid_backdoor_set,
      is_structure_supported, anticipation_known_bounded, local_manipulation,
      treatment_reference_level.

    ``checks`` wins over ``checks_factory``; when neither is given, the real
    checks are bound via ``build_default_checks``.

    ``thresholds`` optionally overrides gate knobs per check name (validated
    against ``ADJUSTABLE_PARAMS``); with None, the user-editable
    thresholds.json is consulted (see ``cais.config.load_thresholds``).
    """
    traversal = _Traversal(
        properties=properties,
        checks=checks,
        checks_factory=checks_factory,
        description=description,
        prompt_callback=prompt_callback,
        df=df,
        llm=llm,
        thresholds=thresholds,
    )
    return traversal.run()


# ---------------------------------------------------------------------------
# Default check wiring (real implementations)
# ---------------------------------------------------------------------------

def _fit_propensity_scores(vars) -> Any:
    from sklearn.linear_model import LogisticRegression

    df = vars.df
    cols = list(vars.covariates)  # treatment must NOT be a feature of its own propensity model
    x = df[cols].astype(float).values
    y = df[vars.treatment].astype(int).values
    model = LogisticRegression(max_iter=1000).fit(x, y)
    return model.predict_proba(x)[:, 1]


# Grant knobs per check: exactly these parameter names may be overridden
# from the user-editable thresholds file (or a caller-supplied dict).
ADJUSTABLE_PARAMS: Dict[str, set] = {
    "parallel_trends": {"alpha"},
    "no_anticipation": {"alpha"},
    "iv_relevance": {"f_threshold"},
    "positivity": {"overlap_threshold", "extreme_ps_bounds", "max_extreme_pct"},
    "rdd_no_manipulation": {"alpha"},
    "rdd_covariate_continuity": {"alpha"},
    "frontdoor_positivity": {"min_count"},
    "baseline_outcome_balance": {"smd_threshold"},
    "cond_ignorability": {"smd_threshold"},
}


def _clean(overrides):
    """Normalize JSON-loaded threshold overrides (lists -> tuples for bounds)."""
    out = {}
    for key, value in dict(overrides or {}).items():
        out[key] = tuple(value) if (isinstance(value, list) and key.endswith("_bounds")) else value
    return out


def build_default_checks(properties: Dict[str, Any], df, description, llm, thresholds=None):
    """Bind the real pre-model assumption checks for the traversal.

    Imports are local so the tree module stays importable without the full
    stats stack when tests inject stubs.

    ``thresholds`` maps check names to keyword overrides, validated strictly
    against ``ADJUSTABLE_PARAMS`` (unknown check names or unknown parameter
    keys raise ``ValueError``). When None, the user-editable thresholds.json
    is consulted via ``cais.config.load_thresholds``.
    """
    thresholds = load_thresholds() if thresholds is None else dict(thresholds or {})
    unknown_names = set(thresholds) - set(ADJUSTABLE_PARAMS)
    if unknown_names:
        raise ValueError(
            f"Unknown threshold checks: {sorted(unknown_names)}. "
            f"Adjustable checks: {sorted(ADJUSTABLE_PARAMS)}"
        )
    for name, kwargs in thresholds.items():
        bad = set(kwargs) - ADJUSTABLE_PARAMS[name]
        if bad:
            raise ValueError(
                f"Invalid threshold parameter(s) {sorted(bad)} for '{name}'. "
                f"Adjustable: {sorted(ADJUSTABLE_PARAMS[name])}"
            )

    from cais.models import AssumptionVariables
    from cais.methods import pre_model_assumption_utils as pm

    instruments = []
    if properties.get("instrument_variable"):
        instruments = [properties["instrument_variable"]]

    vars = AssumptionVariables(
        df=df,
        treatment=properties.get("treatment_variable"),
        outcome=properties.get("outcome_variable"),
        covariates=list(properties.get("covariates") or []),
        instruments=instruments,
        running_variable=properties.get("running_variable"),
        cutoff=properties.get("cutoff_value"),
        time_var=properties.get("time_variable"),
        group_var=properties.get("group_variable"),
        unit_var=properties.get("unit_variable"),
        mediator=properties.get("mediator_variable"),
        treatment_period_start=properties.get("treatment_period_start"),
        placebo_period_start=properties.get("placebo_period_start"),
        anticipation_window_start=properties.get("anticipation_window_start"),
        dataset_description=description,
        variables_summary={
            "treatment": properties.get("treatment_variable"),
            "outcome": properties.get("outcome_variable"),
            "covariates": properties.get("covariates"),
            "instrument": properties.get("instrument_variable"),
            "mediator": properties.get("mediator_variable"),
            "running_variable": properties.get("running_variable"),
        },
    )

    _ps_cache: Dict[str, Any] = {}

    def propensity_scores():
        if "ps" not in _ps_cache:
            _ps_cache["ps"] = _fit_propensity_scores(vars)
        return _ps_cache["ps"]

    def placeholder(reason: str):
        return lambda: AssumptionResult(passed=None, reasoning=reason)

    checks: Dict[str, Callable[[], AssumptionResult]] = {
        "sutva": lambda: pm.check_sutva(vars, llm=llm),
        "stable_group_composition": lambda: pm.check_stable_group_composition(vars, llm=llm),
        "baseline_outcome_balance": lambda: pm.check_baseline_outcome_balance(
            vars, **_clean(thresholds.get("baseline_outcome_balance"))
        ),
        "parallel_trends": lambda: pm.check_parallel_trends(
            vars, **_clean(thresholds.get("parallel_trends"))
        ),
        "no_anticipation": lambda: pm.check_no_anticipation(
            vars, **_clean(thresholds.get("no_anticipation"))
        ),
        "iv_relevance": lambda: pm.check_iv_relevance(
            vars, **_clean(thresholds.get("iv_relevance"))
        ),
        "iv_exclusion": lambda: pm.check_iv_exclusion(vars, llm=llm),
        "iv_exogeneity": lambda: pm.check_iv_exogeneity(vars, llm=llm),
        "iv_monotonicity": lambda: pm.check_iv_monotonicity(vars, llm=llm),
        "cond_ignorability": lambda: pm.check_cond_ignorability(
            vars, **_clean(thresholds.get("cond_ignorability"))
        ),
        "rdd_no_manipulation": lambda: pm.check_rdd_no_manipulation(
            vars, **_clean(thresholds.get("rdd_no_manipulation"))
        ),
        "rdd_covariate_continuity": lambda: pm.check_rdd_covariate_continuity(
            vars, **_clean(thresholds.get("rdd_covariate_continuity"))
        ),
        "rdd_continuity_potential_outcomes": lambda: pm.check_rdd_continuity_potential_outcomes(vars, llm=llm),
        "frontdoor_full_mediation": lambda: pm.check_frontdoor_full_mediation(vars, llm=llm),
        "frontdoor_no_tm_confounding": lambda: pm.check_frontdoor_no_TM_confounding(vars, llm=llm),
        "frontdoor_t_blocks_my": lambda: pm.check_frontdoor_T_blocks_MY(vars, llm=llm),
        "frontdoor_positivity": lambda: pm.check_frontdoor_positivity(
            vars, **_clean(thresholds.get("frontdoor_positivity"))
        ),
        "gps_cond_ignorability": placeholder("GPS pre-model cond. ignorability check is not implemented yet."),
        "gps_positivity": placeholder("GPS pre-model positivity check is not implemented yet."),
    }
    # positivity is wired separately so the propensity scores are only fitted
    # when that check is actually requested.
    checks["positivity"] = lambda: pm.check_positivity(
        vars, propensity_scores=propensity_scores(), **_clean(thresholds.get("positivity"))
    )
    return checks
