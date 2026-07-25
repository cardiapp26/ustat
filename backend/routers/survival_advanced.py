"""Advanced Survival Analyses router (thin HTTP adapter).

Validates the request and delegates to services.survival_advanced_service.
See that module for the statistical implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services import survival_advanced_service as _svc

router = APIRouter()


class MICERequest(BaseModel):
    session_id: str
    columns: List[str]
    n_imputations: int = 5
    max_iter: int = 10
    random_state: int = 42
    mechanism: str = "unknown"  # unknown, MCAR, MAR, MNAR
    # When true, keep the original columns and write imputed values to new
    # "<col>_imp" columns instead of filling in place.
    new_columns: bool = False


class MICEPreviewRequest(BaseModel):
    session_id: str
    columns: List[str]
    max_iter: int = 10
    random_state: int = 42
    mechanism: str = "unknown"


class MICETransferRequest(BaseModel):
    session_id: str
    preview_rows: List[Dict[str, Any]]


@router.post("/mice")
def mice_imputation(req: MICERequest):
    return _svc.fit_mice(req)


@router.post("/mice_preview")
def mice_preview(req: MICEPreviewRequest):
    return _svc.mice_preview(req)


@router.post("/mice_transfer")
def mice_transfer(req: MICETransferRequest):
    return _svc.mice_transfer(req)


class FineGrayRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    event_of_interest: int = 1
    group_col: Optional[str] = None
    # Subdistribution-hazard regression (Fine-Gray 1999) via the Geskus 2011
    # IPCW-weighted Cox recipe. When non-empty the endpoint also returns a
    # `regression_result` block with subdistribution hazard ratios (sHR).
    predictors: Optional[List[str]] = None
    imputation: Optional[str] = "listwise"


@router.post("/fine_gray")
def fine_gray(req: FineGrayRequest):
    return _svc.fit_fine_gray(req)


class EValueRequest(BaseModel):
    estimate: float
    ci_low: float
    ci_high: float
    measure_type: str = "OR"  # OR, HR, RR
    baseline_risk: float = 0.1  # p0, used for OR→RR conversion


@router.post("/evalue")
def evalue(req: EValueRequest):
    return _svc.fit_evalue(req)


class CausalSensitivityRequest(BaseModel):
    observed_estimate: float = Field(1.0, gt=0)
    ci_low: Optional[float] = Field(None, gt=0)
    ci_high: Optional[float] = Field(None, gt=0)
    measure: str = "rr"  # rr | or | hr
    rare_outcome: bool = False
    baseline_risk: Optional[float] = Field(None, ge=0.001, le=0.99)
    smd: Optional[float] = None

    confounding_strength: float = Field(2.0, gt=0)
    prevalence_exposed: float = Field(0.5, ge=0, le=1)
    prevalence_unexposed: float = Field(0.5, ge=0, le=1)
    unmeasured_confounders: List[Dict[str, Any]] = Field(default_factory=list)

    session_id: Optional[str] = None
    treatment_col: Optional[str] = None
    outcome_col: Optional[str] = None
    monotone_treatment_response: bool = False
    p_y1_treated: Optional[float] = Field(None, ge=0, le=1)
    p_y1_control: Optional[float] = Field(None, ge=0, le=1)
    p_treated: Optional[float] = Field(None, ge=0, le=1)

    match_id_col: Optional[str] = None
    rosenbaum_gamma_max: float = Field(3.0, gt=1)
    rosenbaum_n_gamma: int = Field(60, ge=2, le=500)

    negative_control_outcome_col: Optional[str] = None
    negative_control_covariates: List[str] = Field(default_factory=list)
    imputation: Optional[str] = "listwise"


@router.post("/causal_sensitivity")
def causal_sensitivity(req: CausalSensitivityRequest):
    return _svc.fit_causal_sensitivity(req)


class LandmarkRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    landmark_time: float
    group_col: Optional[str] = None
    predictors: Optional[List[str]] = None
    imputation: Optional[str] = "listwise"


@router.post("/landmark")
def landmark_analysis(req: LandmarkRequest):
    return _svc.fit_landmark(req)


class RMSTRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    tau: float                                   # restriction time-horizon
    group_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


@router.post("/rmst")
def rmst(req: RMSTRequest):
    return _svc.fit_rmst(req)


class RecurrentLWYYRequest(BaseModel):
    session_id: str
    id_col: str
    start_col: str
    stop_col: str
    event_col: str
    predictors: List[str]
    group_col: Optional[str] = None          # for the mean cumulative function plot
    model_type: str = "lwyy"                 # lwyy | wlw | both | mcf_only
    event_order_col: Optional[str] = None    # optional recurrence number for WLW strata
    time_scale: str = "total"                # total | gap | calendar
    terminal_time_col: Optional[str] = None
    terminal_event_col: Optional[str] = None
    include_negative_binomial: bool = False
    include_joint_frailty_spec: bool = False
    imputation: Optional[str] = "listwise"


@router.post("/recurrent_lwyy")
def recurrent_lwyy(req: RecurrentLWYYRequest):
    return _svc.fit_recurrent_lwyy(req)


class MultistateRequest(BaseModel):
    session_id: str
    id_col: str = "id"
    from_state_col: str = "from_state"
    to_state_col: str = "to_state"
    entry_col: str = "entry"
    exit_col: str = "exit"
    event_col: str = "event"
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    transition_model_type: Optional[str] = "cox"


@router.post("/multistate")
def multistate(req: MultistateRequest):
    return _svc.fit_multistate(req)


class DynamicPredictionRequest(BaseModel):
    session_id: str
    landmark_time: float
    current_state: int = 0
    id_col: str = "id"
    from_state_col: str = "from_state"
    to_state_col: str = "to_state"
    entry_col: str = "entry"
    exit_col: str = "exit"
    event_col: str = "event"
    predictors: List[str]
    horizon: float = 5.0
    n_points: int = 20
    transition_model_type: Optional[str] = "cox"
    run_bootstrap: Optional[bool] = False
    n_bootstrap: Optional[int] = 50
    run_microsimulation: Optional[bool] = False
    n_simulations: Optional[int] = 1000


@router.post("/dynamic_prediction")
def dynamic_prediction(req: DynamicPredictionRequest):
    return _svc.fit_dynamic_prediction(req)


class JointModelRequest(BaseModel):
    session_id_long: str
    session_id_surv: Optional[str] = None   # if None, assume same session
    id_col: str = "id"
    time_col: str = "time"
    y_cols: List[str] = Field(default_factory=lambda: ["Y"])
    long_predictors: List[str] = Field(default_factory=list)
    surv_predictors: List[str] = Field(default_factory=list)
    duration_col: str = "duration"
    event_col: str = "event"
    association: List[str] = Field(default_factory=lambda: ["value"])
    time_spline: bool = False
    latent_classes: int = 0


@router.post("/joint_model")
def joint_model(req: JointModelRequest):
    return _svc.fit_joint_model(req)


class ExternalValidationRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predicted_lp_col: str
    time_points: Optional[List[float]] = None
    survival_probs: Optional[List[List[float]]] = None  # n_samples x len(time_points)
    dev_metrics: Optional[Dict[str, float]] = None


@router.post("/external_validation")
def external_validation(req: ExternalValidationRequest):
    return _svc.fit_external_validation(req)


class SurvivalMLBenchmarkRequest(BaseModel):
    session_id: str
    duration_col: str = "duration"
    event_col: str = "event"
    predictors: Optional[List[str]] = None
    n_estimators: int = 300
    nested_cv: bool = False
    repeated_cv_repeats: int = 1
    cv_folds: int = 5
    inner_cv_folds: int = 3
    hyperparameter_iter: int = 12
    include_shap: bool = False
    include_partial_dependence: bool = True
    include_competing_risks_ml: bool = False
    optimization_method: str = "random"  # random | bayesian


@router.post("/ml_survival_benchmark")
def ml_survival_benchmark(req: SurvivalMLBenchmarkRequest):
    return _svc.fit_ml_survival_benchmark(req)


class FrailtyRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    cluster_col: str
    predictors: List[str]
    penalizer: float = 0.05
    frailty_distribution: str = "gamma"
    estimation_method: str = "penalized"
    nested_cluster_cols: List[str] = Field(default_factory=list)
    correlated_cluster_col: Optional[str] = None
    baseline_hazard: str = "semi_parametric"
    include_diagnostics: bool = True
    imputation: Optional[str] = "listwise"


@router.post("/frailty")
def shared_frailty(req: FrailtyRequest):
    return _svc.fit_shared_frailty(req)


class IntervalCensoredRequest(BaseModel):
    session_id: str
    lower_col: str                 # left bracket L (≥0)
    upper_col: str                 # right bracket R (blank/inf ⇒ right-censored)
    covariates: List[str] = Field(default_factory=list)
    group_col: Optional[str] = None


@router.post("/interval_censored")
def interval_censored(req: IntervalCensoredRequest):
    from services import interval_censored as _ic
    return _ic.interval_censored_analysis(req)


