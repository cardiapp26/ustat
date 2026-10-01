from fastapi import APIRouter
from .linear import router as linear_router
from .logistic import router as logistic_router
from .cox import router as cox_router
from .glm import router as glm_router
from .multi_outcome_regression import router as multi_outcome_regression_router
from .multinomial import router as multinomial_router
from .psm_iptw import router as psm_iptw_router
from .sensitivity import router as sensitivity_router

# Aggregate all modular model sub-routers under a unified parent router.
# This router will be included in main.py with the prefix "/api/models".
router = APIRouter()

router.include_router(linear_router)
router.include_router(logistic_router)
router.include_router(cox_router)
router.include_router(glm_router)
router.include_router(multi_outcome_regression_router)
router.include_router(multinomial_router)
router.include_router(psm_iptw_router)
router.include_router(sensitivity_router)
