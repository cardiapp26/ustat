from fastapi import APIRouter
from .descriptive import router as descriptive_router
from .inferential import router as inferential_router
from .nonparametric import router as nonparametric_router
from .correlation import router as correlation_router
from .normality import router as normality_router

# Aggregate all modular statistical sub-routers under a unified parent router.
# This router will be included in main.py with the prefix "/api/stats".
router = APIRouter()

router.include_router(descriptive_router)
router.include_router(inferential_router)
router.include_router(nonparametric_router)
router.include_router(correlation_router)
router.include_router(normality_router)
