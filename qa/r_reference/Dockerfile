# Pinned R build for regenerating uSTAT's R reference literals.
#
# The pin here (4.6.0) should track whatever R version webR ships in its
# wasm distribution, NOT necessarily the R version on any one contributor's
# laptop -- see README.md in this directory for why.
FROM r-base:4.6.0

# System libraries needed to compile the R packages below from source
# (survival/MASS/Matrix/lme4 etc. all have compiled components; several of
# the other packages here link against GDAL/GEOS/XML/curl/ssl at the OS level).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libblas-dev \
    liblapack-dev \
    libuv1-dev \
    && rm -rf /var/lib/apt/lists/*

# Every package qa/models_audit/reference.R, qa/power_audit/reference.R and
# qa/tests_audit/reference.R touch via library()/requireNamespace()/::, plus
# jsonlite (not used by the hand-rolled JSON emitters, but harmless to have
# and useful for ad hoc inspection).
#
# models_audit/reference.R:  survival, MASS, logistf, ordinal, lme4, geepack,
#                             MatchIt, survey, car, pROC, rms
# power_audit/reference.R:   pwr, powerSurvEpi
# tests_audit/reference.R:   base R only (no extra packages)
COPY install_packages.R /tmp/install_packages.R
RUN Rscript /tmp/install_packages.R

WORKDIR /work
