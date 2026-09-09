#!/usr/bin/env Rscript
# Installs every package the three reference.R scripts need and prints a
# version table so exact versions land in the `docker build` log.
#
# models_audit/reference.R:  survival, MASS, logistf, ordinal, lme4, geepack,
#                             MatchIt, survey, car, pROC, rms
# power_audit/reference.R:   pwr, powerSurvEpi
# tests_audit/reference.R:   base R only (no extra packages)

pkgs <- c("survival", "MASS", "logistf", "ordinal", "lme4", "geepack",
          "MatchIt", "survey", "car", "pROC", "rms", "pwr",
          "powerSurvEpi", "jsonlite")

# Dated CRAN snapshot (Posit Package Manager). Installing from a moving
# repository would mean two builds of this image on different days can
# disagree about what "R says" without any change in this repo; printing the
# version table below records versions, it does not pin them. Bump this date
# deliberately, together with the R version pin in the Dockerfile, and re-run
# the audits to see what moved.
SNAPSHOT_REPO <- "https://packagemanager.posit.co/cran/2026-09-01"

status <- character(length(pkgs))
version <- character(length(pkgs))

for (i in seq_along(pkgs)) {
  p <- pkgs[i]
  ok <- tryCatch({
    install.packages(p, repos = SNAPSHOT_REPO)
    TRUE
  }, error = function(e) {
    message("INSTALL FAILED: ", p, " -- ", conditionMessage(e))
    FALSE
  })
  if (ok && requireNamespace(p, quietly = TRUE)) {
    status[i] <- "installed"
    version[i] <- as.character(packageVersion(p))
  } else {
    status[i] <- "FAILED"
    version[i] <- NA_character_
  }
}

results <- data.frame(package = pkgs, status = status, version = version,
                       stringsAsFactors = FALSE)

cat("\n===== PACKAGE VERSION TABLE =====\n")
print(results, row.names = FALSE)
cat("==================================\n\n")

failed <- results$package[results$status == "FAILED"]
if (length(failed) > 0) {
  cat("ERROR: the following packages failed to install:\n")
  cat(paste(" -", failed, collapse = "\n"), "\n")
  # Fail the docker build. An image missing a reference package would not be
  # the pinned environment, and the audits would fall over later and further
  # from the cause.
  quit(status = 1)
}
