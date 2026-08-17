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

status <- character(length(pkgs))
version <- character(length(pkgs))

for (i in seq_along(pkgs)) {
  p <- pkgs[i]
  ok <- tryCatch({
    install.packages(p, repos = "https://cloud.r-project.org")
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
  cat("WARNING: the following packages failed to install and were skipped:\n")
  cat(paste(" -", failed, collapse = "\n"), "\n")
}
