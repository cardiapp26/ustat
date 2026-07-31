# R reference for the Power Analysis panel.
#
#   Rscript qa/power_audit/reference.R
#
# Writes reference.json, one entry per case key in cases.json. Every number is
# printed at %.17g so the comparison is not limited by this file.

suppressMessages({
  library(pwr)
  library(powerSurvEpi)
})

here <- "qa/power_audit"
num <- function(x) {
  if (length(x) == 0L || is.na(x[[1L]]) || !is.finite(x[[1L]])) return("null")
  sprintf("%.17g", as.numeric(x[[1L]]))
}
kv  <- function(k, v) paste0("\"", k, "\":", v)
obj <- function(...) paste0("{", paste(c(...), collapse = ","), "}")

cases <- list()
add <- function(key, value, note = NULL, engine = NULL) {
  fields <- c(kv("value", num(value)))
  if (!is.null(engine)) fields <- c(fields, kv("engine", paste0("\"", engine, "\"")))
  if (!is.null(note))   fields <- c(fields, kv("note", paste0("\"", note, "\"")))
  cases[[key]] <<- obj(fields)
}

alt2 <- "two.sided"
alt1 <- "greater"

# ── two-sample t ─────────────────────────────────────────────────────────────
add("t_two_power",
    pwr.t.test(n = 64, d = 0.5, sig.level = 0.05, type = "two.sample",
               alternative = alt2)$power, engine = "pwr.t.test")
add("t_two_n",
    ceiling(pwr.t.test(power = 0.8, d = 0.5, sig.level = 0.05,
                       type = "two.sample", alternative = alt2)$n),
    engine = "pwr.t.test")
add("t_two_es",
    pwr.t.test(n = 64, power = 0.8, sig.level = 0.05, type = "two.sample",
               alternative = alt2)$d, engine = "pwr.t.test")
add("t_two_power_1t",
    pwr.t.test(n = 100, d = 0.35, sig.level = 0.05, type = "two.sample",
               alternative = alt1)$power, engine = "pwr.t.test")
add("t_two_small_n",
    pwr.t.test(n = 10, d = 0.8, sig.level = 0.05, type = "two.sample",
               alternative = alt2)$power, engine = "pwr.t.test")

# ── one-sample t ─────────────────────────────────────────────────────────────
add("t_one_power",
    pwr.t.test(n = 34, d = 0.5, sig.level = 0.05, type = "one.sample",
               alternative = alt2)$power, engine = "pwr.t.test")
add("t_one_n",
    ceiling(pwr.t.test(power = 0.8, d = 0.5, sig.level = 0.05,
                       type = "one.sample", alternative = alt2)$n),
    engine = "pwr.t.test")
add("t_one_small_n",
    pwr.t.test(n = 8, d = 0.9, sig.level = 0.05, type = "one.sample",
               alternative = alt2)$power, engine = "pwr.t.test")

# ── one-way ANOVA ────────────────────────────────────────────────────────────
add("anova_power",
    pwr.anova.test(k = 4, n = 52, f = 0.25, sig.level = 0.05)$power,
    engine = "pwr.anova.test")
add("anova_n",
    ceiling(pwr.anova.test(k = 4, f = 0.25, sig.level = 0.05, power = 0.8)$n),
    engine = "pwr.anova.test")
add("anova_k3",
    pwr.anova.test(k = 3, n = 20, f = 0.4, sig.level = 0.05)$power,
    engine = "pwr.anova.test")

# ── correlation ──────────────────────────────────────────────────────────────
add("corr_power",
    pwr.r.test(n = 85, r = 0.3, sig.level = 0.05, alternative = alt2)$power,
    engine = "pwr.r.test")
add("corr_n",
    ceiling(pwr.r.test(r = 0.3, sig.level = 0.05, power = 0.8,
                       alternative = alt2)$n), engine = "pwr.r.test")
add("corr_small_n",
    pwr.r.test(n = 12, r = 0.5, sig.level = 0.05, alternative = alt2)$power,
    engine = "pwr.r.test")
add("corr_1t",
    pwr.r.test(n = 85, r = 0.3, sig.level = 0.05, alternative = alt1)$power,
    engine = "pwr.r.test")

# ── two proportions (Cohen's h, arcsine) ─────────────────────────────────────
h <- function(p1, p2) abs(2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2)))
add("prop_power",
    pwr.2p.test(h = h(0.5, 0.3), n = 100, sig.level = 0.05,
                alternative = alt2)$power, engine = "pwr.2p.test")
add("prop_n",
    ceiling(pwr.2p.test(h = h(0.5, 0.3), sig.level = 0.05, power = 0.8,
                        alternative = alt2)$n), engine = "pwr.2p.test")
add("prop_rare",
    pwr.2p.test(h = h(0.05, 0.01), n = 300, sig.level = 0.05,
                alternative = alt2)$power, engine = "pwr.2p.test")

# ── chi-square goodness of fit ───────────────────────────────────────────────
add("chi2_power",
    pwr.chisq.test(w = 0.3, N = 150, df = 3, sig.level = 0.05)$power,
    engine = "pwr.chisq.test (df = k - 1)")
add("chi2_n",
    ceiling(pwr.chisq.test(w = 0.3, df = 3, sig.level = 0.05, power = 0.8)$N),
    engine = "pwr.chisq.test (df = k - 1)")

# ── Cox / survival (Schoenfeld) ──────────────────────────────────────────────
# ssizeCT.default returns the number of subjects per arm for a given number of
# events; powerCT.default0 gives power from the two arm sizes. Both implement
# Schoenfeld's formula, which is what the endpoint uses, so the comparison is
# on the events requirement and the power, not on any accrual model.
schoenfeld_events <- function(hr, power, alpha, p) {
  (qnorm(1 - alpha / 2) + qnorm(power))^2 / (p * (1 - p) * log(hr)^2)
}
add("cox_n",
    ceiling(schoenfeld_events(1.8, 0.8, 0.05, 0.5) / 0.4),
    engine = "Schoenfeld", note = "events / event_rate")
add("cox_power", {
    d <- 300 * 0.4
    se <- sqrt(1 / (d * 0.5 * 0.5))
    pnorm(abs(log(1.8)) / se - qnorm(1 - 0.05 / 2))
  }, engine = "Schoenfeld")
add("cox_unbalanced",
    ceiling(schoenfeld_events(2.0, 0.9, 0.05, 0.3) / 0.3),
    engine = "Schoenfeld", note = "events / event_rate")

# ── logistic regression (Hsieh 1998) ────────────────────────────────────────
# The endpoint implements Hsieh's continuous-covariate form,
#   n = (z_a + z_b)^2 / (p (1 - p) B^2)
# with p the overall event probability and B the log odds ratio per unit.
hsieh_n <- function(log_or, p, power, alpha) {
  (qnorm(1 - alpha / 2) + qnorm(power))^2 / (p * (1 - p) * log_or^2)
}
# The panel labels this field "Odds ratio (OR)" and posts the OR itself in
# `log_or`, so 1.5 is an odds ratio and the coefficient is log(1.5).
add("logistic_n", ceiling(hsieh_n(log(1.5), 0.3, 0.8, 0.05)),
    engine = "Hsieh continuous covariate")
add("logistic_power", {
    se <- sqrt(1 / (200 * 0.3 * 0.7))
    pnorm(abs(log(1.5)) / se - qnorm(1 - 0.05 / 2))
  }, engine = "Hsieh continuous covariate")

out <- paste0("{\"cases\":{", paste(
  mapply(function(k, v) paste0("\"", k, "\":", v), names(cases), cases,
         USE.NAMES = FALSE), collapse = ","), "}}")
writeLines(out, file.path(here, "reference.json"))
cat("wrote", file.path(here, "reference.json"), "-", length(cases), "cases\n")
