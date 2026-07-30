#!/usr/bin/env Rscript
# Independent reference values for the uSTAT "advanced" survival endpoints.
#
# This is the GROUND-TRUTH side of the audit. It fits exactly the models the
# brief specifies, with no improvisation, and emits their numbers to JSON at
# full precision so nothing is lost to rounding on the way out.
#
#   Rscript qa/models_audit_30x50/reference_advanced.R
#
# Writes:   qa/models_audit_30x50/reference_advanced.json
# Writes:   no other file.

suppressPackageStartupMessages({
  library(survival)
  library(cmprsk)
  library(survRM2)
  library(icenReg)
  library(mstate)
})

# ── locate the data dir (works whether run from project root or the dir) ───────
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
if (length(here) == 0 || here == "") here <- "qa/models_audit_30x50"

# na.strings matters: without it an empty cell in a text column arrives as ""
# and becomes a category of its own, which is the very mistake this harness
# exists to catch.
rd <- function(f) read.csv(file.path(here, f), stringsAsFactors = FALSE,
                           na.strings = c("", "NA"))

d        <- rd("dataset.csv")            # time, status, cmp_status, ic_l, ic_r, age, arm, sex, site, bmi, ...
d$arm    <- factor(d$arm)                # reference level: control
d$sex    <- factor(d$sex)                # reference level: F
d$site   <- factor(d$site)
drec     <- rd("dataset_recurrent.csv")  # pid, start, stop, event, arm, age, sex
drec$arm <- factor(drec$arm)
dmulti   <- rd("dataset_multistate.csv") # id, from_state, to_state, entry, exit, event, age, arm, sex
dmulti$arm <- factor(dmulti$arm)
dext     <- rd("dataset_external.csv")   # ... time, status, pred_lp

Z <- qnorm(0.975) # 1.959963984540054, for 95% CIs we build by hand

# ── JSON emission (no jsonlite dependency; preserve all digits) ────────────────
num <- function(x) {
  if (is.null(x) || length(x) == 0) return("null")
  x <- x[[1]]
  if (is.na(x) || is.nan(x) || is.infinite(x)) return("null")
  sprintf("%.17g", as.numeric(x))
}
str_ <- function(x) paste0('"', gsub('"', '\\\\"', as.character(x)), '"')
kv  <- function(k, v) paste0(str_(k), ": ", v)
obj <- function(...) paste0("{", paste(c(...), collapse = ", "), "}")
arr <- function(v) paste0("[", paste(v, collapse = ", "), "]")
err_obj <- function(e) {
  msg <- conditionMessage(e)
  obj(kv("error", str_(msg)))
}

# ── shared term builders ──────────────────────────────────────────────────────
cox_term <- function(co, ci, i) {
  # co: summary(coxph)$coefficients row (has coef, se(coef), z, Pr(>|z|), exp(coef))
  # ci: summary(coxph)$conf.int row (has exp(coef), lower .95, upper .95)
  obj(kv("term", str_(rownames(co)[i])),
      kv("estimate", num(co[i, "coef"])),
      kv("se", num(co[i, "se(coef)"])),
      kv("statistic", num(co[i, "z"])),
      kv("p", num(co[i, "Pr(>|z|)"])),
      kv("hr", num(co[i, "exp(coef)"])),
      kv("hr_ci_low", num(ci[i, "lower .95"])),
      kv("hr_ci_high", num(ci[i, "upper .95"])))
}

models <- list()

# A model string is "in error" only if the TOP-LEVEL object is {"error": ...}.
# (e.g. multistate legitimately carries per-transition error sub-objects while
# the model itself ran fine.) The status line must reflect the overall attempt.
is_err <- function(s) {
  s <- trimws(s)
  grepl('^\\{\\s*"error"\\s*:', s)
}

# ── 1. km ─────────────────────────────────────────────────────────────────────
models$km <- tryCatch({
    # conf.type="log-log" on purpose. survfit DEFAULTS to "log"; the app uses
  # the log-log (cloglog) transform, which is what SAS and Stata default to
  # and the one that keeps the limits inside [0, 1]. Comparing against the R
  # default would flag a convention as an error.
  km <- survfit(Surv(time, status) ~ arm, data = d, conf.type = "log-log")
  tb <- summary(km)$table
  strata_rows <- vapply(seq_len(nrow(tb)), function(i)
    obj(kv("stratum", str_(rownames(tb)[i])),
        kv("n", num(tb[i, "records"])),
        kv("n_events", num(tb[i, "events"])),
        kv("median", num(tb[i, "median"])),
        kv("median_ci_low", num(tb[i, "0.95LCL"])),
        kv("median_ci_high", num(tb[i, "0.95UCL"]))), character(1))
  st <- summary(km, times = c(5, 10), extend = FALSE)
  at_times <- vapply(seq_len(length(st$time)), function(i)
    obj(kv("stratum", str_(as.character(st$strata[i]))),
        kv("time", num(st$time[i])),
        kv("surv", num(st$surv[i])),
        kv("se", num(st$std.err[i])),
        kv("ci_low", num(st$lower[i])),
        kv("ci_high", num(st$upper[i]))), character(1))
  sd_ <- survdiff(Surv(time, status) ~ arm, data = d)
  df_ <- length(sd_$n) - 1
  obj(kv("strata", arr(strata_rows)),
      kv("at_times", arr(at_times)),
      kv("survdiff_chisq", num(sd_$chisq)),
      kv("survdiff_df", num(df_)),
      kv("survdiff_p", num(pchisq(sd_$chisq, df_, lower.tail = FALSE))))
}, error = err_obj)

cat(sprintf("km                 %s\n", if (is_err(models$km)) "ERROR" else "OK"))

# ── 2. cox ────────────────────────────────────────────────────────────────────
models$cox <- tryCatch({
  fit <- coxph(Surv(time, status) ~ age + arm + sex, data = d)
  s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
  terms <- vapply(seq_len(nrow(co)), function(i) cox_term(co, ci, i), character(1))
  zph <- cox.zph(fit)
  zph_rows <- vapply(seq_len(nrow(zph$table)), function(i)
    obj(kv("term", str_(rownames(zph$table)[i])),
        kv("chisq", num(zph$table[i, "chisq"])),
        kv("df", num(zph$table[i, "df"])),
        kv("p", num(zph$table[i, "p"]))), character(1))
  obj(kv("terms", arr(terms)),
      kv("concordance", num(s$concordance[["C"]])),
      kv("loglik", num(s$loglik[2])),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)),
      kv("zph", arr(zph_rows)))
}, error = err_obj)

cat(sprintf("cox                %s\n", if (is_err(models$cox)) "ERROR" else "OK"))

# ── 3. cox_missing ────────────────────────────────────────────────────────────
models$cox_missing <- tryCatch({
  fit <- coxph(Surv(time, status) ~ age + arm + bmi, data = d)
  s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
  terms <- vapply(seq_len(nrow(co)), function(i) cox_term(co, ci, i), character(1))
  obj(kv("terms", arr(terms)),
      kv("concordance", num(s$concordance[["C"]])),
      kv("loglik", num(s$loglik[2])),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)))
}, error = err_obj)

cat(sprintf("cox_missing        %s\n", if (is_err(models$cox_missing)) "ERROR" else "OK"))

# ── 4. cox_horizons ───────────────────────────────────────────────────────────
models$cox_horizons <- tryCatch({
  horizon_fit <- function(h) {
    t_h <- pmin(d$time, h)
    s_h <- ifelse(d$time <= h, d$status, 0)
    fit <- coxph(Surv(t_h, s_h) ~ arm + age, data = d)
    s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
    i <- which(rownames(co) == "armtreat")
    obj(kv("estimate", num(co[i, "coef"])),
        kv("se", num(co[i, "se(coef)"])),
        kv("p", num(co[i, "Pr(>|z|)"])),
        kv("hr", num(co[i, "exp(coef)"])),
        kv("hr_ci_low", num(ci[i, "lower .95"])),
        kv("hr_ci_high", num(ci[i, "upper .95"])),
        kv("n", num(s$n)),
        kv("nevent", num(s$nevent)))
  }
  # full follow-up (uncensored) fit
  fit_full <- coxph(Surv(time, status) ~ arm + age, data = d)
  s_full <- summary(fit_full); co_full <- s_full$coefficients; ci_full <- s_full$conf.int
  i_full <- which(rownames(co_full) == "armtreat")
  full_obj <- obj(kv("estimate", num(co_full[i_full, "coef"])),
                  kv("se", num(co_full[i_full, "se(coef)"])),
                  kv("p", num(co_full[i_full, "Pr(>|z|)"])),
                  kv("hr", num(co_full[i_full, "exp(coef)"])),
                  kv("hr_ci_low", num(ci_full[i_full, "lower .95"])),
                  kv("hr_ci_high", num(ci_full[i_full, "upper .95"])),
                  kv("n", num(s_full$n)),
                  kv("nevent", num(s_full$nevent)))
  obj(kv("5", horizon_fit(5)),
      kv("10", horizon_fit(10)),
      kv("full", full_obj))
}, error = err_obj)

cat(sprintf("cox_horizons       %s\n", if (is_err(models$cox_horizons)) "ERROR" else "OK"))

# ── 5. cox_tv ─────────────────────────────────────────────────────────────────
models$cox_tv <- tryCatch({
  fit <- coxph(Surv(start, stop, event) ~ age + arm, data = drec)  # NO cluster term
  s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
  terms <- vapply(seq_len(nrow(co)), function(i) cox_term(co, ci, i), character(1))
  obj(kv("terms", arr(terms)),
      kv("concordance", num(s$concordance[["C"]])),
      kv("loglik", num(s$loglik[2])),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)))
}, error = err_obj)

cat(sprintf("cox_tv             %s\n", if (is_err(models$cox_tv)) "ERROR" else "OK"))

# ── 6. recurrent_lwyy ─────────────────────────────────────────────────────────
models$recurrent_lwyy <- tryCatch({
  fit <- coxph(Surv(start, stop, event) ~ age + arm + cluster(pid), data = drec)
  s <- summary(fit); co <- s$coefficients
  # HR and CI based on the ROBUST se: exp(coef) and exp(coef +/- z * robust se)
  terms <- vapply(seq_len(nrow(co)), function(i) {
    beta <- co[i, "coef"]; rse <- co[i, "robust se"]
    obj(kv("term", str_(rownames(co)[i])),
        kv("estimate", num(beta)),
        kv("se_robust", num(rse)),
        kv("statistic", num(co[i, "z"])),
        kv("p", num(co[i, "Pr(>|z|)"])),
        kv("hr", num(exp(beta))),
        kv("hr_ci_low", num(exp(beta - Z * rse))),
        kv("hr_ci_high", num(exp(beta + Z * rse))))
  }, character(1))
  obj(kv("terms", arr(terms)),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)))
}, error = err_obj)

cat(sprintf("recurrent_lwyy     %s\n", if (is_err(models$recurrent_lwyy)) "ERROR" else "OK"))

# ── 7. fine_gray ──────────────────────────────────────────────────────────────
models$fine_gray <- tryCatch({
  cov1 <- model.matrix(~ age + arm, data = d)[, -1, drop = FALSE]
  fit <- cmprsk::crr(ftime = d$time, fstatus = d$cmp_status, cov1 = cov1,
                     failcode = 1, cencode = 0)
  s <- summary(fit)
  co <- s$coef          # coef, exp(coef), se(coef), z, p-value
  ci <- s$conf.int      # exp(coef), exp(-coef), 2.5%, 97.5%   (subdistribution HR CI)
  terms <- vapply(seq_len(nrow(co)), function(i)
    obj(kv("term", str_(rownames(co)[i])),
        kv("estimate", num(co[i, "coef"])),
        kv("se", num(co[i, "se(coef)"])),
        kv("statistic", num(co[i, "z"])),
        kv("p", num(co[i, "p-value"])),
        kv("subdist_hr", num(co[i, "exp(coef)"])),
        kv("subdist_hr_ci_low", num(ci[i, "2.5%"])),
        kv("subdist_hr_ci_high", num(ci[i, "97.5%"]))), character(1))
  obj(kv("terms", arr(terms)),
      kv("n", num(s$n)),
      kv("converged", str_(as.character(s$converged))))
}, error = err_obj)

cat(sprintf("fine_gray          %s\n", if (is_err(models$fine_gray)) "ERROR" else "OK"))

# ── 8. rmst ───────────────────────────────────────────────────────────────────
models$rmst <- tryCatch({
  r <- survRM2::rmst2(time = d$time, status = d$status,
                      arm = as.integer(d$arm == "treat"), tau = 10)
  rmst1_obj <- function(x) {  # x is a per-arm rmst1 object
    obj(kv("rmst", num(x$rmst["Est."])),
        kv("rmst_se", num(x$rmst["se"])),
        kv("rmst_ci_low", num(x$rmst["lower .95"])),
        kv("rmst_ci_high", num(x$rmst["upper .95"])),
        kv("rmtl", num(x$rmtl["Est."])),
        kv("rmtl_ci_low", num(x$rmtl["lower .95"])),
        kv("rmtl_ci_high", num(x$rmtl["upper .95"])),
        kv("tau", num(x$tau)))
  }
  u <- r$unadjusted.result
  d_row <- "RMST (arm=1)-(arm=0)"
  r_row <- "RMST (arm=1)/(arm=0)"
  diff_obj <- obj(kv("rmst", num(u[d_row, "Est."])),
                  kv("rmst_ci_low", num(u[d_row, "lower .95"])),
                  kv("rmst_ci_high", num(u[d_row, "upper .95"])),
                  kv("p", num(u[d_row, "p"])))
  ratio_obj <- obj(kv("rmst", num(u[r_row, "Est."])),
                   kv("rmst_ci_low", num(u[r_row, "lower .95"])),
                   kv("rmst_ci_high", num(u[r_row, "upper .95"])),
                   kv("p", num(u[r_row, "p"])))
  obj(kv("treat",   rmst1_obj(r$RMST.arm1)),   # arm = 1 (treat)
      kv("control", rmst1_obj(r$RMST.arm0)),   # arm = 0 (control)
      kv("difference", diff_obj),
      kv("ratio", ratio_obj),
      kv("tau", num(r$tau)))
}, error = err_obj)

cat(sprintf("rmst               %s\n", if (is_err(models$rmst)) "ERROR" else "OK"))

# ── 9. landmark ───────────────────────────────────────────────────────────────
models$landmark <- tryCatch({
  keep <- d$time >= 5
  n_kept <- sum(keep); n_excluded <- sum(!keep)
  sub <- d[keep, ]
  sub$time2 <- sub$time - 5

  sd_ <- survdiff(Surv(time2, status) ~ arm, data = sub)
  sdf <- length(sd_$n) - 1
  km <- survfit(Surv(time2, status) ~ arm, data = sub)
  tb <- summary(km)$table
  med_rows <- vapply(seq_len(nrow(tb)), function(i)
    obj(kv("stratum", str_(rownames(tb)[i])),
        kv("median", num(tb[i, "median"])),
        kv("median_ci_low", num(tb[i, "0.95LCL"])),
        kv("median_ci_high", num(tb[i, "0.95UCL"]))), character(1))
  fit <- coxph(Surv(time2, status) ~ age + arm, data = sub)
  s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
  terms <- vapply(seq_len(nrow(co)), function(i) cox_term(co, ci, i), character(1))
  obj(kv("n_kept", num(n_kept)),
      kv("n_excluded", num(n_excluded)),
      kv("survdiff_chisq", num(sd_$chisq)),
      kv("survdiff_df", num(sdf)),
      kv("survdiff_p", num(pchisq(sd_$chisq, sdf, lower.tail = FALSE))),
      kv("median_per_arm", arr(med_rows)),
      kv("cox_terms", arr(terms)))
}, error = err_obj)

cat(sprintf("landmark           %s\n", if (is_err(models$landmark)) "ERROR" else "OK"))

# ── 10. frailty ───────────────────────────────────────────────────────────────
models$frailty <- tryCatch({
  fit <- coxph(Surv(time, status) ~ age + arm + frailty(site, distribution = "gamma"), data = d)
  s <- summary(fit)
  co <- s$coefficients        # last row is the frailty term (Chisq, DF, p)
  ci <- s$conf.int            # fixed-effect rows only
  fixed_names <- intersect(rownames(co), rownames(ci))   # age, armtreat
  terms <- vapply(fixed_names, function(nm) {
    i <- which(rownames(co) == nm); j <- which(rownames(ci) == nm)
    obj(kv("term", str_(nm)),
        kv("estimate", num(co[i, "coef"])),
        kv("se", num(co[i, "se(coef)"])),
        kv("p", num(co[i, "p"])),
        kv("hr", num(ci[j, "exp(coef)"])),
        kv("hr_ci_low", num(ci[j, "lower .95"])),
        kv("hr_ci_high", num(ci[j, "upper .95"])))
  }, character(1))
  # frailty term: the row whose name starts with "frailty("
  fr_row <- which(grepl("^frailty\\(", rownames(co)))
  fr_chisq <- if (length(fr_row)) co[fr_row[1], "Chisq"] else NA
  fr_df    <- if (length(fr_row)) co[fr_row[1], "DF"]    else NA
  fr_p     <- if (length(fr_row)) co[fr_row[1], "p"]     else NA
  # theta: the converged frailty variance, stored in fit$history[[1]]$theta
  theta <- NA
  if (length(fit$history) >= 1 && !is.null(fit$history[[1]]$theta))
    theta <- fit$history[[1]]$theta
  obj(kv("terms", arr(terms)),
      kv("frailty_theta", num(theta)),
      kv("frailty_chisq", num(fr_chisq)),
      kv("frailty_df", num(fr_df)),
      kv("frailty_p", num(fr_p)),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)))
}, error = err_obj)

cat(sprintf("frailty            %s\n", if (is_err(models$frailty)) "ERROR" else "OK"))

# ── 10b. frailty_cluster_robust ───────────────────────────────────────────────
# What the app's frailty endpoint actually reports: the unpenalized partial
# likelihood with a cluster-robust sandwich. Kept as a separate key so the
# comparison is against what is being computed rather than against a different
# estimator that happens to share the panel's name.
models$frailty_cluster_robust <- tryCatch({
  fit <- coxph(Surv(time, status) ~ age + arm + cluster(site), data = d)
  s <- summary(fit)
  co <- s$coefficients
  terms <- vapply(rownames(co), function(nm) {
    obj(kv("term", str_(nm)),
        kv("estimate", num(co[nm, "coef"])),
        kv("se", num(co[nm, "robust se"])),
        kv("statistic", num(co[nm, "z"])),
        kv("p", num(co[nm, "Pr(>|z|)"])),
        kv("hr", num(co[nm, "exp(coef)"])))
  }, character(1), USE.NAMES = FALSE)
  obj(kv("terms", arr(terms)),
      kv("n", num(s$n)),
      kv("nevent", num(s$nevent)))
}, error = err_obj)

cat(sprintf("frailty_cluster_robust %s\n",
            if (is_err(models$frailty_cluster_robust)) "ERROR" else "OK"))

# ── 11. interval_censored (ic_par + ic_sp) ────────────────────────────────────
ic <- d
ic$L <- ic$ic_l
ic$U <- ifelse(is.na(ic$ic_r), Inf, ic$ic_r)

models$interval_censored <- tryCatch({
  fit <- icenReg::ic_par(cbind(L, U) ~ arm, data = ic, model = "ph", dist = "weibull")
  sp <- summary(fit)$summaryParameters
  i <- which(rownames(sp) == "armtreat")
  obj(kv("term", str_("armtreat")),
      kv("estimate", num(sp[i, "Estimate"])),
      kv("se", num(sp[i, "Std.Error"])),
      kv("statistic", num(sp[i, "z-value"])),
      kv("p", num(sp[i, "p"])),
      kv("hr", num(sp[i, "Exp(Est)"])),
      kv("dist", str_("weibull")))
}, error = err_obj)

cat(sprintf("interval_censored  %s\n", if (is_err(models$interval_censored)) "ERROR" else "OK"))

models$interval_censored_sp <- tryCatch({
  set.seed(1)  # ic_sp bootstraps for its standard error
  fit <- icenReg::ic_sp(cbind(L, U) ~ arm, data = ic, model = "ph", bs_samples = 1000)
  sp <- summary(fit)$summaryParameters
  i <- which(rownames(sp) == "armtreat")
  obj(kv("term", str_("armtreat")),
      kv("estimate", num(sp[i, "Estimate"])),
      kv("se", num(sp[i, "Std.Error"])),
      kv("statistic", num(sp[i, "z-value"])),
      kv("p", num(sp[i, "p"])),
      kv("hr", num(sp[i, "Exp(Est)"])),
      kv("baseline", str_("semi-parametric")))
}, error = err_obj)

cat(sprintf("interval_censored_sp %s\n", if (is_err(models$interval_censored_sp)) "ERROR" else "OK"))

# ── 12. multistate ────────────────────────────────────────────────────────────
models$multistate <- tryCatch({
  trans_key <- function(a, b) sprintf("%d_to_%d", a, b)
  out <- list()
  for (tr in list(c(1, 2), c(1, 3), c(2, 3))) {
    a <- tr[1]; b <- tr[2]
    key <- trans_key(a, b)
    out[[key]] <- tryCatch(
      withCallingHandlers({
        sub <- dmulti[dmulti$from_state == a & dmulti$to_state == b, ]
        fit <- coxph(Surv(entry, exit, event) ~ age + arm, data = sub)
        s <- summary(fit); co <- s$coefficients; ci <- s$conf.int
        terms <- vapply(seq_len(nrow(co)), function(i) cox_term(co, ci, i), character(1))
        obj(kv("from", num(a)), kv("to", num(b)),
            kv("terms", arr(terms)),
            kv("n", num(s$n)), kv("nevent", num(s$nevent)))
      }, warning = function(w) {
        # A coxph that "ran out of iterations" / did not converge returns an
        # object whose coefficients are numerically meaningless; surface that
        # as a failure for this transition rather than emitting Inf HRs.
        stop(simpleError(conditionMessage(w), call = NULL))
      }),
      error = err_obj)
  }
  obj(vapply(names(out), function(n) kv(n, out[[n]]), character(1)))
}, error = err_obj)

cat(sprintf("multistate         %s\n", if (is_err(models$multistate)) "ERROR" else "OK"))

# ── 13. external_validation ───────────────────────────────────────────────────
models$external_validation <- tryCatch({
  # Harrell's C on the FROZEN linear predictor (reverse=TRUE: higher lp = higher risk)
  co <- survival::concordance(Surv(time, status) ~ pred_lp, data = dext, reverse = TRUE)
  c_stat <- co$concordance
  c_se   <- sqrt(co$var)

  # calibration slope: the single coefficient of pred_lp (1.0 = perfectly calibrated)
  cal <- coxph(Surv(time, status) ~ pred_lp, data = dext)
  sc <- summary(cal)$coefficients
  sc_ci <- summary(cal)$conf.int
  beta <- sc["pred_lp", "coef"]; rse <- sc["pred_lp", "se(coef)"]

  # null-model check: pred_lp held fixed via offset(), only its loglik is reported
  null_fit <- coxph(Surv(time, status) ~ offset(pred_lp), data = dext)
  null_ll <- as.numeric(null_fit$loglik[1])

  obj(kv("harrell_c", num(c_stat)),
      kv("harrell_c_se", num(c_se)),
      kv("calibration_slope", obj(
         kv("estimate", num(beta)),
         kv("se", num(rse)),
         kv("statistic", num(sc["pred_lp", "z"])),
         kv("p", num(sc["pred_lp", "Pr(>|z|)"])),
         kv("ci_low", num(beta - Z * rse)),
         kv("ci_high", num(beta + Z * rse)),
         kv("hr", num(sc_ci["pred_lp", "exp(coef)"])))),
      kv("null_offset_loglik", num(null_ll)),
      kv("n", num(cal$n)),
      kv("nevent", num(cal$nevent)))
}, error = err_obj)

cat(sprintf("external_validation %s\n", if (is_err(models$external_validation)) "ERROR" else "OK"))

# ── assemble + write ──────────────────────────────────────────────────────────
pkgs <- obj(vapply(
  c("survival", "cmprsk", "survRM2", "icenReg", "mstate"),
  function(p) kv(p, str_(if (requireNamespace(p, quietly = TRUE))
    as.character(packageVersion(p)) else "absent")), character(1)))
meta <- obj(kv("r_version", str_(R.version.string)), kv("packages", pkgs))
body <- obj(kv("meta", meta),
            kv("models", obj(vapply(names(models), function(n) kv(n, models[[n]]),
                                    character(1)))))

out_path <- file.path(here, "reference_advanced.json")
writeLines(body, out_path)
cat("wrote", out_path, "-", length(models), "models\n")
