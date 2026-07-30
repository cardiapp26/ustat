#!/usr/bin/env Rscript
# Independent reference values for the uSTAT Models endpoints.
#
# Base R + survival / MASS / lme4 / logistf / ordinal only. Emits JSON by
# hand at full precision so nothing is lost to rounding on the way out.
#
#   Rscript qa/models_audit/reference.R

suppressPackageStartupMessages({
  library(survival)
  library(MASS)
})

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
if (length(here) == 0 || here == "") here <- "qa/models_audit"

d <- read.csv(file.path(here, "dataset.csv"), stringsAsFactors = FALSE)
long <- read.csv(file.path(here, "dataset_long.csv"), stringsAsFactors = FALSE)

d$sex <- factor(d$sex)                      # reference F
d$arm <- factor(d$arm)                      # reference control
d$stage <- factor(d$stage)                  # reference I
long$arm <- factor(long$arm)
long$visit <- factor(long$visit)

# ── JSON emission (no jsonlite dependency) ───────────────────────────────────
num <- function(x) {
  if (is.null(x) || length(x) == 0) return("null")
  if (is.na(x)) return("null")
  if (is.infinite(x)) return("null")
  sprintf("%.17g", as.numeric(x))
}
str_ <- function(x) paste0('"', gsub('"', '\\\\"', as.character(x)), '"')
kv <- function(k, v) paste0(str_(k), ": ", v)
obj <- function(...) paste0("{", paste(c(...), collapse = ", "), "}")
arr <- function(v) paste0("[", paste(v, collapse = ", "), "]")

term_obj <- function(name, est, se, stat, p, extra = NULL) {
  parts <- c(kv("term", str_(name)), kv("estimate", num(est)), kv("se", num(se)),
             kv("statistic", num(stat)), kv("p", num(p)))
  if (!is.null(extra)) for (nm in names(extra)) parts <- c(parts, kv(nm, num(extra[[nm]])))
  obj(parts)
}

models <- list()

# ── 1. linear ────────────────────────────────────────────────────────────────
fit <- lm(sbp ~ age + bmi + arm + sex, data = d)
s <- summary(fit)
co <- s$coefficients
terms <- vapply(seq_len(nrow(co)), function(i)
  term_obj(rownames(co)[i], co[i, 1], co[i, 2], co[i, 3], co[i, 4]), character(1))
models$linear <- obj(
  kv("reference_levels", obj(kv("arm", str_("control")), kv("sex", str_("F")))),
  kv("terms", arr(terms)),
  kv("r_squared", num(s$r.squared)), kv("adj_r_squared", num(s$adj.r.squared)),
  kv("f_statistic", num(s$fstatistic[["value"]])),
  kv("f_numdf", num(s$fstatistic[["numdf"]])), kv("f_dendf", num(s$fstatistic[["dendf"]])),
  kv("p", num(pf(s$fstatistic[["value"]], s$fstatistic[["numdf"]],
                 s$fstatistic[["dendf"]], lower.tail = FALSE))),
  kv("sigma", num(s$sigma)), kv("aic", num(AIC(fit))), kv("bic", num(BIC(fit))),
  kv("n", num(nobs(fit))),
  kv("n_params_aic", num(attr(logLik(fit), "df"))),
  kv("note", str_("AIC/BIC count the residual variance as a parameter (k = p + 1)")))

# ── 2. polynomial (raw, degree 2) ────────────────────────────────────────────
fitp <- lm(sbp ~ age + I(age^2) + arm, data = d)
cp <- summary(fitp)$coefficients
models$polynomial <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cp)), function(i)
    term_obj(rownames(cp)[i], cp[i, 1], cp[i, 2], cp[i, 3], cp[i, 4]), character(1)))),
  kv("r_squared", num(summary(fitp)$r.squared)), kv("n", num(nobs(fitp))))

fitp0 <- lm(sbp ~ age + I(age^2), data = d)
cp0 <- summary(fitp0)$coefficients
models$polynomial_numeric_only <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cp0)), function(i)
    term_obj(rownames(cp0)[i], cp0[i, 1], cp0[i, 2], cp0[i, 3], cp0[i, 4]), character(1)))),
  kv("r_squared", num(summary(fitp0)$r.squared)), kv("n", num(nobs(fitp0))))

# ── 3. stepwise (AIC, both directions) ───────────────────────────────────────
full <- lm(sbp ~ age + bmi + arm + sex + biomarker, data = d)
step_fit <- MASS::stepAIC(full, direction = "both", trace = FALSE)
models$stepwise <- obj(
  kv("formula", str_(paste(deparse(formula(step_fit)), collapse = ""))),
  kv("selected", arr(vapply(attr(terms(step_fit), "term.labels"), str_, character(1)))),
  kv("aic", num(AIC(step_fit))),
  kv("aic_stepwise_scale", num(extractAIC(step_fit)[2])))

# ── 4. logistic ──────────────────────────────────────────────────────────────
fitg <- glm(event_binary ~ age + bmi + arm + sex, family = binomial, data = d)
cg <- summary(fitg)$coefficients
terms <- vapply(seq_len(nrow(cg)), function(i)
  term_obj(rownames(cg)[i], cg[i, 1], cg[i, 2], cg[i, 3], cg[i, 4],
           extra = list(odds_ratio = exp(cg[i, 1]),
                        or_ci_low = exp(cg[i, 1] - 1.959963984540054 * cg[i, 2]),
                        or_ci_high = exp(cg[i, 1] + 1.959963984540054 * cg[i, 2]))),
  character(1))
models$logistic <- obj(
  kv("terms", arr(terms)),
  kv("null_deviance", num(fitg$null.deviance)), kv("deviance", num(fitg$deviance)),
  kv("aic", num(AIC(fitg))), kv("bic", num(BIC(fitg))),
  kv("log_likelihood", num(as.numeric(logLik(fitg)))),
  kv("n", num(nobs(fitg))))

# ── 5. Firth ─────────────────────────────────────────────────────────────────
if (requireNamespace("logistf", quietly = TRUE)) {
  ff <- logistf::logistf(event_binary ~ age + arm, data = d)
  se_f <- sqrt(diag(vcov(ff)))
  terms <- vapply(seq_along(ff$coefficients), function(i)
    term_obj(names(ff$coefficients)[i], ff$coefficients[i], se_f[i], NA, ff$prob[i],
             extra = list(ci_low = ff$ci.lower[i], ci_high = ff$ci.upper[i],
                          odds_ratio = exp(ff$coefficients[i]))),
    character(1))
  models$firth_logistic <- obj(
    kv("terms", arr(terms)), kv("n", num(ff$n)),
    kv("note", str_(paste("logistf p-values and CIs come from the penalised",
                          "PROFILE likelihood, not from a Wald statistic"))))
}

# ── 6. Poisson (with and without an exposure offset) ─────────────────────────
fpo <- glm(admissions ~ age + arm, family = poisson, data = d)
cpo <- summary(fpo)$coefficients
models$poisson <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cpo)), function(i)
    term_obj(rownames(cpo)[i], cpo[i, 1], cpo[i, 2], cpo[i, 3], cpo[i, 4],
             extra = list(rate_ratio = exp(cpo[i, 1]),
                          ci_low = exp(cpo[i, 1] - 1.959963984540054 * cpo[i, 2]),
                          ci_high = exp(cpo[i, 1] + 1.959963984540054 * cpo[i, 2]))),
    character(1)))),
  kv("deviance", num(fpo$deviance)), kv("aic", num(AIC(fpo))), kv("n", num(nobs(fpo))),
  kv("pearson_chisq", num(sum(residuals(fpo, type = "pearson")^2))),
  kv("df_residual", num(fpo$df.residual)))

fpoff <- glm(admissions ~ age + arm + offset(log(followup_years)),
             family = poisson, data = d)
cpf <- summary(fpoff)$coefficients
models$poisson_offset <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cpf)), function(i)
    term_obj(rownames(cpf)[i], cpf[i, 1], cpf[i, 2], cpf[i, 3], cpf[i, 4]),
    character(1)))),
  kv("aic", num(AIC(fpoff))))

# ── 7. Gamma (log link) ──────────────────────────────────────────────────────
fga <- glm(cost ~ age + arm, family = Gamma(link = "log"), data = d)
sga <- summary(fga)
cga <- sga$coefficients
models$gamma <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cga)), function(i)
    term_obj(rownames(cga)[i], cga[i, 1], cga[i, 2], cga[i, 3], cga[i, 4]),
    character(1)))),
  kv("dispersion", num(sga$dispersion)), kv("deviance", num(fga$deviance)),
  kv("df_residual", num(fga$df.residual)),
  kv("aic", num(AIC(fga))), kv("n", num(nobs(fga))),
  kv("note", str_(paste("the dispersion is estimated, so each coefficient is",
                        "tested with a t on df.residual, not a z"))))

# ── 8. Negative binomial ─────────────────────────────────────────────────────
fnb <- MASS::glm.nb(visits ~ age + arm, data = d)
cnb <- summary(fnb)$coefficients
models$negbinom <- obj(
  kv("terms", arr(vapply(seq_len(nrow(cnb)), function(i)
    term_obj(rownames(cnb)[i], cnb[i, 1], cnb[i, 2], cnb[i, 3], cnb[i, 4]),
    character(1)))),
  kv("theta", num(fnb$theta)), kv("se_theta", num(fnb$SE.theta)),
  kv("alpha", num(1 / fnb$theta)),
  kv("deviance", num(fnb$deviance)), kv("aic", num(AIC(fnb))), kv("n", num(nobs(fnb))),
  kv("note", str_("theta is estimated by maximum likelihood jointly with the betas")))

# ── 9. Ordinal (polr, and clm if available) ──────────────────────────────────
d$grade_f <- factor(d$grade, levels = c("mild", "moderate", "severe"), ordered = TRUE)
fpo2 <- MASS::polr(grade_f ~ age + arm, data = d, Hess = TRUE, method = "logistic")
cpo2 <- summary(fpo2)$coefficients
n_coef <- length(fpo2$coefficients)
terms <- vapply(seq_len(n_coef), function(i)
  term_obj(rownames(cpo2)[i], cpo2[i, 1], cpo2[i, 2], cpo2[i, 3],
           2 * pnorm(-abs(cpo2[i, 3])), extra = list(odds_ratio = exp(cpo2[i, 1]))),
  character(1))
thr <- vapply(seq_len(nrow(cpo2) - n_coef), function(j) {
  i <- n_coef + j
  term_obj(rownames(cpo2)[i], cpo2[i, 1], cpo2[i, 2], cpo2[i, 3],
           2 * pnorm(-abs(cpo2[i, 3])))
}, character(1))
models$ordinal_polr <- obj(
  kv("terms", arr(terms)), kv("thresholds", arr(thr)),
  kv("aic", num(AIC(fpo2))), kv("n", num(nobs(fpo2))),
  kv("note", str_("polr reports no p-value; computed here as 2*pnorm(-|t|)")))

if (requireNamespace("ordinal", quietly = TRUE)) {
  fclm <- ordinal::clm(grade_f ~ age + arm, data = d)
  scl <- summary(fclm)$coefficients
  models$ordinal_clm <- obj(
    kv("terms", arr(vapply(seq_len(nrow(scl)), function(i)
      term_obj(rownames(scl)[i], scl[i, 1], scl[i, 2], scl[i, 3], scl[i, 4]),
      character(1)))),
    kv("aic", num(AIC(fclm))))
}

# ── 10. Cox ──────────────────────────────────────────────────────────────────
fcx <- coxph(Surv(time, status) ~ age + arm + sex, data = d)
scx <- summary(fcx)
ccx <- scx$coefficients
cix <- scx$conf.int
terms <- vapply(seq_len(nrow(ccx)), function(i)
  term_obj(rownames(ccx)[i], ccx[i, "coef"], ccx[i, "se(coef)"],
           ccx[i, "z"], ccx[i, "Pr(>|z|)"],
           extra = list(hr = ccx[i, "exp(coef)"],
                        hr_ci_low = cix[i, "lower .95"],
                        hr_ci_high = cix[i, "upper .95"])),
  character(1))
zph <- cox.zph(fcx)
zrows <- vapply(seq_len(nrow(zph$table)), function(i)
  obj(kv("term", str_(rownames(zph$table)[i])),
      kv("chisq", num(zph$table[i, "chisq"])),
      kv("df", num(zph$table[i, "df"])),
      kv("p", num(zph$table[i, "p"]))), character(1))
models$cox <- obj(
  kv("terms", arr(terms)),
  kv("concordance", num(scx$concordance[["C"]])),
  kv("concordance_se", num(scx$concordance[["se(C)"]])),
  kv("log_likelihood", num(as.numeric(logLik(fcx)))),
  kv("n", num(scx$n)), kv("n_events", num(scx$nevent)),
  kv("lrt", num(scx$logtest[["test"]])), kv("lrt_df", num(scx$logtest[["df"]])),
  kv("lrt_p", num(scx$logtest[["pvalue"]])),
  kv("wald", num(scx$waldtest[["test"]])), kv("wald_p", num(scx$waldtest[["pvalue"]])),
  kv("score", num(scx$sctest[["test"]])), kv("score_p", num(scx$sctest[["pvalue"]])),
  kv("zph", arr(zrows)))

# ── 11. Kaplan-Meier ─────────────────────────────────────────────────────────
km <- survfit(Surv(time, status) ~ arm, data = d)
tb <- summary(km)$table
strata_rows <- vapply(seq_len(nrow(tb)), function(i)
  obj(kv("stratum", str_(rownames(tb)[i])),
      kv("n", num(tb[i, "records"])), kv("events", num(tb[i, "events"])),
      kv("median", num(tb[i, "median"])),
      kv("median_ci_low", num(tb[i, "0.95LCL"])),
      kv("median_ci_high", num(tb[i, "0.95UCL"]))), character(1))
st <- summary(km, times = c(5, 10))
at_times <- vapply(seq_along(st$time), function(i)
  obj(kv("stratum", str_(as.character(st$strata[i]))),
      kv("time", num(st$time[i])),
      kv("n_risk", num(st$n.risk[i])),
      kv("surv", num(st$surv[i])),
      kv("ci_low", num(st$lower[i])), kv("ci_high", num(st$upper[i]))),
  character(1))
sd_ <- survdiff(Surv(time, status) ~ arm, data = d)
models$km <- obj(
  kv("strata", arr(strata_rows)), kv("at_times", arr(at_times)),
  kv("logrank_chisq", num(sd_$chisq)),
  kv("logrank_df", num(length(sd_$n) - 1)),
  kv("logrank_p", num(pchisq(sd_$chisq, length(sd_$n) - 1, lower.tail = FALSE))))

# ── 12. Linear mixed model ───────────────────────────────────────────────────
if (requireNamespace("lme4", quietly = TRUE)) {
  suppressPackageStartupMessages(library(lme4))
  flme <- lmer(score ~ visit + arm + age + (1 | pid), data = long, REML = TRUE)
  cl <- summary(flme)$coefficients
  vc <- as.data.frame(VarCorr(flme))
  models$lmm <- obj(
    kv("terms", arr(vapply(seq_len(nrow(cl)), function(i)
      term_obj(rownames(cl)[i], cl[i, 1], cl[i, 2], cl[i, 3], NA), character(1)))),
    kv("sd_pid", num(vc$sdcor[vc$grp == "pid"])),
    kv("sd_residual", num(vc$sdcor[vc$grp == "Residual"])),
    kv("var_pid", num(vc$vcov[vc$grp == "pid"])),
    kv("var_residual", num(vc$vcov[vc$grp == "Residual"])),
    kv("n", num(nobs(flme))), kv("n_groups", num(ngrps(flme)[["pid"]])),
    kv("note", str_("lme4 reports no p-values for fixed effects")))
}

# ── 13. GEE (geepack::geeglm) ────────────────────────────────────────────────
if (requireNamespace("geepack", quietly = TRUE)) {
  suppressPackageStartupMessages(library(geepack))
  lg <- long[order(long$pid), ]
  for (cs in c("independence", "exchangeable", "ar1")) {
    fg <- geepack::geeglm(score ~ visit + arm + age, id = pid, data = lg,
                          family = gaussian, corstr = cs)
    cgee <- summary(fg)$coefficients
    key <- paste0("gee_", cs)
    models[[key]] <- obj(
      kv("terms", arr(vapply(seq_len(nrow(cgee)), function(i)
        term_obj(rownames(cgee)[i], cgee[i, 1], cgee[i, 2], cgee[i, 3], cgee[i, 4]),
        character(1)))),
      kv("corstr", str_(cs)),
      kv("n", num(nrow(lg))),
      kv("n_clusters", num(length(unique(lg$pid)))),
      kv("note", str_("geeglm reports sandwich (robust) standard errors and a Wald chi-square")))
  }
}

# ── 14. Propensity score matching (MatchIt) ──────────────────────────────────
if (requireNamespace("MatchIt", quietly = TRUE)) {
  suppressPackageStartupMessages(library(MatchIt))
  d$treat01 <- as.integer(d$arm == "treat")
  mi <- MatchIt::matchit(treat01 ~ age + bmi + sex, data = d,
                         method = "nearest", distance = "glm",
                         caliper = 0.2, ratio = 1, replace = FALSE)
  sm <- summary(mi, standardize = TRUE)
  bal_rows <- vapply(seq_len(nrow(sm$sum.matched)), function(i)
    obj(kv("covariate", str_(rownames(sm$sum.matched)[i])),
        kv("smd_matched", num(sm$sum.matched[i, "Std. Mean Diff."])),
        kv("smd_all", num(sm$sum.all[i, "Std. Mean Diff."]))), character(1))
  md <- MatchIt::match.data(mi)
  models$psm <- obj(
    kv("n_treated_all", num(sum(d$treat01 == 1))),
    kv("n_control_all", num(sum(d$treat01 == 0))),
    kv("n_matched_pairs", num(sum(md$treat01 == 1))),
    kv("n_matched_total", num(nrow(md))),
    kv("balance", arr(bal_rows)),
    kv("note", str_("nearest-neighbour on the logit propensity, caliper 0.2 SD, 1:1, no replacement")))
}

# ── 15. IPTW (survey) ────────────────────────────────────────────────────────
if (requireNamespace("survey", quietly = TRUE)) {
  suppressPackageStartupMessages(library(survey))
  d$treat01 <- as.integer(d$arm == "treat")
  ps_fit <- glm(treat01 ~ age + bmi + sex, family = binomial, data = d)
  ps <- fitted(ps_fit)
  # Stabilised ATE weights, the same estimand uSTAT defaults to.
  pt <- mean(d$treat01)
  w_stab <- ifelse(d$treat01 == 1, pt / ps, (1 - pt) / (1 - ps))
  w_raw <- ifelse(d$treat01 == 1, 1 / ps, 1 / (1 - ps))
  dw <- d
  dw$w <- w_stab
  des <- survey::svydesign(ids = ~1, weights = ~w, data = dw)
  fit_w <- survey::svyglm(event_binary ~ treat01, design = des,
                          family = quasibinomial())
  cw <- summary(fit_w)$coefficients
  models$iptw <- obj(
    kv("terms", arr(vapply(seq_len(nrow(cw)), function(i)
      term_obj(rownames(cw)[i], cw[i, 1], cw[i, 2], cw[i, 3], cw[i, 4]),
      character(1)))),
    kv("ps_mean", num(mean(ps))), kv("ps_min", num(min(ps))), kv("ps_max", num(max(ps))),
    kv("w_stab_mean", num(mean(w_stab))), kv("w_stab_max", num(max(w_stab))),
    kv("w_raw_mean", num(mean(w_raw))), kv("w_raw_max", num(max(w_raw))),
    kv("ess", num(sum(w_stab)^2 / sum(w_stab^2))),
    kv("n", num(nrow(d))),
    kv("note", str_("stabilised ATE weights; svyglm quasibinomial with design-based (robust) SEs")))
}

# ── 16. VIF (car) ────────────────────────────────────────────────────────────
if (requireNamespace("car", quietly = TRUE)) {
  suppressPackageStartupMessages(library(car))
  vl <- car::vif(lm(sbp ~ age + bmi + arm + sex, data = d))
  vg <- car::vif(glm(event_binary ~ age + bmi + arm + sex, family = binomial, data = d))
  vrow <- function(v) arr(vapply(names(v), function(n)
    obj(kv("term", str_(n)), kv("vif", num(v[[n]]))), character(1)))
  models$vif <- obj(
    kv("linear", vrow(vl)), kv("logistic", vrow(vg)),
    kv("note", str_(paste("car::vif runs the auxiliary regressions on the model's",
                          "design matrix, intercept included"))))
}

# ── 17. ROC (pROC) ───────────────────────────────────────────────────────────
if (requireNamespace("pROC", quietly = TRUE)) {
  suppressPackageStartupMessages(library(pROC))
  r1 <- pROC::roc(d$event_binary, d$age, quiet = TRUE)
  r2 <- pROC::roc(d$event_binary, d$bmi, quiet = TRUE)
  ci1 <- pROC::ci.auc(r1)
  co <- pROC::coords(r1, "best", best.method = "youden", transpose = FALSE)
  tt <- pROC::roc.test(r1, r2, method = "delong")
  models$roc <- obj(
    kv("auc_age", num(as.numeric(pROC::auc(r1)))),
    kv("auc_age_ci_low", num(ci1[1])), kv("auc_age_ci_high", num(ci1[3])),
    kv("auc_bmi", num(as.numeric(pROC::auc(r2)))),
    kv("youden_sensitivity", num(co$sensitivity)),
    kv("youden_specificity", num(co$specificity)),
    kv("delong_z", num(as.numeric(tt$statistic))),
    kv("delong_p", num(tt$p.value)),
    kv("note", str_("DeLong CI and DeLong two-sample test")))
}

# ── 18. Restricted cubic spline (rms) ────────────────────────────────────────
if (requireNamespace("rms", quietly = TRUE)) {
  suppressPackageStartupMessages(library(rms))
  dd <- rms::datadist(d); options(datadist = "dd")
  fr <- rms::lrm(event_binary ~ rcs(age, 4), data = d)
  ar <- anova(fr)
  models$rcs <- obj(
    kv("knots", arr(vapply(as.numeric(attr(rcs(d$age, 4), "parms")), num, character(1)))),
    kv("nonlinear_chisq", num(ar[grep("Nonlinear", rownames(ar))[1], "Chi-Square"])),
    kv("nonlinear_df", num(ar[grep("Nonlinear", rownames(ar))[1], "d.f."])),
    kv("nonlinear_p", num(ar[grep("Nonlinear", rownames(ar))[1], "P"])),
    kv("overall_chisq", num(ar["age", "Chi-Square"])),
    kv("overall_p", num(ar["age", "P"])),
    kv("note", str_("Harrell knot placement at the default quantiles")))
  options(datadist = NULL)
}

# ── write ────────────────────────────────────────────────────────────────────
pkgs <- obj(vapply(c("survival", "MASS", "lme4", "logistf", "ordinal"), function(p)
  kv(p, str_(if (requireNamespace(p, quietly = TRUE))
    as.character(packageVersion(p)) else "absent")), character(1)))
meta <- obj(kv("r_version", str_(R.version.string)), kv("packages", pkgs))
body <- obj(kv("meta", meta),
            kv("models", obj(vapply(names(models), function(n) kv(n, models[[n]]),
                                    character(1)))))
writeLines(body, file.path(here, "reference.json"))
cat("wrote", file.path(here, "reference.json"), "-", length(models), "models\n")
