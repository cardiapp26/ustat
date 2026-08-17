# Effect sizes, magnitude labels, descriptives, and the two assumption checks.
#
# The R half of ustat_engine/stats/effect_sizes.py and stats/assumptions.py.
# Everything a t-test reports beside t and p lives here so analyses/ttest.R can
# stay what it is meant to be: a mapping layer.
#
# WHAT IS A CALL AND WHAT IS A TRANSCRIPTION, AND WHY
# ---------------------------------------------------
# The house rule is that a statistic has one implementation and this engine maps
# onto R's own. Where R ships the statistic, that is exactly what happens:
#
#   Levene (Brown-Forsythe)  -> stats::oneway.test on |x - group median|
#   Shapiro-Wilk             -> stats::shapiro.test
#   Lilliefors               -> nortest::lillie.test
#   skewness (g1)            -> moments::skewness
#   t quantile               -> stats::qt
#   sd / median / quantile   -> stats::*
#
# Two things are transcribed from the Python source instead, and the reason is
# the same in both cases: R has no function for them, and the packages that do
# (effsize, psych) each define the small-sample correction slightly differently,
# so calling one would introduce a THIRD number for a quantity that already has
# an authoritative one. Those two are:
#
#   ustat_cohen_d / ustat_cohen_d_one_sample -- Hedges' g, its J correction and
#       the Hedges & Olkin large-sample standard error. Copied expression for
#       expression from effect_sizes.py, comment about the CI width bug
#       included, and pinned by fixtures on both sides.
#   ustat_group_summary -- descriptives, which are only "a statistic" in the
#       loosest sense, but whose rounding has to match to the fourth decimal.
#
# ROUNDING. round() appears here because the Python engine rounds: cohen_d and
# group_summary both return round(x, 4), so those rounded values ARE the
# contract, not a formatting choice. Both languages round half to even at the
# same width, and the audit dataset's twenty-odd rounded fields agree exactly.
# analyses/ files are held to a stricter rule -- no rounding at all -- because
# nothing there has that excuse.

# ---------------------------------------------------------------------------
# Magnitude labels (effect_sizes.py::_es_magnitude)
# ---------------------------------------------------------------------------

# The shared vocabulary for every effect size in the app, ported whole even
# though the t-test reaches only two of its branches. Splitting it -- half here,
# half there -- would be exactly the drift this transcription exists to avoid.
ustat_es_magnitude <- function(name, val) {
  v <- abs(val)
  if (!is.finite(v)) {
    return("")
  }
  cuts <- NULL
  if (name %in% c("cohen_d", "hedges_g")) {
    cuts <- c(0.2, 0.5, 0.8)
  } else if (identical(name, "cohen_f")) {
    cuts <- c(0.10, 0.25, 0.40)
  } else if (name %in% c("r", "pearson_r", "point_biserial_r")) {
    cuts <- c(0.10, 0.30, 0.50)
  } else if (name %in% c("eta_squared", "eta2")) {
    cuts <- c(0.01, 0.06, 0.14)
  } else if (name %in% c("cramers_v", "cramer_v")) {
    cuts <- c(0.10, 0.30, 0.50)
  } else if (identical(name, "odds_ratio")) {
    cuts <- c(1.5, 2.5, 4.0)
  } else if (identical(name, "rank_biserial_r")) {
    cuts <- c(0.10, 0.30, 0.50)
  }
  if (is.null(cuts)) {
    return("")
  }
  if (v < cuts[1]) {
    return("negligible")
  }
  if (v < cuts[2]) {
    return("small")
  }
  if (v < cuts[3]) {
    return("medium")
  }
  "large"
}

# ---------------------------------------------------------------------------
# Effect sizes and descriptives
# ---------------------------------------------------------------------------

ustat_cohen_d <- function(g1, g2) {
  n1 <- length(g1)
  n2 <- length(g2)
  m1 <- mean(g1)
  m2 <- mean(g2)
  s1 <- stats::sd(g1)
  s2 <- stats::sd(g2)
  s_pooled <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
  if (!is.na(s_pooled) && s_pooled == 0) {
    return(list(
      name = "cohen_d", value = 0, ci_low = 0, ci_high = 0,
      magnitude = "negligible"
    ))
  }
  d <- (m1 - m2) / s_pooled
  # Hedges' correction for small samples.
  j <- 1 - 3 / (4 * (n1 + n2 - 2) - 1)
  g <- d * j
  # Large-sample standard error of a two-sample d (Hedges & Olkin):
  #     sqrt((n1 + n2) / (n1 * n2) + g^2 / (2 * (n1 + n2)))
  # The Python source records that this used to be written in a form whose
  # second term was missing its (n1 + n2) divisor, giving intervals 5.5x too
  # wide at n = 100 per arm. Transcribed in the corrected form.
  se <- sqrt((n1 + n2) / (n1 * n2) + g^2 / (2 * (n1 + n2)))
  # t rather than 1.96 -- with small groups the normal quantile is too short.
  crit <- stats::qt(0.975, max(n1 + n2 - 2, 1))
  list(
    name = "hedges_g",
    value = round(g, 4),
    ci_low = round(g - crit * se, 4),
    ci_high = round(g + crit * se, 4),
    magnitude = ustat_es_magnitude("hedges_g", g)
  )
}

ustat_cohen_d_one_sample <- function(x, mu) {
  n <- length(x)
  s <- stats::sd(x)
  d <- if (!is.na(s) && s > 0) (mean(x) - mu) / s else 0
  se <- sqrt(1 / n + d^2 / (2 * n))
  crit <- stats::qt(0.975, max(n - 1, 1))
  list(
    name = "cohen_d",
    value = round(d, 4),
    ci_low = round(d - crit * se, 4),
    ci_high = round(d + crit * se, 4),
    magnitude = ustat_es_magnitude("cohen_d", d)
  )
}

# numpy's default percentile interpolation is "linear", which is R's quantile
# type 7 -- also R's default, stated here rather than relied on.
ustat_group_summary <- function(x, label = "Sample") {
  list(
    label = as.character(label),
    n = length(x),
    mean = round(mean(x), 4),
    sd = round(stats::sd(x), 4),
    median = round(stats::median(x), 4),
    q1 = round(unname(stats::quantile(x, 0.25, type = 7)), 4),
    q3 = round(unname(stats::quantile(x, 0.75, type = 7)), 4),
    min = round(min(x), 4),
    max = round(max(x), 4)
  )
}

# ---------------------------------------------------------------------------
# Assumption checks (assumptions.py)
# ---------------------------------------------------------------------------

# Tier 1: n < 50        -> Shapiro-Wilk (most powerful for small samples)
# Tier 2: 50 <= n <= 2000 -> Kolmogorov-Smirnov with Lilliefors correction
# Tier 3: n > 2000      -> CLT skewness bypass (|skew| <= 1.5), else Lilliefors
#
# THE LILLIEFORS P DIVERGES FROM PYTHON'S, ON PURPOSE AND UNAVOIDABLY. The
# statistic is the same to 1e-15 -- it is just sup|F_n - Phi| with mu and sigma
# estimated -- but its null distribution has no closed form, so every
# implementation ships an approximation of it. nortest::lillie.test uses the
# Dallal-Wilkinson (1986) analytic form with Stephens' (1974) modification;
# statsmodels interpolates a table of simulated critical values. On the audit
# dataset that is p = 0.5265 here against 0.6002 there for the same 0.048315
# statistic. This feeds only the assumption LINE next to the result -- which
# test ran is decided by Levene, never by normality -- and both sides agree on
# `met` unless a p sits across 0.05. It is recorded in qa/parity/ttest.json as a
# divergence rather than papered over; papering over it would mean hand-rolling
# a third approximation to make two disagreeing ones look like one.
ustat_check_normality <- function(x, label = "Sample") {
  name <- paste0("Normality (", label, ")")
  n <- length(x)
  if (n < 3L) {
    return(list(name = name, met = TRUE, detail = "Too few obs to test"))
  }
  if (stats::sd(x) == 0) {
    return(list(name = name, met = TRUE, detail = "Constant values (no variation)"))
  }

  if (n < 50L) {
    p <- stats::shapiro.test(x)$p.value
    if (is.na(p)) {
      return(list(name = name, met = TRUE, detail = "Test inconclusive"))
    }
    test_name <- "Shapiro-Wilk"
  } else if (n <= 2000L) {
    p <- nortest::lillie.test(x)$p.value
    test_name <- "Kolmogorov-Smirnov (Lilliefors)"
  } else {
    skew <- moments::skewness(x)
    if (abs(skew) <= 1.5) {
      return(list(
        name = name,
        met = TRUE,
        detail = sprintf("CLT bypass (n=%d, |skewness|=%.2f \u2264 1.5)", n, abs(skew))
      ))
    }
    p <- nortest::lillie.test(x)$p.value
    test_name <- "Kolmogorov-Smirnov (Lilliefors)"
  }

  list(
    name = name,
    met = isTRUE(p >= 0.05),
    detail = sprintf("%s: p = %.4f", test_name, p)
  )
}

# Levene's test for homogeneity of variances, median-centred -- the
# Brown-Forsythe form, which is scipy.stats.levene's default and therefore what
# the Python engine computes. Median-centring makes it a one-way ANOVA on
# |x - median(group)|, and R has that: stats::oneway.test(var.equal = TRUE) is
# the classic equal-variance F. Nothing is computed here that R does not.
#
# This one is load-bearing rather than decorative: when the caller asks for
# "auto" it is this `met` that decides Student versus Welch, so a Levene that
# differed between the two engines would not print a different sentence, it
# would run a different test and report a different p.
#
# `on_violation` states what the CALLER does when the assumption fails, because
# reporting a correction that was never applied is a false methods claim.
ustat_check_equal_variances <- function(groups, names, on_violation = "") {
  if (length(groups) < 2L) {
    return(list(name = "Equal variances", met = TRUE, detail = "Single group"))
  }
  spread <- unlist(
    lapply(groups, function(g) abs(g - stats::median(g))),
    use.names = FALSE
  )
  membership <- factor(rep(
    seq_along(groups),
    vapply(groups, length, integer(1))
  ))
  fit <- stats::oneway.test(spread ~ membership, var.equal = TRUE)
  stat <- unname(fit$statistic)
  p <- unname(fit$p.value)
  violated <- isTRUE(p < 0.05)
  suffix <- if (violated && nzchar(on_violation)) {
    paste0(" \u2014 violated, ", on_violation)
  } else if (violated) {
    " \u2014 violated"
  } else {
    ""
  }
  list(
    name = "Equal variances (Levene)",
    met = !violated,
    detail = paste0(sprintf("F = %.3f, p = %.4f", stat, p), suffix)
  )
}
