# Methods, Results, interpretation and R-code prose.
#
# The R half of ustat_engine/text/ (common.py, numbers.py, ttest.py), plus the
# two interpretation sentences that the Python engine builds inline inside
# run_ttest. They are pulled out into named functions here because analyses/ is
# held to a no-formatting rule -- see test_r_engine_isolation.py. Same strings
# either way; only the seam moved.
#
# Every function whose output is prose ends in _text, which is what the lint
# keys on.
#
# Non-ASCII characters are written as \u escapes so the whole tree stays 7-bit:
# the em dash in the missing marker, the en dash between confidence limits, the
# "less than or equal" in the methods paragraph. See the note in errors.R.

USTAT_DASH <- "\u2014"

# numbers.py::format_p. Canonical p-value string: "<0.001" or an exact
# 3-decimal. A second copy that rounded to "0.000" would contradict the `p`
# field sitting beside it in the same payload.
ustat_format_p_text <- function(p, prefix = FALSE) {
  ok <- !is.null(p) && length(p) == 1L && !is.na(p) &&
    is.finite(suppressWarnings(as.numeric(p)))
  if (!ok) {
    return(USTAT_DASH)
  }
  n <- as.numeric(p)
  if (n < 0.001) {
    return(if (prefix) "p<0.001" else "<0.001")
  }
  body <- sprintf("%.3f", n)
  if (prefix) paste0("p=", body) else body
}

# common.py::_es_str
ustat_es_text <- function(es) {
  name <- gsub("_", " ", as.character(es$name), fixed = TRUE)
  out <- sprintf("%s = %.3f", name, as.numeric(es$value))
  if (!is.null(es$ci_low) && !is.null(es$ci_high)) {
    out <- paste0(out, sprintf(
      " (95%% CI: %.3f\u2013%.3f)", as.numeric(es$ci_low), as.numeric(es$ci_high)
    ))
  }
  mag <- if (is.null(es$magnitude)) "" else as.character(es$magnitude)
  if (nzchar(mag)) {
    out <- paste0(out, sprintf(" [%s]", mag))
  }
  out
}

# common.py::_df_str -- a whole df prints bare, a Welch-Satterthwaite df keeps
# two decimals.
ustat_df_text <- function(df) {
  if (!is.numeric(df)) {
    return(as.character(df))
  }
  f <- as.numeric(df)
  if (is.na(f)) {
    return("")
  }
  if (f == floor(f)) as.character(as.integer(f)) else sprintf("%.2f", f)
}

# Python's str() applied to the number as JSON would have delivered it.
#
# This exists for one field: `mu`, which is echoed into three sentences. JSON
# has one number type; Python's json module splits it, so `"mu": 140` arrives as
# the int 140 and prints "140" while `"mu": 140.5` arrives as a float and prints
# "140.5". R cannot see that distinction, so it is reconstructed from the value:
# no fractional part means it came in as an int.
#
# `as_float = TRUE` covers the one case where Python is holding a float that
# looks whole -- an absent mu, whose default in the request shim is the literal
# 0.0, printed as "0.0".
#
# Residual divergence, stated: R's as.character() on a double emits up to 15
# significant digits where Python's repr emits the shortest round-trip form, so
# a mu of 1/3 would print "0.333333333333333" here and "0.3333333333333333"
# there. It affects prose only, and only for a hypothesised value nobody types.
ustat_py_num_text <- function(x, as_float = FALSE) {
  if (is.null(x) || length(x) != 1L || is.na(x)) {
    return("None")
  }
  v <- as.numeric(x)
  if (!is.finite(v)) {
    return(if (is.nan(v)) "nan" else if (v > 0) "inf" else "-inf")
  }
  if (v == floor(v) && abs(v) < 1e15) {
    whole <- sprintf("%.0f", v)
    return(if (as_float) paste0(whole, ".0") else whole)
  }
  as.character(v)
}

# ---------------------------------------------------------------------------
# text/ttest.py
# ---------------------------------------------------------------------------

ustat_methods_ttest_ind_text <- function(col, group_col, welch = FALSE) {
  variant <- if (welch) "Welch's" else "Student's"
  paste0(
    "Group differences in ", col, " were compared between levels of ",
    group_col, " using an independent-samples ", variant, " t-test. ",
    "Effect size was quantified with Hedges' g and its 95% confidence interval. ",
    "Normality of each group was assessed with the Shapiro-Wilk test (n < 50), ",
    "the Lilliefors-corrected Kolmogorov-Smirnov test ",
    "(50 \u2264 n \u2264 2000), ",
    "or a skewness/CLT criterion (n > 2000). ",
    "Homogeneity of variances was checked with Levene's test."
  )
}

ustat_methods_ttest_one_text <- function(col, mu_text) {
  paste0(
    "A one-sample t-test was used to compare the mean of ", col,
    " against the hypothesized value of ", mu_text, ". ",
    "Effect size was quantified with Cohen's d."
  )
}

ustat_results_ttest_ind_text <- function(result) {
  g1 <- if (is.null(result$group1)) "Group 1" else result$group1
  g2 <- if (is.null(result$group2)) "Group 2" else result$group2
  t <- if (is.null(result$t)) 0 else result$t
  p <- if (is.null(result$p)) 1 else result$p
  df <- ustat_df_text(if (is.null(result$df)) "" else result$df)
  m1 <- if (is.null(result$mean1)) 0 else result$mean1
  m2 <- if (is.null(result$mean2)) 0 else result$mean2
  sig <- isTRUE(result$significant)
  es_list <- result$effect_sizes
  es_text <- if (length(es_list) > 0L) paste0(", ", ustat_es_text(es_list[[1]])) else ""

  paste0(
    "The ", g1, " group (M = ", sprintf("%.2f", m1), ") ",
    if (sig) "significantly differed from" else "did not significantly differ from",
    " the ", g2, " group (M = ", sprintf("%.2f", m2), "), t(", df, ") = ",
    sprintf("%.3f", t), ", p = ", ustat_format_p_text(p), es_text, "."
  )
}

ustat_results_ttest_one_text <- function(result) {
  mu_text <- if (is.null(result$mu_text)) ustat_py_num_text(result$mu) else result$mu_text
  mean <- if (is.null(result$mean)) 0 else result$mean
  t <- if (is.null(result$t)) 0 else result$t
  p <- if (is.null(result$p)) 1 else result$p
  df <- if (is.null(result$df)) "" else as.character(result$df)
  sig <- isTRUE(result$significant)
  es_list <- result$effect_sizes
  es_text <- if (length(es_list) > 0L) paste0(", ", ustat_es_text(es_list[[1]])) else ""

  paste0(
    "The sample mean (M = ", sprintf("%.2f", mean), ") ",
    if (sig) "was significantly different from" else "did not significantly differ from",
    " the test value of ", mu_text, ", t(", df, ") = ", sprintf("%.3f", t),
    ", p = ", ustat_format_p_text(p), es_text, "."
  )
}

# var.equal = TRUE is Student's pooled-variance test; Welch needs FALSE. The
# snippet is the reproducibility claim, so it is generated from the same
# use_welch the test actually took rather than reconstructed by whoever renders
# the result.
ustat_r_ttest_ind_text <- function(col, group_col, welch = FALSE) {
  paste0(
    "t.test(", col, " ~ ", group_col, ", data = data, var.equal = ",
    if (welch) "FALSE" else "TRUE", ")"
  )
}

ustat_r_ttest_one_text <- function(col, mu_text) {
  paste0("t.test(data$", col, ", mu = ", mu_text, ")")
}

# ---------------------------------------------------------------------------
# The two interpretation sentences, built inline in stats/ttest.py
# ---------------------------------------------------------------------------

# ttest.py spells the p in these two sentences with 4 decimals, not through
# format_p's 3. Transcribed as it is, not as it arguably should be.
ustat_interp_p_text <- function(p) {
  if (p < 0.001) "<0.001" else sprintf("%.4f", p)
}

ustat_interpretation_ttest_ind_text <- function(sig, stat, p, es) {
  paste0(
    if (sig) "Significant" else "No significant",
    " difference between groups (t = ", sprintf("%.3f", stat),
    ", p = ", ustat_interp_p_text(p),
    ", Hedges' g = ", sprintf("%.3f", as.numeric(es$value)),
    " [", as.character(es$magnitude), "])"
  )
}

ustat_interpretation_ttest_one_text <- function(sig, stat, p, es, mu_text) {
  paste0(
    "Mean ", if (sig) "differs from" else "does not differ from", " ", mu_text,
    " (t = ", sprintf("%.3f", stat),
    ", p = ", ustat_interp_p_text(p),
    ", Cohen's d = ", sprintf("%.3f", as.numeric(es$value)),
    " [", as.character(es$magnitude), "])"
  )
}
