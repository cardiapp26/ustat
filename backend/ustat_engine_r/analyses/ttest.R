# Independent-samples and one-sample t-test.
#
# The R half of ustat_engine/stats/ttest.py, and deliberately the thinnest file
# in the tree. Every number it returns comes out of stats::t.test, out of the
# helpers in runtime/stats.R, or out of the frame -- there is no arithmetic
# here. That is what "two engines, both thin" means: this file decides which
# test to run, on which rows, and what to call the answer; R decides what the
# answer is.
#
# WHAT MUST MATCH, AND WHAT DOES NOT
# -----------------------------------
# The KEYS of the returned list are a contract: HypothesisPanel destructures
# them with no optional chaining, so a missing one is a runtime crash rather
# than a type error, and the browser has to be able to show an R result in the
# panel a Python result came out of. test_r_ttest_local.py asserts set equality
# against run_ttest's own keys for the same params.
#
# The VALUES match to about 1e-15 for everything that comes from t.test, and
# exactly for the counts, the means and the rounded effect sizes. Two do not,
# both recorded in qa/parity/ttest.json:
#
#   - the Lilliefors p in the normality assumption (different approximation of
#     the same null distribution -- see runtime/stats.R);
#   - the wording around `mu` when a caller sends it as a JSON integer (see
#     ustat_py_num_text in runtime/text.R).
#
# PRECEDENCE for Student vs Welch is copied exactly: an explicit `method` wins,
# then the legacy `equal_var`, then Levene. Getting that order wrong would not
# show up as an error, it would silently run the other test.

USTAT_TTEST_METHODS <- c("auto", "student", "welch")

ustat_ttest_params <- function(params) {
  column <- params$column
  if (is.null(column) || length(column) != 1L || is.na(column) ||
      !nzchar(as.character(column))) {
    ustat_stop("Field 'column' is required.", 422L)
  }

  # AliasChoices("group_column", "group_col") -- either spelling arrives from a
  # caller that predates the rename.
  group_column <- params$group_column
  if (is.null(group_column)) {
    group_column <- params$group_col
  }
  if (!is.null(group_column) &&
      (is.na(group_column) || !nzchar(as.character(group_column)))) {
    group_column <- NULL
  }

  method <- params$method
  if (is.null(method) || is.na(method) || !nzchar(as.character(method))) {
    method <- "auto"
  }
  method <- as.character(method)
  if (!(method %in% USTAT_TTEST_METHODS)) {
    ustat_stop(
      "Field 'method' must be one of ('auto', 'student', 'welch').", 422L
    )
  }

  # mu defaults to the float 0.0 in the Python request shim, and NULL means "not
  # supplied" for equal_var -- it used to default to TRUE while the handler
  # ignored it entirely and always let Levene decide.
  mu_supplied <- !is.null(params$mu)
  list(
    column = as.character(column),
    group_column = if (is.null(group_column)) NULL else as.character(group_column),
    method = method,
    equal_var = params$equal_var,
    mu = if (mu_supplied) as.numeric(params$mu) else 0,
    mu_text = ustat_py_num_text(
      if (mu_supplied) as.numeric(params$mu) else 0,
      as_float = !mu_supplied
    )
  )
}

ustat_ttest_column <- function(frame, name) {
  if (!(name %in% names(frame))) {
    ustat_stop(paste0("Unknown column: '", name, "'"), 400L)
  }
  frame[[name]]
}

ustat_ttest_ind <- function(req, frame) {
  # _two_level_work: normalise the grouping column, coerce the value column,
  # then drop any row missing either. This is what fixes n1 and n2.
  cleaned <- ustat_clean_two_level(
    ustat_ttest_column(frame, req$group_column), name = req$group_column
  )
  value <- ustat_to_numeric(ustat_ttest_column(frame, req$column))
  grp <- cleaned$series
  keep <- !is.na(value) & !is.na(grp)
  value <- value[keep]
  grp <- grp[keep]

  groups <- ustat_sorted_groups(grp)
  if (length(groups) != 2L) {
    ustat_stop("Group column must have exactly 2 groups", 400L)
  }
  g1 <- value[grp == groups[1]]
  g2 <- value[grp == groups[2]]
  name1 <- as.character(groups[1])
  name2 <- as.character(groups[2])

  assumptions <- list(
    ustat_check_normality(g1, name1),
    ustat_check_normality(g2, name2),
    ustat_check_equal_variances(
      list(g1, g2), c(name1, name2), "Welch correction applied"
    )
  )

  if (identical(req$method, "welch")) {
    use_welch <- TRUE
    chosen_by <- "request (method)"
  } else if (identical(req$method, "student")) {
    use_welch <- FALSE
    chosen_by <- "request (method)"
  } else if (!is.null(req$equal_var)) {
    use_welch <- !isTRUE(as.logical(req$equal_var))
    chosen_by <- "request (equal_var)"
  } else {
    use_welch <- !isTRUE(assumptions[[3]]$met)
    chosen_by <- "auto (Levene)"
  }

  tt <- stats::t.test(g1, g2, var.equal = !use_welch)
  stat <- unname(tt$statistic)
  p <- unname(tt$p.value)
  sig <- isTRUE(p < 0.05)
  es <- ustat_cohen_d(g1, g2)

  ret <- list(
    test = paste0(
      "Independent samples t-test", if (use_welch) " (Welch)" else ""
    ),
    group1 = name1, n1 = length(g1), mean1 = mean(g1),
    group2 = name2, n2 = length(g2), mean2 = mean(g2),
    t = stat,
    p = p,
    # df must match the test that produced t and p. t.test reports the
    # fractional Satterthwaite df for Welch and n1 + n2 - 2 for the pooled
    # test, which is exactly the pairing the Python side assembles by hand.
    df = unname(tt$parameter),
    df_method = if (use_welch) "welch_satterthwaite" else "pooled",
    variance_assumption = if (use_welch) "welch" else "student",
    variance_assumption_selected_by = chosen_by,
    significant = sig,
    effect_sizes = list(es),
    assumptions = assumptions,
    summary = stats::setNames(
      list(ustat_group_summary(g1, name1), ustat_group_summary(g2, name2)),
      c(name1, name2)
    ),
    interpretation = ustat_interpretation_ttest_ind_text(sig, stat, p, es),
    methods_text = ustat_methods_ttest_ind_text(
      req$column, req$group_column, use_welch
    ),
    r_code = ustat_r_ttest_ind_text(req$column, req$group_column, use_welch)
  )
  if (length(cleaned$warnings) > 0L) {
    ret$warnings <- cleaned$warnings
  }
  ret$result_text <- ustat_results_ttest_ind_text(ret)
  ret
}

ustat_ttest_one <- function(req, frame) {
  x <- ustat_to_numeric(ustat_ttest_column(frame, req$column))
  x <- x[!is.na(x)]

  tt <- stats::t.test(x, mu = req$mu)
  stat <- unname(tt$statistic)
  p <- unname(tt$p.value)
  sig <- isTRUE(p < 0.05)
  es <- ustat_cohen_d_one_sample(x, req$mu)

  ret <- list(
    test = "One-sample t-test",
    mu = req$mu, n = length(x),
    mean = mean(x), std = stats::sd(x),
    t = stat, p = p, df = length(x) - 1L,
    significant = sig,
    effect_sizes = list(es),
    assumptions = list(ustat_check_normality(x, req$column)),
    summary = list(sample = ustat_group_summary(x, "Sample")),
    interpretation = ustat_interpretation_ttest_one_text(
      sig, stat, p, es, req$mu_text
    ),
    methods_text = ustat_methods_ttest_one_text(req$column, req$mu_text),
    r_code = ustat_r_ttest_one_text(req$column, req$mu_text)
  )
  # mu_text is passed rather than stored: it is a rendering of a field the
  # payload already carries, and a second spelling of mu in the result would be
  # one more thing the two engines could disagree about.
  ret$result_text <- ustat_results_ttest_one_text(
    c(ret, list(mu_text = req$mu_text))
  )
  ret
}

ustat_ttest <- function(params, frame) {
  req <- ustat_ttest_params(params)
  if (is.null(req$group_column)) {
    ustat_ttest_one(req, frame)
  } else {
    ustat_ttest_ind(req, frame)
  }
}

ustat_register(list(
  id = "stats.ttest",
  needs_frame = TRUE,
  # Base R carries t.test, shapiro.test and the one-way F this engine's Levene
  # goes through. These two are here for one branch each of check_normality, and
  # for the same reason the Python spec declares statsmodels reluctantly: a
  # package that is not in the boot plan cannot be fetched from inside a
  # synchronous call, so any group of 50 or more would crash a local run that
  # declared only base R.
  packages = c("moments", "nortest"),
  columns_for = function(params) {
    group_column <- params$group_column
    if (is.null(group_column)) group_column <- params$group_col
    c(params$column, group_column)
  },
  fn = ustat_ttest
))
