# Gray's K-sample test of equal cumulative incidence (competing risks).
#
# A thin wrapper over cmprsk::cuminc (Gray 1988, rho = 0). Every number comes
# out of cuminc; this file only reads the columns, refuses inputs cuminc would
# mis-handle, and names the answer.
#
# WHY THIS RUNS ONLY IN R
# -----------------------
# The server's Fine-Gray panel reports a cause-specific log-rank and points
# here. Gray's score is easy to reproduce, but his variance has no published
# finite-sample form that matches cmprsk: three asymptotically equivalent
# estimators derived from the paper agreed with cmprsk under the null and
# differed by 10-30% on the statistic under alternatives, enough to move a p
# across 0.05. cmprsk is GPL and uSTAT is MIT, so its estimator cannot be
# ported either. The test is therefore cmprsk's or nothing.
#
# PARAMS
#   duration_col        follow-up time column (numeric, >= 0)
#   event_col           0 = censored, any other integer code = an event type
#   group_col           grouping column (>= 2 groups)
#   event_of_interest   the event code whose cumulative incidence is compared
#   censor_code         optional, default 0

ustat_gray_params <- function(params) {
  need <- c("duration_col", "event_col", "group_col")
  for (field in need) {
    v <- params[[field]]
    if (is.null(v) || length(v) != 1L || is.na(v) || !nzchar(as.character(v))) {
      ustat_stop(paste0("Field '", field, "' is required."), 422L)
    }
  }
  eoi <- params$event_of_interest
  if (is.null(eoi) || length(eoi) != 1L || is.na(suppressWarnings(as.numeric(eoi)))) {
    ustat_stop("Field 'event_of_interest' is required (an event code, e.g. 1).", 422L)
  }
  censor <- params$censor_code
  if (is.null(censor)) censor <- 0
  list(
    duration_col = as.character(params$duration_col),
    event_col = as.character(params$event_col),
    group_col = as.character(params$group_col),
    event_of_interest = as.numeric(eoi),
    censor_code = as.numeric(censor)
  )
}

ustat_gray <- function(params, frame) {
  req <- ustat_gray_params(params)
  cols <- c(req$duration_col, req$event_col, req$group_col)
  missing <- cols[!(cols %in% names(frame))]
  if (length(missing)) {
    ustat_stop(paste0("Column(s) not found: ", paste(missing, collapse = ", ")), 422L)
  }

  time <- ustat_to_numeric(frame[[req$duration_col]])
  status <- ustat_to_numeric(frame[[req$event_col]])
  group <- frame[[req$group_col]]
  if (is.factor(group)) group <- as.character(group)
  keep <- !is.na(time) & !is.na(status) & !is.na(group)
  time <- time[keep]
  status <- status[keep]
  group <- as.character(group[keep])
  n_dropped <- sum(!keep)

  if (any(time < 0)) {
    ustat_stop(paste0("'", req$duration_col, "' has negative follow-up times."), 422L)
  }
  if (any(status %% 1 != 0)) {
    ustat_stop(paste0(
      "'", req$event_col, "' must hold integer event codes (",
      req$censor_code, " = censored)."
    ), 422L)
  }
  causes <- sort(unique(status[status != req$censor_code]))
  if (!(req$event_of_interest %in% causes)) {
    ustat_stop(paste0(
      "Event code ", req$event_of_interest, " does not occur in '", req$event_col,
      "'. Event codes present: ", paste(causes, collapse = ", "), "."
    ), 422L)
  }
  levels <- ustat_sorted_groups(group)
  if (length(levels) < 2L) {
    ustat_stop(paste0("'", req$group_col, "' needs at least two groups."), 422L)
  }

  fit <- cmprsk::cuminc(
    ftime = time, fstatus = status,
    group = factor(group, levels = levels),
    cencode = req$censor_code
  )
  tests <- fit$Tests
  by_cause <- lapply(rownames(tests), function(code) list(
    event = as.numeric(code),
    statistic = unname(tests[code, "stat"]),
    p = unname(tests[code, "pv"]),
    df = as.integer(tests[code, "df"])
  ))
  row <- as.character(req$event_of_interest)
  if (!(row %in% rownames(tests))) {
    # cuminc names rows by the codes as they print; 1 and 1.0 print alike.
    row <- rownames(tests)[match(req$event_of_interest, as.numeric(rownames(tests)))]
  }

  counts <- lapply(levels, function(lv) {
    s <- status[group == lv]
    list(
      group = lv,
      n = length(s),
      event_of_interest = sum(s == req$event_of_interest),
      competing_events = sum(s != req$censor_code & s != req$event_of_interest),
      censored = sum(s == req$censor_code)
    )
  })

  list(
    test = "Gray's test",
    statistic = unname(tests[row, "stat"]),
    df = as.integer(tests[row, "df"]),
    p = unname(tests[row, "pv"]),
    event_of_interest = req$event_of_interest,
    hypothesis = paste0(
      "Equal cumulative incidence of event ", req$event_of_interest,
      " across ", req$group_col, " groups"
    ),
    groups = counts,
    by_cause = by_cause,
    n = length(time),
    n_excluded = n_dropped,
    engine = "R cmprsk::cuminc (rho = 0)",
    r_code = paste0(
      "library(cmprsk)\n",
      "ci <- cuminc(ftime = df$", req$duration_col, ", fstatus = df$", req$event_col,
      ", group = df$", req$group_col, ", cencode = ", req$censor_code, ")\n",
      "ci$Tests"
    )
  )
}

ustat_register(list(
  id = "survival.gray",
  needs_frame = TRUE,
  packages = c("cmprsk"),
  columns_for = function(params) {
    c(params$duration_col, params$event_col, params$group_col)
  },
  fn = ustat_gray
))
