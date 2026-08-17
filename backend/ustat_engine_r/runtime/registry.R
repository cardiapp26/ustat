# The list of analyses this engine can run, and the single entry point.
#
# The R half of ustat_engine/registry.py and spec.py. A registry rather than
# direct dispatch, because the host has to answer "can this run here, and what
# does it need loaded first?" before it has evaluated anything that could
# answer it.
#
# An analysis is declared by a list with these entries:
#
#   id           character   unique, e.g. "stats.ttest"
#   needs_frame  logical     TRUE when fn takes (params, frame)
#   packages     character   R packages the HOST must install and attach before
#                            calling fn -- webr::install(packages, repos =
#                            "/webr/repo"). Never library() from a source file.
#   columns_for  function    (params) -> character, the dataset columns this run
#                            will read. A consented server-side run uploads the
#                            answer to this and nothing else.
#   fn           function    (params, frame) or (params) -> named list
#
# This file is concatenated LAST of the runtime modules, so every analyses/ file
# that follows it can call ustat_register at its own file scope.

.ustat_analyses <- new.env(parent = emptyenv())

# Called by ustat_run and by ustat_frame_from_envelope; safe to call again.
ustat_init <- function() {
  ustat_set_c_collation()
  invisible(TRUE)
}

ustat_register <- function(spec) {
  if (!is.list(spec) || is.null(spec$id)) {
    stop("ustat_register(): an analysis spec needs an 'id'")
  }
  id <- as.character(spec$id)
  if (!is.function(spec$fn)) {
    stop(paste0("ustat_register(): analysis '", id, "' has no callable 'fn'"))
  }
  # Silently replacing would mean two analyses answering to one name, with
  # whichever module was concatenated last deciding which of them runs.
  if (exists(id, envir = .ustat_analyses, inherits = FALSE)) {
    stop(paste0("analysis id '", id, "' is already registered"))
  }
  if (is.null(spec$needs_frame)) spec$needs_frame <- TRUE
  if (is.null(spec$packages)) spec$packages <- character(0)
  assign(id, spec, envir = .ustat_analyses)
  invisible(spec)
}

ustat_analyses <- function() {
  sort(ls(.ustat_analyses), method = "radix")
}

ustat_get <- function(analysis_id) {
  analysis_id <- as.character(analysis_id)
  if (!exists(analysis_id, envir = .ustat_analyses, inherits = FALSE)) {
    ustat_stop(paste0("Unknown analysis: ", analysis_id), 404L)
  }
  get(analysis_id, envir = .ustat_analyses)
}

ustat_packages <- function(analysis_id) {
  as.character(ustat_get(analysis_id)$packages)
}

ustat_columns_for <- function(analysis_id, params = list()) {
  spec <- ustat_get(analysis_id)
  if (!is.function(spec$columns_for)) {
    return(character(0))
  }
  cols <- as.character(spec$columns_for(params))
  cols[!is.na(cols) & nzchar(cols)]
}

# Refuse a frame that was cut under a different Select Cases.
#
# A worker keeps a frame resident between runs; the user's filter does not have
# to stay still while it does. When the caller states which filter the analysis
# assumes, the frame has to agree, because the failure mode otherwise is
# invisible: a perfectly ordinary-looking result computed over the wrong
# patients. Same message and same 409 as registry.py, deliberately -- the two
# engines answer this identically or the guard means nothing.
ustat_check_filter <- function(frame, params) {
  expected <- params[["__filter_fingerprint"]]
  if (is.null(expected)) {
    return(invisible(NULL))
  }
  actual <- attr(frame, "filter_fingerprint")
  if (!identical(as.character(actual), as.character(expected))) {
    ustat_stop("frame does not match the active Select Cases", 409L)
  }
  invisible(NULL)
}

ustat_run <- function(analysis_id, params = list(), frame = NULL) {
  ustat_init()
  spec <- ustat_get(analysis_id)
  if (is.null(params)) params <- list()

  if (isTRUE(spec$needs_frame)) {
    if (is.null(frame)) {
      ustat_stop(
        paste0(analysis_id, " needs a dataset and none was supplied"), 400L
      )
    }
    ustat_check_filter(frame, params)
    return(ustat_sanitize(spec$fn(params, frame)))
  }
  ustat_sanitize(spec$fn(params))
}

# The JSON-in, JSON-out form the browser calls, so a host never has to build R
# lists by hand or decide how an R condition should be shaped.
#
#   {"ok": true,  "result": { ... }}
#   {"ok": false, "error": {"message": "...", "status_hint": 409}}
#
# Every failure becomes the second shape, including one raised by R itself
# rather than by this engine -- those carry status_hint 500, which is the
# caller's cue that it found a bug rather than a bad request.
ustat_run_json <- function(analysis_id, params_json = "{}", frame = NULL) {
  out <- tryCatch(
    {
      params <- ustat_from_json(params_json)
      list(ok = TRUE, result = ustat_run(analysis_id, params, frame))
    },
    ustat_error = function(e) {
      list(ok = FALSE, error = list(
        message = conditionMessage(e),
        status_hint = ustat_status_hint(e, 400L)
      ))
    },
    error = function(e) {
      list(ok = FALSE, error = list(
        message = conditionMessage(e),
        status_hint = 500L
      ))
    }
  )
  as.character(ustat_to_json(out))
}

# What this bundle can do, for a host deciding whether to run locally. The
# bundle's own identity -- version, source fingerprint, sha256 -- is in the
# manifest.json written beside it, not in here: a file that asserted its own
# hash could not be checked against anything.
ustat_identity <- function() {
  ids <- ustat_analyses()
  # as.list, not the bare vectors: toJSON(auto_unbox = TRUE) collapses a
  # length-1 atomic vector to a scalar, so with exactly one analysis registered
  # `analyses` would serialise as "stats.ttest" rather than ["stats.ttest"] and
  # a caller doing analyses.includes(...) would get a string method instead of
  # an array one. A list is never unboxed.
  list(
    schema = USTAT_FRAME_SCHEMA,
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    analyses = as.list(ids),
    packages = as.list(as.character(sort(unique(
      unlist(lapply(ids, ustat_packages), use.names = FALSE)
    ))))
  )
}

ustat_identity_json <- function() {
  as.character(ustat_to_json(ustat_sanitize(ustat_identity())))
}
