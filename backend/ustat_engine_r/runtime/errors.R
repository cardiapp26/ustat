# The R engine's only way to reject a request.
#
# The mirror of ustat_engine/errors.py, and it exists for the same reason: the
# code that detected the problem is the only code that knows whether it was the
# caller's input (422), a missing analysis (404), or a frame that no longer
# matches the active Select Cases (409). The status travels with the error
# rather than being re-derived at the boundary.
#
# It is a real R condition with class "ustat_error", so a host can
# tryCatch(ustat_error = ...) it apart from an ordinary R error -- which is the
# difference between "the user asked for something we refuse" and "the engine
# broke". webR surfaces both as a rejected promise otherwise, indistinguishably.
#
# NOTE ON ENCODING: every .R file in this tree is pure ASCII. Where the Python
# engine's strings carry a character above U+007F -- the em dash in the missing
# marker, the en dash in a confidence interval, the "less than or equal" in the
# methods paragraph -- the R source writes it as a \u escape instead. The bundle
# is fetched, decoded and eval'd by a browser whose text pipeline we do not
# control end to end, and a mangled multibyte character would show up as a
# corrupted sentence in a methods section rather than as an error.

ustat_error <- function(message, status_hint = 400L) {
  structure(
    class = c("ustat_error", "error", "condition"),
    list(
      message = as.character(message),
      call = NULL,
      status_hint = as.integer(status_hint)
    )
  )
}

ustat_stop <- function(message, status_hint = 400L) {
  stop(ustat_error(message, status_hint))
}

ustat_status_hint <- function(cond, default = 500L) {
  hint <- cond$status_hint
  if (is.null(hint)) as.integer(default) else as.integer(hint)
}
