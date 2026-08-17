# One implementation of "make this safe to serialise", matching jsonsafe.py.
#
# ustat_sanitize() is the R half of ustat_engine/jsonsafe.py::sanitize. Same
# rule, same reason: NaN and the infinities are not representable in JSON, and
# emitting them produces a document a strict parser rejects outright. They
# become null, which is what a missing value in a result table means. NA gets
# the same treatment -- it is R's spelling of the same thing.
#
# WHY digits = I(17), AND WHY NOT sprintf("%.17g")
# ------------------------------------------------
# The whole point of two engines is that their numbers can be compared. A
# serialiser that quietly drops precision makes that comparison meaningless
# below its own resolution, so this was measured rather than assumed. Over 2008
# doubles (the audit dataset's t/df/p literals, pi, 1/3, DBL_MIN, 1e308 and 2000
# rnorm draws), round-tripped through toJSON/fromJSON under jsonlite 2.0.0:
#
#   digits = I(17)      0 values changed   <- exact, this is what we use
#   digits = NA      1874 values changed   (caps at 15 significant digits)
#   sprintf("%.17g")  367 values changed   (R's sprintf is not a shortest-
#                                           round-trip formatter)
#
# So jsonlite's I(n) form is the only one of the three that survives the trip,
# and 17 significant digits is the width at which an IEEE double always does.
# digits = NA is the trap the R0 spike found: it looks like "no rounding" and is
# in fact 15 digits, which loses the last two bits of every p-value.
#
# jsonlite is called through `::` rather than library(), because a package is
# declared by an analysis and loaded by the host -- see registry.R.

ustat_sanitize <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  # A named list becomes a JSON object, an unnamed one a JSON array. Both
  # recurse. data.frames are lists too, and would degrade to an object of
  # columns; no analysis returns one, and none should.
  if (is.list(x)) {
    out <- lapply(x, ustat_sanitize)
    names(out) <- names(x)
    return(out)
  }

  if (is.factor(x)) {
    x <- as.character(x)
  }

  if (is.character(x) || is.logical(x)) {
    x <- unname(x)
    if (length(x) == 1L) {
      return(if (is.na(x)) NULL else x)
    }
    return(lapply(x, function(v) if (is.na(v)) NULL else v))
  }

  if (is.numeric(x)) {
    x <- unname(x)
    if (length(x) == 1L) {
      return(if (is.finite(x)) x else NULL)
    }
    # An all-finite vector is left as a vector so toJSON writes a plain array;
    # only a vector that actually holds a non-finite value has to be split into
    # a list, because a JSON array can hold null but an R numeric cannot.
    if (all(is.finite(x))) {
      return(x)
    }
    return(lapply(x, function(v) if (is.finite(v)) v else NULL))
  }

  x
}

ustat_to_json <- function(x, pretty = FALSE) {
  jsonlite::toJSON(
    x,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = I(17),
    pretty = pretty
  )
}

# The inverse, for the two things that arrive as text: the frame envelope and
# the analysis params. simplifyVector = FALSE is not a preference -- it is what
# keeps a JSON null inside an array as a NULL element of an R list, which is how
# a missing numeric survives the trip at all. With simplification on, jsonlite
# would coerce [1, null, 3] into something it decided the shape of.
ustat_from_json <- function(txt) {
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}
