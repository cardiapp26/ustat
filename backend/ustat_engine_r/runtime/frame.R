# One filtered dataset, rebuilt from the ustat.frame/1 envelope.
#
# The R half of ustat_engine/frame/envelope.py::frame_from_envelope, plus the
# two things in frame/levels.py and frame/category_health.py that decide *which
# rows exist* and *which group comes first*: ustat_sorted_groups and
# ustat_clean_two_level. Those are not presentation. clean_two_level folds
# M/male/Male into one level and treats "n/a" as missing, so it fixes n1 and n2
# before the test sees them; sorted_groups fixes which arm is group1. An R copy
# that differed on either would not produce a rounding difference, it would test
# a different set of patients and report a confidently wrong result.
#
# NOTHING HERE PARSES A CSV. read.csv was the R0 spike's richest source of
# silent wrongness -- it mangles column names unless check.names = FALSE, and
# its stringsAsFactors and na.strings defaults each decide something the
# envelope has already decided properly. The envelope carries typed columns,
# declared kinds, explicit level orders and the filter's fingerprint; this file
# copies them across and re-derives none of them.
#
# LEVEL ORDER IS NEVER LEFT TO R. factor() with no levels= argument sorts by the
# collation of whatever locale the process happens to be in, and Turkish
# collation orders dotted and dotless i differently from code point -- so the same
# dataset would produce a different group1 on a Turkish machine. Every factor
# built here is given explicit levels= in the envelope's order, and every
# comparison sort goes through order(method = "radix"), which R documents as
# always using the C locale.

USTAT_FRAME_SCHEMA <- "ustat.frame/1"

# Belt and braces next to the radix sorts above: anything that reaches a
# locale-sensitive comparison anyway sees the C collation.
ustat_set_c_collation <- function() {
  suppressWarnings(try(Sys.setlocale("LC_COLLATE", "C"), silent = TRUE))
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# Envelope -> data.frame
# ---------------------------------------------------------------------------

ustat_num_vector <- function(payload) {
  if (is.null(payload)) {
    return(numeric(0))
  }
  if (is.numeric(payload)) {
    return(as.numeric(payload))
  }
  nulls <- vapply(payload, is.null, logical(1))
  if (any(nulls)) {
    payload[nulls] <- NA_real_
  }
  as.numeric(unlist(payload, use.names = FALSE))
}

ustat_chr_vector <- function(payload) {
  if (is.null(payload)) {
    return(character(0))
  }
  nulls <- vapply(payload, is.null, logical(1))
  if (any(nulls)) {
    payload[nulls] <- NA_character_
  }
  as.character(unlist(payload, use.names = FALSE))
}

ustat_int_vector <- function(payload) {
  if (is.null(payload)) {
    return(integer(0))
  }
  if (is.numeric(payload)) {
    return(as.integer(payload))
  }
  nulls <- vapply(payload, is.null, logical(1))
  if (any(nulls)) {
    payload[nulls] <- NA_integer_
  }
  as.integer(unlist(payload, use.names = FALSE))
}

ustat_factor_from_payload <- function(payload, spec) {
  levels_chr <- ustat_chr_vector(payload$levels)
  if (length(levels_chr) == 0L) {
    levels_chr <- ustat_chr_vector(spec$levels)
  }
  codes <- ustat_int_vector(payload$codes)
  # -1 is the envelope's spelling of "missing"; anything outside the declared
  # level range is treated the same way rather than silently pointing at a
  # neighbouring level.
  codes[!is.na(codes) & (codes < 0L | codes >= length(levels_chr))] <- NA_integer_

  ordered <- isTRUE(spec$ordered)
  if (length(levels_chr) == 0L) {
    return(factor(rep(NA_character_, length(codes)), levels = character(0), ordered = ordered))
  }

  out <- factor(
    codes + 1L,
    levels = seq_along(levels_chr),
    labels = levels_chr,
    ordered = ordered
  )

  # The envelope names the reference category -- the level a model coefficient
  # is relative to -- and in R that is expressed by making it the FIRST level,
  # because model.matrix() takes level one as the baseline. An unordered factor
  # is releveled here so the declared reference survives into any lm()/glm()
  # that reads this frame. Presentation order is not lost by doing so: nothing
  # in this engine reads a factor's level order to decide which group comes
  # first -- ustat_sorted_groups re-derives that below, the same way Python's
  # sorted_groups does. An ordered factor is left alone; relevel() refuses one,
  # and for an ordinal column the declared order IS the meaning.
  reference <- spec$reference
  if (!ordered && !is.null(reference) && length(reference) == 1L &&
      !is.na(reference) && as.character(reference) %in% levels_chr) {
    out <- stats::relevel(out, ref = as.character(reference))
  }
  out
}

ustat_frame_from_envelope <- function(env) {
  ustat_set_c_collation()

  if (!is.list(env) || !identical(env$schema, USTAT_FRAME_SCHEMA)) {
    got <- if (is.list(env) && !is.null(env$schema)) {
      paste0("'", as.character(env$schema), "'")
    } else {
      paste0("'", class(env)[1], "'")
    }
    ustat_stop(paste0("expected a ", USTAT_FRAME_SCHEMA, " envelope, got ", got), 422L)
  }

  specs <- env$columns
  if (is.null(specs)) specs <- list()
  data <- env$data
  if (is.null(data)) data <- list()

  cols <- vector("list", length(specs))
  nms <- character(length(specs))

  for (i in seq_along(specs)) {
    spec <- specs[[i]]
    name <- as.character(spec$name)
    storage <- if (is.null(spec$storage)) "str" else as.character(spec$storage)
    payload <- data[[name]]

    cols[[i]] <- if (identical(storage, "cat")) {
      ustat_factor_from_payload(payload, spec)
    } else if (identical(storage, "f64") || identical(storage, "f64_epoch_ms")) {
      # epoch milliseconds stay a double here. The Python side turns them into a
      # tz-aware datetime; no analysis in this engine reads a date column yet,
      # and inventing a conversion nothing consumes would only be a second place
      # for the two runtimes to disagree.
      ustat_num_vector(payload)
    } else {
      ustat_chr_vector(payload)
    }
    nms[i] <- name
  }

  rows <- if (is.null(env$rows)) NA_integer_ else as.integer(env$rows)
  if (is.na(rows)) {
    rows <- if (length(cols) > 0L) length(cols[[1]]) else 0L
  }

  # The row index is the rows' positions in the UNFILTERED frame, so a result
  # that names a row names the same row the grid does.
  row_index <- ustat_int_vector(env$row_index)
  if (length(row_index) != rows) {
    row_index <- seq_len(rows)
  }

  # structure() rather than data.frame(): data.frame() runs check.names on the
  # column names and would turn a Turkish header into "Ya..y.l.". Names travel
  # verbatim -- spaces, parentheses, Turkish letters -- because they are keys,
  # never identifiers.
  df <- structure(
    cols,
    names = nms,
    row.names = as.integer(row_index),
    class = "data.frame"
  )

  # The filter this frame was cut with, so a later run can prove it is asking
  # about the same patients.
  attr(df, "filter_fingerprint") <- if (is.null(env$filter$fingerprint)) {
    NULL
  } else {
    as.character(env$filter$fingerprint)
  }
  df
}

ustat_frame_from_json <- function(txt) {
  ustat_frame_from_envelope(ustat_from_json(txt))
}

ustat_filter_fingerprint <- function(frame) {
  attr(frame, "filter_fingerprint")
}

# ---------------------------------------------------------------------------
# Column vocabulary: how values are read and how levels are ordered
# ---------------------------------------------------------------------------

# pandas' pd.to_numeric(errors="coerce"). A numeric column passes through
# untouched -- deliberately NOT via as.character(), which would round-trip every
# double through decimal for no reason.
ustat_to_numeric <- function(x) {
  if (is.numeric(x)) {
    return(as.numeric(x))
  }
  if (is.factor(x)) {
    x <- as.character(x)
  }
  suppressWarnings(as.numeric(x))
}

# frame/levels.py::sorted_groups. Sort by the underlying value code numerically
# when every distinct value coerces, else lexicographically by string. Without
# it, groups follow their order of appearance in the data and results come out
# scrambled relative to the value labels.
#
# order(method = "radix") is stable and, for character input, documented to sort
# in the C locale regardless of the session's -- which is byte order, which for
# UTF-8 is code point order, which is what Python's sorted() on str does. That
# equivalence is the whole reason this can be a one-liner.
ustat_sorted_groups <- function(x) {
  if (is.factor(x)) {
    x <- as.character(x)
  }
  vals <- unique(x[!is.na(x)])
  if (length(vals) == 0L) {
    return(vals)
  }
  nums <- suppressWarnings(as.numeric(as.character(vals)))
  if (!any(is.na(nums))) {
    return(vals[order(nums, method = "radix")])
  }
  vals[order(as.character(vals), method = "radix")]
}

# ---------------------------------------------------------------------------
# frame/category_health.py::clean_two_level
# ---------------------------------------------------------------------------

USTAT_MISSING_TOKENS <- c(
  "", ".", "-", "--", "?", "na", "n/a", "nan", "null",
  "missing", "unknown", "unk"
)

# Blanks and punctuation are unambiguous and dropping them silently is what a
# user expects. A word like "unknown" or "n/a" may well be a category the user
# meant to keep, so those get named in a warning instead.
USTAT_SILENT_MISSING_TOKENS <- c("", ".", "-", "--", "?")

USTAT_SEX_MAP <- c(
  f = "Female", female = "Female", woman = "Female", women = "Female",
  m = "Male", male = "Male", man = "Male", men = "Male"
)

USTAT_BINARY_MAP <- c(
  "0" = "0", "1" = "1",
  no = "0", n = "0", "false" = "0", negative = "0", neg = "0", absent = "0",
  yes = "1", y = "1", "true" = "1", positive = "1", pos = "1", present = "1"
)

ustat_level_counts <- function(values) {
  if (length(values) == 0L) {
    return(list(levels = character(0), n = integer(0)))
  }
  tab <- table(values)
  # Descending count, ties broken by level name. pandas' value_counts() sorts
  # descending too but leaves tied counts in an unspecified order, so this is
  # the one place the two engines can legitimately list the same warning's
  # levels in a different sequence.
  idx <- order(-as.integer(tab), names(tab), method = "radix")
  list(levels = names(tab)[idx], n = as.integer(tab)[idx])
}

ustat_dropped_levels <- function(counts) {
  lapply(seq_along(counts$levels), function(i) {
    list(level = counts$levels[i], n = counts$n[i])
  })
}

# Returns list(series = <character with NA for missing>, warnings = <list>).
#
# Only the two fields run_ttest reads are returned. CategoryCleanResult also
# carries `levels` and `n_dropped`, which no caller in either engine touches;
# computing them here would only widen the surface the fingerprint has to cover.
ustat_clean_two_level <- function(series, name = NULL) {
  raw_na <- is.na(series)
  text <- trimws(as.character(series))
  lowered <- tolower(text)

  token_missing <- !raw_na & !is.na(lowered) & (lowered %in% USTAT_MISSING_TOKENS)
  missing <- raw_na | token_missing

  cleaned <- text
  cleaned[missing] <- NA_character_
  warnings <- list()

  spoken <- lowered[token_missing & !(lowered %in% USTAT_SILENT_MISSING_TOKENS)]
  if (length(spoken) > 0L) {
    counts <- ustat_level_counts(spoken)
    warnings[[length(warnings) + 1L]] <- list(
      variable = if (is.null(name)) NULL else as.character(name),
      dropped_levels = ustat_dropped_levels(counts),
      note = paste0(
        "'", as.character(name), "': ", sum(counts$n), " row(s) hold a value ",
        "that reads as missing (",
        paste(paste0("'", counts$levels, "'"), collapse = ", "),
        ") and were excluded from the counts and the test. If any of these is ",
        "a real category, recode it before analysing."
      )
    )
  }

  observed <- unique(lowered[!missing])
  observed <- observed[!is.na(observed)]
  sex_labels <- sort(unique(unname(USTAT_SEX_MAP[observed[observed %in% names(USTAT_SEX_MAP)]])))

  mapper <- NULL
  if (identical(sex_labels, c("Female", "Male")) ||
      (length(observed) > 0L && all(observed %in% names(USTAT_SEX_MAP)))) {
    mapper <- USTAT_SEX_MAP
  } else if (length(observed) > 0L && all(observed %in% names(USTAT_BINARY_MAP))) {
    mapper <- USTAT_BINARY_MAP
  }

  if (!is.null(mapper)) {
    mapped <- unname(mapper[lowered])
    known <- !is.na(mapped)
    cleaned <- ifelse(known, mapped, NA_character_)
    unknown <- lowered[!missing & !known]
    unknown <- unknown[!is.na(unknown)]
    if (length(unknown) > 0L) {
      counts <- ustat_level_counts(unknown)
      warnings[[length(warnings) + 1L]] <- list(
        variable = if (is.null(name)) NULL else as.character(name),
        dropped_levels = ustat_dropped_levels(counts),
        note = paste0(
          "Unrecognized values were treated as missing after normalizing the ",
          "two-level variable."
        )
      )
    }
  }

  list(series = cleaned, warnings = warnings)
}
