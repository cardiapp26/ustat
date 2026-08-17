#!/usr/bin/env Rscript
# Independent reference values for the webR feasibility spike.
#
# Two-sample t-tests on the synthetic Models-audit dataset: Welch (the R
# default) and Student (var.equal = TRUE). Base R only; JSON is emitted by
# hand at %.17g so nothing is lost to rounding on the way out -- jsonlite's
# default `digits = 4` would silently truncate these to four significant
# figures and turn a real parity check into a rounded one.
#
#   Rscript qa/tests_audit/reference.R

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
if (length(here) == 0 || here == "") here <- "qa/tests_audit"

d <- read.csv(file.path(here, "..", "models_audit", "dataset.csv"),
              stringsAsFactors = FALSE)
d$arm <- factor(d$arm)   # levels: control, treat -- control is the reference

# ── JSON emission (no jsonlite dependency, full precision) ───────────────────
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

# Every field the spike compares, named exactly as htest names it so the
# browser side can read them straight off the R object without a mapping
# table that could itself be wrong.
tt_obj <- function(tt) {
  obj(
    kv("statistic", num(tt$statistic[["t"]])),
    kv("parameter", num(tt$parameter[["df"]])),
    kv("p.value", num(tt$p.value)),
    kv("conf.int", arr(c(num(tt$conf.int[1]), num(tt$conf.int[2])))),
    kv("estimate", arr(c(num(tt$estimate[[1]]), num(tt$estimate[[2]])))),
    kv("estimate_names", arr(vapply(names(tt$estimate), str_, character(1)))),
    kv("conf_level", num(attr(tt$conf.int, "conf.level"))),
    kv("alternative", str_(tt$alternative)),
    kv("method", str_(tt$method))
  )
}

welch   <- t.test(sbp ~ arm, data = d, var.equal = FALSE)
student <- t.test(sbp ~ arm, data = d, var.equal = TRUE)

meta <- obj(
  kv("r_version", str_(R.version.string)),
  kv("call", str_("t.test(sbp ~ arm, data = d, var.equal = <FALSE|TRUE>)")),
  kv("dataset", str_("qa/models_audit/dataset.csv")),
  kv("n", num(nrow(d))),
  kv("arm_levels", arr(vapply(levels(d$arm), str_, character(1)))),
  kv("n_by_arm", obj(vapply(levels(d$arm), function(l)
    kv(l, num(sum(d$arm == l))), character(1)))),
  kv("note", str_(paste("emitted with sprintf('%.17g'); jsonlite's default",
                        "digits = 4 would truncate these")))
)

body <- obj(kv("meta", meta),
            kv("welch", tt_obj(welch)),
            kv("student", tt_obj(student)))

writeLines(body, file.path(here, "reference.json"))
cat("wrote", file.path(here, "reference.json"), "- welch + student\n")
