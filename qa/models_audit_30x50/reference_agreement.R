# Reference values for the Correlation panel's four tabs — Pairwise, Matrix,
# ICC and Cohen's kappa — plus Fleiss for three raters.
#
# Writes reference_agreement.json.  Run from the project root:
#   Rscript qa/models_audit_30x50/reference_agreement.R

suppressMessages({
  library(irr)
})

here <- "qa/models_audit_30x50"
# na.strings matters: an empty cell in a text column would otherwise arrive as
# "" and become a rating category of its own.
d  <- read.csv(file.path(here, "dataset.csv"), stringsAsFactors = FALSE,
               na.strings = c("", "NA"))
ag <- read.csv(file.path(here, "dataset_raters.csv"), stringsAsFactors = FALSE,
               na.strings = c("", "NA"))

num <- function(x) {
  if (length(x) == 0L || is.na(x[[1L]]) || !is.finite(x[[1L]])) return("null")
  sprintf("%.17g", as.numeric(x[[1L]]))
}
kv  <- function(k, v) paste0("\"", k, "\":", v)
obj <- function(...) paste0("{", paste(c(...), collapse = ","), "}")

models <- list()

# ── Pairwise: Pearson, Spearman, Kendall on age vs bmi (bmi has 3 NA) ────────
ct <- cor.test(d$age, d$bmi, method = "pearson")
models$corr_pearson <- obj(
  kv("r", num(unname(ct$estimate))), kv("p", num(ct$p.value)),
  kv("ci_low", num(ct$conf.int[1])), kv("ci_high", num(ct$conf.int[2])),
  kv("n", num(unname(ct$parameter) + 2)))

cs <- suppressWarnings(cor.test(d$age, d$bmi, method = "spearman"))
models$corr_spearman <- obj(
  kv("r", num(unname(cs$estimate))), kv("p", num(cs$p.value)))

ck <- suppressWarnings(cor.test(d$age, d$bmi, method = "kendall"))
models$corr_kendall <- obj(
  kv("r", num(unname(ck$estimate))), kv("p", num(ck$p.value)))

# ── Matrix: pairwise-complete Pearson over four numeric columns ──────────────
cols <- c("age", "bmi", "sbp", "score1")
# complete.obs = listwise, which is what the endpoint does: it drops any
# row missing on ANY selected variable, so every cell shares one n.
m <- cor(d[, cols], use = "complete.obs")
matrix_rows <- character(0)
for (i in cols) for (j in cols) {
  matrix_rows <- c(matrix_rows, kv(paste0(i, "|", j), num(m[i, j])))
}
models$corr_matrix <- obj(matrix_rows)

# ── ICC: two-way random, single measure. BOTH forms, because the point
#    estimate and the interval have to belong to the same one. ──────────────
r_ag <- irr::icc(ag[, c("rater_a", "rater_b")], model = "twoway",
                 type = "agreement", unit = "single")
r_cs <- irr::icc(ag[, c("rater_a", "rater_b")], model = "twoway",
                 type = "consistency", unit = "single")
models$icc <- obj(
  kv("icc", num(r_ag$value)),
  kv("ci_low", num(r_ag$lbound)), kv("ci_high", num(r_ag$ubound)),
  kv("f_stat", num(r_ag$Fvalue)), kv("n", num(r_ag$subjects)),
  kv("consistency_icc", num(r_cs$value)),
  kv("consistency_ci_low", num(r_cs$lbound)),
  kv("consistency_ci_high", num(r_cs$ubound)))

# ── Cohen's kappa ────────────────────────────────────────────────────────────
k2 <- irr::kappa2(ag[, c("cat_a", "cat_b")])
sub <- na.omit(ag[, c("cat_a", "cat_b")])
tab <- table(sub$cat_a, sub$cat_b)
n_k <- sum(tab)
po <- sum(diag(tab)) / n_k
pe <- sum(rowSums(tab) * colSums(tab)) / n_k^2
models$cohens_kappa <- obj(
  kv("kappa", num(k2$value)), kv("z", num(k2$statistic)),
  kv("p", num(k2$p.value)), kv("n", num(k2$subjects)),
  kv("po", num(po)), kv("pe", num(pe)))

# ── Fleiss kappa ─────────────────────────────────────────────────────────────
kf <- irr::kappam.fleiss(ag[, c("cat_a", "cat_b", "cat_c")])
models$fleiss_kappa <- obj(
  kv("kappa", num(kf$value)), kv("z", num(kf$statistic)),
  kv("p", num(kf$p.value)), kv("n_subjects", num(kf$subjects)),
  kv("se", num(kf$value / kf$statistic)))

out <- paste0("{\"models\":{", paste(
  mapply(function(k, v) paste0("\"", k, "\":", v), names(models), models,
         USE.NAMES = FALSE), collapse = ","), "}}")
writeLines(out, file.path(here, "reference_agreement.json"))
cat("wrote", file.path(here, "reference_agreement.json"), "-",
    length(models), "models\n")
