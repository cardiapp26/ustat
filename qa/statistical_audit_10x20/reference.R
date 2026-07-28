args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript reference.R dataset.csv")
}

d <- read.csv(args[[1]], na.strings = c(""))
emit <- function(name, value) {
  cat(name, format(as.numeric(value), digits = 17, scientific = TRUE), sep = "\t")
  cat("\n")
}

# Independent and one-sample tests.
one_t <- t.test(d$biomarker_normal, mu = 10)
emit("one_sample_t", one_t$statistic)
emit("one_sample_t_p", one_t$p.value)

a <- d$biomarker_normal[d$arm == "A" & !is.na(d$biomarker_normal)]
b <- d$biomarker_normal[d$arm == "B" & !is.na(d$biomarker_normal)]
ind_t <- t.test(a, b, var.equal = TRUE)
emit("independent_student_t", ind_t$statistic)
emit("independent_student_p", ind_t$p.value)

fit_aov <- aov(biomarker_skew ~ group3, data = d)
aov_tab <- summary(fit_aov)[[1]]
emit("anova_f", aov_tab["group3", "F value"])
emit("anova_p", aov_tab["group3", "Pr(>F)"])

emit(
  "mann_whitney_p",
  wilcox.test(a, b, alternative = "two.sided", exact = TRUE)$p.value
)
emit(
  "kruskal_p",
  kruskal.test(biomarker_skew ~ group3, data = d)$p.value
)

# Contingency tests. R and SciPy both apply Yates correction by default to 2x2.
ct <- table(d$category_binary, d$arm)
emit("chi_square", unname(chisq.test(ct, correct = TRUE)$statistic))
emit("chi_square_p", chisq.test(ct, correct = TRUE)$p.value)
emit("fisher_p", fisher.test(ct)$p.value)

# Paired and repeated-measures tests.
paired <- complete.cases(d[, c("pre_score", "post_score")])
paired_t <- t.test(
  d$pre_score[paired], d$post_score[paired],
  paired = TRUE
)
emit("paired_t", paired_t$statistic)
emit("paired_t_p", paired_t$p.value)

friedman_cc <- complete.cases(
  d[, c("pre_score", "post_score", "followup_score")]
)
wide <- as.matrix(
  d[friedman_cc, c("pre_score", "post_score", "followup_score")]
)
fried <- friedman.test(wide)
emit("friedman_chi2", fried$statistic)
emit("friedman_p", fried$p.value)

# ANCOVA: drop1 gives partial F for arm in additive model, matching Type II.
anc <- lm(ancova_outcome ~ arm + age, data = d)
anc_tab <- drop1(anc, test = "F")
emit("ancova_f", anc_tab["arm", "F value"])
emit("ancova_p", anc_tab["arm", "Pr(>F)"])

# Partial multivariate group effect: compare reduced and full multivariate LMs.
manc_reduced <- lm(cbind(ancova_outcome, outcome2) ~ age, data = d)
manc_full <- lm(cbind(ancova_outcome, outcome2) ~ age + arm, data = d)
manc_cmp <- anova(manc_reduced, manc_full, test = "Pillai")
emit("mancova_pillai", manc_cmp[2, "Pillai"])
emit("mancova_f", manc_cmp[2, "approx F"])
emit("mancova_p", manc_cmp[2, "Pr(>F)"])

# Correct split-plot error strata for repeated and mixed ANOVA.
cc <- d[friedman_cc, c(
  "id", "arm", "pre_score", "post_score", "followup_score"
)]
long <- reshape(
  cc,
  varying = c("pre_score", "post_score", "followup_score"),
  v.names = "score",
  timevar = "timepoint",
  times = c("pre", "post", "followup"),
  direction = "long"
)
long$id <- factor(long$id)
long$arm <- factor(long$arm)
long$timepoint <- factor(
  long$timepoint,
  levels = c("pre", "post", "followup")
)
mixed <- summary(aov(
  score ~ arm * timepoint + Error(id / timepoint),
  data = long
))
rm_only <- summary(aov(
  score ~ timepoint + Error(id / timepoint),
  data = long
))
between <- mixed[["Error: id"]][[1]]
within <- mixed[["Error: id:timepoint"]][[1]]
rm_within <- rm_only[["Error: id:timepoint"]][[1]]
emit("mixed_arm_f", between["arm", "F value"])
emit("mixed_arm_p", between["arm", "Pr(>F)"])
emit("rm_time_f", rm_within["timepoint", "F value"])
emit("rm_time_p", rm_within["timepoint", "Pr(>F)"])
emit("mixed_time_f", within["timepoint", "F value"])
emit("mixed_time_p", within["timepoint", "Pr(>F)"])
emit("mixed_interaction_f", within["arm:timepoint", "F value"])
emit("mixed_interaction_p", within["arm:timepoint", "Pr(>F)"])

# Correct one-sided non-inferiority p-values for upper-bound hypotheses.
test_y <- d$ancova_outcome[d$arm == "B"]
ref_y <- d$ancova_outcome[d$arm == "A"]
estimate <- mean(test_y) - mean(ref_y)
test_var_term <- var(test_y) / length(test_y)
ref_var_term <- var(ref_y) / length(ref_y)
se <- sqrt(test_var_term + ref_var_term)
welch_df <- (test_var_term + ref_var_term)^2 / (
  test_var_term^2 / (length(test_y) - 1) +
    ref_var_term^2 / (length(ref_y) - 1)
)
emit("ni_cont_estimate", estimate)
emit("ni_cont_welch_df", welch_df)
emit("ni_cont_upper_p", pt((estimate - 20) / se, df = welch_df))

test_e <- sum(d$event_binary[d$arm == "B"])
ref_e <- sum(d$event_binary[d$arm == "A"])
test_n <- sum(d$arm == "B")
ref_n <- sum(d$arm == "A")
rr <- (test_e / test_n) / (ref_e / ref_n)
se_log_rr <- sqrt(
  (1 - test_e / test_n) / test_e +
  (1 - ref_e / ref_n) / ref_e
)
emit("ni_binary_rr", rr)
emit("ni_binary_upper_p", pnorm((log(rr) - log(3)) / se_log_rr))
