data_path <- "qa/models_audit_30x50/dataset.csv"
long_path <- "qa/models_audit_30x50/dataset_long.csv"
output_path <- "qa/models_audit_30x50/reference_regression.json"

# na.strings matters: without it an empty cell in a text column arrives as ""
# and becomes a category of its own.
data <- read.csv(data_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
long <- read.csv(long_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))

json_string <- function(x) {
  if (length(x) == 0L || is.na(x[[1L]])) {
    return("null")
  }
  encodeString(enc2utf8(as.character(x[[1L]])), quote = "\"", justify = "none")
}

json_number <- function(x) {
  if (length(x) == 0L || is.na(x[[1L]]) || !is.finite(x[[1L]])) {
    return("null")
  }
  sprintf("%.17g", as.numeric(x[[1L]]))
}

json_boolean <- function(x) {
  if (length(x) == 0L || is.na(x[[1L]])) {
    return("null")
  }
  if (isTRUE(x[[1L]])) "true" else "false"
}

json_array <- function(values) {
  if (length(values) == 0L) {
    return("[]")
  }
  paste0("[", paste(unlist(values, use.names = FALSE), collapse = ","), "]")
}

json_object <- function(fields) {
  if (length(fields) == 0L) {
    return("{}")
  }
  parts <- mapply(
    function(key, value) paste0(json_string(key), ":", value),
    names(fields),
    fields,
    SIMPLIFY = TRUE,
    USE.NAMES = FALSE
  )
  paste0("{", paste(parts, collapse = ","), "}")
}

term_json <- function(term, estimate, se, statistic, p, extras = list()) {
  json_object(c(
    list(
      term = json_string(term),
      estimate = json_number(estimate),
      se = json_number(se),
      statistic = json_number(statistic),
      p = json_number(p)
    ),
    extras
  ))
}

matrix_terms_json <- function(coefficient_matrix, term_names = rownames(coefficient_matrix),
                              p_column = 4L) {
  rows <- lapply(term_names, function(term) {
    present <- term %in% rownames(coefficient_matrix)
    estimate <- if (present) coefficient_matrix[term, 1L] else NA_real_
    se <- if (present && ncol(coefficient_matrix) >= 2L) {
      coefficient_matrix[term, 2L]
    } else {
      NA_real_
    }
    statistic <- if (present && ncol(coefficient_matrix) >= 3L) {
      coefficient_matrix[term, 3L]
    } else {
      NA_real_
    }
    p <- if (present && ncol(coefficient_matrix) >= p_column) {
      coefficient_matrix[term, p_column]
    } else {
      NA_real_
    }
    term_json(term, estimate, se, statistic, p)
  })
  json_array(rows)
}

model_terms_json <- function(fit) {
  coefficient_matrix <- coef(summary(fit))
  term_names <- names(coef(fit))
  rows <- lapply(term_names, function(term) {
    present <- term %in% rownames(coefficient_matrix)
    estimate <- coef(fit)[[term]]
    se <- if (present) coefficient_matrix[term, 2L] else NA_real_
    statistic <- if (present) coefficient_matrix[term, 3L] else NA_real_
    p <- if (present && ncol(coefficient_matrix) >= 4L) {
      coefficient_matrix[term, 4L]
    } else {
      NA_real_
    }
    term_json(term, estimate, se, statistic, p)
  })
  json_array(rows)
}

named_numbers_json <- function(values) {
  fields <- lapply(values, json_number)
  names(fields) <- names(values)
  json_object(fields)
}

models <- list()

run_model <- function(key, computation) {
  result <- tryCatch(
    list(ok = TRUE, json = computation()),
    error = function(e) list(ok = FALSE, message = conditionMessage(e))
  )
  if (isTRUE(result$ok)) {
    models[[key]] <<- result$json
    cat(key, " OK\n", sep = "")
  } else {
    models[[key]] <<- json_object(list(error = json_string(result$message)))
    cat(key, " ERROR\n", sep = "")
  }
}

run_model("linear", function() {
  fit <- lm(sbp ~ age + bmi + arm + sex, data = data)
  fit_summary <- summary(fit)
  f_statistic <- fit_summary$fstatistic
  overall_p <- pf(
    f_statistic[["value"]],
    f_statistic[["numdf"]],
    f_statistic[["dendf"]],
    lower.tail = FALSE
  )
  json_object(list(
    terms = model_terms_json(fit),
    r_squared = json_number(fit_summary$r.squared),
    adj_r_squared = json_number(fit_summary$adj.r.squared),
    f_statistic = json_number(f_statistic[["value"]]),
    f_numdf = json_number(f_statistic[["numdf"]]),
    f_dendf = json_number(f_statistic[["dendf"]]),
    p = json_number(overall_p),
    sigma = json_number(fit_summary$sigma),
    aic = json_number(AIC(fit)),
    bic = json_number(BIC(fit)),
    n = json_number(nobs(fit))
  ))
})

run_model("linear_collinear", function() {
  fit <- lm(sbp ~ score1 + score2 + age, data = data)
  vif_values <- car::vif(fit)
  json_object(list(
    terms = model_terms_json(fit),
    vif = named_numbers_json(vif_values)
  ))
})

run_model("linear_constant", function() {
  fit <- lm(sbp ~ age + const_num, data = data)
  json_object(list(
    terms = model_terms_json(fit),
    n = json_number(nobs(fit))
  ))
})

run_model("linear_rare", function() {
  fit <- lm(sbp ~ age + rare_grp, data = data)
  json_object(list(
    terms = model_terms_json(fit),
    n = json_number(nobs(fit))
  ))
})

run_model("linear_mar", function() {
  fit <- lm(sbp ~ egfr + crp + age, data = data)
  fit_summary <- summary(fit)
  json_object(list(
    terms = model_terms_json(fit),
    r_squared = json_number(fit_summary$r.squared),
    n = json_number(nobs(fit))
  ))
})

run_model("linear_robust", function() {
  fit <- lm(sbp ~ age + bmi + arm, data = data)
  residual_df <- df.residual(fit)
  robust_tests <- lapply(c("HC0", "HC1", "HC3"), function(hc_type) {
    lmtest::coeftest(
      fit,
      vcov. = sandwich::vcovHC(fit, type = hc_type),
      df = residual_df
    )
  })
  names(robust_tests) <- c("hc0", "hc1", "hc3")
  ordinary <- coef(summary(fit))
  # The app's robust_se option is HC3 with the residual-df t, so HC3 is what
  # goes in the primary se/statistic/p fields; the ordinary and the other HC
  # types ride along as extras for reference.
  rows <- lapply(names(coef(fit)), function(term) {
    extras <- list(
      ordinary_se = json_number(ordinary[term, 2L]),
      ordinary_statistic = json_number(ordinary[term, 3L]),
      ordinary_p = json_number(ordinary[term, 4L])
    )
    for (hc_name in names(robust_tests)) {
      test <- robust_tests[[hc_name]]
      extras[[paste0(hc_name, "_se")]] <- json_number(test[term, 2L])
      extras[[paste0(hc_name, "_statistic")]] <- json_number(test[term, 3L])
      extras[[paste0(hc_name, "_p")]] <- json_number(test[term, 4L])
    }
    hc3 <- robust_tests[["hc3"]]
    term_json(
      term,
      coef(fit)[[term]],
      hc3[term, 2L],
      hc3[term, 3L],
      hc3[term, 4L],
      extras
    )
  })
  json_object(list(
    terms = json_array(rows),
    residual_df = json_number(residual_df)
  ))
})

run_model("polynomial", function() {
  fit <- lm(sbp ~ age + I(age^2) + arm, data = data)
  json_object(list(terms = model_terms_json(fit)))
})

run_model("logistic", function() {
  fit <- glm(event_binary ~ age + bmi + arm, data = data, family = binomial)
  json_object(list(
    terms = model_terms_json(fit),
    aic = json_number(AIC(fit)),
    bic = json_number(BIC(fit)),
    log_likelihood = json_number(logLik(fit)),
    n = json_number(nobs(fit))
  ))
})

run_model("logistic_separated", function() {
  warning_messages <- character()
  fit <- withCallingHandlers(
    glm(sep_binary ~ prior_tx, data = data, family = binomial),
    warning = function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  json_object(list(
    terms = model_terms_json(fit),
    iterations = json_number(fit$iter),
    warning_raised = json_boolean(length(warning_messages) > 0L),
    warnings = json_array(lapply(warning_messages, json_string))
  ))
})

logistf_terms_json <- function(fit) {
  estimates <- fit$coefficients
  standard_errors <- sqrt(diag(fit$var))
  # logistf returns prob / ci.lower / ci.upper as UNNAMED vectors in
  # coefficient order, so index them by position. Looking them up by name
  # is a subscript-out-of-bounds error, not a missing value.
  terms <- names(estimates)
  rows <- lapply(seq_along(terms), function(i) {
    estimate <- unname(estimates[i])
    se <- unname(standard_errors[i])
    term_json(
      terms[i],
      estimate,
      se,
      estimate / se,
      unname(fit$prob[i]),
      list(
        ci_low = json_number(unname(fit$ci.lower[i])),
        ci_high = json_number(unname(fit$ci.upper[i]))
      )
    )
  })
  json_array(rows)
}

run_model("firth_logistic", function() {
  fit <- logistf::logistf(event_binary ~ age + arm, data = data)
  json_object(list(terms = logistf_terms_json(fit)))
})

run_model("firth_separated", function() {
  fit <- logistf::logistf(sep_binary ~ prior_tx, data = data)
  json_object(list(terms = logistf_terms_json(fit)))
})

run_model("poisson", function() {
  fit <- glm(admissions ~ age + arm, data = data, family = poisson)
  json_object(list(terms = model_terms_json(fit)))
})

run_model("gamma", function() {
  fit <- glm(cost ~ age + arm, data = data, family = Gamma(link = "log"))
  fit_summary <- summary(fit)
  json_object(list(
    terms = model_terms_json(fit),
    dispersion = json_number(fit_summary$dispersion),
    residual_df = json_number(df.residual(fit)),
    aic = json_number(AIC(fit))
  ))
})

run_model("negbinom", function() {
  fit <- MASS::glm.nb(visits ~ age + arm, data = data)
  json_object(list(
    terms = model_terms_json(fit),
    theta = json_number(fit$theta),
    se_theta = json_number(fit$SE.theta),
    aic = json_number(AIC(fit))
  ))
})

run_model("ordinal", function() {
  ordinal_data <- data
  ordinal_data$g <- factor(
    ordinal_data$grade,
    levels = c("mild", "moderate", "severe"),
    ordered = TRUE
  )
  fit <- MASS::polr(g ~ age + arm, data = ordinal_data, Hess = TRUE)
  coefficient_matrix <- coef(summary(fit))
  beta_names <- names(coef(fit))
  threshold_names <- names(fit$zeta)
  json_object(list(
    terms = matrix_terms_json(coefficient_matrix, beta_names),
    thresholds = matrix_terms_json(coefficient_matrix, threshold_names)
  ))
})

run_model("ordinal_clm", function() {
  ordinal_data <- data
  ordinal_data$g <- factor(
    ordinal_data$grade,
    levels = c("mild", "moderate", "severe"),
    ordered = TRUE
  )
  fit <- ordinal::clm(g ~ age + arm, data = ordinal_data)
  coefficient_matrix <- coef(summary(fit))
  json_object(list(
    terms = matrix_terms_json(coefficient_matrix, names(fit$beta)),
    thresholds = matrix_terms_json(coefficient_matrix, names(fit$alpha))
  ))
})

gee_terms_json <- function(fit) {
  coefficient_matrix <- summary(fit)$coefficients
  matrix_terms_json(coefficient_matrix)
}

run_model("gee", function() {
  selected <- c("score", "visit", "arm", "age", "pid")
  gee_data <- long[complete.cases(long[, selected]), selected, drop = FALSE]
  gee_data <- gee_data[order(gee_data$pid), , drop = FALSE]
  fit <- geepack::geeglm(
    score ~ visit + arm + age,
    id = pid,
    data = gee_data,
    family = gaussian,
    corstr = "exchangeable"
  )
  json_object(list(
    terms = gee_terms_json(fit),
    alpha = json_number(fit$geese$alpha),
    scale = json_number(fit$geese$gamma)
  ))
})

run_model("gee_binomial", function() {
  selected <- c("resp", "visit", "arm", "pid")
  gee_data <- long[complete.cases(long[, selected]), selected, drop = FALSE]
  gee_data <- gee_data[order(gee_data$pid), , drop = FALSE]
  fit <- geepack::geeglm(
    resp ~ visit + arm,
    id = pid,
    data = gee_data,
    family = binomial,
    corstr = "exchangeable"
  )
  json_object(list(
    terms = gee_terms_json(fit),
    alpha = json_number(fit$geese$alpha),
    scale = json_number(fit$geese$gamma)
  ))
})

run_model("lmm", function() {
  fit <- lme4::lmer(
    score ~ visit + arm + age + (1 | pid),
    data = long,
    REML = TRUE
  )
  estimates <- lme4::fixef(fit)
  # `diag(as.matrix(vcov(fit)))` on lme4's sparse vcov raises "long vectors
  # not supported"; the fixed-effect table already carries the SEs.
  fixed_table <- summary(fit)$coefficients
  standard_errors <- fixed_table[, "Std. Error"]
  t_values <- fixed_table[, "t value"]
  rows <- lapply(names(estimates), function(term) {
    term_json(
      term,
      estimates[[term]],
      standard_errors[[term]],
      t_values[[term]],
      NA_real_
    )
  })
  variance_components <- as.data.frame(lme4::VarCorr(fit))
  pid_variance <- variance_components$vcov[
    variance_components$grp == "pid" & is.na(variance_components$var2)
  ][1L]
  residual_variance <- variance_components$vcov[
    variance_components$grp == "Residual"
  ][1L]
  json_object(list(
    terms = json_array(rows),
    pid_variance = json_number(pid_variance),
    residual_variance = json_number(residual_variance),
    reml_criterion = json_number(lme4::REMLcrit(fit))
  ))
})

cat(
  json_object(list(models = json_object(models))),
  "\n",
  file = output_path,
  sep = ""
)
