# ==============================================================================
# Distribution Fitting: AIC, BIC, KS Test & Diagnostic Plots
# SA Credit Dataset Simulation — N = 100,000
# ==============================================================================
#
# Fits six candidate distributions to the simulated loan duration and credit
# amount columns by maximum likelihood, computes AIC, BIC, and the
# Kolmogorov-Smirnov test for each, and produces three diagnostic plots per
# variable (PDF overlay, Q-Q plot, CDF overlay).
#
# Requires the simulated dataset produced by sa_simulation.R to already exist.
#
# Install required packages (run once):
#   install.packages(c("readr", "MASS", "fitdistrplus", "actuar", "ggplot2",
#                      "gridExtra", "scales"))
#
# Usage:
#   source("distribution_fitting.R")
#   -> prints AIC/BIC/KS tables to the console
#   -> writes six PNG files to the working directory
# ==============================================================================

library(readr)
library(MASS)           # fitdistr()
library(fitdistrplus)   # fitdist(), gofstat()
library(actuar)         # burr distribution
library(ggplot2)
library(gridExtra)
library(scales)

# ------------------------------------------------------------------------------
# 0. LOAD DATA
# ------------------------------------------------------------------------------
# Update this path to match your sa_sims.csv location.
DATA_PATH <- "C:/Users/23108/Downloads/sa_sims.csv"

df <- read_csv(DATA_PATH, show_col_types = FALSE)
cat(sprintf("Loaded: %d rows x %d columns\n", nrow(df), ncol(df)))

duration      <- df$Duration
credit_amount <- df$`Credit amount`

cat(sprintf("\nDuration      — mean: %.2f  median: %.2f  sd: %.2f  min: %d  max: %d\n",
            mean(duration), median(duration), sd(duration),
            min(duration), max(duration)))
cat(sprintf("Credit amount — mean: R%.0f  median: R%.0f  sd: R%.0f  min: R%d  max: R%d\n",
            mean(credit_amount), median(credit_amount), sd(credit_amount),
            min(credit_amount), max(credit_amount)))

# ------------------------------------------------------------------------------
# 1. DISTRIBUTION FITTING FUNCTION
# ------------------------------------------------------------------------------
# Fits a named list of distributions to a numeric vector, computes AIC, BIC,
# and the two-sided KS test statistic and p-value for each.
# Returns a data.frame ranked by AIC.
#
# dist_list: named list of lists, each with:
#   $dist  — distribution name string accepted by fitdist()
#   $start — named list of starting values for optim (NULL for auto-start)
#   $fixed — named list of fixed parameters (e.g. list(loc = 0))
# ------------------------------------------------------------------------------
fit_distributions <- function(data, dist_list, label) {

  results <- lapply(names(dist_list), function(name) {
    spec  <- dist_list[[name]]
    tryCatch({
      # Fit by MLE
      if (is.null(spec$start)) {
        fit <- fitdist(data, distr = spec$dist, fix.arg = spec$fixed,
                       method = "mle", optim.method = "L-BFGS-B",
                       lower = 1e-6)
      } else {
        fit <- fitdist(data, distr = spec$dist, fix.arg = spec$fixed,
                       start = spec$start, method = "mle",
                       optim.method = "L-BFGS-B", lower = 1e-6)
      }

      # AIC and BIC (fitdist computes these internally)
      aic_val <- fit$aic
      bic_val <- fit$bic

      # KS test: compare empirical CDF against the fitted theoretical CDF
      # Build the CDF function from the fitted parameters
      params     <- as.list(fit$estimate)
      fixed_args <- spec$fixed
      all_args   <- c(params, fixed_args)

      ks_result <- ks.test(
        data,
        spec$cdf_fn,
        all_args
      )

      data.frame(
        Distribution = name,
        AIC          = aic_val,
        BIC          = bic_val,
        KS_stat      = ks_result$statistic,
        KS_p         = ks_result$p.value,
        stringsAsFactors = FALSE,
        fit_obj      = I(list(fit)),
        params_list  = I(list(all_args))
      )
    }, error = function(e) {
      message(sprintf("  [%s] %s FAILED: %s", label, name, e$message))
      NULL
    })
  })

  # Drop failures and combine
  results <- do.call(rbind, Filter(Negate(is.null), results))
  results <- results[order(results$AIC), ]
  results$Rank  <- seq_len(nrow(results))
  best_aic      <- results$AIC[1]
  best_bic      <- results$BIC[1]
  results$dAIC  <- results$AIC - best_aic
  results$dBIC  <- results$BIC - best_bic
  results
}

# CDF wrapper functions for ks.test()
# ks.test() needs a function of the form f(x) returning P(X <= x)
make_cdf <- function(pfun) {
  function(x, args) do.call(pfun, c(list(x), args))
}

cdf_lnorm  <- function(x, args) plnorm(x,  meanlog = args$meanlog, sdlog   = args$sdlog)
cdf_gamma  <- function(x, args) pgamma(x,  shape   = args$shape,   rate    = args$rate)
cdf_weib   <- function(x, args) pweibull(x,shape   = args$shape,   scale   = args$scale)
cdf_exp    <- function(x, args) pexp(x,    rate    = args$rate)
cdf_burr   <- function(x, args) pburr(x,   shape1  = args$shape1,  shape2  = args$shape2,
                                           scale   = args$scale)

# ------------------------------------------------------------------------------
# 2. DEFINE DISTRIBUTION LIST
# ------------------------------------------------------------------------------
# Each entry: dist name (fitdistrplus), starting values, fixed args, CDF fn.
# loc = 0 is fixed throughout (data is strictly positive; no shift needed).
# ------------------------------------------------------------------------------

make_dist_list <- function() {
  list(
    "Log-Normal" = list(
      dist    = "lnorm",
      start   = NULL,
      fixed   = list(),
      cdf_fn  = cdf_lnorm
    ),
    "Gamma" = list(
      dist    = "gamma",
      start   = list(shape = 2, rate = 0.1),
      fixed   = list(),
      cdf_fn  = cdf_gamma
    ),
    "Weibull" = list(
      dist    = "weibull",
      start   = list(shape = 1.5, scale = 20),
      fixed   = list(),
      cdf_fn  = cdf_weib
    ),
    "Exponential" = list(
      dist    = "exp",
      start   = NULL,
      fixed   = list(),
      cdf_fn  = cdf_exp
    ),
    "Burr (XII)" = list(
      dist    = "burr",
      start   = list(shape1 = 2, shape2 = 1, scale = 10),
      fixed   = list(),
      cdf_fn  = cdf_burr
    )
  )
}

# ------------------------------------------------------------------------------
# 3. SIMPLER APPROACH: use MASS::fitdistr + manual AIC/BIC/KS
# ------------------------------------------------------------------------------
# fitdistrplus with actuar::burr can be finicky on large N. The approach below
# uses fitdistr() for the four standard distributions and a direct log-normal
# fit, then computes AIC/BIC manually from the log-likelihood.
# ------------------------------------------------------------------------------

fit_and_rank <- function(data, label, x_label, x_unit = "") {

  n <- length(data)
  rows <- list()

  # ── Log-Normal ──────────────────────────────────────────────────────────────
  tryCatch({
    fit    <- fitdistr(data, "lognormal")
    params <- fit$estimate                          # meanlog, sdlog
    ll     <- fit$loglik
    k      <- length(params)
    ks     <- ks.test(data, "plnorm",
                      meanlog = params["meanlog"], sdlog = params["sdlog"])
    rows[["Log-Normal"]] <- list(
      dist = "Log-Normal", params = params, ll = ll,
      k = k, ks_stat = ks$statistic, ks_p = ks$p.value,
      pdf_fn  = function(x, p) dlnorm(x, p["meanlog"], p["sdlog"]),
      cdf_fn  = function(x, p) plnorm(x, p["meanlog"], p["sdlog"]),
      ppf_fn  = function(p2, p) qlnorm(p2, p["meanlog"], p["sdlog"])
    )
  }, error = function(e) message("Log-Normal failed: ", e$message))

  # ── Gamma ────────────────────────────────────────────────────────────────────
  tryCatch({
    fit    <- fitdistr(data, "gamma")
    params <- fit$estimate                          # shape, rate
    ll     <- fit$loglik
    k      <- length(params)
    ks     <- ks.test(data, "pgamma",
                      shape = params["shape"], rate = params["rate"])
    rows[["Gamma"]] <- list(
      dist = "Gamma", params = params, ll = ll,
      k = k, ks_stat = ks$statistic, ks_p = ks$p.value,
      pdf_fn  = function(x, p) dgamma(x, p["shape"], p["rate"]),
      cdf_fn  = function(x, p) pgamma(x, p["shape"], p["rate"]),
      ppf_fn  = function(p2, p) qgamma(p2, p["shape"], p["rate"])
    )
  }, error = function(e) message("Gamma failed: ", e$message))

  # ── Weibull ──────────────────────────────────────────────────────────────────
  tryCatch({
    fit    <- fitdistr(data, "weibull")
    params <- fit$estimate                          # shape, scale
    ll     <- fit$loglik
    k      <- length(params)
    ks     <- ks.test(data, "pweibull",
                      shape = params["shape"], scale = params["scale"])
    rows[["Weibull"]] <- list(
      dist = "Weibull", params = params, ll = ll,
      k = k, ks_stat = ks$statistic, ks_p = ks$p.value,
      pdf_fn  = function(x, p) dweibull(x, p["shape"], p["scale"]),
      cdf_fn  = function(x, p) pweibull(x, p["shape"], p["scale"]),
      ppf_fn  = function(p2, p) qweibull(p2, p["shape"], p["scale"])
    )
  }, error = function(e) message("Weibull failed: ", e$message))

  # ── Exponential ──────────────────────────────────────────────────────────────
  tryCatch({
    fit    <- fitdistr(data, "exponential")
    params <- fit$estimate                          # rate
    ll     <- fit$loglik
    k      <- length(params)
    ks     <- ks.test(data, "pexp", rate = params["rate"])
    rows[["Exponential"]] <- list(
      dist = "Exponential", params = params, ll = ll,
      k = k, ks_stat = ks$statistic, ks_p = ks$p.value,
      pdf_fn  = function(x, p) dexp(x, p["rate"]),
      cdf_fn  = function(x, p) pexp(x, p["rate"]),
      ppf_fn  = function(p2, p) qexp(p2, p["rate"])
    )
  }, error = function(e) message("Exponential failed: ", e$message))

  # ── Burr (XII) via actuar ────────────────────────────────────────────────────
  tryCatch({
    library(actuar)
    fit  <- fitdist(data, "burr", start = list(shape1 = 2, shape2 = 1, scale = median(data)),
                    method = "mle", optim.method = "Nelder-Mead")
    params <- fit$estimate                          # shape1, shape2, scale
    ll     <- -fit$aic/2 + length(params)           # recover ll from AIC
    ll     <- logLik(fit)[1]
    k      <- length(params)
    ks     <- ks.test(data, "pburr",
                      shape1 = params["shape1"],
                      shape2 = params["shape2"],
                      scale  = params["scale"])
    rows[["Burr (XII)"]] <- list(
      dist = "Burr (XII)", params = params, ll = ll,
      k = k, ks_stat = ks$statistic, ks_p = ks$p.value,
      pdf_fn  = function(x, p) dburr(x, p["shape1"], p["shape2"], p["scale"]),
      cdf_fn  = function(x, p) pburr(x, p["shape1"], p["shape2"], p["scale"]),
      ppf_fn  = function(p2, p) qburr(p2, p["shape1"], p["shape2"], p["scale"])
    )
  }, error = function(e) message("Burr (XII) failed: ", e$message))

  # ── Compute AIC and BIC ──────────────────────────────────────────────────────
  # AIC = 2k - 2 * log-likelihood
  # BIC = k * log(n) - 2 * log-likelihood
  results <- lapply(names(rows), function(nm) {
    r   <- rows[[nm]]
    aic <- 2 * r$k - 2 * r$ll
    bic <- r$k * log(n) - 2 * r$ll
    data.frame(
      Distribution = r$dist,
      AIC          = aic,
      BIC          = bic,
      KS_stat      = r$ks_stat,
      KS_p         = r$ks_p,
      stringsAsFactors = FALSE
    )
  })

  df_res <- do.call(rbind, results)
  df_res <- df_res[order(df_res$AIC), ]
  df_res$Rank <- seq_len(nrow(df_res))
  best_aic    <- df_res$AIC[1]
  best_bic    <- df_res$BIC[1]
  df_res$dAIC <- df_res$AIC - best_aic
  df_res$dBIC <- df_res$BIC - best_bic
  df_res      <- df_res[, c("Rank","Distribution","AIC","dAIC","BIC","dBIC","KS_stat","KS_p")]

  # Print table
  cat(sprintf("\n%s\n%s\n", strrep("=", 80), label))
  cat(sprintf("%-14s %14s %10s %14s %10s %8s %8s\n",
              "Distribution","AIC","dAIC","BIC","dBIC","KS stat","KS p"))
  cat(strrep("-", 80), "\n")
  for (i in seq_len(nrow(df_res))) {
    r <- df_res[i, ]
    cat(sprintf("%-14s %14.2f %10.2f %14.2f %10.2f %8.4f %8.4f%s\n",
                r$Distribution, r$AIC, r$dAIC, r$BIC, r$dBIC,
                r$KS_stat, r$KS_p,
                ifelse(r$KS_p > 0.05, " *", "")))
  }
  cat("* = not rejected by KS test at alpha = 0.05\n")

  list(table = df_res, fits = rows, data = data,
       label = label, x_label = x_label, x_unit = x_unit, n = n)
}

# ------------------------------------------------------------------------------
# 4. RUN FITTING
# ------------------------------------------------------------------------------
cat("\nFitting distributions — this may take 30-60 seconds at N = 100,000...\n")
dur_results  <- fit_and_rank(duration,      "DURATION (months)",     "Duration (months)")
cred_results <- fit_and_rank(credit_amount, "CREDIT AMOUNT (ZAR)",   "Credit amount", "ZAR")

# ------------------------------------------------------------------------------
# 5. PLOT FUNCTION
# ------------------------------------------------------------------------------
# Produces three plots per variable and saves as a combined PNG:
#   (a) Histogram + PDF overlay (top 3 distributions)
#   (b) Q-Q plot for the best-fitting distribution
#   (c) Empirical CDF + fitted CDF overlay (top 3 distributions)
# ------------------------------------------------------------------------------

COLOURS <- c("Log-Normal"  = "#d62728",
             "Burr (XII)"  = "#2ca02c",
             "Gamma"       = "#1f77b4",
             "Weibull"     = "#9467bd",
             "Exponential" = "#8c564b")

LINETYPES <- c("Log-Normal"  = "solid",
               "Burr (XII)"  = "dashed",
               "Gamma"       = "dotted",
               "Weibull"     = "dotdash",
               "Exponential" = "longdash")

plot_diagnostics <- function(res, output_file, top_n = 3) {

  data    <- res$data
  fits    <- res$fits
  df_tbl  <- res$table
  x_label <- res$x_label
  n       <- res$n

  # Top 3 distributions by AIC
  top3_names <- df_tbl$Distribution[seq_len(min(top_n, nrow(df_tbl)))]
  best_name  <- top3_names[1]

  # x-axis grid for PDF/CDF
  x_lo <- quantile(data, 0.001)
  x_hi <- quantile(data, 0.995)
  x_grid <- seq(x_lo, x_hi, length.out = 1000)

  # ── (a) Histogram + PDF overlay ──────────────────────────────────────────────
  pdf_df <- do.call(rbind, lapply(top3_names, function(nm) {
    r <- fits[[nm]]
    data.frame(x = x_grid,
               y = r$pdf_fn(x_grid, r$params),
               Distribution = nm,
               stringsAsFactors = FALSE)
  }))

  p_pdf <- ggplot() +
    geom_histogram(aes(x = data, y = after_stat(density)),
                   bins = 60, fill = "steelblue", alpha = 0.5, colour = NA) +
    geom_line(data = pdf_df,
              aes(x = x, y = y, colour = Distribution, linetype = Distribution),
              linewidth = 0.9) +
    scale_colour_manual(values = COLOURS) +
    scale_linetype_manual(values = LINETYPES) +
    labs(title = paste0(x_label, ": PDF overlay"),
         x = x_label, y = "Density",
         colour = NULL, linetype = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom",
          legend.key.width = unit(1.2, "cm"),
          plot.title = element_text(size = 10, face = "bold"))

  # ── (b) Q-Q plot for best-fitting distribution ───────────────────────────────
  # Theoretical quantiles from fitted distribution vs empirical quantiles
  # Subsample to 3000 points for plotting speed (N=100k would overplot)
  set.seed(1)
  idx      <- sort(sample(seq_len(n), min(3000, n)))
  emp_sort <- sort(data)[idx]
  probs    <- (idx) / (n + 1)   # plotting positions (Hazen formula)

  best_fit <- fits[[best_name]]
  theo     <- best_fit$ppf_fn(probs, best_fit$params)

  qq_df <- data.frame(theoretical = theo, empirical = emp_sort)

  # 45-degree reference line
  ref_lo <- min(theo[is.finite(theo)], emp_sort)
  ref_hi <- max(quantile(theo[is.finite(theo)], 0.99), quantile(emp_sort, 0.99))

  p_qq <- ggplot(qq_df, aes(x = theoretical, y = empirical)) +
    geom_point(alpha = 0.25, size = 0.8, colour = "steelblue") +
    geom_abline(slope = 1, intercept = 0, colour = "#d62728",
                linetype = "dashed", linewidth = 1) +
    coord_cartesian(xlim = c(ref_lo, ref_hi), ylim = c(ref_lo, ref_hi)) +
    labs(title = paste0(x_label, ": Q-Q (", best_name, ")"),
         x = "Theoretical quantiles", y = "Empirical quantiles") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(size = 10, face = "bold"))

  # ── (c) Empirical CDF + fitted CDF overlay ───────────────────────────────────
  # Empirical CDF: step function
  ecdf_x  <- sort(data)
  ecdf_y  <- seq_len(n) / n

  # Subsample ECDF for plotting (every 50th point is enough at N=100k)
  sub_idx  <- seq(1, n, by = 50)
  ecdf_sub <- data.frame(x = ecdf_x[sub_idx], y = ecdf_y[sub_idx],
                          Distribution = "Empirical CDF")

  cdf_df <- do.call(rbind, lapply(top3_names, function(nm) {
    r <- fits[[nm]]
    data.frame(x = x_grid,
               y = r$cdf_fn(x_grid, r$params),
               Distribution = nm,
               stringsAsFactors = FALSE)
  }))

  # Combine colours/linetypes for legend
  all_colours   <- c("Empirical CDF" = "steelblue",     COLOURS)
  all_linetypes <- c("Empirical CDF" = "solid",          LINETYPES)
  all_sizes     <- c("Empirical CDF" = 0.6,
                     setNames(rep(0.9, length(top3_names)), top3_names))

  p_cdf <- ggplot() +
    geom_line(data = ecdf_sub,
              aes(x = x, y = y, colour = Distribution, linetype = Distribution),
              linewidth = 0.7) +
    geom_line(data = cdf_df,
              aes(x = x, y = y, colour = Distribution, linetype = Distribution),
              linewidth = 0.9) +
    scale_colour_manual(values   = all_colours,   breaks = c("Empirical CDF", top3_names)) +
    scale_linetype_manual(values = all_linetypes,  breaks = c("Empirical CDF", top3_names)) +
    labs(title = paste0(x_label, ": CDF overlay"),
         x = x_label, y = "Cumulative probability",
         colour = NULL, linetype = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom",
          legend.key.width = unit(1.2, "cm"),
          plot.title = element_text(size = 10, face = "bold"))

  # ── Combine and save ──────────────────────────────────────────────────────────
  combined <- arrangeGrob(p_pdf, p_qq, p_cdf, ncol = 3,
    top = grid::textGrob(
      paste0("Distribution Fitting Diagnostics — ", x_label,
             "  (N = ", format(n, big.mark = ","), ")"),
      gp = grid::gpar(fontsize = 12, fontface = "bold")
    )
  )

  ggsave(output_file, combined, width = 15, height = 5.5, dpi = 150, bg = "white")
  message(sprintf("Saved: %s", output_file))
}

# ── Combined 2-row figure (both variables) ────────────────────────────────────
plot_combined <- function(dur_res, cred_res, output_file, top_n = 3) {

  make_plots <- function(res) {
    data       <- res$data
    fits       <- res$fits
    df_tbl     <- res$table
    x_label    <- res$x_label
    n          <- res$n
    top3_names <- df_tbl$Distribution[seq_len(min(top_n, nrow(df_tbl)))]
    best_name  <- top3_names[1]

    x_lo   <- quantile(data, 0.001)
    x_hi   <- quantile(data, 0.995)
    x_grid <- seq(x_lo, x_hi, length.out = 1000)

    # PDF
    pdf_df <- do.call(rbind, lapply(top3_names, function(nm) {
      r <- fits[[nm]]
      data.frame(x = x_grid, y = r$pdf_fn(x_grid, r$params),
                 Distribution = nm, stringsAsFactors = FALSE)
    }))
    p1 <- ggplot() +
      geom_histogram(aes(x = data, y = after_stat(density)),
                     bins = 60, fill = "steelblue", alpha = 0.5, colour = NA) +
      geom_line(data = pdf_df,
                aes(x = x, y = y, colour = Distribution, linetype = Distribution),
                linewidth = 0.9) +
      scale_colour_manual(values = COLOURS) +
      scale_linetype_manual(values = LINETYPES) +
      labs(title = paste0(x_label, ": PDF overlay"),
           x = x_label, y = "Density", colour = NULL, linetype = NULL) +
      theme_bw(base_size = 9) +
      theme(legend.position = "bottom", legend.key.width = unit(1, "cm"),
            plot.title = element_text(size = 9, face = "bold"))

    # Q-Q
    set.seed(1)
    idx      <- sort(sample(seq_len(n), min(3000, n)))
    emp_sort <- sort(data)[idx]
    probs    <- idx / (n + 1)
    best_fit <- fits[[best_name]]
    theo     <- best_fit$ppf_fn(probs, best_fit$params)
    ref_hi   <- max(quantile(theo[is.finite(theo)], 0.99), quantile(emp_sort, 0.99))
    ref_lo   <- min(theo[is.finite(theo)], emp_sort)
    qq_df    <- data.frame(theoretical = theo, empirical = emp_sort)
    p2 <- ggplot(qq_df, aes(x = theoretical, y = empirical)) +
      geom_point(alpha = 0.2, size = 0.6, colour = "steelblue") +
      geom_abline(slope = 1, intercept = 0, colour = "#d62728",
                  linetype = "dashed", linewidth = 1) +
      coord_cartesian(xlim = c(ref_lo, ref_hi), ylim = c(ref_lo, ref_hi)) +
      labs(title = paste0(x_label, ": Q-Q (", best_name, ")"),
           x = "Theoretical quantiles", y = "Empirical quantiles") +
      theme_bw(base_size = 9) +
      theme(plot.title = element_text(size = 9, face = "bold"))

    # CDF
    ecdf_x  <- sort(data)
    ecdf_y  <- seq_len(n) / n
    sub_idx <- seq(1, n, by = 50)
    ecdf_sub <- data.frame(x = ecdf_x[sub_idx], y = ecdf_y[sub_idx],
                            Distribution = "Empirical CDF")
    cdf_df <- do.call(rbind, lapply(top3_names, function(nm) {
      r <- fits[[nm]]
      data.frame(x = x_grid, y = r$cdf_fn(x_grid, r$params),
                 Distribution = nm, stringsAsFactors = FALSE)
    }))
    all_colours   <- c("Empirical CDF" = "steelblue", COLOURS)
    all_linetypes <- c("Empirical CDF" = "solid",      LINETYPES)
    p3 <- ggplot() +
      geom_line(data = ecdf_sub,
                aes(x = x, y = y, colour = Distribution, linetype = Distribution),
                linewidth = 0.6) +
      geom_line(data = cdf_df,
                aes(x = x, y = y, colour = Distribution, linetype = Distribution),
                linewidth = 0.9) +
      scale_colour_manual(values   = all_colours,
                          breaks   = c("Empirical CDF", top3_names)) +
      scale_linetype_manual(values = all_linetypes,
                            breaks = c("Empirical CDF", top3_names)) +
      labs(title = paste0(x_label, ": CDF overlay"),
           x = x_label, y = "Cumulative probability",
           colour = NULL, linetype = NULL) +
      theme_bw(base_size = 9) +
      theme(legend.position = "bottom", legend.key.width = unit(1, "cm"),
            plot.title = element_text(size = 9, face = "bold"))

    list(p1, p2, p3)
  }

  dur_plots  <- make_plots(dur_res)
  cred_plots <- make_plots(cred_res)

  combined <- arrangeGrob(
    grobs  = c(dur_plots, cred_plots),
    ncol   = 3, nrow = 2,
    top    = grid::textGrob(
      "Distribution Fitting — SA Simulated Data (N = 100,000)\nLog-Normal vs Gamma vs Weibull",
      gp = grid::gpar(fontsize = 13, fontface = "bold")
    )
  )
  ggsave(output_file, combined, width = 16, height = 10, dpi = 150, bg = "white")
  message(sprintf("Saved: %s", output_file))
}

# ------------------------------------------------------------------------------
# 6. GENERATE PLOTS
# ------------------------------------------------------------------------------
cat("\nGenerating plots...\n")

# Individual plots per variable
plot_diagnostics(dur_results,  "plot_duration_diagnostics.png")
plot_diagnostics(cred_results, "plot_credit_amount_diagnostics.png")

# Combined 2-row figure (matches the Word document Figure 1)
plot_combined(dur_results, cred_results, "plot_combined_diagnostics.png")

# ------------------------------------------------------------------------------
# 7. SUMMARY TABLE (formatted for copy-paste into thesis)
# ------------------------------------------------------------------------------
cat("\n", strrep("=", 80), "\n", sep = "")
cat("FINAL AIC / BIC / KS COMPARISON TABLE\n")
cat(strrep("=", 80), "\n", sep = "")
cat(sprintf("%-16s %14s %10s %14s %10s %8s %8s\n",
            "", "Duration", "", "Credit amount", "", "", ""))
cat(sprintf("%-16s %14s %10s %14s %10s %8s %8s\n",
            "Distribution", "AIC", "ΔAIC", "AIC", "ΔAIC", "KS p (D)", "KS p (C)"))
cat(strrep("-", 80), "\n")

dur_tbl  <- dur_results$table
cred_tbl <- cred_results$table
all_dists <- union(dur_tbl$Distribution, cred_tbl$Distribution)

for (d in all_dists) {
  dr <- dur_tbl[dur_tbl$Distribution == d, ]
  cr <- cred_tbl[cred_tbl$Distribution == d, ]
  d_aic  <- if (nrow(dr) > 0) sprintf("%14.2f", dr$AIC)  else sprintf("%14s", "—")
  d_daic <- if (nrow(dr) > 0) sprintf("%10.2f", dr$dAIC) else sprintf("%10s", "—")
  c_aic  <- if (nrow(cr) > 0) sprintf("%14.2f", cr$AIC)  else sprintf("%14s", "—")
  c_daic <- if (nrow(cr) > 0) sprintf("%10.2f", cr$dAIC) else sprintf("%10s", "—")
  d_ksp  <- if (nrow(dr) > 0) sprintf("%8.4f", dr$KS_p)  else sprintf("%8s", "—")
  c_ksp  <- if (nrow(cr) > 0) sprintf("%8.4f", cr$KS_p)  else sprintf("%8s", "—")
  cat(sprintf("%-16s %s %s %s %s %s %s\n", d, d_aic, d_daic, c_aic, c_daic, d_ksp, c_ksp))
}
cat(strrep("-", 80), "\n")
cat("KS p > 0.05 = distribution not rejected at alpha = 0.05\n")
cat("D = Duration column, C = Credit amount column\n")

cat("\nDone. Output files written to working directory:\n")
cat("  plot_duration_diagnostics.png\n")
cat("  plot_credit_amount_diagnostics.png\n")
cat("  plot_combined_diagnostics.png\n")
