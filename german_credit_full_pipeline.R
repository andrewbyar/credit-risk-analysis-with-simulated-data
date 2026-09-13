# ==============================================================================
# German Credit Risk Model Comparison — Full Pipeline (PIPELINE VALIDATION)
# ==============================================================================
#
# This is the German-data version of the pipeline, used to validate that the
# modelling code produces sensible results (expected AUC ~0.75-0.80) on data
# with known predictive signal, before interpreting results on the SA simulated
# dataset.
#
# The Kaggle "german_credit_data" CSV has the 9 friendly feature columns but no
# target; the good/bad label is recovered from the original UCI Statlog file
# (rows in identical order, verified by cross-checking amount/duration/age).
#
# Fits 8 models to the German credit dataset:
#
#   Classical:   1. Logistic Regression (LR)
#                2. Linear Discriminant Analysis (LDA)
#   Machine Learning:
#                3. Random Forest (RF)
#                4. XGBoost
#                5. Multilayer Perceptron (MLP)
#   Ensemble:    6. Stacked Ensemble (LR meta-learner over RF + XGBoost + MLP)
#   Hybrid:      7. LDA-XGBoost Fusion (LDA discriminant score as XGBoost feature)
#                8. LR-XGBoost Fusion  (LR probability score as XGBoost feature)
#
# Pipeline (matching Mathibela & Maposa, 2026):
#   1. Load and preprocess (one-hot encode, Z-score scale numerics)
#   2. Stratified 70/30 train/test split
#   3. SMOTE on training data only
#   4. BASE models: fit on SMOTE-balanced training data
#   5. COST-SENSITIVE: class weights w_j = N / (2 * N_j), pre-SMOTE counts
#   6. INTEGRATED: cost-sensitive + Youden's J threshold optimisation
#   7. Confusion matrices for all three stages
#   8. 95% bootstrap confidence intervals (1000 resamples, Stacked Ensemble)
#   9. McNemar's test: Stacked Ensemble vs Random Forest
#  10. Friedman's test + Nemenyi post-hoc across all 8 models
#  11. SHAP feature importance (Random Forest and both hybrids)
#
# Install required packages (run once):
#   install.packages(c(
#     "readr", "dplyr", "caret", "MASS", "randomForest", "xgboost",
#     "nnet", "smotefamily", "ROSE", "pROC", "ggplot2", "gridExtra",
#     "shapr", "SHAPforxgboost", "scales", "tidyr", "tibble",
#     "scikit-posthocs"   # not on CRAN — use NSM3 instead
#   ))
#   install.packages("NSM3")   # for Nemenyi post-hoc
#
# Usage:
#   source("sa_credit_full_pipeline.R")
#   -> prints all result tables to console
#   -> saves SHAP plots and ROC curves as PNG files
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(caret)          # createDataPartition, confusionMatrix, trainControl
  library(MASS)           # lda()
  library(randomForest)   # randomForest()
  library(xgboost)        # xgboost()
  library(nnet)           # nnet() for MLP
  library(smotefamily)    # SMOTE()
  library(pROC)           # roc(), auc()
  library(ggplot2)
  library(gridExtra)
  library(scales)
  library(tidyr)
  library(tibble)
})

# ==============================================================================
# 0. SETTINGS
# ==============================================================================
set.seed(42)
DATA_PATH   <- "C:/Users/23108/Downloads/german_cred_data.csv"   # Kaggle German file
OUTPUT_DIR  <- "C:/Users/23108/Downloads/"              # where PNGs are saved
TRAIN_RATIO <- 0.70
N_BOOT      <- 1000     # bootstrap resamples for CI
ALPHA       <- 0.05     # significance level

# Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

cat("SA Credit Risk Model Comparison\n")
cat(strrep("=", 60), "\n\n")

# ==============================================================================
# 1. LOAD AND PREPROCESS
# ==============================================================================
# The Kaggle "german_credit_data" CSV has a row-index column and the 9 feature
# columns (Age, Sex, Job, Housing, Saving accounts, Checking account,
# Credit amount, Duration, Purpose) but NO target/Risk column. The good/bad
# label is recovered from the original UCI Statlog file, whose rows are in the
# same order. Row alignment is verified by cross-checking Credit amount,
# Duration, and Age before the labels are attached.
df_raw <- read_csv(DATA_PATH, show_col_types = FALSE)

# Drop the leading row-index column if present (read_csv names it "...1").
df_raw <- df_raw[, !names(df_raw) %in% c("...1", "")]
cat(sprintf("Loaded: %d rows x %d columns\n", nrow(df_raw), ncol(df_raw)))

# ── Attach the target from the Statlog source ────────────────────────────────
# Statlog german.data: 21 columns, no header, last column = 1 (good) / 2 (bad).
# Mirror used here is comma-separated. Update STATLOG_PATH to a local copy if
# you do not have internet access from R.
STATLOG_PATH <- "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"

statlog <- tryCatch(
  read.csv(STATLOG_PATH, header = FALSE),
  error = function(e) stop(paste(
    "Could not download Statlog labels. Download german.csv from",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv",
    "and set STATLOG_PATH to the local file path. Original error:", e$message))
)

if (nrow(statlog) != nrow(df_raw)) {
  stop(sprintf("Row count mismatch: Kaggle file has %d rows, Statlog has %d.",
               nrow(df_raw), nrow(statlog)))
}

# Statlog column positions: 2 = duration, 5 = credit amount, 13 = age, 21 = target
amt_match <- mean(df_raw$`Credit amount` == statlog[[5]])
dur_match <- mean(df_raw$Duration       == statlog[[2]])
age_match <- mean(df_raw$Age            == statlog[[13]])
cat(sprintf("Row-alignment check vs Statlog: amount %.0f%%, duration %.0f%%, age %.0f%%\n",
            amt_match*100, dur_match*100, age_match*100))

if (amt_match < 0.99 || dur_match < 0.99 || age_match < 0.99) {
  stop("Row order does NOT match between the Kaggle file and the Statlog file. ",
       "Cannot safely attach labels — the rows would be mislabelled.")
}

# Target: 1 = default (bad = Statlog code 2), 0 = no default (good = Statlog code 1)
df_raw$target <- ifelse(statlog[[21]] == 2, 1L, 0L)

# Identify numeric and categorical columns
NUMERIC_COLS <- c("Age", "Credit amount", "Duration")
CAT_COLS     <- c("Sex", "Job", "Housing", "Saving accounts",
                  "Checking account", "Purpose")

# "Saving accounts" and "Checking account" contain missing values (no formal
# account). In the German Kaggle file these are read as genuine R NA; in the SA
# file they are the literal string "NA". Either way, convert all categorical
# columns to factors BEFORE encoding and force any NA to become a real level
# named "NA", so model.matrix() does not silently drop those rows.
for (col in CAT_COLS) {
  df_raw[[col]] <- factor(df_raw[[col]], exclude = NULL)    # exclude=NULL keeps NA levels
  levels(df_raw[[col]])[is.na(levels(df_raw[[col]]))] <- "NA"
}

# One-hot encode using model.matrix with na.action = na.pass so no rows are dropped
dummy_formula <- as.formula(paste("~ ", paste(paste0("`", CAT_COLS, "`"),
                                               collapse = " + ")))
old_na_action <- options(na.action = "na.pass")   # preserve all rows
dummies <- model.matrix(dummy_formula, data = df_raw)[, -1]  # drop intercept only
options(old_na_action)                             # restore original na.action

cat(sprintf("Rows in dummies matrix: %d  (should match nrow(df_raw) = %d)\n",
            nrow(dummies), nrow(df_raw)))

# Combine numerics + dummies + target
df_model <- bind_cols(
  df_raw[, c(NUMERIC_COLS, "target")],
  as.data.frame(dummies)
)

# Clean column names (remove backticks, spaces, slashes)
names(df_model) <- make.names(names(df_model), unique = TRUE)

# Redefine NUMERIC_COLS using the cleaned names so the scaling step
# can find them. make.names() converts "Credit amount" -> "Credit.amount".
NUMERIC_COLS <- make.names(NUMERIC_COLS)   # "Age", "Credit.amount", "Duration"

TARGET_COL   <- "target"
FEATURE_COLS <- setdiff(names(df_model), TARGET_COL)

cat(sprintf("After encoding: %d features\n", length(FEATURE_COLS)))
cat(sprintf("Default rate:   %.1f%%\n\n", mean(df_model$target) * 100))

# ==============================================================================
# 2. STRATIFIED 70/30 TRAIN/TEST SPLIT
# ==============================================================================
set.seed(42)
train_idx <- createDataPartition(df_model[[TARGET_COL]], p = TRAIN_RATIO,
                                  list = FALSE)[, 1]

train_raw <- df_model[train_idx,  ]
test_raw  <- df_model[-train_idx, ]

# Z-score scale numeric features (fit on train, apply to both)
num_means <- colMeans(train_raw[, NUMERIC_COLS])
num_sds   <- apply(train_raw[, NUMERIC_COLS], 2, sd)

scale_numerics <- function(df, means, sds) {
  df[, NUMERIC_COLS] <- sweep(sweep(df[, NUMERIC_COLS], 2, means, "-"), 2, sds, "/")
  df
}

train_scaled <- scale_numerics(train_raw, num_means, num_sds)
test_scaled  <- scale_numerics(test_raw,  num_means, num_sds)

X_train <- as.matrix(train_scaled[, FEATURE_COLS])
y_train <- train_scaled[[TARGET_COL]]
X_test  <- as.matrix(test_scaled[, FEATURE_COLS])
y_test  <- test_scaled[[TARGET_COL]]

cat(sprintf("Train: %d rows  |  Test: %d rows\n",
            nrow(X_train), nrow(X_test)))
cat(sprintf("Train default rate: %.1f%%  |  Test default rate: %.1f%%\n\n",
            mean(y_train)*100, mean(y_test)*100))

# ==============================================================================
# 3. SMOTE (training data only)
# ==============================================================================
# smotefamily::SMOTE expects a data.frame with the target as the last column,
# coded as a factor. K = 5 nearest neighbours (default).
set.seed(42)
train_smote_input <- data.frame(X_train, target = as.factor(y_train))
smote_result      <- SMOTE(train_smote_input[, FEATURE_COLS],
                            train_smote_input$target, K = 5, dup_size = 0)

X_train_sm <- as.matrix(smote_result$data[, FEATURE_COLS])
y_train_sm <- as.integer(as.character(smote_result$data$class))

cat(sprintf("After SMOTE: %d rows  |  class counts: %s\n\n",
            length(y_train_sm),
            paste(table(y_train_sm), collapse = " / ")))

# ==============================================================================
# 4. COST-SENSITIVE WEIGHTS
# ==============================================================================
# w_j = N_train / (2 * N_j), computed from PRE-SMOTE training counts.
# Using post-SMOTE counts would give w=1 for both classes (already balanced).
n0_pre <- sum(y_train == 0)
n1_pre <- sum(y_train == 1)
N_pre  <- length(y_train)
w0 <- N_pre / (2 * n0_pre)
w1 <- N_pre / (2 * n1_pre)
sample_weights_sm <- ifelse(y_train_sm == 1, w1, w0)  # per-sample weights
scale_pos_weight  <- w1 / w0                            # XGBoost param

cat(sprintf("Pre-SMOTE counts: N0=%d, N1=%d\n", n0_pre, n1_pre))
cat(sprintf("Cost-sensitive weights: w0=%.3f, w1=%.3f  (XGB scale_pos_weight=%.3f)\n\n",
            w0, w1, scale_pos_weight))

# ==============================================================================
# 5. EVALUATION HELPERS
# ==============================================================================
# evaluate(): returns a named vector of metrics given predicted probabilities.
# youden_threshold(): finds the threshold maximising sensitivity + specificity - 1.
# conf_mat_str(): formats TP/TN/FP/FN as a string.

evaluate <- function(probs, labels, threshold = 0.5) {
  preds <- as.integer(probs >= threshold)
  tp <- sum(preds == 1 & labels == 1)
  tn <- sum(preds == 0 & labels == 0)
  fp <- sum(preds == 1 & labels == 0)
  fn <- sum(preds == 0 & labels == 1)
  acc  <- (tp + tn) / length(labels)
  prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  rec  <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  spec <- ifelse(tn + fp == 0, 0, tn / (tn + fp))
  f1   <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  auc  <- as.numeric(pROC::auc(pROC::roc(labels, probs, quiet = TRUE)))
  c(Accuracy = acc, Precision = prec, Recall = rec,
    Specificity = spec, F1 = f1, AUC = auc,
    TP = tp, TN = tn, FP = fp, FN = fn)
}

youden_threshold <- function(probs, labels) {
  roc_obj <- pROC::roc(labels, probs, quiet = TRUE)
  coords  <- pROC::coords(roc_obj, "best", best.method = "youden",
                           ret = c("threshold", "sensitivity", "specificity"))
  coords$threshold[1]
}

conf_mat_str <- function(metrics) {
  sprintf("TP=%d  TN=%d  FP=%d  FN=%d",
          metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"])
}

print_table <- function(mat, title) {
  cat("\n", strrep("=", 65), "\n", sep = "")
  cat(title, "\n")
  cat(strrep("=", 65), "\n", sep = "")
  metrics_to_show <- c("Accuracy","Precision","Recall","Specificity","F1","AUC")
  df_show <- round(mat[, metrics_to_show, drop = FALSE], 3)
  print(df_show)
}

# ==============================================================================
# 6. MODEL FITTING
# ==============================================================================
# Store fitted models, SMOTE-trained probabilities, and cost-sensitive variants.
# Each entry: list(base = model, cs = model_cost_sensitive)
# "base" = trained on SMOTE data with uniform weights
# "cs"   = trained on SMOTE data with cost-sensitive weights
# ==============================================================================

cat(strrep("-", 60), "\n")
cat("Fitting models...\n")
cat(strrep("-", 60), "\n\n")

models      <- list()
probs_base  <- list()  # predicted probs on test set, base
probs_cs    <- list()  # predicted probs on test set, cost-sensitive

# ── Helper: XGBoost predict ──────────────────────────────────────────────────
xgb_predict <- function(model, X) {
  predict(model, xgb.DMatrix(data = X))
}

# ── Helper: train XGBoost ────────────────────────────────────────────────────
# Uses the current XGBoost API (>= 2.0): 'x' instead of 'data', 'learning_rate'
# instead of 'eta', and verbosity controlled via params not a top-level argument.
train_xgb <- function(X, y, weights = NULL, scale_pos = 1,
                       nrounds = 100) {
  dtrain <- xgb.DMatrix(data = X, label = y,
                         weight = if (is.null(weights)) rep(1, length(y)) else weights)
  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.9,
    scale_pos_weight = scale_pos,
    verbosity        = 0
  )
  xgb.train(params = params, data = dtrain, nrounds = nrounds,
             verbose = 0)
}

# ── 6.1 Logistic Regression ──────────────────────────────────────────────────
cat("  [1/8] Logistic Regression\n")
lr_base <- glm(target ~ ., data = data.frame(X_train_sm, target = y_train_sm),
               family = binomial(link = "logit"), weights = rep(1, nrow(X_train_sm)))
lr_cs   <- glm(target ~ ., data = data.frame(X_train_sm, target = y_train_sm),
               family = binomial(link = "logit"),
               weights = ifelse(y_train_sm == 1, w1, w0))
models[["Logistic Regression"]] <- list(base = lr_base, cs = lr_cs)
probs_base[["Logistic Regression"]] <- predict(lr_base, newdata = data.frame(X_test), type = "response")
probs_cs[["Logistic Regression"]]   <- predict(lr_cs,   newdata = data.frame(X_test), type = "response")

# ── 6.2 LDA ─────────────────────────────────────────────────────────────────
cat("  [2/8] Linear Discriminant Analysis\n")
lda_base <- lda(x = X_train_sm, grouping = factor(y_train_sm))
lda_cs   <- lda(x = X_train_sm, grouping = factor(y_train_sm),
                 prior = c(w0 / (w0 + w1), w1 / (w0 + w1)))
models[["LDA"]] <- list(base = lda_base, cs = lda_cs)

# Defensive posterior extraction: look up the "default" column by name.
# SMOTE can reorder factor levels; never assume column index 2 = class 1.
lda_default_col <- as.character(1)
lda_post_base   <- predict(lda_base, newdata = X_test)$posterior
lda_post_cs     <- predict(lda_cs,   newdata = X_test)$posterior
if (!lda_default_col %in% colnames(lda_post_base)) {
  stop(paste("LDA posterior has no column '1'. Available columns:",
             paste(colnames(lda_post_base), collapse = ", ")))
}
probs_base[["LDA"]] <- lda_post_base[, lda_default_col]
probs_cs[["LDA"]]   <- lda_post_cs[,   lda_default_col]

# Store LDA discriminant scores on training data (for Hybrid 7)
lda_score_train <- predict(lda_base, newdata = X_train_sm)$x[, 1]  # LD1
lda_score_test  <- predict(lda_base, newdata = X_test)$x[, 1]

# Store LR probability scores on training data (for Hybrid 8)
lr_score_train  <- predict(lr_base, newdata = data.frame(X_train_sm), type = "response")
lr_score_test   <- predict(lr_base, newdata = data.frame(X_test),     type = "response")

# ── 6.3 Random Forest ───────────────────────────────────────────────────────
cat("  [3/8] Random Forest\n")
rf_base <- randomForest(x = X_train_sm, y = factor(y_train_sm),
                         ntree = 100, maxnodes = NULL,
                         classwt = c("0" = 1, "1" = 1))
rf_cs   <- randomForest(x = X_train_sm, y = factor(y_train_sm),
                         ntree = 100, classwt = c("0" = w0, "1" = w1))
models[["Random Forest"]] <- list(base = rf_base, cs = rf_cs)
probs_base[["Random Forest"]] <- predict(rf_base, newdata = X_test, type = "prob")[, "1"]
probs_cs[["Random Forest"]]   <- predict(rf_cs,   newdata = X_test, type = "prob")[, "1"]

# ── 6.4 XGBoost ─────────────────────────────────────────────────────────────
cat("  [4/8] XGBoost\n")
xgb_base <- train_xgb(X_train_sm, y_train_sm)
xgb_cs   <- train_xgb(X_train_sm, y_train_sm,
                        weights   = sample_weights_sm,
                        scale_pos = scale_pos_weight)
models[["XGBoost"]] <- list(base = xgb_base, cs = xgb_cs)
probs_base[["XGBoost"]] <- xgb_predict(xgb_base, X_test)
probs_cs[["XGBoost"]]   <- xgb_predict(xgb_cs,   X_test)

# ── 6.5 MLP ─────────────────────────────────────────────────────────────────
cat("  [5/8] MLP (Neural Network)\n")
# nnet() implements a single hidden layer MLP.
# size = number of hidden units; decay = L2 regularisation (alpha in sklearn).
# maxit = max iterations; MaxNWts = max weights (needs to be large for wide input).
mlp_base <- nnet(x = X_train_sm, y = y_train_sm,
                  size = 100, decay = 0.001, maxit = 500,
                  MaxNWts = 100000, trace = FALSE, linout = FALSE)
# nnet has no native class weighting; use case weights via the 'weights' arg
mlp_cs   <- nnet(x = X_train_sm, y = y_train_sm,
                  size = 100, decay = 0.001, maxit = 500,
                  MaxNWts = 100000, trace = FALSE, linout = FALSE,
                  weights = sample_weights_sm)
models[["MLP"]] <- list(base = mlp_base, cs = mlp_cs)
probs_base[["MLP"]] <- as.vector(predict(mlp_base, newdata = X_test, type = "raw"))
probs_cs[["MLP"]]   <- as.vector(predict(mlp_cs,   newdata = X_test, type = "raw"))

# ── 6.6 Stacked Ensemble ────────────────────────────────────────────────────
# Meta-features: out-of-fold predictions from RF, XGBoost, MLP on training data.
# Meta-learner: Logistic Regression.
cat("  [6/8] Stacked Ensemble\n")

# Generate out-of-fold predictions using 5-fold CV on the SMOTE training set
n_sm   <- nrow(X_train_sm)
k_fold <- 5
folds  <- createFolds(factor(y_train_sm), k = k_fold, list = TRUE, returnTrain = FALSE)

oof_rf  <- numeric(n_sm)
oof_xgb <- numeric(n_sm)
oof_mlp <- numeric(n_sm)

set.seed(42)
for (fold_idx in seq_len(k_fold)) {
  val_idx   <- folds[[fold_idx]]
  trn_idx   <- setdiff(seq_len(n_sm), val_idx)
  Xf_trn    <- X_train_sm[trn_idx, ]
  yf_trn    <- y_train_sm[trn_idx]
  Xf_val    <- X_train_sm[val_idx, ]

  # RF
  rf_f <- randomForest(x = Xf_trn, y = factor(yf_trn), ntree = 100)
  oof_rf[val_idx]  <- predict(rf_f, newdata = Xf_val, type = "prob")[, "1"]

  # XGBoost
  xgb_f <- train_xgb(Xf_trn, yf_trn)
  oof_xgb[val_idx] <- xgb_predict(xgb_f, Xf_val)

  # MLP
  mlp_f <- nnet(x = Xf_trn, y = yf_trn, size = 100, decay = 0.001,
                 maxit = 300, MaxNWts = 100000, trace = FALSE, linout = FALSE)
  oof_mlp[val_idx] <- as.vector(predict(mlp_f, newdata = Xf_val, type = "raw"))
}

# Meta-feature matrix for training
meta_train <- data.frame(rf = oof_rf, xgb = oof_xgb, mlp = oof_mlp,
                          target = y_train_sm)

# Train meta-learner on OOF predictions
meta_lr_base <- glm(target ~ ., data = meta_train, family = binomial())
meta_lr_cs   <- glm(target ~ ., data = meta_train, family = binomial(),
                     weights = sample_weights_sm)

# Generate test-set meta-features using fully fitted base models
meta_test <- data.frame(
  rf  = probs_base[["Random Forest"]],
  xgb = probs_base[["XGBoost"]],
  mlp = probs_base[["MLP"]]
)

models[["Stacked Ensemble"]] <- list(
  base = meta_lr_base, cs = meta_lr_cs,
  meta_test_base = meta_test,
  meta_test_cs   = data.frame(
    rf  = probs_cs[["Random Forest"]],
    xgb = probs_cs[["XGBoost"]],
    mlp = probs_cs[["MLP"]]
  )
)

probs_base[["Stacked Ensemble"]] <- predict(meta_lr_base,
  newdata = models[["Stacked Ensemble"]]$meta_test_base, type = "response")
probs_cs[["Stacked Ensemble"]]   <- predict(meta_lr_cs,
  newdata = models[["Stacked Ensemble"]]$meta_test_cs,   type = "response")

# ── 6.7 Hybrid 1: LDA-XGBoost Fusion ────────────────────────────────────────
# Augment training and test feature matrices with the LDA discriminant score.
cat("  [7/8] Hybrid: LDA-XGBoost Fusion\n")

X_train_lda <- cbind(X_train_sm, lda_score = lda_score_train)
X_test_lda  <- cbind(X_test,     lda_score = lda_score_test)

lda_xgb_base <- train_xgb(X_train_lda, y_train_sm)
lda_xgb_cs   <- train_xgb(X_train_lda, y_train_sm,
                            weights   = sample_weights_sm,
                            scale_pos = scale_pos_weight)
models[["LDA-XGBoost"]] <- list(base = lda_xgb_base, cs = lda_xgb_cs)
probs_base[["LDA-XGBoost"]] <- xgb_predict(lda_xgb_base, X_test_lda)
probs_cs[["LDA-XGBoost"]]   <- xgb_predict(lda_xgb_cs,   X_test_lda)

# ── 6.8 Hybrid 2: LR-XGBoost Fusion ─────────────────────────────────────────
# Augment training and test feature matrices with the LR predicted probability.
cat("  [8/8] Hybrid: LR-XGBoost Fusion\n\n")

X_train_lr <- cbind(X_train_sm, lr_score = lr_score_train)
X_test_lr  <- cbind(X_test,     lr_score = lr_score_test)

lr_xgb_base <- train_xgb(X_train_lr, y_train_sm)
lr_xgb_cs   <- train_xgb(X_train_lr, y_train_sm,
                           weights   = sample_weights_sm,
                           scale_pos = scale_pos_weight)
models[["LR-XGBoost"]] <- list(base = lr_xgb_base, cs = lr_xgb_cs)
probs_base[["LR-XGBoost"]] <- xgb_predict(lr_xgb_base, X_test_lr)
probs_cs[["LR-XGBoost"]]   <- xgb_predict(lr_xgb_cs,   X_test_lr)

# ==============================================================================
# 7. EVALUATE ALL MODELS
# ==============================================================================
MODEL_NAMES <- c("Logistic Regression", "LDA", "Random Forest", "XGBoost",
                  "MLP", "Stacked Ensemble", "LDA-XGBoost", "LR-XGBoost")

METRIC_COLS <- c("Accuracy","Precision","Recall","Specificity","F1","AUC")

# ── 7.1 Base performance ─────────────────────────────────────────────────────
results_base <- do.call(rbind, lapply(MODEL_NAMES, function(nm) {
  evaluate(probs_base[[nm]], y_test)
}))
rownames(results_base) <- MODEL_NAMES
print_table(results_base, "BASE MODEL PERFORMANCE  (Table 3 equivalent)")

# ── 7.2 Cost-sensitive performance ──────────────────────────────────────────
results_cs <- do.call(rbind, lapply(MODEL_NAMES, function(nm) {
  evaluate(probs_cs[[nm]], y_test)
}))
rownames(results_cs) <- MODEL_NAMES
print_table(results_cs, "COST-SENSITIVE PERFORMANCE  (Table 4 equivalent)")

# ── 7.3 Threshold optimisation (Youden's J) ──────────────────────────────────
opt_thresholds <- sapply(MODEL_NAMES, function(nm) {
  youden_threshold(probs_cs[[nm]], y_test)
})

results_opt <- do.call(rbind, lapply(MODEL_NAMES, function(nm) {
  t  <- unname(opt_thresholds[nm])   # strip name so column stays "Threshold"
  m  <- evaluate(probs_cs[[nm]], y_test, threshold = t)
  c(m, Threshold = t)
}))
rownames(results_opt) <- MODEL_NAMES

cat("\n", strrep("=", 65), "\n", sep = "")
cat("INTEGRATED: cost-sensitive + threshold-optimised  (Table 5 equivalent)\n")
cat(strrep("=", 65), "\n", sep = "")
print(round(results_opt[, c(METRIC_COLS, "Threshold")], 3))

# ── 7.4 Confusion matrices ───────────────────────────────────────────────────
cat("\n", strrep("=", 65), "\n", sep = "")
cat("BASE CONFUSION MATRICES\n")
cat(strrep("=", 65), "\n", sep = "")
for (nm in MODEL_NAMES) {
  m <- evaluate(probs_base[[nm]], y_test)
  cat(sprintf("  %-22s %s\n", nm, conf_mat_str(m)))
}

cat("\n", strrep("=", 65), "\n", sep = "")
cat("COST-SENSITIVE + THRESHOLD-OPTIMISED CONFUSION MATRICES\n")
cat(strrep("=", 65), "\n", sep = "")
for (nm in MODEL_NAMES) {
  m <- evaluate(probs_cs[[nm]], y_test, threshold = opt_thresholds[nm])
  cat(sprintf("  %-22s %s\n", nm, conf_mat_str(m)))
}

# ==============================================================================
# 8. BOOTSTRAP 95% CONFIDENCE INTERVALS (Stacked Ensemble, base)
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat(sprintf("95%% BOOTSTRAP CONFIDENCE INTERVALS (Stacked Ensemble, %d resamples)\n",
            N_BOOT))
cat(strrep("=", 65), "\n", sep = "")

set.seed(42)
boot_metrics <- c("Accuracy","Precision","Recall","AUC")
boot_results <- matrix(NA, nrow = N_BOOT, ncol = length(boot_metrics),
                        dimnames = list(NULL, boot_metrics))

se_probs  <- probs_base[["Stacked Ensemble"]]
n_test    <- length(y_test)

for (b in seq_len(N_BOOT)) {
  idx <- sample(seq_len(n_test), n_test, replace = TRUE)
  pb  <- se_probs[idx]
  lb  <- y_test[idx]
  if (length(unique(lb)) < 2) next
  m   <- evaluate(pb, lb)
  boot_results[b, "Accuracy"]  <- m["Accuracy"]
  boot_results[b, "Precision"] <- m["Precision"]
  boot_results[b, "Recall"]    <- m["Recall"]
  boot_results[b, "AUC"]       <- m["AUC"]
}

for (metric in boot_metrics) {
  vals <- boot_results[, metric]
  vals <- vals[!is.na(vals)]
  cat(sprintf("  %-12s [%.3f, %.3f]\n", metric,
              quantile(vals, 0.025), quantile(vals, 0.975)))
}

# ==============================================================================
# 9. McNEMAR'S TEST: Stacked Ensemble vs Random Forest
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("McNEMAR'S TEST: Stacked Ensemble vs Random Forest\n")
cat(strrep("=", 65), "\n", sep = "")

pred_stack <- as.integer(probs_base[["Stacked Ensemble"]] >= 0.5)
pred_rf    <- as.integer(probs_base[["Random Forest"]]    >= 0.5)

correct_stack <- (pred_stack == y_test)
correct_rf    <- (pred_rf    == y_test)

n01 <- sum( correct_stack & !correct_rf)  # Stack right, RF wrong
n10 <- sum(!correct_stack &  correct_rf)  # RF right, Stack wrong

# McNemar with continuity correction: chi2 = (|n01-n10| - 1)^2 / (n01+n10)
mc_stat  <- (abs(n01 - n10) - 1)^2 / (n01 + n10)
mc_pval  <- pchisq(mc_stat, df = 1, lower.tail = FALSE)

cat(sprintf("  n01 (Stack correct, RF wrong) = %d\n", n01))
cat(sprintf("  n10 (RF correct, Stack wrong) = %d\n", n10))
cat(sprintf("  chi2 = %.4f,  p-value = %.4f\n", mc_stat, mc_pval))
cat(sprintf("  Conclusion: %s\n",
            ifelse(mc_pval < ALPHA,
                   "Significant difference (p < 0.05)",
                   "No significant difference (p >= 0.05)")))

# ==============================================================================
# 10. FRIEDMAN'S TEST + NEMENYI POST-HOC
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("FRIEDMAN'S TEST + NEMENYI POST-HOC\n")
cat(strrep("=", 65), "\n", sep = "")

# Metric matrix: rows = models, cols = 5 metrics
metric_mat <- results_base[, METRIC_COLS[1:5]]   # exclude AUC (6th)
cat("\nMetric matrix (base models):\n")
print(round(metric_mat, 3))

# Friedman test: treat models as "blocks", metrics as "treatments"
# friedman.test() expects: response ~ group | block
# Here: metric value ~ metric_name | model_name
melt_mat <- as.data.frame(metric_mat) %>%
  tibble::rownames_to_column("Model") %>%
  tidyr::pivot_longer(-Model, names_to = "Metric", values_to = "Value")

friedman_result <- friedman.test(Value ~ Model | Metric, data = melt_mat)
cat(sprintf("\nFriedman chi2 = %.4f,  p-value = %.6f\n",
            friedman_result$statistic, friedman_result$p.value))

# Average ranks (lower = better)
ranks <- apply(-metric_mat, 2, rank)   # rank within each metric (ascending = best)
avg_ranks <- sort(rowMeans(ranks))
cat("\nAverage ranks across metrics (lower = better):\n")
for (nm in names(avg_ranks)) {
  cat(sprintf("  %-22s %.2f\n", nm, avg_ranks[nm]))
}

# Nemenyi post-hoc (manual implementation using critical difference)
# Uses the Nemenyi test for multiple comparisons of ranked data.
# CD = q_alpha * sqrt(k*(k+1) / (6*n)) where k=models, n=metrics
if (friedman_result$p.value < ALPHA) {
  cat("\nFriedman significant — running Nemenyi post-hoc...\n")

  k_models  <- nrow(metric_mat)
  n_metrics <- ncol(metric_mat)

  # q_alpha for alpha=0.05, infinite df (two-tailed Studentised range / sqrt(2))
  # Table values from Demsar (2006): q_0.05 for k comparisons
  q_table <- c(2, 1.960, 2.344, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164)
  q_alpha <- if (k_models <= length(q_table)) q_table[k_models] else 3.164

  cd <- q_alpha * sqrt(k_models * (k_models + 1) / (6 * n_metrics))
  cat(sprintf("  Critical difference (CD) at alpha=%.2f: %.4f\n", ALPHA, cd))

  rank_means <- rowMeans(ranks)
  pairs <- combn(names(rank_means), 2, simplify = FALSE)
  sig_pairs <- Filter(function(p) abs(rank_means[p[1]] - rank_means[p[2]]) > cd, pairs)

  if (length(sig_pairs) == 0) {
    cat("  No pairs significantly different at CD threshold.\n")
  } else {
    cat("  Significantly different pairs (|rank diff| > CD):\n")
    for (p in sig_pairs) {
      cat(sprintf("    %s vs %s  |diff| = %.3f\n",
                  p[1], p[2], abs(rank_means[p[1]] - rank_means[p[2]])))
    }
  }
} else {
  cat("  Friedman not significant at p < 0.05 — Nemenyi post-hoc skipped.\n")
}

# ==============================================================================
# 11. SHAP FEATURE IMPORTANCE
# ==============================================================================
# Computes SHAP values for XGBoost, LDA-XGBoost, and LR-XGBoost using the
# built-in XGBoost SHAP implementation (predict with predcontrib = TRUE).
# Saves a bar chart of mean |SHAP| for each model's top 15 features.
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("SHAP FEATURE IMPORTANCE\n")
cat(strrep("=", 65), "\n", sep = "")

shap_plot <- function(model, X_matrix, feature_names, model_label, output_file) {
  dmat  <- xgb.DMatrix(data = X_matrix)
  shap  <- predict(model, dmat, predcontrib = TRUE)
  # predcontrib returns matrix with ncol = nfeatures + 1 (last col = BIAS)
  shap  <- shap[, seq_len(ncol(shap) - 1)]
  colnames(shap) <- feature_names

  mean_abs <- sort(colMeans(abs(shap)), decreasing = TRUE)
  top15    <- head(mean_abs, 15)

  df_plot <- data.frame(
    Feature   = factor(names(top15), levels = rev(names(top15))),
    MeanAbsSHAP = as.numeric(top15)
  )

  cat(sprintf("\n  %s — Top 10 features by mean |SHAP|:\n", model_label))
  for (i in seq_len(min(10, nrow(df_plot)))) {
    cat(sprintf("    %-45s %.4f\n", df_plot$Feature[nrow(df_plot)+1-i],
                df_plot$MeanAbsSHAP[nrow(df_plot)+1-i]))
  }

  p <- ggplot(df_plot, aes(x = Feature, y = MeanAbsSHAP)) +
    geom_col(fill = "#2E75B6", alpha = 0.85) +
    coord_flip() +
    labs(title    = paste0("SHAP Feature Importance — ", model_label),
         subtitle = sprintf("Top 15 features by mean |SHAP value|  (N test = %d)", nrow(X_matrix)),
         x = NULL, y = "Mean |SHAP value|") +
    theme_bw(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", size = 12),
          plot.subtitle = element_text(size = 10, color = "gray40"))

  ggsave(output_file, p, width = 8, height = 6, dpi = 150, bg = "white")
  message(sprintf("  Saved: %s", output_file))
  invisible(mean_abs)
}

shap_plot(xgb_base,     X_test,     colnames(X_test),    "XGBoost",
          file.path(OUTPUT_DIR, "shap_xgboost.png"))
shap_plot(lda_xgb_base, X_test_lda, colnames(X_test_lda), "LDA-XGBoost Fusion",
          file.path(OUTPUT_DIR, "shap_lda_xgboost.png"))
shap_plot(lr_xgb_base,  X_test_lr,  colnames(X_test_lr),  "LR-XGBoost Fusion",
          file.path(OUTPUT_DIR, "shap_lr_xgboost.png"))

# ==============================================================================
# 12. ROC CURVES
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("ROC CURVES\n")
cat(strrep("=", 65), "\n", sep = "")

# Colours: classical=blues, ML=greens, ensemble=purple, hybrids=oranges
roc_colours <- c(
  "Logistic Regression" = "#1F4E79",
  "LDA"                 = "#2E75B6",
  "Random Forest"       = "#1A7A4A",
  "XGBoost"             = "#2CA02C",
  "MLP"                 = "#98DF8A",
  "Stacked Ensemble"    = "#9467BD",
  "LDA-XGBoost"         = "#E07B00",
  "LR-XGBoost"          = "#FF7F0E"
)

# Compute ROC objects
roc_list <- lapply(MODEL_NAMES, function(nm) {
  pROC::roc(y_test, probs_base[[nm]], quiet = TRUE)
})
names(roc_list) <- MODEL_NAMES

# Build data.frame for ggplot
roc_df <- do.call(rbind, lapply(MODEL_NAMES, function(nm) {
  r   <- roc_list[[nm]]
  auc <- round(as.numeric(pROC::auc(r)), 3)
  data.frame(
    FPR   = 1 - r$specificities,
    TPR   = r$sensitivities,
    Model = paste0(nm, " (AUC=", auc, ")"),
    Group = nm,
    stringsAsFactors = FALSE
  )
}))

# Map colours to the AUC-labelled model names
auc_labels  <- unique(roc_df$Model)
group_labels <- unique(roc_df$Group)
col_map <- setNames(roc_colours[group_labels], auc_labels)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, colour = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "gray60", linewidth = 0.5) +
  scale_colour_manual(values = col_map) +
  labs(title    = "ROC Curves — All 8 Models (Base, no resampling correction)",
       subtitle = "SA simulated credit dataset  |  N test set",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       colour = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position  = "right",
        legend.text      = element_text(size = 9),
        plot.title       = element_text(face = "bold", size = 12),
        plot.subtitle    = element_text(size = 10, colour = "gray40"))

roc_file <- file.path(OUTPUT_DIR, "roc_curves_all_models.png")
ggsave(roc_file, p_roc, width = 10, height = 6.5, dpi = 150, bg = "white")
cat(sprintf("  Saved: %s\n", roc_file))

# Print AUC summary
cat("\n  AUC summary (base models):\n")
auc_vec <- sapply(MODEL_NAMES, function(nm) round(as.numeric(pROC::auc(roc_list[[nm]])), 4))
auc_sorted <- sort(auc_vec, decreasing = TRUE)
for (nm in names(auc_sorted)) {
  cat(sprintf("    %-22s AUC = %.4f\n", nm, auc_sorted[nm]))
}

# ==============================================================================
# 13. SUMMARY
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("DONE\n")
cat(strrep("=", 65), "\n", sep = "")
cat("Output files saved to:", OUTPUT_DIR, "\n")
cat("  shap_xgboost.png\n")
cat("  shap_lda_xgboost.png\n")
cat("  shap_lr_xgboost.png\n")
cat("  roc_curves_all_models.png\n\n")
cat("Compare results against Mathibela & Maposa (2026) Tables 3-7.\n")
cat("Exact values will differ (different data and seed) but rankings\n")
cat("and the effect of cost-sensitive weighting and threshold\n")
cat("optimisation should follow the same pattern.\n")
