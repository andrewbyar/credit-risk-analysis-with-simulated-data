#Results########################################################################
# ==============================================================================
# South African Credit Dataset Simulation
# ==============================================================================
#
# Takes the German Credit Dataset and re-parameterises it to reflect South
# African consumer credit market conditions, sourced from:
#
#   - National Credit Regulator (NCR) Consumer Credit Market Report Q4 2024
#   - TransUnion SA Industry Insights Report Q1 2024
#   - South African Reserve Bank Quarterly Bulletin 2024
#   - South African Business Matters / NDCA (2025): average personal loan
#     in 2024 was just over R30,000; max NCA interest rate 27.5% p.a.
#
# Install required packages (run once):
#   install.packages(c("readr", "dplyr"))
#
# Usage:
#   source("sa_simulation.R")
#   -> writes: german_credit_sa_simulated.csv (1000 rows)
#   -> prints: side-by-side comparison of key statistics
# ==============================================================================

library(readr)
library(dplyr)

set.seed(99)
N <- 100000

# ==============================================================================
# 1. LOAD ORIGINAL GERMAN DATASET
# ==============================================================================
# Update this path if the file is elsewhere on your machine.
INPUT_PATH  <- "C:/Users/23108/Downloads/german_credit_data (2).csv"
OUTPUT_PATH <- "C:/Users/23108/Downloads/sa_sims.csv"

df_orig <- read_csv(INPUT_PATH, show_col_types = FALSE)
cat(sprintf("Loaded original: %d rows, %d columns\n", nrow(df_orig), ncol(df_orig)))

# ==============================================================================
# 2. SIMULATE EACH COLUMN
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 Age
# SA profile: younger borrowers dominate; Gen Z driving originations growth
# (TransUnion Q1 2024). Mean ~32, min 18 (NCA requires majority), max 70.
# ------------------------------------------------------------------------------
age <- as.integer(pmax(18, pmin(70, round(rnorm(N, mean = 32, sd = 10)))))

# ------------------------------------------------------------------------------
# 2.2 Sex
# NCA prohibits use of gender in credit decisions, but kept for structural
# parity with the original dataset.
# Approximate SA adult population split: 48% male / 52% female (StatsSA 2023)
# ------------------------------------------------------------------------------
sex <- sample(c("male", "female"), size = N, replace = TRUE, prob = c(0.48, 0.52))

# ------------------------------------------------------------------------------
# 2.3 Job
# SA unemployment ~32.1% (StatsSA Q4 2024); large informal sector.
# Mapping (matching original integer codes):
#   0 = unemployed/unskilled non-resident -> 15%
#   1 = unskilled/resident (informal)     -> 30%
#   2 = skilled/official                  -> 40%
#   3 = highly skilled/management         -> 15%
# German original: 0=2%, 1=20%, 2=63%, 3=15%
# ------------------------------------------------------------------------------
job <- sample(c(0L, 1L, 2L, 3L), size = N, replace = TRUE,
              prob = c(0.15, 0.30, 0.40, 0.15))

# ------------------------------------------------------------------------------
# 2.4 Housing
# SA homeownership ~55% (StatsSA GHS 2022). Significant informal/free housing.
# own=55%, rent=25%, free/informal=20%
# German original: own=71%, rent=18%, free=11%
# ------------------------------------------------------------------------------
housing <- sample(c("own", "rent", "free"), size = N, replace = TRUE,
                  prob = c(0.55, 0.25, 0.20))

# ------------------------------------------------------------------------------
# 2.5 Saving accounts
# ~30% of SA adults remain underbanked (FinScope SA 2023); higher NA rate.
# NA=35%, little=45%, moderate=12%, quite rich=5%, rich=3%
# German original: NA=18%, little=60%, moderate=10%, quite rich=6%, rich=5%
# ------------------------------------------------------------------------------
saving_accounts <- sample(
  c("NA", "little", "moderate", "quite rich", "rich"),
  size = N, replace = TRUE,
  prob = c(0.35, 0.45, 0.12, 0.05, 0.03)
)

# ------------------------------------------------------------------------------
# 2.6 Checking account
# Same underbanking logic; higher NA proportion.
# NA=45%, little=30%, moderate=20%, rich=5%
# German original: NA=39%, little=27%, moderate=27%, rich=6%
# ------------------------------------------------------------------------------
checking_account <- sample(
  c("NA", "little", "moderate", "rich"),
  size = N, replace = TRUE,
  prob = c(0.45, 0.30, 0.20, 0.05)
)

# ------------------------------------------------------------------------------
# 2.7 Credit amount (Rand)
# SA average personal loan ~R30,000 (NCR/NDCA 2025).
# Range R500-R250,000 (NCA-regulated lenders per RandWallet 2024).
# Log-normal distribution: right-skewed, matching SA unsecured credit market.
# Target: mean ~R30,000, median ~R22,000, max ~R250,000.
# ------------------------------------------------------------------------------
# Log-normal fitted to German Credit amount data (MLE, floc=0):
#   sdlog=0.7761, scale=2413.16 -> mean=3,261 EUR (German)
# SA shift: mean target=R30,000 -> new meanlog=log(22199)=10.0078, sdlog unchanged
# Log-normal outperforms Gamma (ΔAIC=106.2) and Weibull (ΔAIC=164.8).
SA_CRED_MEANLOG <- 10.0078  # log(22199), SA-shifted
SA_CRED_SDLOG   <- 0.7761   # shape unchanged from German fit
credit_amount <- as.integer(pmax(500, pmin(250000, round(rlnorm(N, meanlog = SA_CRED_MEANLOG, sdlog = SA_CRED_SDLOG)))))

# ------------------------------------------------------------------------------
# 2.8 Duration (months)
# SA unsecured loans: 1-72 months (NCA cap).
# Short-term products dominate by volume (NCR Q1 2025).
# Gamma distribution gives realistic right skew; mean ~18 months.
# ------------------------------------------------------------------------------
# Log-normal fitted to German Credit duration data (MLE, floc=0):
#   sdlog=0.5819, scale=17.7612 -> mean=21.0 months (German)
# SA shift: mean target=18 months -> new meanlog=log(15.1965)=2.7211, sdlog unchanged
# Log-normal outperforms Gamma (ΔAIC=13.9) and Weibull (ΔAIC=73.9).
# NOTE: 75% of German durations are multiples of 6 months (standard
# loan terms). No continuous distribution passes KS at n=1000 on
# this data. Log-normal is chosen as the best-fitting continuous
# approximation; the discrete rounding in pmax/pmin partially restores
# the original term structure.
SA_DUR_MEANLOG <- 2.7211   # log(15.1965), SA-shifted
SA_DUR_SDLOG   <- 0.5819   # shape unchanged from German fit
duration <- as.integer(pmax(1, pmin(72, round(rlnorm(N, meanlog = SA_DUR_MEANLOG, sdlog = SA_DUR_SDLOG)))))

# ------------------------------------------------------------------------------
# 2.9 Purpose
# Restructured to match SA NCR product mix:
#   personal/debt consolidation: 35%
#   car/vehicle finance:         25%
#   home improvement:            15%
#   education:                   10%
#   business:                     8%
#   furniture/appliances:         5%
#   other:                        2%
# German original: car=34%, radio/TV=28%, furniture=18%, business=10%, education=6%
# ------------------------------------------------------------------------------
purpose <- sample(
  c("personal/debt consolidation", "car", "home improvement",
    "education", "business", "furniture/appliances", "other"),
  size = N, replace = TRUE,
  prob = c(0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02)
)

# ==============================================================================
# 3. SIMULATE TARGET (default/no default)
# ==============================================================================
# SA: NCR Q1 2025 reports 36.04% of credit-active consumers have impaired
# records. We target ~35% default rate (vs German 30%).
#
# Default probability is NOT uniform -- it is higher for:
#   - No/little savings or checking account (liquidity-constrained)
#   - Unemployed or unskilled job categories
#   - Younger borrowers (less credit history)
#   - Renting or free/informal housing (less stable)
#   - Larger loan amounts relative to SA income norms
#   - Longer durations
# ==============================================================================

default_score <- numeric(N)

# Saving account contribution
saving_map <- c("NA" = 0.25, "little" = 0.10, "moderate" = 0.00,
                "quite rich" = -0.10, "rich" = -0.15)
default_score <- default_score + saving_map[saving_accounts]

# Checking account contribution
checking_map <- c("NA" = 0.20, "little" = 0.10, "moderate" = 0.00, "rich" = -0.10)
default_score <- default_score + checking_map[checking_account]

# Job contribution
job_map <- c("0" = 0.20, "1" = 0.10, "2" = 0.00, "3" = -0.10)
default_score <- default_score + job_map[as.character(job)]

# Housing contribution
housing_map <- c("own" = -0.05, "rent" = 0.05, "free" = 0.10)
default_score <- default_score + housing_map[housing]

# Age contribution: younger = slightly higher default risk
age_score <- ifelse(age < 25, 0.10, ifelse(age < 35, 0.05, 0.00))
default_score <- default_score + age_score

# Credit amount contribution: larger loans relative to SA norms = higher risk
amount_score <- ifelse(credit_amount > 80000, 0.10,
                ifelse(credit_amount > 40000, 0.05, 0.00))
default_score <- default_score + amount_score

# Duration contribution: longer = higher risk
duration_score <- ifelse(duration > 48, 0.10,
                  ifelse(duration > 24, 0.05, 0.00))
default_score <- default_score + duration_score

# Convert to probability via sigmoid
# Intercept of -1.35 calibrated to produce ~35% default rate with this seed
intercept <- -1.35
logit <- intercept + default_score * 2.0
prob_default <- 1 / (1 + exp(-logit))

# Draw target
target_raw <- rbinom(N, size = 1, prob = prob_default)
cat(sprintf("Default rate: %.1f%% (target: ~35%%)\n", mean(target_raw) * 100))

# ==============================================================================
# 4. ASSEMBLE DATAFRAME
# ==============================================================================
df_sa <- data.frame(
  Age               = age,
  Sex               = sex,
  Job               = job,
  Housing           = housing,
  `Saving accounts` = saving_accounts,
  `Checking account`= checking_account,
  `Credit amount`   = credit_amount,
  Duration          = duration,
  Purpose           = purpose,
  Risk              = ifelse(target_raw == 1, "bad", "good"),
  check.names       = FALSE,
  stringsAsFactors  = FALSE
)

# ==============================================================================
# 5. PRINT COMPARISON
# ==============================================================================
cat("\n", strrep("=", 65), "\n", sep = "")
cat("COMPARISON: German Original vs SA Simulated\n")
cat(strrep("=", 65), "\n", sep = "")

cat(sprintf("\n%-35s %12s %12s\n", "Statistic", "German", "SA Simulated"))
cat(strrep("-", 60), "\n", sep = "")
cat(sprintf("%-35s %12d %12d\n",   "N",                     nrow(df_orig), nrow(df_sa)))
cat(sprintf("%-35s %12s %11.1f%%\n","Default rate",          "~30%",        mean(target_raw) * 100))
cat(sprintf("%-35s %12.1f %12.1f\n","Mean age",              mean(df_orig$Age, na.rm=TRUE), mean(df_sa$Age)))
cat(sprintf("%-35s %12.1f %12.1f\n","Age std",               sd(df_orig$Age, na.rm=TRUE),   sd(df_sa$Age)))
cat(sprintf("%-35s %12s %12s\n",   "Credit amount unit",     "EUR",         "ZAR"))
cat(sprintf("%-35s %12.0f %12.0f\n","Mean credit amount",    mean(df_orig$`Credit amount`, na.rm=TRUE), mean(df_sa$`Credit amount`)))
cat(sprintf("%-35s %12.0f %12.0f\n","Median credit amount",  median(df_orig$`Credit amount`, na.rm=TRUE), median(df_sa$`Credit amount`)))
cat(sprintf("%-35s %12.1f %12.1f\n","Mean duration (months)",mean(df_orig$Duration, na.rm=TRUE), mean(df_sa$Duration)))

cat("\n")
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Housing (own %)",
    mean(df_orig$Housing == "own",  na.rm=TRUE)*100,
    mean(df_sa$Housing   == "own")*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Housing (rent %)",
    mean(df_orig$Housing == "rent", na.rm=TRUE)*100,
    mean(df_sa$Housing   == "rent")*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Housing (free %)",
    mean(df_orig$Housing == "free", na.rm=TRUE)*100,
    mean(df_sa$Housing   == "free")*100))

cat("\n")
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Saving acct (NA %)",
    mean(is.na(df_orig$`Saving accounts`))*100,
    mean(df_sa$`Saving accounts` == "NA")*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Checking acct (NA %)",
    mean(is.na(df_orig$`Checking account`))*100,
    mean(df_sa$`Checking account` == "NA")*100))

cat("\n")
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Job 0 (unemployed %)",
    mean(df_orig$Job == 0, na.rm=TRUE)*100,
    mean(df_sa$Job   == 0)*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Job 1 (unskilled %)",
    mean(df_orig$Job == 1, na.rm=TRUE)*100,
    mean(df_sa$Job   == 1)*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Job 2 (skilled %)",
    mean(df_orig$Job == 2, na.rm=TRUE)*100,
    mean(df_sa$Job   == 2)*100))
cat(sprintf("%-35s %11.1f%% %11.1f%%\n", "Job 3 (management %)",
    mean(df_orig$Job == 3, na.rm=TRUE)*100,
    mean(df_sa$Job   == 3)*100))

cat("\nPurpose distribution (SA):\n")
print(sort(table(df_sa$Purpose), decreasing = TRUE))

# ==============================================================================
# 6. SAVE
# ==============================================================================
write_csv(df_sa, OUTPUT_PATH)
cat(sprintf("\nSaved: %s  (%d rows x %d cols)\n", OUTPUT_PATH, nrow(df_sa), ncol(df_sa)))
cat("\nNOTE: This is a synthetic simulation for academic purposes only.\n")
cat("It is NOT real South African credit data.\n")
cat("Cite as: Simulated dataset based on NCR (2024), TransUnion SA (2024),\n")
cat("and SARB (2024) parameters, derived from Hofmann (1994) German Credit Data.\n")
