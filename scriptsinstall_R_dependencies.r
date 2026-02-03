#!/usr/bin/env Rscript
# Install R dependencies for the risk assessment framework

cat("Installing R dependencies for the Integrated Risk Assessment Framework...\n")

# Required CRAN packages
cran_packages <- c(
  "httk",          # High-throughput toxicokinetics
  "mrgsolve",      # PBPK modeling
  "ggplot2",       # Visualization
  "dplyr",         # Data manipulation
  "tidyr",         # Data tidying
  "readr",         # Fast data reading
  "jsonlite",      # JSON handling
  "yaml",          # YAML configuration
  "future",        # Parallel processing
  "furrr",         # Parallel map functions
  "purrr",         # Functional programming
  "tibble",        # Modern data frames
  "stringr",       # String manipulation
  "lubridate",     # Date handling
  "reshape2",      # Data reshaping
  "cowplot",       # Plot arrangement
  "ggpubr",        # Publication-ready plots
  "gridExtra",     # Grid graphics
  "RColorBrewer",  # Color palettes
  "viridis",       # Color scales
  "scales",        # Scale functions
  "knitr",         # Dynamic report generation
  "rmarkdown",     # R Markdown
  "bookdown",      # Book-style documents
  "devtools"       # Package development
)

# Install CRAN packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("Installing %s...\n", pkg))
      install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
    } else {
      cat(sprintf("%s is already installed.\n", pkg))
    }
  }
}

# Install CRAN packages
install_if_missing(cran_packages)

# Install GitHub packages if needed
github_packages <- c(
  # Add any GitHub packages here
  # "user/repo"
)

if (length(github_packages) > 0) {
  if (!require("devtools", character.only = TRUE, quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org")
  }
  
  for (pkg in github_packages) {
    cat(sprintf("Installing %s from GitHub...\n", pkg))
    devtools::install_github(pkg)
  }
}

# Verify installations
cat("\nVerifying package installations...\n")
all_packages <- c(cran_packages, basename(github_packages))
for (pkg in all_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("✓ %s loaded successfully\n", pkg))
  } else {
    cat(sprintf("✗ Failed to load %s\n", pkg))
  }
}

# Create R profile for project
cat("\nCreating .Rprofile for project settings...\n")
rprofile_content <- '
# Project-specific R settings
options(
  # Repositories
  repos = c(CRAN = "https://cloud.r-project.org"),
  
  # Memory
  mc.cores = parallel::detectCores() - 1,
  
  # Display
  digits = 4,
  scipen = 999,
  
  # Packages
  install.packages.check.source = "no",
  install.packages.compile.from.source = "never"
)

# Load commonly used packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
})

# Set ggplot2 theme
theme_set(theme_minimal(base_size = 11) +
            theme(
              panel.grid.minor = element_blank(),
              panel.border = element_rect(fill = NA, color = "gray50"),
              axis.ticks = element_line(color = "gray50")
            ))

cat("Integrated Risk Assessment Framework R environment loaded\\n")
'

writeLines(rprofile_content, ".Rprofile")

cat("\n========================================\n")
cat("R dependency installation complete!\n")
cat("Next steps:\n")
cat("1. Restart R session to apply .Rprofile settings\n")
cat("2. Test the installation with: library(httk); library(mrgsolve)\n")
cat("========================================\n")