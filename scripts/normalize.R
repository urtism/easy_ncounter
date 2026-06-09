#!/usr/bin/env Rscript
project_lib <- Sys.getenv("NC_R_LIB", unset = file.path(getwd(), "r-lib"))
if (dir.exists(project_lib)) {
  .libPaths(c(project_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(yaml)
})

source("r/ncounter_utils.R")

opt <- parse_cli_args()
cfg <- read_pipeline_config(opt$config)

results_dir <- cfg$outputs$results_dir
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

analysis_counts_path <- if (file.exists(file.path(results_dir, "counts_filtered.csv"))) {
  file.path(results_dir, "counts_filtered.csv")
} else if (file.exists(file.path(results_dir, "counts_background_corrected.csv"))) {
  file.path(results_dir, "counts_background_corrected.csv")
} else {
  file.path(results_dir, "counts_raw.csv")
}

counts <- readr::read_csv(analysis_counts_path, show_col_types = FALSE)
count_data <- read_counts_matrix(analysis_counts_path)
matrix <- count_data$matrix
probe_info <- count_data$probe_info

method <- cfg$normalization$method %||% "nanostringnorm"
normalized <- NULL

if (identical(tolower(method), "nanostringnorm") && requireNamespace("NanoStringNorm", quietly = TRUE)) {
  ns_counts <- readr::read_csv(file.path(results_dir, "counts_raw.csv"), show_col_types = FALSE)
  ns_input <- ns_counts %>%
    dplyr::rename(Code.Class = CodeClass, Gene.Name = Name, Accession = Name)
  ns_norm <- NanoStringNorm::NanoStringNorm(
    x = as.data.frame(ns_input),
    CodeCount = "geo.mean",
    Background = "mean.2sd",
    SampleContent = "housekeeping.geo.mean",
    round.values = FALSE,
    take.log = FALSE
  )
  normalized <- ns_norm %>%
    dplyr::rename(CodeClass = Code.Class, Name = Gene.Name) %>%
    dplyr::filter(Name %in% rownames(matrix)) %>%
    dplyr::select(CodeClass, Name, dplyr::all_of(colnames(matrix)))
} else {
  lib_size <- colSums(matrix, na.rm = TRUE)
  scale_factor <- median(lib_size) / lib_size
  normalized_matrix <- sweep(matrix, 2, scale_factor, "*")
  normalized <- cbind(probe_info, as.data.frame(normalized_matrix, check.names = FALSE))
}

readr::write_csv(normalized, file.path(results_dir, "counts_normalized.csv"))
message("Wrote ", file.path(results_dir, "counts_normalized.csv"))
