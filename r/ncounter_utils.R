project_lib <- Sys.getenv("NC_R_LIB", unset = file.path(getwd(), "r-lib"))
if (dir.exists(project_lib)) {
  .libPaths(c(project_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

parse_cli_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  config <- "config/pipeline.yaml"
  config_index <- match("--config", args)
  if (!is.na(config_index) && length(args) >= config_index + 1) {
    config <- args[[config_index + 1]]
  }
  list(config = config)
}

read_pipeline_config <- function(path) {
  yaml::read_yaml(path)
}

read_counts_matrix <- function(path) {
  counts <- readr::read_csv(path, show_col_types = FALSE)
  probe_info <- counts %>% dplyr::select(CodeClass, Name)
  matrix <- counts %>%
    dplyr::select(-CodeClass) %>%
    tibble::column_to_rownames("Name") %>%
    as.matrix()
  storage.mode(matrix) <- "numeric"
  list(probe_info = probe_info, matrix = matrix)
}

filter_low_counts <- function(matrix, min_count, min_samples) {
  keep <- rowSums(matrix >= min_count, na.rm = TRUE) >= min_samples
  matrix[keep, , drop = FALSE]
}

write_matrix_csv <- function(matrix, path) {
  output <- as.data.frame(matrix)
  output <- tibble::rownames_to_column(output, "Name")
  readr::write_csv(output, path)
}
