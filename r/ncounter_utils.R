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

analysis_group_columns <- function(cfg) {
  group_columns <- cfg$analysis$group_columns
  if (is.null(group_columns) || length(group_columns) == 0) {
    group_columns <- cfg$analysis$group_column
  }
  unlist(group_columns, use.names = FALSE)
}

composite_group_column_name <- function(group_columns) {
  if (length(group_columns) == 1) {
    return(group_columns[[1]])
  }
  safe_parts <- gsub("[^A-Za-z0-9_-]", "_", group_columns)
  paste0("__group__", paste(safe_parts, collapse = "__"))
}

add_analysis_group_column <- function(metadata, cfg) {
  group_columns <- analysis_group_columns(cfg)
  group_col <- cfg$analysis$group_column %||% composite_group_column_name(group_columns)
  missing_columns <- setdiff(group_columns, colnames(metadata))
  if (length(missing_columns) > 0) {
    stop("Missing metadata columns for analysis groups: ", paste(missing_columns, collapse = ", "))
  }
  if (length(group_columns) > 1) {
    metadata[[group_col]] <- apply(
      metadata[, group_columns, drop = FALSE],
      1,
      function(values) {
        if (any(is.na(values)) || any(trimws(as.character(values)) == "")) {
          return(NA_character_)
        }
        paste(paste0(group_columns, "=", values), collapse = " | ")
      }
    )
  }
  metadata
}

normalize_analysis_contrasts <- function(contrasts) {
  normalized <- list()
  used_ids <- character()
  for (index in seq_along(contrasts)) {
    item <- contrasts[[index]]
    reference_group <- as.character(item$reference_group %||% "")
    case_group <- as.character(item$case_group %||% "")
    comparison_id <- as.character(item$comparison_id %||% "")
    reference_samples <- normalize_sample_list(item$reference_samples %||% character())
    case_samples <- normalize_sample_list(item$case_samples %||% character())
    if (comparison_id == "" && reference_group != "" && case_group != "") {
      comparison_id <- paste0(case_group, "_vs_", reference_group)
    }
    if (comparison_id == "") {
      comparison_id <- paste0("comparison_", index)
    }
    has_sample_groups <- length(reference_samples) > 0 && length(case_samples) > 0
    if (!has_sample_groups && (reference_group == "" || case_group == "" || reference_group == case_group)) {
      next
    }
    safe_id <- unique_comparison_id(sanitize_comparison_id(comparison_id), used_ids)
    used_ids <- c(used_ids, safe_id)
    normalized[[length(normalized) + 1]] <- list(
      comparison_id = safe_id,
      reference_group = reference_group,
      case_group = case_group,
      reference_samples = reference_samples,
      case_samples = case_samples
    )
  }
  normalized
}

normalize_sample_list <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return(character())
  }
  if (is.character(value) && length(value) == 1 && grepl(";", value, fixed = TRUE)) {
    value <- unlist(strsplit(value, ";", fixed = TRUE), use.names = FALSE)
  }
  value <- trimws(as.character(unlist(value, use.names = FALSE)))
  unique(value[value != ""])
}

unique_comparison_id <- function(value, used_ids) {
  if (!value %in% used_ids) {
    return(value)
  }
  suffix <- 2
  candidate <- paste0(value, "_", suffix)
  while (candidate %in% used_ids) {
    suffix <- suffix + 1
    candidate <- paste0(value, "_", suffix)
  }
  candidate
}

sanitize_comparison_id <- function(value) {
  safe <- gsub("[^A-Za-z0-9_-]", "_", as.character(value))
  safe <- gsub("_+", "_", safe)
  safe <- gsub("^_+|_+$", "", safe)
  if (safe == "") {
    safe <- "comparison"
  }
  safe
}
