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

safe_geomean <- function(values) {
  values <- as.numeric(values)
  values <- values[is.finite(values) & values > 0]
  if (length(values) == 0) {
    return(0)
  }
  exp(mean(log(values)))
}

housekeeper_names <- function(results_dir, counts) {
  parsed_path <- file.path(results_dir, "probe_annotation_parsed.csv")
  if (file.exists(parsed_path)) {
    parsed <- readr::read_csv(parsed_path, show_col_types = FALSE)
    if (all(c("probe_id", "is_housekeeper") %in% colnames(parsed))) {
      return(parsed %>%
        dplyr::filter(.data$is_housekeeper %in% c(TRUE, "TRUE", "true", 1, "1")) %>%
        dplyr::pull(probe_id) %>%
        as.character())
    }
  }
  counts %>%
    dplyr::filter(grepl("housekeep|housekeeping|reference", CodeClass, ignore.case = TRUE)) %>%
    dplyr::pull(Name) %>%
    as.character()
}

select_geNorm_like_housekeepers <- function(hk_matrix, min_selected = 2, max_selected = NULL) {
  if (is.null(max_selected)) {
    max_selected <- nrow(hk_matrix)
  }
  valid <- rowSums(hk_matrix > 0, na.rm = TRUE) >= ceiling(ncol(hk_matrix) * 0.5)
  selected_table <- tibble::tibble(
    housekeeper = rownames(hk_matrix),
    used_for_normalization = FALSE,
    stability_score = NA_real_,
    reason = ifelse(valid, "excluded_unstable", "excluded_low_signal")
  )
  valid_matrix <- hk_matrix[valid, , drop = FALSE]
  if (nrow(valid_matrix) < min_selected) {
    selected_table$used_for_normalization <- rownames(hk_matrix) %in% rownames(valid_matrix)
    selected_table$reason <- ifelse(
      selected_table$used_for_normalization,
      "fallback_all_housekeepers",
      selected_table$reason
    )
    return(list(selected = rownames(valid_matrix), table = selected_table, fallback = TRUE))
  }

  log_hk <- log2(valid_matrix + 1)
  scores <- vapply(seq_len(nrow(log_hk)), function(index) {
    others <- setdiff(seq_len(nrow(log_hk)), index)
    variations <- vapply(others, function(other_index) {
      stats::sd(log_hk[index, ] - log_hk[other_index, ], na.rm = TRUE)
    }, numeric(1))
    mean(variations, na.rm = TRUE)
  }, numeric(1))
  names(scores) <- rownames(log_hk)
  ranked <- names(sort(scores, decreasing = FALSE))
  selected <- head(ranked, min(max(min_selected, length(ranked)), max_selected))

  selected_table$stability_score <- scores[match(selected_table$housekeeper, names(scores))]
  selected_table$used_for_normalization <- selected_table$housekeeper %in% selected
  selected_table$reason <- dplyr::case_when(
    selected_table$used_for_normalization ~ "selected_by_geNorm_like_stability",
    selected_table$reason == "excluded_low_signal" ~ "excluded_low_signal",
    TRUE ~ "excluded_unstable"
  )
  list(selected = selected, table = selected_table, fallback = FALSE)
}

normalize_with_housekeepers <- function(matrix, probe_info, results_dir, method, cfg) {
  hk_source_path <- if (file.exists(file.path(results_dir, "counts_background_corrected.csv"))) {
    file.path(results_dir, "counts_background_corrected.csv")
  } else {
    file.path(results_dir, "counts_raw.csv")
  }
  hk_counts <- readr::read_csv(hk_source_path, show_col_types = FALSE)
  hk_names <- intersect(housekeeper_names(results_dir, hk_counts), hk_counts$Name)
  sample_cols <- colnames(matrix)
  selected_table <- tibble::tibble(
    housekeeper = hk_names,
    used_for_normalization = TRUE,
    stability_score = NA_real_,
    reason = "selected_by_all_housekeepers"
  )
  selected <- hk_names
  fallback <- FALSE

  if (length(hk_names) == 0) {
    warning("No housekeeping probes found; falling back to library_size normalization.")
    return(NULL)
  }
  hk_matrix <- hk_counts %>%
    dplyr::filter(Name %in% hk_names) %>%
    dplyr::select(dplyr::all_of(c("Name", sample_cols))) %>%
    tibble::column_to_rownames("Name") %>%
    as.matrix()
  storage.mode(hk_matrix) <- "numeric"

  if (identical(tolower(method), "hk_geomean_genorm")) {
    min_selected <- cfg$normalization$geNorm_min_housekeepers %||% 2
    max_selected <- cfg$normalization$geNorm_max_housekeepers %||% nrow(hk_matrix)
    selection <- select_geNorm_like_housekeepers(
      hk_matrix,
      min_selected = min_selected,
      max_selected = max_selected
    )
    selected <- selection$selected
    selected_table <- selection$table
    fallback <- isTRUE(selection$fallback)
    if (length(selected) < 2) {
      warning("Fewer than two valid housekeeping probes for geNorm-like selection; using all available housekeeping probes.")
      selected <- hk_names
      selected_table$used_for_normalization <- selected_table$housekeeper %in% selected
      selected_table$reason <- ifelse(
        selected_table$used_for_normalization,
        "fallback_all_housekeepers",
        selected_table$reason
      )
      fallback <- TRUE
    }
  }

  selected_hk_matrix <- hk_matrix[selected, , drop = FALSE]
  hk_geomean <- apply(selected_hk_matrix, 2, safe_geomean)
  if (any(!is.finite(hk_geomean) | hk_geomean <= 0)) {
    warning("Invalid housekeeping geomean found; falling back to library_size normalization.")
    return(NULL)
  }
  normalization_factor <- stats::median(hk_geomean, na.rm = TRUE) / hk_geomean
  normalized_matrix <- sweep(matrix, 2, normalization_factor, "*")
  normalized <- cbind(probe_info, as.data.frame(normalized_matrix, check.names = FALSE))
  readr::write_csv(selected_table, file.path(results_dir, "selected_housekeepers.csv"))
  readr::write_csv(
    tibble::tibble(
      sample_id = names(normalization_factor),
      normalization_method = ifelse(fallback, paste0(method, "_fallback_all"), method),
      normalization_factor = as.numeric(normalization_factor),
      housekeeping_geomean = as.numeric(hk_geomean),
      n_housekeepers_available = length(hk_names),
      n_housekeepers_selected = length(selected),
      background_correction_applied = file.exists(file.path(results_dir, "counts_background_corrected.csv"))
    ),
    file.path(results_dir, "normalization_audit.csv")
  )
  normalized
}

normalize_with_library_size <- function(matrix, probe_info, results_dir, method = "library_size") {
  lib_size <- colSums(matrix, na.rm = TRUE)
  scale_factor <- median(lib_size) / lib_size
  normalized_matrix <- sweep(matrix, 2, scale_factor, "*")
  readr::write_csv(
    tibble::tibble(
      sample_id = names(scale_factor),
      normalization_method = method,
      normalization_factor = as.numeric(scale_factor),
      housekeeping_geomean = NA_real_,
      n_housekeepers_available = NA_integer_,
      n_housekeepers_selected = NA_integer_,
      background_correction_applied = file.exists(file.path(results_dir, "counts_background_corrected.csv"))
    ),
    file.path(results_dir, "normalization_audit.csv")
  )
  cbind(probe_info, as.data.frame(normalized_matrix, check.names = FALSE))
}

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

method_lower <- tolower(method)
if (method_lower %in% c("hk_geomean_all", "hk_geomean_genorm")) {
  audit_method <- if (identical(method_lower, "hk_geomean_genorm")) "hk_geomean_geNorm" else method_lower
  normalized <- normalize_with_housekeepers(matrix, probe_info, results_dir, audit_method, cfg)
  if (is.null(normalized)) {
    normalized <- normalize_with_library_size(matrix, probe_info, results_dir, "library_size_fallback")
  }
} else if (identical(method_lower, "nanostringnorm") && requireNamespace("NanoStringNorm", quietly = TRUE)) {
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
  readr::write_csv(
    tibble::tibble(
      sample_id = colnames(matrix),
      normalization_method = "nanostringnorm",
      normalization_factor = NA_real_,
      housekeeping_geomean = NA_real_,
      n_housekeepers_available = NA_integer_,
      n_housekeepers_selected = NA_integer_,
      background_correction_applied = TRUE
    ),
    file.path(results_dir, "normalization_audit.csv")
  )
} else {
  normalized <- normalize_with_library_size(matrix, probe_info, results_dir, "library_size")
}

readr::write_csv(normalized, file.path(results_dir, "counts_normalized.csv"))
message("Wrote ", file.path(results_dir, "counts_normalized.csv"))
