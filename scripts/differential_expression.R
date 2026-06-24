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
stale_result_files <- list.files(
  results_dir,
  pattern = "^differential_expression(__.*)?\\.csv$|^comparison_summary\\.csv$",
  full.names = TRUE
)
if (length(stale_result_files) > 0) {
  unlink(stale_result_files)
}

metadata_path <- if (file.exists(file.path(results_dir, "metadata.csv"))) {
  file.path(results_dir, "metadata.csv")
} else {
  cfg$inputs$metadata
}
group_col <- cfg$analysis$group_column
configured_contrasts <- cfg$analysis$contrasts
if (is.null(configured_contrasts) || length(configured_contrasts) == 0) {
  configured_contrasts <- list(list(
    comparison_id = paste0(cfg$analysis$case_group, "_vs_", cfg$analysis$reference_group),
    reference_group = cfg$analysis$reference_group,
    case_group = cfg$analysis$case_group
  ))
}

metadata <- readr::read_csv(metadata_path, show_col_types = FALSE)
metadata <- add_analysis_group_column(metadata, cfg)
count_data <- read_counts_matrix(file.path(results_dir, "counts_normalized.csv"))
matrix <- count_data$matrix

skip_low_count_filter <- isTRUE(cfg$analysis$skip_low_count_filter)
if (!skip_low_count_filter) {
  matrix <- filter_low_counts(
    matrix,
    min_count = cfg$analysis$min_count %||% 10,
    min_samples = cfg$analysis$min_samples %||% 2
  )
} else {
  message("Skipping low-count gene filter for final processed counts.")
}

sample_order <- intersect(metadata$sample_id, colnames(matrix))
metadata <- metadata %>% dplyr::filter(sample_id %in% sample_order)
matrix <- matrix[, metadata$sample_id, drop = FALSE]

contrasts <- normalize_analysis_contrasts(configured_contrasts)
if (length(contrasts) == 0) {
  stop("No valid differential-expression contrast is configured.")
}

summary_rows <- list()
first_result <- TRUE
for (contrast_config in contrasts) {
  comparison_id <- sanitize_comparison_id(contrast_config$comparison_id)
  reference_group <- contrast_config$reference_group
  case_group <- contrast_config$case_group
  explicit_reference_samples <- intersect(contrast_config$reference_samples, colnames(matrix))
  explicit_case_samples <- intersect(contrast_config$case_samples, colnames(matrix))
  has_explicit_samples <- length(explicit_reference_samples) > 0 && length(explicit_case_samples) > 0

  if (has_explicit_samples) {
    overlap <- intersect(explicit_reference_samples, explicit_case_samples)
    if (length(overlap) > 0) {
      warning("Skipping contrast with samples in both reference and case: ", comparison_id)
      next
    }
    reference_samples <- explicit_reference_samples
    case_samples <- explicit_case_samples
    contrast_metadata <- metadata %>%
      dplyr::filter(sample_id %in% c(reference_samples, case_samples)) %>%
      dplyr::mutate(.contrast_group = dplyr::case_when(
        sample_id %in% reference_samples ~ reference_group,
        sample_id %in% case_samples ~ case_group,
        TRUE ~ NA_character_
      ))
    contrast_group_col <- ".contrast_group"
  } else {
    contrast_metadata <- metadata %>%
      dplyr::filter(.data[[group_col]] %in% c(reference_group, case_group))
    reference_samples <- contrast_metadata %>%
      dplyr::filter(.data[[group_col]] == reference_group) %>%
      dplyr::pull(sample_id)
    case_samples <- contrast_metadata %>%
      dplyr::filter(.data[[group_col]] == case_group) %>%
      dplyr::pull(sample_id)
    contrast_group_col <- group_col
  }

  if (length(reference_samples) == 0 || length(case_samples) == 0) {
    warning(
      "Skipping contrast with missing samples: ",
      comparison_id,
      " (reference n=",
      length(reference_samples),
      ", case n=",
      length(case_samples),
      ")"
    )
    next
  }

  contrast_sample_order <- c(reference_samples, case_samples)
  contrast_metadata <- contrast_metadata %>%
    dplyr::filter(sample_id %in% contrast_sample_order) %>%
    dplyr::mutate(.sample_order = match(sample_id, contrast_sample_order)) %>%
    dplyr::arrange(.sample_order)
  contrast_matrix <- matrix[, contrast_metadata$sample_id, drop = FALSE]
  contrast_metadata[[contrast_group_col]] <- factor(
    contrast_metadata[[contrast_group_col]],
    levels = c(reference_group, case_group)
  )
  if (nrow(contrast_metadata) != ncol(contrast_matrix)) {
    stop("Sample metadata and count matrix are not aligned after filtering contrast samples.")
  }
  design <- model.matrix(~ 0 + contrast_metadata[[contrast_group_col]])
  colnames(design) <- levels(contrast_metadata[[contrast_group_col]])
  no_residual_df <- nrow(contrast_metadata) <= ncol(design)
  if (no_residual_df) {
    message(
      "No residual degrees of freedom are available for ",
      comparison_id,
      ". Writing descriptive logFC tables with P.Value and adj.P.Val set to NA."
    )
  } else {
    if (!requireNamespace("limma", quietly = TRUE)) {
      stop("Package 'limma' is required for differential expression.")
    }
    fit <- limma::lmFit(log2(contrast_matrix + 1), design)
  }

  if (no_residual_df) {
    contrast_samples <- c(reference_samples, case_samples)
    log_reference <- rowMeans(log2(contrast_matrix[, reference_samples, drop = FALSE] + 1), na.rm = TRUE)
    log_case <- rowMeans(log2(contrast_matrix[, case_samples, drop = FALSE] + 1), na.rm = TRUE)
    de <- tibble::tibble(
      Name = rownames(contrast_matrix),
      logFC = log_case - log_reference,
      AveExpr = rowMeans(log2(contrast_matrix[, contrast_samples, drop = FALSE] + 1), na.rm = TRUE),
      t = NA_real_,
      P.Value = NA_real_,
      adj.P.Val = NA_real_,
      B = NA_real_
    ) %>%
      dplyr::arrange(dplyr::desc(abs(logFC))) %>%
      dplyr::mutate(
        comparison_id = comparison_id,
        reference_group = reference_group,
        case_group = case_group,
        .before = 1
      )
  } else {
    contrast <- matrix(0, nrow = ncol(design), ncol = 1, dimnames = list(colnames(design), comparison_id))
    contrast[case_group, 1] <- 1
    contrast[reference_group, 1] <- -1
    fit2 <- limma::eBayes(limma::contrasts.fit(fit, contrast))
    de <- limma::topTable(fit2, number = Inf, sort.by = "P", adjust.method = "BH") %>%
      tibble::rownames_to_column("Name") %>%
      dplyr::mutate(
        comparison_id = comparison_id,
        reference_group = reference_group,
        case_group = case_group,
        .before = 1
      )
  }

  result_file <- paste0("differential_expression__", comparison_id, ".csv")
  readr::write_csv(de, file.path(results_dir, result_file))
  if (first_result) {
    readr::write_csv(de, file.path(results_dir, "differential_expression.csv"))
    first_result <- FALSE
  }

  significant <- de %>% dplyr::filter(adj.P.Val <= 0.05)
  summary_rows[[length(summary_rows) + 1]] <- tibble::tibble(
    comparison_id = comparison_id,
    reference_group = reference_group,
    case_group = case_group,
    n_reference = length(reference_samples),
    n_case = length(case_samples),
    reference_samples = paste(reference_samples, collapse = ";"),
    case_samples = paste(case_samples, collapse = ";"),
    n_genes = nrow(de),
    n_significant = nrow(significant),
    n_up = sum(significant$logFC >= 1, na.rm = TRUE),
    n_down = sum(significant$logFC <= -1, na.rm = TRUE),
    top_gene = if (nrow(de) > 0) de$Name[[1]] else NA_character_,
    result_file = result_file
  )
  message("Wrote ", file.path(results_dir, result_file))
}

if (length(summary_rows) == 0) {
  stop("No valid differential-expression contrast could be run.")
}
summary <- dplyr::bind_rows(summary_rows)
readr::write_csv(summary, file.path(results_dir, "comparison_summary.csv"))
message("Wrote ", file.path(results_dir, "comparison_summary.csv"))
message("Wrote ", file.path(results_dir, "differential_expression.csv"))
