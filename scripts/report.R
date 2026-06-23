#!/usr/bin/env Rscript
project_lib <- Sys.getenv("NC_R_LIB", unset = file.path(getwd(), "r-lib"))
if (dir.exists(project_lib)) {
  .libPaths(c(project_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(yaml)
})

source("r/ncounter_utils.R")

safe_zscore_rows <- function(matrix) {
  scaled <- t(scale(t(matrix)))
  scaled[!is.finite(scaled)] <- 0
  scaled
}

write_score_module <- function(matrix, annotation, annotation_column, prefix, label, reports_dir, results_dir) {
  if (!annotation_column %in% colnames(annotation)) {
    message("Skipping ", label, " scoring: no annotation column found.")
    return(FALSE)
  }
  ann <- annotation %>%
    dplyr::filter(!is.na(.data[[annotation_column]]), trimws(as.character(.data[[annotation_column]])) != "")
  if (nrow(ann) == 0) {
    message("Skipping ", label, " scoring: annotation is empty.")
    return(FALSE)
  }
  score_rows <- list()
  for (term in sort(unique(as.character(ann[[annotation_column]])))) {
    probes <- intersect(ann$probe_id[as.character(ann[[annotation_column]]) == term], rownames(matrix))
    if (length(probes) == 0) {
      next
    }
    score_rows[[length(score_rows) + 1]] <- c(feature = term, colMeans(matrix[probes, , drop = FALSE], na.rm = TRUE))
  }
  if (length(score_rows) == 0) {
    message("Skipping ", label, " scoring: no annotated probes were found in normalized counts.")
    return(FALSE)
  }
  scores <- as.data.frame(do.call(rbind, score_rows), stringsAsFactors = FALSE)
  colnames(scores)[1] <- ifelse(prefix == "pathway", "pathway", "cell_type")
  sample_cols <- setdiff(colnames(scores), colnames(scores)[1])
  scores[sample_cols] <- lapply(scores[sample_cols], as.numeric)
  readr::write_csv(scores, file.path(results_dir, paste0(prefix, "_scores.csv")))

  score_matrix <- scores %>%
    tibble::column_to_rownames(colnames(scores)[1]) %>%
    as.matrix()
  z <- safe_zscore_rows(score_matrix)
  z_out <- as.data.frame(z) %>% tibble::rownames_to_column(colnames(scores)[1])
  readr::write_csv(z_out, file.path(results_dir, paste0(prefix, "_scores_z.csv")))

  if (requireNamespace("pheatmap", quietly = TRUE) && nrow(z) >= 2 && ncol(z) >= 2) {
    png(file.path(reports_dir, paste0(prefix, "_heatmap.png")), width = 1000, height = 850)
    pheatmap::pheatmap(z, fontsize_row = 8, main = label)
    dev.off()
  }
  TRUE
}

plot_count_distribution <- function(path, output_path, title) {
  if (!file.exists(path)) {
    message("Skipping distribution plot for missing matrix: ", path)
    return()
  }
  data <- read_counts_matrix(path)
  values <- as.numeric(data$matrix)
  values <- values[is.finite(values)]
  if (length(values) == 0) {
    return()
  }
  plot <- ggplot(data.frame(log_count = log2(values + 1)), aes(x = log_count)) +
    geom_density(fill = "#7A9E9F", alpha = 0.45, color = "#4F6D7A") +
    labs(x = "log2(count + 1)", y = "Density", title = title) +
    theme_minimal(base_size = 11)
  ggsave(output_path, plot = plot, width = 7, height = 4.5, dpi = 150)
}

safe_plot_name <- function(value) {
  safe <- gsub("[^A-Za-z0-9_-]", "_", as.character(value))
  safe <- gsub("_+", "_", safe)
  gsub("^_+|_+$", "", safe)
}

opt <- parse_cli_args()
cfg <- read_pipeline_config(opt$config)

results_dir <- cfg$outputs$results_dir
reports_dir <- cfg$outputs$reports_dir
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)

qc <- readr::read_csv(file.path(results_dir, "qc_summary.csv"), show_col_types = FALSE)
qc_plot <- ggplot(qc, aes(x = reorder(sample_id, library_size), y = library_size)) +
  geom_col(fill = "#3A7CA5") +
  coord_flip() +
  labs(x = NULL, y = "Library size") +
  theme_minimal(base_size = 11)
ggsave(
  file.path(reports_dir, "qc_library_size.png"),
  plot = qc_plot,
  width = 8,
  height = 5,
  dpi = 150
)

if ("hk_geomean" %in% colnames(qc) && any(is.finite(qc$hk_geomean))) {
  hk_plot <- ggplot(qc, aes(x = hk_geomean)) +
    geom_histogram(bins = 20, fill = "#6C8EAD", color = "white") +
    labs(x = "Housekeeper geomean", y = "Samples") +
    theme_minimal(base_size = 11)
  ggsave(file.path(reports_dir, "housekeeper_geomean_qc.png"), plot = hk_plot, width = 6, height = 4, dpi = 150)
}

count_data <- read_counts_matrix(file.path(results_dir, "counts_normalized.csv"))
matrix <- log2(count_data$matrix + 1)
metadata_path <- if (file.exists(file.path(results_dir, "metadata.csv"))) {
  file.path(results_dir, "metadata.csv")
} else {
  cfg$inputs$metadata
}
metadata <- readr::read_csv(metadata_path, show_col_types = FALSE)
metadata <- add_analysis_group_column(metadata, cfg)
group_col <- cfg$analysis$group_column

row_vars <- apply(matrix, 1, stats::var, na.rm = TRUE)
variable_matrix <- matrix[is.finite(row_vars) & row_vars > 0, , drop = FALSE]

if (nrow(variable_matrix) >= 2 && ncol(variable_matrix) >= 2) {
  pca <- prcomp(t(variable_matrix), scale. = TRUE)
  pca_df <- as.data.frame(pca$x[, 1:2, drop = FALSE]) %>%
    tibble::rownames_to_column("sample_id")
  pca_df <- dplyr::left_join(pca_df, metadata, by = "sample_id")

  pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = .data[[group_col]], label = sample_id)) +
    geom_point(size = 3) +
    labs(color = group_col) +
    theme_minimal(base_size = 11)
  ggsave(file.path(reports_dir, "pca.png"), plot = pca_plot, width = 6, height = 5, dpi = 150)

  mds_points <- cmdscale(stats::dist(t(variable_matrix)), k = 2)
  mds_df <- as.data.frame(mds_points)
  colnames(mds_df) <- c("Dim1", "Dim2")
  mds_df <- mds_df %>% tibble::rownames_to_column("sample_id") %>% dplyr::left_join(metadata, by = "sample_id")
  mds_plot <- ggplot(mds_df, aes(x = Dim1, y = Dim2, color = .data[[group_col]], label = sample_id)) +
    geom_point(size = 3) +
    labs(color = group_col) +
    theme_minimal(base_size = 11)
  ggsave(file.path(reports_dir, "mds_plot.png"), plot = mds_plot, width = 6, height = 5, dpi = 150)
} else {
  message("Skipping PCA: fewer than two variable genes or samples after filtering zero-variance genes.")
}

if (requireNamespace("pheatmap", quietly = TRUE) && nrow(variable_matrix) >= 2 && ncol(variable_matrix) >= 2) {
  top_n <- cfg$analysis$top_variable_genes %||% 50
  vars <- sort(apply(variable_matrix, 1, stats::var), decreasing = TRUE)
  selected <- names(head(vars, min(top_n, length(vars))))
  png(file.path(reports_dir, "heatmap_top_variable.png"), width = 1000, height = 900)
  pheatmap::pheatmap(variable_matrix[selected, , drop = FALSE], scale = "row", fontsize_row = 7)
  dev.off()

  correlation <- stats::cor(variable_matrix, method = "pearson", use = "pairwise.complete.obs")
  png(file.path(reports_dir, "sample_correlation_heatmap.png"), width = 900, height = 800)
  pheatmap::pheatmap(correlation, main = "Sample correlation")
  dev.off()
}

plot_count_distribution(file.path(results_dir, "counts_raw.csv"), file.path(reports_dir, "count_distribution_raw.png"), "Raw counts")
plot_count_distribution(
  file.path(results_dir, "counts_background_corrected.csv"),
  file.path(reports_dir, "count_distribution_background_corrected.png"),
  "Background-corrected counts"
)
plot_count_distribution(
  file.path(results_dir, "counts_normalized.csv"),
  file.path(reports_dir, "count_distribution_normalized.png"),
  "Normalized counts"
)

raw_counts_path <- file.path(results_dir, "counts_raw.csv")
if (file.exists(raw_counts_path)) {
  raw_counts <- readr::read_csv(raw_counts_path, show_col_types = FALSE)
  positives <- raw_counts %>% dplyr::filter(grepl("positive", CodeClass, ignore.case = TRUE))
  if (nrow(positives) > 0) {
    sample_cols <- setdiff(colnames(positives), c("CodeClass", "Name"))
    positive_long <- data.frame(
      sample_id = rep(sample_cols, each = nrow(positives)),
      control = rep(positives$Name, times = length(sample_cols)),
      count = as.numeric(as.matrix(positives[, sample_cols, drop = FALSE]))
    )
    positive_plot <- ggplot(positive_long, aes(x = control, y = count, group = sample_id, color = sample_id)) +
      geom_line(alpha = 0.6) +
      geom_point(size = 1.5) +
      coord_flip() +
      labs(x = NULL, y = "Raw positive-control count") +
      theme_minimal(base_size = 10) +
      theme(legend.position = "none")
    ggsave(file.path(reports_dir, "positive_control_qc.png"), plot = positive_plot, width = 7, height = 5, dpi = 150)
  }
}

annotation_path <- file.path(results_dir, "probe_annotation_parsed.csv")
if (file.exists(annotation_path)) {
  annotation <- readr::read_csv(annotation_path, show_col_types = FALSE)
  write_score_module(matrix, annotation, "pathway", "pathway", "Pathway marker scores", reports_dir, results_dir)
  if (write_score_module(matrix, annotation, "cell_type", "cell_type", "Cell type marker-expression scores", reports_dir, results_dir)) {
    message("Cell type scores are marker-expression scores, not cell abundance deconvolution.")
  }
} else {
  message("Skipping pathway/cell-type scoring: parsed probe annotation is missing.")
}

de_path <- file.path(results_dir, "differential_expression.csv")
single_probe_plotted <- FALSE
if (file.exists(de_path)) {
  de <- readr::read_csv(de_path, show_col_types = FALSE) %>%
    dplyr::mutate(
      plot_adj_p = dplyr::if_else(is.na(adj.P.Val), 1, adj.P.Val),
      neg_log10_adj_p = -log10(pmax(plot_adj_p, .Machine$double.xmin)),
      regulation = dplyr::case_when(
        adj.P.Val <= 0.05 & logFC >= 1 ~ "up",
        adj.P.Val <= 0.05 & logFC <= -1 ~ "down",
        TRUE ~ "not_significant"
      )
    )
  volcano_plot <- ggplot(de, aes(x = logFC, y = neg_log10_adj_p, color = regulation)) +
    geom_point(alpha = 0.75, size = 1.8) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "#777777", linewidth = 0.4) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "#777777", linewidth = 0.4) +
    scale_color_manual(
      values = c(up = "#B23A48", down = "#2F6690", not_significant = "#8A8F98")
    ) +
    labs(x = "log2 fold change", y = "-log10 adjusted p-value", color = NULL) +
    theme_minimal(base_size = 11)
  ggsave(file.path(reports_dir, "volcano_plot.png"), plot = volcano_plot, width = 7, height = 5, dpi = 150)

  single_dir <- file.path(reports_dir, "single_probe_plots")
  dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)
  top_n <- cfg$analysis$single_probe_top_n %||% 20
  selected_genes <- head(de$Name[order(de$adj.P.Val, -abs(de$logFC), na.last = TRUE)], top_n)
  interest <- cfg$analysis$single_probe_genes %||% character()
  selected_genes <- unique(c(selected_genes, unlist(interest, use.names = FALSE)))
  selected_genes <- intersect(selected_genes, rownames(count_data$matrix))
  for (gene in selected_genes) {
    expression_df <- tibble::tibble(
      sample_id = colnames(count_data$matrix),
      expression = as.numeric(log2(count_data$matrix[gene, ] + 1))
    ) %>%
      dplyr::left_join(metadata, by = "sample_id")
    probe_plot <- ggplot(expression_df, aes(x = .data[[group_col]], y = expression, color = .data[[group_col]])) +
      geom_boxplot(outlier.shape = NA, alpha = 0.3) +
      geom_jitter(width = 0.12, size = 2) +
      labs(x = group_col, y = "log2(normalized count + 1)", title = gene) +
      theme_minimal(base_size = 11) +
      theme(legend.position = "none")
    ggsave(file.path(single_dir, paste0(safe_plot_name(gene), ".png")), plot = probe_plot, width = 5.5, height = 4.5, dpi = 150)
  }
  single_probe_plotted <- length(selected_genes) > 0
}

if (!single_probe_plotted && nrow(variable_matrix) > 0) {
  single_dir <- file.path(reports_dir, "single_probe_plots")
  dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)
  top_n <- cfg$analysis$single_probe_top_n %||% 20
  vars <- sort(apply(variable_matrix, 1, stats::var), decreasing = TRUE)
  selected_genes <- names(head(vars, min(top_n, length(vars))))
  for (gene in selected_genes) {
    expression_df <- tibble::tibble(
      sample_id = colnames(count_data$matrix),
      expression = as.numeric(log2(count_data$matrix[gene, ] + 1))
    ) %>%
      dplyr::left_join(metadata, by = "sample_id")
    probe_plot <- ggplot(expression_df, aes(x = .data[[group_col]], y = expression, color = .data[[group_col]])) +
      geom_boxplot(outlier.shape = NA, alpha = 0.3) +
      geom_jitter(width = 0.12, size = 2) +
      labs(x = group_col, y = "log2(normalized count + 1)", title = gene) +
      theme_minimal(base_size = 11) +
      theme(legend.position = "none")
    ggsave(file.path(single_dir, paste0(safe_plot_name(gene), ".png")), plot = probe_plot, width = 5.5, height = 4.5, dpi = 150)
  }
}

message("Wrote report figures to ", reports_dir)
