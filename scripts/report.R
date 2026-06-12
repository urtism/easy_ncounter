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
}

de_path <- file.path(results_dir, "differential_expression.csv")
if (file.exists(de_path)) {
  de <- readr::read_csv(de_path, show_col_types = FALSE) %>%
    dplyr::mutate(
      neg_log10_adj_p = -log10(pmax(adj.P.Val, .Machine$double.xmin)),
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
}

message("Wrote report figures to ", reports_dir)
