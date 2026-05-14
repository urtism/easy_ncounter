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
pca <- prcomp(t(matrix), scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2, drop = FALSE]) %>%
  tibble::rownames_to_column("sample_id")
metadata <- readr::read_csv(cfg$inputs$metadata, show_col_types = FALSE)
pca_df <- dplyr::left_join(pca_df, metadata, by = "sample_id")
group_col <- cfg$analysis$group_column

pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = .data[[group_col]], label = sample_id)) +
  geom_point(size = 3) +
  labs(color = group_col) +
  theme_minimal(base_size = 11)
ggsave(file.path(reports_dir, "pca.png"), plot = pca_plot, width = 6, height = 5, dpi = 150)

if (requireNamespace("pheatmap", quietly = TRUE)) {
  top_n <- cfg$analysis$top_variable_genes %||% 50
  vars <- sort(apply(matrix, 1, stats::var), decreasing = TRUE)
  selected <- names(head(vars, min(top_n, length(vars))))
  png(file.path(reports_dir, "heatmap_top_variable.png"), width = 1000, height = 900)
  pheatmap::pheatmap(matrix[selected, , drop = FALSE], scale = "row", fontsize_row = 7)
  dev.off()
}

message("Wrote report figures to ", reports_dir)
