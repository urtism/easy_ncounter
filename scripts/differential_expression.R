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
metadata_path <- cfg$inputs$metadata
group_col <- cfg$analysis$group_column
reference_group <- cfg$analysis$reference_group
case_group <- cfg$analysis$case_group

metadata <- readr::read_csv(metadata_path, show_col_types = FALSE)
count_data <- read_counts_matrix(file.path(results_dir, "counts_normalized.csv"))
matrix <- count_data$matrix

matrix <- filter_low_counts(
  matrix,
  min_count = cfg$analysis$min_count %||% 10,
  min_samples = cfg$analysis$min_samples %||% 2
)

sample_order <- intersect(metadata$sample_id, colnames(matrix))
metadata <- metadata %>% dplyr::filter(sample_id %in% sample_order)
matrix <- matrix[, metadata$sample_id, drop = FALSE]

if (!requireNamespace("limma", quietly = TRUE)) {
  stop("Package 'limma' is required for differential expression.")
}

metadata[[group_col]] <- factor(metadata[[group_col]], levels = c(reference_group, case_group))
design <- model.matrix(stats::as.formula(paste0("~ 0 + ", group_col)), data = metadata)
colnames(design) <- levels(metadata[[group_col]])
fit <- limma::lmFit(log2(matrix + 1), design)
contrast <- limma::makeContrasts(contrasts = paste0(case_group, "-", reference_group), levels = design)
fit2 <- limma::eBayes(limma::contrasts.fit(fit, contrast))
de <- limma::topTable(fit2, number = Inf, sort.by = "P") %>%
  tibble::rownames_to_column("Name")

readr::write_csv(de, file.path(results_dir, "differential_expression.csv"))
message("Wrote ", file.path(results_dir, "differential_expression.csv"))
