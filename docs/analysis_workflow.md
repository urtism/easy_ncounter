# easy-ncounter Analysis Workflow

This document explains the `easy-ncounter` analysis workflow step by step: what
each stage does, why it is performed, and which outputs are produced.

The goal of the workflow is to make NanoString nCounter gene expression analysis
traceable and reproducible. Every major transformation writes an intermediate
file, so users can inspect how the final statistics were obtained.

Easy_nCounter is not a ROSALIND clone and is not official Bruker/NanoString
software. It is an independent local workflow partially aligned with nCounter
analysis principles, with optional Bruker-like QC summaries and ROSALIND-like
exploratory outputs.

## 1. Input Selection

`easy-ncounter` supports multiple entry points because NanoString data can arrive
in different forms.

Supported modes:

- `RCC + RLF`: raw NanoString RCC files, with an optional panel design file.
- `Samplesheet RCC`: an nf-core-style samplesheet pointing to RCC files.
- `Counts + Metadata`: a raw count matrix plus a metadata table.
- `Final processed counts`: a normalized or otherwise analysis-ready count
  matrix plus metadata.

Why this is done:

- raw RCC files allow the most complete technical QC;
- raw count matrices are useful when RCC parsing has already been done
  elsewhere;
- final processed counts let users continue from external tools such as NACHO or
  other NanoString workflows without repeating preprocessing.

Main outputs:

- copied input files under `ui_runs/<analysis_name>/input/`;
- `pipeline.yaml`, which records the selected parameters and file paths.

## 2. Samplesheet and Metadata Preparation

The UI creates or imports a samplesheet and aligns it with the metadata table.
In RCC mode, sample IDs are generated from the uploaded RCC files and can be
edited by the user. The metadata table always contains a locked `sample_id`
column and user-defined experimental columns such as `condition`, `TIME`,
`TREATMENT`, `BATCH`, or any other grouping variable.

Why this is done:

- statistical comparisons require sample identities to match exactly between
  count data and metadata;
- keeping `sample_id` locked prevents accidental desynchronization;
- custom metadata columns make it possible to analyze the same dataset through
  different biological questions.

Main outputs:

- `metadata.csv`;
- a synchronized samplesheet for RCC-based analyses.

## 3. Count Import

For RCC-based analyses, the pipeline parses probe counts from each RCC file and
builds a sample-by-gene count matrix. For count-matrix inputs, the matrix is
read directly. Probe annotation is kept when available, especially `CodeClass`
and `Name`.

Why this is done:

- all downstream steps need a consistent count table format;
- `CodeClass` identifies endogenous genes, positive controls, negative controls,
  and housekeeping probes;
- preserving probe names and classes makes QC, filtering, and interpretation
  more transparent.

Main output:

- `counts_raw.csv`.
- `probe_annotation.csv`;
- `probe_annotation_parsed.csv`.

## 4. Sample-Level Technical QC

The pipeline calculates technical QC metrics for each sample. These include
library size, endogenous library size, detected probes, detected endogenous
probes, positive-control signal, negative-control background, and the number of
endogenous probes above background. When RCC metrics are available, the pipeline
also uses fields such as counted FOV fraction and binding density.

Samples are labeled as:

- `PASS`: no major technical concern;
- `WARN`: suspicious but still retained;
- `FAIL`: severe technical concern and excluded from downstream filtered
  analyses.

Why this is done:

- weak or technically compromised samples can distort normalization,
  fold-changes, and p-values;
- separating `WARN` from `FAIL` keeps the analysis conservative without removing
  borderline samples too aggressively;
- writing explicit QC flags lets users review which samples influenced the final
  results.

Main outputs:

- `qc_summary.csv`;
- `sample_qc.csv`;
- `sample_qc_decisions.csv`;
- QC summaries and plots in the report folder.

## 5. Background Estimation and Correction

For each sample, background is estimated from negative controls:

```text
background_threshold = mean(negative controls) + N * sd(negative controls)
```

Counts are then corrected as:

```text
corrected_count = max(raw_count - background_threshold, 0)
```

Why this is done:

- NanoString assays include low-level technical background;
- negative controls provide a sample-specific estimate of that background;
- subtracting background reduces noise before filtering and normalization.

Main output:

- `counts_background_corrected.csv`.

## 6. Gene Filtering

The pipeline focuses differential testing on endogenous genes with enough signal.
Control probes are excluded from differential expression. Genes are retained
when they pass minimum count and sample-detection requirements. The filter also
protects genes that are detected only in one biological group, because those can
be biologically meaningful.

Samples marked as `FAIL` are excluded before the filtered matrix is created.

Why this is done:

- very low-count genes produce unstable fold-changes and unreliable statistics;
- removing unsupported genes reduces multiple-testing burden;
- keeping group-specific detected genes avoids losing biologically relevant
  on/off signals.

Main outputs:

- `counts_filtered.csv`;
- `gene_filter_decisions.csv`.

## 7. Normalization

The normalized matrix is produced from the best available processed count table,
usually `counts_filtered.csv` after background correction and filtering.

When `NanoStringNorm` is available and selected, the pipeline uses
NanoString-specific normalization. If it is not available, the pipeline falls
back to library-size scaling. Optional housekeeping modes can normalize by all
housekeeping probes (`hk_geomean_all`) or by a geNorm-like stable housekeeping
subset (`hk_geomean_geNorm`).

Why this is done:

- samples can differ in total signal because of loading, hybridization, or
  acquisition variability;
- normalization makes samples more comparable before statistical testing;
- using a clear fallback keeps the pipeline executable even when optional R
  packages are not installed.

Main output:

- `counts_normalized.csv`.
- `normalization_audit.csv`;
- `selected_housekeepers.csv` when housekeeping normalization is used.

## 8. Special Handling for Final Processed Counts

When users upload final processed counts, the pipeline treats the matrix as
already analysis-ready. Processing/QC, background correction, gene filtering,
and normalization are skipped. The uploaded matrix is written directly as
`counts_normalized.csv`, and QC is marked as `PREPROCESSED`.

The statistical low-count filter is also skipped in this mode.

Why this is done:

- externally processed data should not be transformed a second time unless the
  user explicitly chooses to reprocess it;
- preserving the uploaded gene set makes comparisons with external tools easier;
- users can move directly to group comparisons and visualization.

Main outputs:

- `counts_normalized.csv`;
- `metadata.csv`;
- minimal `sample_qc.csv` with `PREPROCESSED` status.

## 9. Differential Expression

The differential-expression step reads normalized counts and metadata, then
compares one case group against one reference group selected in the UI. The
current implementation uses `limma` on `log2(count + 1)` transformed values.

The model uses the selected metadata column as the grouping variable and tests:

```text
case group - reference group
```

Why this is done:

- `limma` is robust for small and medium gene-expression datasets;
- log transformation stabilizes variance and makes fold-change interpretation
  more natural;
- explicitly choosing reference and case groups makes the direction of `logFC`
  clear.

Interpretation:

- positive `logFC`: higher expression in the case group;
- negative `logFC`: higher expression in the reference group;
- `P.Value`: nominal p-value;
- `adj.P.Val`: Benjamini-Hochberg FDR-adjusted p-value.

Main output:

- `differential_expression.csv`.

## 10. Optional Marker Score Modules

When parsed probe annotation contains pathway or cell-type marker columns, the
reporting step calculates simple marker scores from normalized counts. Scores
are means of `log2(normalized count + 1)` across probes assigned to each pathway
or marker cell type. Z-scored versions are written for heatmap visualization.

Cell type scores are marker-expression scores only. They are not cell abundance
deconvolution.

Main outputs:

- `pathway_scores.csv`;
- `pathway_scores_z.csv`;
- `pathway_heatmap.png`;
- `cell_type_scores.csv`;
- `cell_type_scores_z.csv`;
- `cell_type_heatmap.png`.

## 11. Static Report Generation

The reporting step creates standard QC and analysis figures, including library
size summaries, PCA, heatmap, and volcano plot. These plots are saved as image
files so the analysis can be reviewed outside the browser UI.

Why this is done:

- static figures provide a reproducible record of the run;
- PCA helps identify global sample structure and possible outliers;
- heatmaps summarize expression patterns across selected genes;
- volcano plots combine effect size and statistical evidence.

Main outputs:

- `qc_library_size.png`;
- `pca.png`;
- `mds_plot.png`;
- `sample_correlation_heatmap.png`;
- count-distribution and control/QC plots when required inputs exist;
- `heatmap_top_variable.png`;
- `volcano_plot.png`.
- `single_probe_plots/`.

## 12. Interactive Results Exploration

The browser UI provides an interactive results page. Users can filter genes by
adjusted p-value and absolute log fold-change, search for any analyzed gene,
click points in the volcano plot, and inspect expression values across samples
and groups.

Dynamic plots can be recalculated from the currently selected gene set:

- volcano plot over all analyzed genes;
- heatmap using genes above the selected thresholds;
- PCA using genes above the selected thresholds;
- expression profile for a selected gene;
- group boxplot for the selected comparison.

Why this is done:

- threshold filters are useful for exploration but should not hide access to
  non-significant genes;
- linking volcano, heatmap, search, and expression plots helps users move from a
  statistical signal to the underlying sample-level data;
- dynamic heatmap and PCA views show whether selected genes separate the chosen
  groups.

Main outputs:

- interactive plots in the Streamlit UI;
- downloadable result tables and generated files.

## 13. Result Interpretation

The final interpretation should combine technical QC, normalization behavior,
statistical output, and biological context.

Recommended review order:

1. Check whether any samples failed QC.
2. Inspect background and normalized count distributions.
3. Confirm that the selected metadata groups are correct.
4. Review PCA for outliers or unexpected clustering.
5. Inspect the volcano plot for strong and statistically supported effects.
6. Open expression plots for genes of interest.
7. Compare filtered and unfiltered gene lists when searching for specific genes.

Why this is done:

- a significant gene can still be driven by one problematic sample;
- a biologically important gene may be non-significant in small experiments;
- looking at raw expression patterns prevents overinterpreting a single p-value.

## 14. Important Notes

Different tools can produce different results from the same experiment because
they may use different background correction, normalization, filtering,
housekeeping handling, and statistical models. When comparing `easy-ncounter`
with NACHO, `nf-core/nanostring`, or other software, check:

- whether the same samples were included;
- whether the same genes were tested;
- whether background correction was applied;
- which normalization method was used;
- whether housekeeping genes were used;
- whether low-count filters were applied;
- which group was the reference and which was the case;
- whether statistics were calculated on raw, normalized, or transformed counts.

The safest comparison is made by aligning these choices explicitly before
interpreting differences in fold-change or p-values.
