# easy-ncounter

Reproducible NanoString nCounter analysis pipeline with Python orchestration,
R-based normalization/statistics, and a local browser interface.

## What It Does

`easy-ncounter` handles a complete NanoString gene expression workflow:

1. imports `.RCC` files, samplesheets, raw count matrices, or final processed
   counts;
2. harmonizes samples, metadata, and probe annotations;
3. computes sample-level technical QC;
4. estimates background from negative controls;
5. writes background-corrected counts;
6. filters low-expression endogenous genes;
7. normalizes counts;
8. runs group comparisons with `limma`;
9. generates tables and plots that can be explored in the browser UI.

The project is not a replacement for `nf-core/nanostring` or NACHO yet, but it
is aligned with that style of workflow: traceable inputs, explicit QC,
controlled normalization, documented filtering, and saved intermediate outputs.

For a step-by-step explanation of the analysis workflow and why each stage is
performed, see [docs/analysis_workflow.md](docs/analysis_workflow.md).

## Installation

Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

R:

```bash
Rscript scripts/install_r_deps.R
```

The R installer creates a project-local `r-lib/` folder. `NanoStringNorm` and
`limma` are recommended. If `NanoStringNorm` is not available, normalization
falls back to library-size scaling.

## Supported Inputs

### RCC + RLF Mode

The main browser UI mode:

- one or more `.RCC` files;
- optional `.RLF` panel design file;
- metadata created or edited in the UI.

The UI automatically generates a samplesheet and copies RCC files into:

```text
ui_runs/<analysis_name>/input/rcc/
```

### Samplesheet Mode

nf-core-style samplesheet with at least:

```csv
RCC_FILE,RCC_FILE_NAME,SAMPLE_ID,condition
rcc/sample1.RCC,sample1.RCC,sample1,control
rcc/sample2.RCC,sample2.RCC,sample2,treated
```

`RCC_FILE` and `SAMPLE_ID` are required. `RCC_FILE_NAME` and metadata columns
are optional.

### Counts + Metadata Mode

Count CSV/TSV file with:

- optional `CodeClass`;
- `Name`;
- one column per sample.

Metadata CSV/TSV file with:

- `sample_id`;
- at least one experimental column, such as `condition`, `TIME`, `TREATMENT`, or
  `BATCH`.

This mode is intended for counts that still need processing. The pipeline runs
QC, filtering, and normalization before statistical analysis.

### Final Processed Counts Mode

Use this mode when you already have a final normalized or otherwise
analysis-ready count matrix.

The count file must contain:

- optional `CodeClass`;
- `Name`;
- one column per sample.

After uploading counts, the UI lets you download a metadata template with one
row per sample. The final metadata file must contain:

- `sample_id`, matching the sample count columns;
- at least one group column, such as `condition`.

In this mode, the Processing/QC module is skipped. The UI writes
`counts_normalized.csv`, `metadata.csv`, and a minimal `PREPROCESSED` QC table,
then you can proceed directly to statistical comparison. The low-count gene
filter is skipped for final processed counts so that the uploaded gene set is
preserved.

## Local Browser UI

Start:

```bash
easy-ncounter-ui
```

Then open:

```text
http://localhost:8501
```

The UI is organized into three modules.

### 1. Samples and Metadata

`Upload` page:

- upload `.RCC` and `.RLF` files;
- or upload counts and metadata;
- or upload final processed counts and download a metadata template;
- or upload a samplesheet;
- view the generated samplesheet immediately;
- edit only `SAMPLE_ID` in the samplesheet.

`Metadata management` page:

- `sample_id` is locked and synchronized with the samplesheet;
- add metadata columns;
- delete selected metadata columns;
- fill experimental groups, batches, timepoints, treatments, or other
  covariates.

Edits remain in memory while you move between modules. They reset only when a
new file set is uploaded.

### 2. Processing and QC

`Processing/QC parameters` page:

- choose normalization;
- choose the metadata column used for QC/PCA coloring;
- set main QC thresholds:
  - minimum library size relative to the median;
  - minimum detected endogenous probes relative to the median;
  - minimum positive controls relative to the median.

`Advanced QC options`:

- background: `mean(negative controls) + N * sd(negative controls)`;
- negative-control threshold: median + `N * MAD`;
- minimum counted FOV;
- minimum and maximum binding density;
- `FAIL` thresholds;
- gene filter: minimum count and minimum number of samples.

`QC evaluation` page:

- `PASS`, `WARN`, and `FAIL` summary;
- problematic samples;
- QC metric plots;
- full QC table.

### 3. Analysis and Plots

`Comparisons` page:

- choose the metadata group column;
- choose reference group;
- choose case group;
- run differential analysis.

`Data visualization` page:

- differential results;
- `adj.P.Val` and `logFC` filters;
- autocomplete gene search from filtered genes or all analyzed genes, including
  non-significant genes;
- interactive plots with volcano, gene search, and expression profile in one
  block;
- click a volcano gene to select it and open the expression plot;
- click a dynamic heatmap gene to select it in the volcano and open its
  expression profile;
- dynamic PCA and heatmap computed only on genes above the selected thresholds;
- normalized expression profiles for one gene or all threshold-selected genes;
- expression boxplots for the two comparison groups;
- static PCA, volcano, and heatmap images;
- table downloads.

## Technical QC

For a detailed explanation of each parameter, how it changes results, and why
the defaults are used, see
[docs/processing_parameters.md](docs/processing_parameters.md).

The pipeline computes QC at two levels.

### RCC Metrics

When available in RCC files:

- `FovCount`;
- `FovCounted`;
- `FovCounted / FovCount` fraction;
- `BindingDensity`;
- scanner/cartridge fields;
- declared RLF.

### Count Metrics

For each sample:

- `library_size`;
- `endogenous_library_size`;
- `detected_probes`;
- `detected_endogenous_probes`;
- `positive_control_sum`;
- `positive_control_cv`;
- `negative_control_mean`;
- `negative_control_sd`;
- `background_threshold`;
- `endogenous_above_background`.

### QC Decision

Each sample receives:

```text
qc_status: PASS / WARN / FAIL
qc_warnings
qc_fail_reasons
```

Example warnings/failures:

- `low_library_size`;
- `very_low_library_size`;
- `low_detected_endogenous_probes`;
- `very_low_detected_endogenous_probes`;
- `low_positive_controls`;
- `high_negative_controls`;
- `low_fov_counted_fraction`;
- `low_binding_density`;
- `high_binding_density`.

Samples marked as `FAIL` are excluded from the filtered counts used for
normalization and statistical analysis.

## Background, Filtering, and Normalization

### Background Correction

Background is estimated per sample from negative controls:

```text
background_threshold = mean(negative controls) + N * sd(negative controls)
```

The pipeline writes:

```text
counts_background_corrected.csv
```

with:

```text
max(raw_count - background_threshold, 0)
```

### Gene Filtering

The pipeline:

- removes positive and negative controls from differential analysis;
- keeps housekeeping and controls separate from endogenous gene testing;
- keeps only `Endogenous` genes;
- filters low-expression genes by minimum count and number of samples;
- excludes samples with `qc_status == FAIL`.

Outputs:

```text
counts_filtered.csv
gene_filter_decisions.csv
```

### Normalization

Preferred normalization input:

```text
counts_filtered.csv
```

If unavailable, the pipeline uses `counts_background_corrected.csv`, then
`counts_raw.csv`.

Methods:

- `NanoStringNorm`, if installed;
- `library_size` fallback if `NanoStringNorm` is unavailable.

Output:

```text
counts_normalized.csv
```

## Statistical Analysis

Differential analysis uses `limma` on:

```text
log2(counts_normalized + 1)
```

The contrast is:

```text
case_group - reference_group
```

Output:

```text
differential_expression.csv
```

Key columns:

- `Name`;
- `logFC`;
- `AveExpr`;
- `t`;
- `P.Value`;
- `adj.P.Val`;
- `B`.

## Main Outputs

```text
counts_raw.csv
metadata.csv
probe_annotation.csv
qc_summary.csv
sample_qc_decisions.csv
counts_background_corrected.csv
counts_filtered.csv
gene_filter_decisions.csv
counts_normalized.csv
differential_expression.csv
```

Static report figures:

```text
qc_library_size.png
pca.png
heatmap_top_variable.png
volcano_plot.png
```

## CLI

Run the full pipeline:

```bash
easy-ncounter run --config config/pipeline.yaml
```

Run individual steps:

```bash
easy-ncounter prepare --config config/pipeline.yaml
easy-ncounter normalize --config config/pipeline.yaml
easy-ncounter differential --config config/pipeline.yaml
easy-ncounter report --config config/pipeline.yaml
```

## Repository Layout

```text
config/                     # Example pipeline configuration
data/                       # Example or local data
docs/                       # Documentation
r/                          # Shared R utilities
scripts/                    # R analysis scripts
src/easy_ncounter/          # Python package and browser UI
tests/                      # Tests
ui_runs/                    # Browser UI run outputs
```

## Notes and Limitations

- Statistical analysis currently supports simple group-vs-group comparisons.
  More complex models and covariate adjustment are future work.
- Technical QC is strongest when RCC metrics and control probes are available.
- Final processed counts mode cannot recalculate RCC-level QC because raw RCC
  metrics are not available.
- Thresholds should be documented when changed from defaults.

## Planned Improvements

1. richer contrast builder for multi-factor designs;
2. manual QC override for samples;
3. richer interactive report export;
4. optional NACHO-style QC summaries;
5. improved support for housekeeping gene selection.
