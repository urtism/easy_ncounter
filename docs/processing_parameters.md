# Processing, QC, and Analysis Parameters

This document explains what the main `easy-ncounter` parameters do, how they
change the final results, and why the standard values are used as conservative
defaults.

The pipeline is designed for NanoString nCounter gene expression data. Defaults
do not replace biological judgement, but they provide a robust starting point
for small and medium-sized datasets where overly aggressive thresholds can
remove useful signal.

## Overall Workflow

For RCC files or raw count matrices, the pipeline runs:

1. count and metadata import;
2. sample-level technical QC;
3. background correction;
4. exclusion of samples marked as `FAIL`;
5. low-expression gene filtering;
6. normalization;
7. differential analysis with `limma`;
8. report and plot generation.

For `Final processed counts` input, the uploaded matrix is treated as already
ready for analysis. In that mode, technical QC, background correction, and
normalization are skipped. The UI writes `counts_normalized.csv` directly and
marks QC as `PREPROCESSED`. The statistical low-count filter is also skipped so
that the gene set uploaded from the external software is preserved.

## Sample QC Parameters

### Minimum Library vs Median

Default: `0.5`

Adds a warning if a sample total library size is below 50% of the sample median.

Effect on results:

- low-count samples can produce unstable estimates;
- samples marked as `FAIL` are removed from the filtered matrix;
- samples marked only as `WARN` remain included but are highlighted.

Why this default:

- it detects clearly weak samples;
- it avoids penalizing biologically heterogeneous datasets too strongly;
- it uses the median, which is less sensitive to outliers.

### FAIL Minimum Library vs Median

Default: `0.25`

Marks a sample as `FAIL` if its library size is below 25% of the median.

Effect on results:

- the sample is excluded from filtered counts;
- it does not contribute to normalization or differential analysis.

Why this default:

- it separates truly problematic samples from merely weak samples;
- it reduces the risk of excluding biologically valid low-signal samples.

### Minimum Detected Probes vs Median

Warning default: `0.5`

Fail default: `0.25`

Compares the number of detected endogenous probes in each sample with the
dataset median.

Effect on results:

- very few detected genes may indicate poor sample quality;
- `FAIL` samples are excluded;
- `WARN` samples remain visible in QC summaries.

Why these defaults:

- `0.5` flags suspicious samples without being too aggressive;
- `0.25` identifies samples with severe signal loss.

### Minimum Positive Controls vs Median

Default: `0.5`

Adds a warning if the positive control sum is below 50% of the sample median.

Effect on results:

- may indicate hybridization, imaging, or acquisition issues;
- currently produces a warning rather than automatic exclusion.

Why this default:

- positive controls are important technical indicators;
- a warning threshold avoids automatic exclusion when the rest of QC is
  acceptable.

### Negative Control Threshold

Default: `median + 3 * MAD`

Detects samples with unusually high negative-control background.

Effect on results:

- adds the `high_negative_controls` warning;
- high background can reduce the number of genes detected after correction.

Why MAD is used:

- MAD is robust to outliers;
- `3 * MAD` is a common conservative threshold for clear deviations.

### Minimum Counted FOV

Default: `0.75`

Used when RCC files contain `FovCount` and `FovCounted`. A sample is marked as
`FAIL` if `FovCounted / FovCount` is below 75%.

Effect on results:

- samples with incomplete imaging/acquisition are excluded;
- the parameter is ignored when RCC metrics are unavailable.

Why this default:

- below this threshold, the measurement may represent too little of the
  acquired field;
- it is a strong technical filter, but only when the metric is available.

### Minimum and Maximum Binding Density

Minimum default: `0.05`

Maximum default: `2.25`

Used when RCC files contain `BindingDensity`.

Effect on results:

- very low values may indicate weak signal;
- very high values may indicate saturation or technical issues;
- in the current pipeline these produce warnings.

Why these defaults:

- they are conservative limits for suspicious technical ranges;
- they do not automatically exclude the sample because interpretation depends on
  the panel and dataset.

## Background Correction

### Negative Mean + N SD

Default: `2.0`

For each sample the pipeline computes:

```text
background_threshold = mean(negative controls) + N * sd(negative controls)
```

Then it writes:

```text
corrected_count = max(raw_count - background_threshold, 0)
```

Effect on results:

- reduces background noise;
- can set weakly expressed genes to zero;
- directly affects downstream low-expression filtering.

Why the default is 2:

- it balances noise removal and retention of weak but plausible signal;
- higher values are stricter and may remove borderline signal;
- lower values keep more noise in the matrix.

## Gene Filtering

### Minimum Count

Default: `10`

A gene is considered detected in a sample if its count is at least 10.

Effect on results:

- removes very low-expression genes;
- reduces false positives and statistical instability;
- can remove biologically interesting but very weak genes.

Why this default:

- it is a simple, cautious threshold separating minimal signal from noise;
- it works well as a starting point for targeted NanoString panels.

### Minimum Samples

Default: `2`

A gene passes the global filter if it reaches the minimum count in at least two
samples.

Effect on results:

- avoids testing genes supported by only one isolated value;
- stabilizes fold-change estimates;
- may need cautious lowering in very small datasets.

Why this default:

- it requires minimal reproducibility;
- it remains permissive for experiments with few replicates.

### Minimum Group Samples

Internal default: `1`

A gene can also be retained if it is detected in at least one sample of a group.

Effect on results:

- protects genes expressed only in one condition;
- avoids losing truly group-specific genes.

Why this default:

- in biological comparisons, a gene can be active only in the case group or only
  in the reference group;
- a global-only filter could remove those genes.

## Normalization

### `nanostringnorm`

UI default: `nanostringnorm`

If the R package `NanoStringNorm` is available, the pipeline uses:

```text
CodeCount = geo.mean
Background = mean.2sd
SampleContent = housekeeping.geo.mean
```

Effect on results:

- uses NanoString-specific controls and sample-content adjustment;
- can produce results different from library-size normalization;
- is closer to established NanoString workflows.

Why this is the default:

- it uses a method designed for NanoString data;
- it is preferred when controls and housekeeping genes are appropriate for the
  panel.

### `library_size`

Automatic fallback if `NanoStringNorm` is not installed, or a manual option.

It computes:

```text
scale_factor = median(library_size) / sample_library_size
```

Effect on results:

- makes samples with different total counts more comparable;
- does not explicitly use positive controls, negative controls, or housekeeping
  genes;
- can be more stable when housekeeping controls are unreliable.

Why it exists:

- it lets the pipeline run without optional R dependencies;
- it is transparent and easy to interpret;
- it is useful for already filtered matrices or panels without robust
  housekeeping genes.

## Statistical Analysis Parameters

### Group Columns

Defines which metadata column, or combination of metadata columns, is used for
differential comparison.

In the UI, selecting multiple columns creates combined subgroups. For example,
selecting `DISEASE` and `TIME` creates labels such as:

```text
DISEASE=control | TIME=T0
DISEASE=control | TIME=T1
DISEASE=disease | TIME=T0
DISEASE=disease | TIME=T1
```

Effect on results:

- changes the contrast being tested;
- also controls colors and boxplots in the interactive plots.

Recommended choice:

- use the main biological variable, such as `condition`, `TREATMENT`, or
  `TIME`;
- select multiple variables when the comparison should be made inside combined
  subgroups, such as `DISEASE + TIME`;
- avoid technical columns such as batch unless the goal is a technical
  comparison.

### Reference Group, Case Group, and Contrasts

The basic contrast is:

```text
case_group - reference_group
```

For multiple comparisons, the pipeline stores a list of contrasts:

```yaml
contrasts:
  - comparison_id: disease_T0_vs_control_T0
    reference_group: DISEASE=control | TIME=T0
    case_group: DISEASE=disease | TIME=T0
  - comparison_id: disease_T1_vs_control_T1
    reference_group: DISEASE=control | TIME=T1
    case_group: DISEASE=disease | TIME=T1
```

Effect on results:

- `logFC > 0` means higher expression in the case group;
- `logFC < 0` means higher expression in the reference group.
- each contrast writes its own differential-expression table;
- `comparison_summary.csv` summarizes sample sizes and significant genes across
  all contrasts.

Recommended choice:

- use control, baseline, untreated, or time zero as the reference;
- use treatment, disease, or the condition under test as the case.
- when several subgroups exist, compare like-with-like where possible, such as
  disease versus control within each timepoint.

### Analysis Min Count and Min Samples

Defaults: `min_count = 10`, `min_samples = 2`

Before `limma`, the differential script reapplies a low-count filter to
`counts_normalized.csv`.

Effect on results:

- avoids statistical tests on genes that are nearly always absent;
- reduces the number of multiple tests;
- can change adjusted p-values because the number of tested genes changes.

Why these defaults match QC filtering:

- they keep filtering consistent between processing and statistics;
- they provide extra protection when starting from raw counts or RCC-derived
  matrices.

Exception:

- for `Final processed counts`, this filter is skipped by default because those
  counts are assumed to have already been processed by another workflow.

### Top Variable Genes

Default: `50`

Number of genes used by the static report heatmap.

Effect on results:

- does not change differential statistics;
- changes only the heatmap/report visualization;
- higher values make the heatmap more complete but less readable.

Why this default:

- it is rich enough to show global patterns;
- it remains readable for most NanoString panels.

## Visualization Thresholds

### P-value Metric and Threshold

UI default metric: `adj.P.Val`

UI default threshold: `0.05`

Filters genes displayed as significant in the volcano plot, dynamic heatmap, and
dynamic PCA. The UI can apply the threshold either to:

- `adj.P.Val`, the multiple-testing-adjusted p-value;
- `P.Value`, the nominal p-value.

Effect on results:

- does not rerun statistics;
- changes only which genes are highlighted and used in dynamic plots.

Why `adj.P.Val` is the default:

- it is a common standard for FDR-controlled results;
- it is consistent with multiple-testing correction.

Why `P.Value` can be useful:

- it helps exploratory review when fold changes are large but adjusted p-values
  are not significant;
- it should be interpreted cautiously because it does not control the false
  discovery rate across all tested genes.

### `|logFC|` Threshold

UI default: `1.0`

Requires an approximately two-fold change on the linear scale.

Effect on results:

- restricts displayed genes to more biologically visible effects;
- does not modify p-values or statistical modeling.

Why this default:

- it combines statistical significance and effect size;
- it makes volcano plots and heatmaps easier to interpret.

## When to Change Defaults

Changing defaults can make sense when:

- the dataset has very few replicates;
- the panel contains many low-expression genes of biological interest;
- negative controls or housekeeping genes are clearly unreliable;
- counts have already been processed by another tool;
- the analysis is exploratory and borderline signal should be inspected.

Practical rule:

- start from defaults for the primary analysis;
- lower visualization thresholds temporarily during exploration;
- document every modified parameter in final reports.

## Final Result Impact

QC and filtering parameters change which samples and genes enter the analysis.
They can therefore change:

- the number of tested genes;
- adjusted p-values;
- estimated log fold changes;
- PCA and heatmaps;
- final biological interpretation.

Visualization parameters change what is shown in the UI, but they do not change
already computed statistical results.
