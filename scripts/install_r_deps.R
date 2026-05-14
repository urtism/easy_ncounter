repos <- c(CRAN = "https://cloud.r-project.org")
user_lib <- Sys.getenv("NC_R_LIB", unset = file.path(getwd(), "r-lib"))
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

cran_packages <- c("yaml", "readr", "dplyr", "tibble", "ggplot2", "pheatmap")

local_packages <- rownames(installed.packages(lib.loc = user_lib))
missing_cran <- setdiff(cran_packages, local_packages)
if (length(missing_cran) > 0) {
  install.packages(missing_cran, repos = repos, lib = user_lib)
}

local_packages <- rownames(installed.packages(lib.loc = user_lib))
if (!"BiocManager" %in% local_packages) {
  install.packages("BiocManager", repos = repos, lib = user_lib)
}

bioc_packages <- c("limma", "NanoStringNorm")
local_packages <- rownames(installed.packages(lib.loc = user_lib))
missing_bioc <- setdiff(bioc_packages, local_packages)
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE, lib = user_lib)
}
