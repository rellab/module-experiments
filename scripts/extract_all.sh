#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
EXTRACT="../scripts/extract.py"

# baseline (NHPP only) — read from cache_offset_metraw
$PYTHON $EXTRACT cache_offset_metraw \
    --model baseline --view all --output csv > result_prediction_offset_nhpp.csv

# best for each method
# runall_offset_metrics.py          → cache_offset_metraw
$PYTHON $EXTRACT cache_offset_metraw \
    --model best --view all --output csv > result_prediction_offset_metrics.csv

# runall_offset_embkernel.py        → cache_offset_embkernel
$PYTHON $EXTRACT cache_offset_embkernel \
    --model best --view all --output csv > result_prediction_offset_embkernel.csv

# runall_offset_embpca.py           → cache_offset_embpca
$PYTHON $EXTRACT cache_offset_embpca \
    --model best --view all --output csv > result_prediction_offset_embpca.csv

# runall_offset_id_embkernel.py     → cache_offset_identity_smetrics_embprecision_l2_emb12
$PYTHON $EXTRACT cache_offset_identity_smetrics_embprecision_l2_emb12 \
    --model best --view all --output csv > result_prediction_offset_id_embkernel.csv

# runall_offset_id_metkernel.py     → cache_offset_identity_smetrics_mprecision_l2
$PYTHON $EXTRACT cache_offset_identity_smetrics_mprecision_l2 \
    --model best --view all --output csv > result_prediction_offset_id_metkernel.csv

# runall_offset_identity.py         → cache_offset_identity_random_effects
$PYTHON $EXTRACT cache_offset_identity_random_effects \
    --model best --view all --output csv > result_prediction_offset_identity.csv

# runall_offset_metkernel.py        → cache_offset_metrics_kernel
$PYTHON $EXTRACT cache_offset_metrics_kernel \
    --model best --view all --output csv > result_prediction_offset_metkernel.csv

# runall_offset_metpca.py           → cache_offset_metpca95
$PYTHON $EXTRACT cache_offset_metpca95 \
    --model best --view all --output csv > result_prediction_offset_metpca.csv

# runall_offset_metrics_embkernel_re.py → cache_offset_metrics_re_embedding_kernel
$PYTHON $EXTRACT cache_offset_metrics_re_embedding_kernel \
    --model best --view all --output csv > result_prediction_offset_metrics_embkernel_re.csv

# runall_offset_metrics_re.py       → cache_offset_metrics_random_effect_penalty
$PYTHON $EXTRACT cache_offset_metrics_random_effect_penalty \
    --model best --view all --output csv > result_prediction_offset_metrics_re.csv
