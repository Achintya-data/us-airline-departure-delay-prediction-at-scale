#!/bin/bash
# --- CONFIGURATION FOR LOW QUOTA (12 CPUs MAX) ---

REGION="us-east1"
ZONE="us-east1-b"
BUCKET="big-data-project-481305-flightdelay"

# Make sure this points to your actual python script location
SCRIPT="gs://${BUCKET}/airline/scripts/scaleout_gbt_job.py"
GOLD_BASE="gs://${BUCKET}/airline/gold/sample_step2_features"
OUT_BASE="gs://${BUCKET}/airline/scaling/scaleout_25pct" 

# USE SMALLER MACHINES (2 CPUs each)
MASTER_TYPE="n1-standard-2"
WORKER_TYPE="n1-standard-2"
IMAGE="2.2-debian12"

# LIMIT WORKERS TO MAX 5
# Calculation: 
# w5 = 1 Master (2) + 5 Workers (10) = 12 CPUs
for W in 2 3 4; do
  CL="airline-scale-w${W}"

  echo "=== Creating cluster: $CL (workers=$W) in $REGION ==="
  gcloud dataproc clusters create "$CL" \
    --region="$REGION" --zone="$ZONE" \
    --image-version="$IMAGE" \
    --master-machine-type="$MASTER_TYPE" \
    --worker-machine-type="$WORKER_TYPE" \
    --num-workers="$W" \
    --master-boot-disk-size=50GB \
    --worker-boot-disk-size=50GB \
    --bucket="$BUCKET" \
    --quiet

  # Calculate parallelism
  CORES=$((W*2))
  # Slightly lower partitions for 25% data to avoid too many tiny tasks
  SHUF=$((CORES*4)) 
  PAR=$((CORES*4))

  echo "=== Submitting job on $CL (shuffle=$SHUF, parallelism=$PAR) ==="
  gcloud dataproc jobs submit pyspark "$SCRIPT" \
    --region="$REGION" --cluster="$CL" \
    --properties="spark:spark.sql.shuffle.partitions=${SHUF},spark:spark.default.parallelism=${PAR}" \
    -- \
    --gold_base="$GOLD_BASE" \
    --out_base="$OUT_BASE" \
    --sample_frac=0.25 \
    --run_tag="w${W}"

  echo "=== Deleting cluster: $CL ==="
  gcloud dataproc clusters delete "$CL" --region="$REGION" --quiet
done