#!/bin/bash
# Usage: ./submit_from.sh <starting_batch_number>
set -e
COMBO_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_BATCHES=10
START=${1:-1}
PREV_JOBID=""
for ((i=START; i<=NUM_BATCHES; i++)); do
  BATCH_DIR="$COMBO_DIR/batch_$i"
  if [ -z "$PREV_JOBID" ]; then
    JOBID=$(sbatch --parsable --chdir="$BATCH_DIR" "$BATCH_DIR/lmp.sub")
  else
    JOBID=$(sbatch --parsable --dependency=afterok:$PREV_JOBID --chdir="$BATCH_DIR" "$BATCH_DIR/lmp.sub")
  fi
  echo "Submitted batch_$i as job $JOBID"
  PREV_JOBID=$JOBID
done
