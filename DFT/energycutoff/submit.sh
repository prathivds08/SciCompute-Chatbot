#!/bin/bash
# This script loops through subdirectories and submits jobs
for dir in */; do
    if [ -f "$dir/qe.sub" ]; then
        echo "Submitting job in $dir"
        cd "$dir"
        sbatch qe.sub
        cd ..
    fi
done
