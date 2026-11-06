#!/usr/bin/env bash
set -euo pipefail

# Controlled one-round CPU profile used for the manuscript's software profile.
# Each invocation is a fresh process; repeat three times with distinct seeds.
for ds in fashion har; do
  for method in fedavg fednova powersgd proposed; do
    for seed in 42 43 44; do
      tag="profile"
      extra=""
      if [[ "$method" == "proposed" && "$ds" == "har" ]]; then extra="--fs off"; fi
      python run.py --dataset "$ds" --method "$method" --seed "$seed" \
        --rounds 1 --local-epochs 5 --tag "$tag" $extra --out results/profile
    done
  done
done

python scripts/summarize_auxiliary.py profile
