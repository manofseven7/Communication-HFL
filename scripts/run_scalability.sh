#!/usr/bin/env bash
set -euo pipefail

# One-round UCI HAR scalability sweep: ten clients per edge, alpha=1.0,
# one local epoch, rank 10, and three seeds.
for clients in 50 100 250 500; do
  edges=$((clients / 10))
  for seed in 42 43 44; do
    python run.py --dataset har --method proposed --seed "$seed" --fs off \
      --clients "$clients" --edges "$edges" --rounds 1 --local-epochs 1 \
      --alpha 1.0 --rank 10 --tag "scale_n${clients}" --out results/scalability
  done
done

python scripts/summarize_auxiliary.py scalability
