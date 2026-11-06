# Reproducing the revision outputs

1. Install dependencies and run `python scripts/unpack_results.py`.
2. Use the five analysis commands in README to rebuild the two tabular comparisons, six-comparison Holm family, HAR component figure, CIFAR summary/figure and auxiliary summaries.
3. Inspect `analysis_outputs/`. The CIFAR analysis validates all 400 rounds, exact reused control prefixes and original hashes before computing summaries.
4. Full training is optional. It requires the public datasets and the declared configuration. The main grid has 135 tabular runs and 12 CIFAR attempts; the auxiliary records comprise 9 retention sweeps, 24 fresh-process profiles and 12 client-count sweeps. The additional CIFAR QSGD seed is not included in the main comparison.

The tabular analysis requires every declared seed and correct mask timing; it does not fill missing entries. The Holm family comprises Proposed versus FedAvg and PowerSGD-EF on Fashion-MNIST, and each HAR Proposed variant versus those two baselines. The HAR shared-mask component contrast is separate. CIFAR is reported descriptively outside that family.

The CNN diagnostic increases all SVD stages to rank 20 or sends the rank-10 central update densely, with seeds 42–44. The dense-downlink intervention also removes central residual handling. Existing rank-10 prefixes provide the controls. The full worker/source definition is in `executed_source_snapshot.zip` under diagnostics50; checkpoints are separately backed up and are not needed for figure regeneration.

## Baseline accounting

Dense means a float32 model-sized message. SCAFFOLD option-II counts model and control variates on all three links. FedNova normalises by actual minibatch-step counts. FedKD-style uses teacher/student widths 256/128, lambda=.3, T=1 and rank10 student uplinks; each model receives five local epochs. PowerSGD-EF is the declared local factorisation adaptation, with warm starts, client residuals, intermediate P broadcasts, edge factors and a dense downlink. QSGD-8 uses 255 magnitude levels plus signs/norms, with other links dense. FedPAQ samples 25/50 clients; FedCOM uses gamma=1.5. Top-k retains10%; Adaptive-TopK retains99% energy bounded to1–20%; both include values and uint32 indices. SignSGD uses a hierarchical majority vote. Proposed sends SVD factors on all three links with central downlink error feedback.

Do not use the historical default five-seed CIFAR scripts as the current protocol: `scripts/analyze_cifar.py` is the declared three-seed analysis. The former `revision_outputs.py --stage cifar` path is explicitly disabled rather than silently expecting unavailable seeds.

## Source identity

Compression, models, feature scoring and method kernels match the executed source snapshot. Release-only changes concern checkpoint path ownership, command-line host/thread options, lazy dataset imports, documentation and rejection of nonfinite future evaluations. For finite trajectories these do not change the algorithm's update equations. The exact historical records remain unchanged. Existing molab checkpoints use the historical source/signature; use that snapshot if continuing a historical checkpoint, and do not overwrite it with a release-generated run.
