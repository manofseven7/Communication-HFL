# Release validation — 2026-11-06

- CPU unit/synthetic integration tests: **22 passed**. Paths cover all thirteen methods, exact mask warm-up timing, epoch/step accounting, compact payload dimensions and checkpoint continuation with explicit and global loader RNG patterns.
- Validation host: Python 3.12.14, PyTorch 2.5.1+cpu, one numerical-library thread. This is a code validation host, not the recorded molab experiment host.
- Recorded-output analysis: complete 135-run tabular group, six paired Holm contrasts, HAR component contrast, CIFAR primary/diagnostic analysis and profile/scalability summaries regenerated successfully.
- CLI help and syntax checks passed. No full-data training was rerun for this release.
- DOI access: existing v2.0 archive retrieved without authentication, MD5 matched; the repository API reports public access. See doi_access_check.json. This verifies the earlier archive, not publication of this v2.1.0 package.
- The revised archive DOI must be assigned after deposit. Numerical reproduction on a different GPU/software stack is subject to the CUDA reproducibility scope recorded in README.
