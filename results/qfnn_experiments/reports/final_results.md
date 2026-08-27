# Final QFNN results

Recipes and checkpoints were selected using validation data from the official training partition. The official test partition was evaluated only after all per-seed checkpoints for a dataset were fixed. Reported +/- values across seeds are sample standard deviations.

Resource columns are exact only at the stated abstract oracle level. They are not compiled gate counts, two-qubit depths, physical runtimes, or claims of a fair hardware-resource match. One abstract reflection pair means one branch-multiplexed reference SELECT and one current-state reflection. Base-data calls use $(1+2d)^L$ for uniform degree $d$ and depth $L$.

| Dataset | Depth/degree | Parameters | Exact test accuracy (%) | Finite-shot test accuracy (%) | Shots | Repeats/seed | Logical qubits | Abstract reflection blocks/forward | Prep/unprep base-data calls/forward | Train+validation wall time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST8 | 2/3 | 409 | 90.57 +/- 0.20 | 90.51 +/- 0.17 | 6000 | 5 | 14 | 12 | 49 | 17063.6 +/- 5573.1 |
| FashionMNIST8 | 2/2 | 409 | 74.55 +/- 0.51 | 73.96 +/- 0.48 | 6000 | 5 | 14 | 8 | 25 | 13453.5 +/- 4963.7 |
| KMNIST8 | 2/2 | 409 | 67.61 +/- 0.65 | 67.21 +/- 0.69 | 6000 | 5 | 14 | 8 | 25 | 12302.4 +/- 4559.3 |
