# Selected per-dataset QFNN recipes

Selection used validation splits from official training data; no test metrics were used. The default sixty-trial, two-seed study is a substantial validation search, but it is not an exhaustive architecture or optimizer search.

| Dataset | Trial | Validation objective (%) | Depth | Degree | Optimizer | Scheduler | LR | Weight decay | Batch | Label smoothing | Entropy penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST8 | 37 | 89.031 | 2 | 3 | rmsprop | onecycle | 0.00464 | 9.79e-05 | 32 | 0.1 | 0.0 |
| FashionMNIST8 | 43 | 75.781 | 2 | 2 | rmsprop | onecycle | 0.00849 | 9.65e-07 | 64 | 0.02 | 0.003 |
| KMNIST8 | 52 | 81.469 | 2 | 2 | rmsprop | onecycle | 0.0124 | 3.69e-07 | 64 | 0.1 | 0.003 |
