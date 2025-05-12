# Evaluation

## Metrics

### Types

- Chamfer Distance (CD)

- Earth Mover's Distance (EMD)

### Scores

- Minimum Matching Distance (MMD) ↓

- Coverage (Cov) ↑

- 1-NN Classification Accuracy ~50%

- Jensen-Shannon Divergence (JSD) ↓

- F1 Score ↑

## Options

### Scheduler

- `ddpm`
- `ddim`

### Normalization

- `none`
- `unit_sphere`

### Steps

`1000 | 500 | 250 | 125 | 63 | 32 | 16 | 8 | 4 | 2 | 1`

### Training

- With ddim scheduler
  - With skip steps
- With ddpm scheduler
  - With/Without stochastic
- With/Without normalization to 0 `new_x = new_x - new_x.mean(dim=1, keepdim=True)`

Perhaps, also train a nn from scratch for same epochs and compare results.

## Time

### To generate the samples

- 1000 steps: 25 mins
- 125 steps: 3 mins

### To compute the metrics

(704 bacthes) - Car

- compute all metrics: 12 * 3 mins

(1317 bacthes) - Chair

- compute all metrics: 43 * 3 mins

## Results

### Airplane

#### No normalization (ddpm)

| Steps | CD-Acc | EMD-Acc | JSD | Size |
|-------|--------|-------| ----| --- |
| 1000 | 74.1 | 71.6 | 0.056 | (5 batches) |
| 1000 | 82.3 | 84.2 | 0.044 | all |

#### To unit sphere (ddpm)

| Steps | CD-Acc | EMD-Acc | JSD | Size |
|-------|--------|-------| ----| --- |
| 1000 | 55.0 | 49.3 | 0.028 | (5 batches) |
| 1000 | 63.4 | 56.3 | 0.012 | all |


### Chair

#### No normalization (ddpm)

| Steps | CD-Acc | EMD-Acc | JSD | Size |
|-------|--------|-------| ----| --- |
| 1000 | 40.3 | 40.6 | 0.014 | (5 batches) |
| 500 | 63.1 | 78.4 | 0.027 | (5 batches) |
| 500 | 77.3 | 86.8 | 0.016 | all |
| 250 | 74.7 | 85.0 | 0.034 | (5 batches) |

#### To unit sphere (ddpm)

| Steps | CD-Acc | EMD-Acc | JSD | Size |
|-------|--------|-------| ----| --- |
| 1000 | 42.9 | 44.1 | 0.020 | (5 batches) |
| 500 | 49.7 | 49.4 | 0.022 | (5 batches) |
| 500 | 64.7 | 62.3 | 0.009 | all |
| 250   | 61.5 | 58.1 | 0.026 | (5 batches) |