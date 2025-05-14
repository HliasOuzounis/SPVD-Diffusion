# Notes

## DDPM vs DDIM

DDPM better for higher steps, more details
DDIM better for lower steps, less noise

## Car

Trained ddim for only 200 epochs for 32, 16, 8, 4, 2 steps.
500, 250, 125, 63 are trained on 1000 epochs.

## Airplane

Trained ddpm for 1000 epochs (with random noise in teacher) for 500, 250, 125, 63 steps. (bad results)
Train on 200 epochs ddim and see.

## Chair

Trained ddpm for 1000 epochs (with random noise in teacher) for 500, 250, 125, 63 steps. (bad results)
Train on 200 epochs ddim and see.