# How to run scripts in the background even when the remote connection is closed

## Pre install on new instances

```bash
conda install scikit-learn
```

## Using `nohup` command

```bash
nohup python distillation_training.py > distillation.log 2>&1 &
```

```bash
nohup python evaluation.py > evaluation.log 2>&1 & 
```

- `nohup` allows the script to continue running even after the terminal is closed.

## Monitoring the script

```bash
cat output.log | grep "steps"
```

## Killing the script

```bash
ps aux | grep distillation_training.py
kill -9 <PID>
```

or

```bash
pkill -f distillation_training.py
pkill -f evaluation.py
```

## Loss per epoch

Loss is saved in src/lightning_logs/version_X/metrics.csv

- version_0 = 500 steps
- version_1 = 250 steps
- version_2 = 125 steps
- version_3 = 63 steps
- version_4 = 32 steps
- version_5 = 16 steps