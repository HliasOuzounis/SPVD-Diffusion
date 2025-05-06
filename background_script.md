# How to run scripts in the background even when the remote connection is closed

## Pre install on new instances
```bash
conda install scikit-learn
```

## Using `nohup` command
```bash
nohup python your_script.py > output.log 2>&1 &
```
- `nohup` allows the script to continue running even after the terminal is closed.

## Killing the script
```bash
ps aux | grep your_script.py
kill -9 <PID>
```