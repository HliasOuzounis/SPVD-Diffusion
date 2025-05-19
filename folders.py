import os

category = "chair"
start_path = './metrics/' + category + '/'

for sched in ("ddim", "ddpm"):
    for t in ("base", "distilled"):
        for i in ("cond", "uncond"):
            for j in ("norm", "no-norm"):
                os.makedirs(os.path.join(start_path, sched, t, i, j), exist_ok=True)
    if sched == "ddim":
        for i in ("cond", "uncond"):
            for j in ("norm", "no-norm"):
                os.makedirs(os.path.join(start_path, sched, "skip", i, j), exist_ok=True)
    if sched == "ddpm":
        for i in ("cond", "uncond"):
            for j in ("norm", "no-norm"):
                os.makedirs(os.path.join(start_path, sched, "retrained", i, j), exist_ok=True)