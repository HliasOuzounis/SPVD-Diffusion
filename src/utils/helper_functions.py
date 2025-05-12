def process_ckpt(ckpt):
    if "state_dict" not in ckpt:
        return ckpt

    new_ckpt = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("teacher."):
            continue
        if k.startswith("student."):
            k = k.replace("student.", "")
            new_ckpt[k] = v
        if k.startswith("model."):
            new_ckpt[k] = v
    
    return new_ckpt
    
    