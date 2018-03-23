

def get_trainable_params(model):
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return params


def show_memusage(device=2):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("GPU: {}/{}".format(item["memory.used"], item["memory.total"]))
