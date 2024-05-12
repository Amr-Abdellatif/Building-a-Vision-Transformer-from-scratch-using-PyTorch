def get_config():
    return {
        "batch_size":64,
        "pin_memory":True,
        "num_workers":0,
        "img_size":28,
        "patch_size":4,
        "in_channels":1,
        "num_classes":10,
        "lr":0.001,
        "epochs":5,
    }