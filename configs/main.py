import yaml


if __name__ == "__main__":
    config = {
        "VitMetaEncoder": {
            "input_size": 32,
            "input_channels": 3,
            "patch_size": 2,
            "token_dim": 512,
            "num_classes": 100,
            "num_layers": 6,
            "num_heads": 8,
            "dim_ffn": 2048,
            "dropout": 0.1,
        },
        "optimizer": {
            "lr": 1e-4,
            "optim_type": "AdamW",
        },
        "lr_scheduler": {
            "scheduler_type": "cosine",
            "eta_min": 1e-8,
        },
    }

    filename = 'default.yml'

    # Save the dictionary to a YAML file
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile,)