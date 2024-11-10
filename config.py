config = {
    "experiment_name": "Iterative column-pruning",
    "device": "cuda",
    "train_val_split": 0.8,
    "batch_size": 32,
    "num_classes": 10,
    "seed": 42,
    "save_model": True,

    "learning_rate": 1e-4,
    "weight_decay": 0,
    "l1_weight": 0,
    "epochs": 30,
    "patience": 5,  # for early-stopping

    # pruning parameters
    "prune_percent": 0.0,
    "prune_type": "per-column",

    # LRA parameters
    "rank": 40,
}
