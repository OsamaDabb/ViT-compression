import torch
import numpy as np
import random

import dataloader
from config import config
from model import createModel
import experiments
from low_rank_approximation import apply_lra_to_model
import sys


def main():
    # setting consistent seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # data is (train, val, test)
    # loading ViT-16 Base pre-trained on ImageNet
    model = createModel(config)
    data = dataloader.getDataloaders(config)

    exp_choice = sys.argv[1]

    if exp_choice == "LRA":
        experiments.low_rank_approximation_experiment(model, data, config)

    elif exp_choice == "prune":
        experiments.iterative_pruning_experiment(model, data, config)

    elif exp_choice == "both":

        experiments.low_rank_and_pruning_test(model, data, config, "iterative_magnitude_pruning-9best_model.pth")

    else:
        print("Not a valid experiment choice")


if __name__ == "__main__":

    main()
