import torch
from pruning import *
from train import fine_tune
from evaluate import evaluate
import low_rank_approximation


def low_rank_and_pruning_test(model, data, config, model_path=None):

    train_loader, val_loader, test_loader = data

    if model_path is not None:
        model.load_state_dict(torch.load("models/" + model_path, weights_only=True))

    initial_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters {initial_model_size / 10 ** 6}M")
    print(f"Post-Pruning Test Loss, Accuracy: {evaluate(model, test_loader, config)}")

    model = low_rank_approximation.apply_lra_to_model(model, config["rank"])
    new_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters {new_model_size / 10 ** 6}M")

    print("Fine-tuning post-LRA")
    fine_tune(model, train_loader, val_loader, config)
    print(f"Post-LRA Test Loss, Accuracy: {evaluate(model, test_loader, config)}")


def low_rank_approximation_experiment(model, data, config):

    train_loader, val_loader, test_loader = data

    initial_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters {initial_model_size/10**6}M")

    print("Pretraining:")
    fine_tune(model, train_loader, val_loader, config)
    print(f"Pre-LRA Test Loss, Accuracy: {evaluate(model, test_loader, config)}")

    if config["prune_percent"] > 0:

        mask = magnitude_prune(model, config["prune_percent"])
        fine_tune(model, train_loader, val_loader, config, mask)
        print(f"Post-Pruning Test Loss, Accuracy: {evaluate(model, test_loader, config)}")

    model = low_rank_approximation.apply_lra_to_model(model, config["rank"])
    new_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters {new_model_size/10**6}M")

    print("Fine-tuning post-LRA")
    fine_tune(model, train_loader, val_loader, config)
    print(f"Post-LRA Test Loss, Accuracy: {evaluate(model, test_loader, config)}")


def iterative_pruning_experiment(model, data, config):

    train_loader, val_loader, test_loader = data

    for truncation in range(10):

        prune_pct = truncation / 10
        config["experiment_name"] = config["experiment_name"][:-2] + f"-{truncation}"
        config["l1_weight"] = 0

        if config["prune_type"] == "unstructured":
            mask = unstructured_magnitude_prune(model, prune_pct)

        else:
            mask = column_magnitude_pruning(model, prune_pct)

        fine_tune(model, train_loader, val_loader, config, mask)

        zeroed, total = 0, 0

        # calculate true pruning rate
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                num_zeros = torch.sum(param == 0).item()
                total_params = param.numel()
                zeroed += num_zeros
                total += total_params

        # output statistics for k-th iteration
        print("*" * 35)
        print(f"Expected {k * 100}% Pruning:")
        print(f"True pruning rate: {zeroed / total}")
        print(f"Loss, Accuracy: {evaluate(model, test_loader, config)}")
        print("*" * 35)


