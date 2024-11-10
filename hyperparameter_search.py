import optuna
import torch
import torchvision
import dataloader
import evaluate
from config import config
from model import createModel
import torchvision.transforms as transforms
from tqdm import tqdm


def hyperparameter_objective(train_loader, val_loader, config, trial):

    model = createModel(config)

    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1, log=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    device = config["device"]

    val_accuracy = 0

    model.train()

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        # Randomly change brightness, contrast, saturation, and hue
        transforms.RandomRotation(degrees=10),  # Randomly rotate images by 15 degrees
        transforms.Resize((224, 224))
    ])

    # early stopping variables
    best_val_loss = float('inf')  # Initialize with a large value
    patience_counter = 0  # Counter for epochs without improvement

    for epoch in range(config["epochs"]):

        for images, labels in tqdm(train_loader):
            images = torch.stack([train_transforms(image) for image in images]).to(device)  # Apply transformations here
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate.evaluate(model, val_loader, config)

        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter when improvement occurs
            print(f"Validation loss improved to {val_loss}. Saving the model.")
            torch.save(model.state_dict(), 'models/' + config["experiment_name"] + 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1  # Increment the counter if no improvement
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        # Check if we should stop
        if patience_counter >= config["patience"]:
            return val_accuracy

    return val_accuracy


def main():

    train_loader, val_loader, _ = dataloader.getDataloaders(config)

    study = optuna.create_study(direction='maximize')  # Maximize accuracy
    study.optimize(lambda x: hyperparameter_objective(train_loader, val_loader, config, x),
                   n_trials=10)  # Perform 10 trials

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")


if __name__ == "__main__":
    main()
