import torch
import torchvision
from evaluate import evaluate
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
from copy import deepcopy
from pruning import apply_mask


def fine_tune(model, train_loader, val_loader, config, mask=None):

    # initializations
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    device = config["device"]
    patience = config["patience"]
    best_model = model.state_dict()

    # create wandb instance for tracking
    wandb.init(project="ViT-hyperopt",
               config={"lr": config["learning_rate"], "batch_size": config["batch_size"],
                       "weight_decay": config["weight_decay"]})
    wandb.run.name = config["experiment_name"]

    # data augmentation transforms for robustness
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        # Randomly change brightness, contrast, saturation, and hue
        transforms.RandomRotation(degrees=10),  # Randomly rotate images by 15 degrees
        transforms.Resize((224, 224))
    ])

    # early stopping variables
    best_val_loss = float('inf')  # Initialize with a large value
    patience_counter = 0  # Counter for epochs without improvement

    # training loop
    model.train()
    for epoch in range(config["epochs"]):

        acc = 0
        running_loss = 0

        # individual epoch train
        for images, labels in tqdm(train_loader):

            images = torch.stack([train_transforms(image) for image in images]).to(device)  # Apply transformations here
            labels = labels.to(device)

            output = model(images)

            acc += torch.mean((torch.argmax(output, dim=1) == labels).detach().float())
            loss = criterion(output, labels)
            running_loss += loss.item()

            # add l1 norm to loss
            if config["l1_weight"] > 0:

                l1_norm = torch.sum(torch.stack(
                            [torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad]
                                    ))

                loss += l1_norm * config["l1_weight"]

            # calculate loss
            optimizer.zero_grad()
            loss.backward()

            # apply mask to zero-ed parameters
            if mask:
                apply_mask(model, mask)

            optimizer.step()

        # epoch statistics calculation
        acc /= len(train_loader)
        running_loss /= len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, config)

        # output results
        print(f"Epoch: {epoch + 1}, Loss: {running_loss}, Accuracy: {acc}")
        print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy}")

        wandb.log({"validation accuracy": val_accuracy, "validation loss": val_loss})

        # Early stopping logic
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter when improvement occurs
            print(f"Validation loss improved to {val_loss}. Saving the model.")

            # keep track of best model so far- will revert model to this at the end
            best_model = deepcopy(model.state_dict())

            if config["save_model"]:
                # Save the best model
                torch.save(model.state_dict(), 'models/' + config["experiment_name"] + 'best_model.pth')

        elif patience > 0:
            patience_counter += 1  # Increment the counter if no improvement
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

            # Check if we should stop
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    # once training is done, return to best model
    model.load_state_dict(best_model)
