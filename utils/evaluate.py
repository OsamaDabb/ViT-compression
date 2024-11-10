import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm


def evaluate(model, val_data, config):
    """return loss and accuracy for the validation data"""

    acc = 0
    loss = 0
    device = config["device"]

    model.eval()

    resize = transforms.Resize((224, 224))

    with torch.no_grad():
        for images, labels in tqdm(val_data):

            images = torch.stack([resize(image) for image in images]).to(device)  # Apply transformations here
            labels = labels.to(device)

            output = model(images)

            acc += torch.mean((torch.argmax(output, dim=1) == labels).detach().float())
            loss += F.cross_entropy(output, labels).detach().float()

    loss = loss.item()
    acc = acc.item()

    return loss / len(val_data), acc / len(val_data)
