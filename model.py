import timm
import torch.nn as nn

def createModel(config):

    model = timm.create_model(model_name="vit_base_patch16_224",
                              pretrained=True)

    model.head = nn.Linear(model.head.in_features, config["num_classes"])

    model = model.to(config["device"])

    return model


if __name__ == "__main__":

    print(timm.list_models("*vit*", pretrained=True))
