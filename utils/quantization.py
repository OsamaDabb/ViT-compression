import torch.quantization


def quantize_model(model, config):

    model_quantized = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # specify the layers to quantize
        dtype=torch.qint8  # set dtype to quantize to
    )

    return model_quantized