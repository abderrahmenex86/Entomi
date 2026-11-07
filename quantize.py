import argparse
import os

import torch

from model import HierarchicalPestClassifier, PestClassifier

NUM_CLASSES = 102
NUM_ORDERS = 8
NUM_FAMILIES = 47


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def build_model(state_dict):
    if any(key.startswith("backbone.") for key in state_dict):
        model = HierarchicalPestClassifier(
            num_orders=NUM_ORDERS,
            num_families=NUM_FAMILIES,
            num_species=NUM_CLASSES,
            pretrained=False,
        )
    else:
        model = PestClassifier(num_species=NUM_CLASSES, pretrained=False)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def quantize_model(checkpoint_path, output_path):
    model = build_model(load_checkpoint(checkpoint_path))

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save(output_path)

    fp32_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"FP32 checkpoint: {fp32_size:.2f} MB")
    print(f"Quantized TorchScript: {int8_size:.2f} MB")
    print(f"Size ratio: {fp32_size / int8_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Quantize an Entomi checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", default="best_pest_model.pth")
    parser.add_argument("--output", default="quantized_pest_model.pt")
    parsed_arguments = parser.parse_args()

    quantize_model(parsed_arguments.checkpoint, parsed_arguments.output)


if __name__ == "__main__":
    main()
