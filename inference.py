import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from model import HierarchicalPestClassifier, PestClassifier

NUM_CLASSES = 102
NUM_ORDERS = 8
NUM_FAMILIES = 47


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_class_names(classes_path):
    names = {}
    if classes_path is None or not Path(classes_path).exists():
        return names

    for line in Path(classes_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            names[int(parts[0]) - 1] = parts[1].strip()
    return names


def load_taxonomy(taxonomy_path):
    if taxonomy_path is None or not Path(taxonomy_path).exists():
        return {}
    return json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))


def build_taxonomy_indices(taxonomy):
    orders = sorted({row["order"] for row in taxonomy.values()})
    families = sorted({row["family"] for row in taxonomy.values()})
    return {name: index for index, name in enumerate(orders)}, {
        name: index for index, name in enumerate(families)
    }


class PestDetector:
    def __init__(self, model_path, taxonomy_path, classes_path, device):
        self.device = device
        self.taxonomy = load_taxonomy(taxonomy_path)
        self.class_names = load_class_names(classes_path)
        self.order_index, self.family_index = build_taxonomy_indices(self.taxonomy)

        state_dict = load_checkpoint(model_path, device)
        self.hierarchical = any(key.startswith("backbone.") for key in state_dict)

        if self.hierarchical:
            self.model = HierarchicalPestClassifier(
                num_orders=NUM_ORDERS,
                num_families=NUM_FAMILIES,
                num_species=NUM_CLASSES,
                pretrained=False,
            )
        else:
            self.model = PestClassifier(num_species=NUM_CLASSES, pretrained=False)

        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_path, top_k=5):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if self.hierarchical:
            species_probabilities = torch.softmax(outputs["species"], dim=1)[0]
            order_probabilities = torch.softmax(outputs["order"], dim=1)[0]
            family_probabilities = torch.softmax(outputs["family"], dim=1)[0]

            scores = []
            for class_id in range(NUM_CLASSES):
                row = self.taxonomy.get(str(class_id), {})
                order_index = self.order_index.get(row.get("order"))
                family_index = self.family_index.get(row.get("family"))

                if order_index is None or family_index is None:
                    scores.append(species_probabilities[class_id].item())
                else:
                    joint_probability = (
                        order_probabilities[order_index]
                        * family_probabilities[family_index]
                        * species_probabilities[class_id]
                    )
                    scores.append(joint_probability.item())
            scores = torch.tensor(scores)
        else:
            scores = torch.softmax(outputs, dim=1)[0]

        values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
        results = []

        for value, index in zip(values.tolist(), indices.tolist()):
            row = self.taxonomy.get(str(index), {})
            results.append(
                {
                    "class_id": index,
                    "label": self.class_names.get(index, f"class_{index}"),
                    "order": row.get("order", "unknown"),
                    "family": row.get("family", "unknown"),
                    "confidence": value * 100.0,
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Run Entomi on a single image.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", default="best_pest_model.pth")
    parser.add_argument("--classes", default="ip102_v1.1/classes.txt")
    parser.add_argument("--taxonomy", default="taxonomy.json")
    parser.add_argument("--top_k", type=int, default=5)
    parsed_arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = PestDetector(
        model_path=parsed_arguments.checkpoint,
        taxonomy_path=parsed_arguments.taxonomy,
        classes_path=parsed_arguments.classes,
        device=device,
    )

    for row in detector.predict(parsed_arguments.image_path, parsed_arguments.top_k):
        print(
            f"class {row['class_id']:3d} | {row['confidence']:5.2f}% | "
            f"{row['order']} / {row['family']} | {row['label']}"
        )


if __name__ == "__main__":
    main()
