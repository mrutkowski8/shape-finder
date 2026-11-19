import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shape_model import MODEL_PATH, SHAPES, ShapeClassifier, grayscale_on_white, prepare_drawing


SAMPLE_FILENAME = re.compile(r"([a-z]+)[0-9]+\.png", re.IGNORECASE)
AUGMENTATIONS_PER_DRAWING = 24
EPOCHS = 24
SEED = 42


@dataclass(frozen=True)
class Sample:
    path: Path
    image: Image.Image
    label: int


def read_samples(folder: Path) -> list[Sample]:
    samples = []
    for path in sorted(folder.iterdir()):
        match = SAMPLE_FILENAME.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        shape = match[1].lower()
        if shape not in SHAPES:
            continue

        with Image.open(path) as image:
            grayscale = grayscale_on_white(image).copy()
        samples.append(Sample(path, grayscale, SHAPES.index(shape)))
    return samples


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.2:
        image = image.filter(ImageFilter.MinFilter(3))
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    size = (
        max(1, round(image.width * rng.uniform(0.88, 1.12))),
        max(1, round(image.height * rng.uniform(0.88, 1.12))),
    )
    image = image.resize(size, Image.Resampling.BICUBIC)
    limit = 25 if rng.random() < 0.5 else 180
    return image.rotate(
        rng.uniform(-limit, limit), Image.Resampling.BICUBIC, expand=True, fillcolor=255
    )


def train_model(samples: list[Sample]) -> ShapeClassifier:
    counts = Counter(sample.label for sample in samples)
    missing = [shape for index, shape in enumerate(SHAPES) if not counts[index]]
    if missing:
        raise ValueError(f"Missing training drawings for {', '.join(missing)}.")

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    images, labels = [], []
    for sample in samples:
        original = prepare_drawing(sample.image)
        if original is None:
            raise ValueError(f"Blank training drawing: {sample.path.name}")
        images.append(original)
        labels.append(sample.label)
        for _ in range(AUGMENTATIONS_PER_DRAWING):
            drawing = prepare_drawing(augment(sample.image, rng))
            if drawing is not None:
                images.append(drawing * rng.uniform(0.45, 1.0))
                labels.append(sample.label)

    dataset = TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))
    batches = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ShapeClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch, targets in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        if epoch == 1 or epoch % 8 == 0:
            print(f"Epoch {epoch:2}/{EPOCHS}   loss {total_loss / len(dataset):.3f}", flush=True)
    return model.eval()


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    samples = read_samples(MODEL_PATH.parent / "samples")
    print(f"Training on {len(samples)} labeled drawings and their augmented copies.", flush=True)
    model = train_model(samples)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as directory:
        checkpoint = Path(directory) / MODEL_PATH.name
        torch.save(model.state_dict(), checkpoint)
        checkpoint.replace(MODEL_PATH)
    print(f"Saved trained weights to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
