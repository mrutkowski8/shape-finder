from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn


SHAPES = ("triangle", "square", "circle")
IMAGE_SIZE = 64
IMAGE_PADDING = 4
MODEL_PATH = Path(__file__).resolve().with_name("shape_model.pt")


def grayscale_on_white(image: Image.Image) -> Image.Image:
    if image.mode == "L" and "transparency" not in image.info:
        return image

    rgba = image.convert("RGBA")
    grayscale = Image.new("L", image.size, 255)
    grayscale.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
    return grayscale


def prepare_drawing(image: Image.Image) -> torch.Tensor | None:
    ink = ImageOps.invert(grayscale_on_white(image))
    bounds = ink.getbbox()
    if bounds is None:
        return None

    ink = ink.crop(bounds)
    scale = (IMAGE_SIZE - 2 * IMAGE_PADDING) / max(ink.size)
    size = (max(1, round(ink.width * scale)), max(1, round(ink.height * scale)))
    ink = ink.resize(size, Image.Resampling.LANCZOS)
    square = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE))
    square.paste(ink, ((IMAGE_SIZE - ink.width) // 2, (IMAGE_SIZE - ink.height) // 2))
    pixels = torch.frombuffer(bytearray(square.tobytes()), dtype=torch.uint8)
    return pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE).float().div_(255)


class ShapeClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, len(SHAPES)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def load_model() -> ShapeClassifier:
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = ShapeClassifier()
    model.load_state_dict(weights)
    return model.eval()


@torch.inference_mode()
def predict_percentages(model: ShapeClassifier, image: Image.Image) -> dict[str, int]:
    drawing = prepare_drawing(image)
    if drawing is None:
        return dict.fromkeys(SHAPES, 0)

    scores = model(drawing.unsqueeze(0)).softmax(dim=1)[0].mul(100).tolist()
    percentages = [int(score) for score in scores]
    # Give leftover points to the largest fractions so the displayed total is 100%.
    order = sorted(
        range(len(scores)),
        key=lambda index: scores[index] - percentages[index],
        reverse=True,
    )
    for index in order[: 100 - sum(percentages)]:
        percentages[index] += 1
    return dict(zip(SHAPES, percentages, strict=True))
