import sys
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision as tv

from config import HyperParameters
from color import RandomColorPicker
from transforms import \
    CreateBackground, DrawRandomCircle, DrawRandomLine, \
    DecorateImage, CompositeMNISTImage, PILnAuxToTensor

hparams = HyperParameters()
color_picker = RandomColorPicker(hparams)

transform = tv.transforms.Compose([
    CreateBackground(hparams, color_picker),
    DecorateImage(
        hparams,
        [
            (DrawRandomCircle(hparams, color_picker), 3),
            (DrawRandomLine(hparams, color_picker), 2)
        ]
    ),
    CompositeMNISTImage(),
    DecorateImage(
        hparams,
        [
            DrawRandomCircle(hparams, color_picker),
            (DrawRandomLine(hparams, color_picker), 2)
        ]
    ),
    PILnAuxToTensor()
])


@torch.no_grad()
def main(writer):
    generator = torch.Generator(hparams.DEVICE)
    generator.manual_seed(hparams.SEED)
    dataset = tv.datasets.MNIST(
        hparams.DATASETS_BASE_DIR,
        transform=transform,
        train=True,
        download=True
    )
    loader = DataLoader(
        dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=hparams.NUM_WORKERS
    )

    img_per_row = hparams.BATCH_SIZE // 2
    for step, ((img, label_addon), label) in enumerate(loader):
        grid = tv.utils.make_grid(img, nrow=img_per_row)
        writer.add_image('COOLBEANS', grid, step)
        labels = [
            f"{start} {end}"
            for start, end in zip(label.tolist(), label_addon)
        ]

        for idx, l in enumerate(labels):
            print(l, end='')
            if (idx % img_per_row) == (img_per_row - 1):
                print()
            else:
                print(" | ", end='')
        print("----------------------------------")
        if step > 18:
            break


if __name__ == '__main__':
    # Reproducibility.
    torch.manual_seed(hparams.SEED)
    np.random.seed(hparams.SEED)
    random.seed(hparams.SEED)

    if len(sys.argv) > 1:
        hparams.TB_BASE_DIR = sys.argv[1]

    with SummaryWriter(
        f"{hparams.TB_BASE_DIR}/"
        f"{hparams.TB_RUNS_DIR}/"
        f"{time.strftime('%Y-%m-%d/%z-%H%M%S', time.localtime())}"
    ) as writer:
        main(writer)
