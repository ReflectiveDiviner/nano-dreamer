import sys
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision as tv

from config import AugmenterParameters
from color import RandomColorPicker
from transforms import \
    CreateBackground, DrawRandomCircle, DrawRandomLine, \
    DecorateImage, CompositeMNISTImage, PILnAuxToTensor
from dataset import MNISTwAnnotatedAugmentations, MNISTwAnnotationsCollate_fn
from utils import set_seeds


def get_dataloader(
    hparams: AugmenterParameters,
    generator: torch.Generator | None=None,
    train: bool=True
):
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

    dataset = MNISTwAnnotatedAugmentations(
        hparams.DATASETS_BASE_DIR,
        transform=transform,
        train=train,
        download=True
    )
    return DataLoader(
        dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=train,
        generator=generator,
        num_workers=hparams.NUM_WORKERS,
        collate_fn=MNISTwAnnotationsCollate_fn
    )


@torch.no_grad()
def main(hparams: AugmenterParameters, writer: SummaryWriter):
    # Reproducibility.
    generator = set_seeds(hparams.SEED)

    loader = get_dataloader(hparams, generator)

    img_per_row = hparams.BATCH_SIZE // 2
    for step, batch in enumerate(loader):
        images = batch['image']
        labels = batch['label']
        annotations = batch['annotation']

        grid = tv.utils.make_grid(images, nrow=img_per_row)
        writer.add_image('COOLBEANS', grid, step)

        for idx, l in enumerate(labels):
            print(l, end='')
            if (idx % img_per_row) == (img_per_row - 1):
                print()
            else:
                print(" | ", end='')
        print("----------------------------------")
        for idx, a in enumerate(annotations):
            print(a, end='')
            if (idx % img_per_row) == (img_per_row - 1):
                print()
            else:
                print(" | ", end='')
        print("++++++++++++++++++++++++++++++++++")
        if step > 18:
            break


if __name__ == '__main__':
    tb_base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs_dir = 'mnist-aug'

    hparams = AugmenterParameters()

    with SummaryWriter(
        f"{tb_base_dir}/"
        f"{runs_dir}/"
        f"{time.strftime('%Y-%m-%d/%z-%H%M%S', time.localtime())}"
    ) as writer:
        main(hparams, writer)
