import sys
import time

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision as tv

from nanodreamer import config, augmentation


@torch.no_grad()
def main(
    hparams: augmentation.AugmenterParameters,
    writer: SummaryWriter,
    batch_size: int,
    num_steps: int
):
    # Reproducibility.
    generator = config.set_seeds()

    loader = augmentation.get_dataloader(hparams, batch_size, generator)

    img_per_row = 3 if any(batch_size == i for i in (6, 9)) else 4

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
        if step > (num_steps - 2):
            break


if __name__ == '__main__':
    tb_base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs_dir = 'mnist-aug'

    hparams = augmentation.AugmenterParameters()

    with SummaryWriter(
        f"{tb_base_dir}/"
        f"{runs_dir}/"
        f"{time.strftime('%Y-%m-%d/%z-%H%M%S', time.localtime())}"
    ) as writer:
        main(hparams, writer, 10, 20)
