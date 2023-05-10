from typing import Any
from PIL import Image

import torch
from torch.utils.data import DataLoader

import torchvision as tv

from ..config import GlobalConfig

from .config import AugmenterParameters
from .color import RandomColorPicker
from .transforms import (
    CreateBackground,
    DrawRandomCircle,
    DrawRandomLine,
    DecorateImage,
    DecorateImageWithPassthrough,
    CompositeMNISTImage,
    PILnAuxToTensor,
)

class MNISTwAnnotatedAugmentations(tv.datasets.MNIST):
    # A version of MNIST dataset intended for use with
    # augmentations from this module.
    # It merges augmentation annotations with the original label.
    @staticmethod
    def _collate_label_and_annotations(
        label: str,
        annotations: tuple[dict[str, Any], ...] | None
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        # Looks silly for now, later each annotation should be
        # a dataclass object with repr generating the next step of
        # image description.
        orig_label_annotation = {
            'type': 'original-label',
            'label-str': str(label)
        }
        if annotations is None:
            return label, (orig_label_annotation,)
        annotations = (orig_label_annotation, *annotations)

        label = ' '.join(annotation['label-str'] for annotation in annotations)

        return label, annotations

    def __getitem__(
        self,
        index: int
    ) -> dict[str, Image.Image | tuple[dict, ...] | str | None]:
        image_and_possibly_annotations, label = super().__getitem__(index)
        if isinstance(image_and_possibly_annotations, Image.Image):
            # There are no annotations.
            image, annotations = image_and_possibly_annotations, None
        else:
            image, annotations = image_and_possibly_annotations
            annotations = ({
                'type': 'full_label_addon',
                'label-str': annotations
            },)

        label, annotations = \
            self._collate_label_and_annotations(label, annotations)

        return {
            'image': image,
            'label': label,
            'annotation': annotations
        }


def MNISTwAnnotationsCollate_fn(
    batch: list[dict[str, str | tuple[dict, ...] | torch.Tensor]]
):
    collated = {
        key: [item[key] for item in batch] for key in batch[0].keys()
    }
    # Sometimes PyLance is silly, that's why I'm telling it to ignore this line.
    collated['image'] = torch.stack(collated['image'], 0)  # type: ignore
    return collated


def get_dataloader(
    hparams: AugmenterParameters,
    batch_size: int,
    generator: torch.Generator | None=None,
    train: bool=True,
    monochrome: bool=False
) -> DataLoader:
    color_picker = RandomColorPicker(hparams)

    transform = tv.transforms.Compose([
        CreateBackground(
            hparams,
            color_picker,
            mode_override=None if monochrome else 'RGB'
        ),
        DecorateImageWithPassthrough(
            hparams,
            [
                DrawRandomCircle(hparams, color_picker),
                DrawRandomLine(hparams, color_picker),
            ]
        ),
        CompositeMNISTImage(),
        DecorateImage(
            hparams,
            [
                DrawRandomCircle(hparams, color_picker),
                DrawRandomLine(hparams, color_picker),
            ]
        ),
        PILnAuxToTensor()
    ])

    dataset = MNISTwAnnotatedAugmentations(
        GlobalConfig.DATASETS_BASE_DIR,
        transform=transform,
        train=train,
        download=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=generator,
        num_workers=GlobalConfig.NUM_WORKERS,
        collate_fn=MNISTwAnnotationsCollate_fn
    )
