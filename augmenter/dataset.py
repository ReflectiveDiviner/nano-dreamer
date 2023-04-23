from typing import Any
from PIL import Image

import torch
import torchvision as tv


class MNISTwAnnotatedAugmentations(tv.datasets.MNIST):
    # A version of MNIST dataset intended for use with
    # augmentations from this module.
    # It merges augmentation annotations with the original label.
    @staticmethod
    def _collate_label_and_annotations(
        label: str,
        annotations: tuple[dict[str, Any], ...] | None):
        # Looks silly for now, later each annotation should be
        # a dataclass object with repr generating the next step of
        # image description.
        orig_label_annotation = {
            'type': 'original-label',
            'label_str': str(label)
        }
        if annotations is None:
            return label, (orig_label_annotation,)
        annotations = (orig_label_annotation, *annotations)

        label = ' '.join(annotation['label_str'] for annotation in annotations)

        return label, annotations

    def __getitem__(
        self,
        index: int
    ) -> dict[str, Image.Image | dict | str | None]:
        image_and_possibly_annotations, label = super().__getitem__(index)
        if isinstance(image_and_possibly_annotations, Image.Image):
            # There are no annotations.
            image, annotations = image_and_possibly_annotations, None
        else:
            image, annotations = image_and_possibly_annotations
            annotations = ({
                'type': 'full_label_addon',
                'label_str': annotations
            },)

        label, annotations = \
            self._collate_label_and_annotations(label, annotations)

        return {
            'image': image,
            'label': label,
            'annotation': annotations
        }

def MNISTwAnnotationsCollate_fn(
    batch: list[dict[str, str | dict | torch.Tensor]]
):
    collated = {
        key: [item[key] for item in batch] for key in batch[0].keys()
    }
    # Sometimes PyLance is silly, that's why I'm telling it to ignore this line.
    collated['image'] = torch.stack(collated['image'], 0)  # type: ignore
    return collated
