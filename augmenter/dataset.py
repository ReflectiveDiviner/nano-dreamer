from PIL import Image

import torchvision as tv


class MNISTwAnnotatedAugmentations(tv.datasets.MNIST):
    # A version of MNIST dataset intended for use with
    # augmentations from this module.
    # It merges augmentation annotations with the original label.
    @staticmethod
    def _collate_label_and_annotations(label: str, annotations: str | None):
        # Looks silly for now, later each annotation should be
        # a dataclass object with repr generating the next step of
        # image description.
        if annotations is None:
            return label
        return f"{label} {annotations}"

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

        label = self._collate_label_and_annotations(label, annotations)

        return {
            'image': image,
            'label': label,
            'annotation': annotations
        }
