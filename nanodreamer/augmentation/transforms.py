import abc
import random
import math

from PIL import Image, ImageDraw, ImageOps

from torch import nn
import torchvision as tv

from .config import AugmenterParameters
from .color import RandomColorPicker
from .utils import truncnorm_in_sample_space


class CreateBackground(nn.Module):
    def __init__(
        self,
        hparams: AugmenterParameters,
        color_picker: RandomColorPicker,
        image_size: tuple[int, int] | None=None,
        mode_override: str | None=None
    ) -> None:
        super().__init__()
        self.mode_override = mode_override
        self.color_picker = color_picker
        self.image_size = image_size

        self.hues = hparams.background_hues

    def forward(
        self,
        passthrough: Image.Image | None=None
    ):
        size = (
            passthrough.size
            if passthrough is not None
            else self.image_size
        )
        if size is None:
            raise ValueError(
                "Running in image creation mode, image_size was not provided. "
                "Provide image_size in the constructor."
            )
        mode = (
            self.mode_override or
            (passthrough.mode if passthrough is not None else None) or
            'RGB'
        )

        color, name = self.color_picker.get_color(self.hues, mode)
        img = Image.new(mode, size, color)

        label = f"on {name} background"

        result = (img, label)
        if passthrough is not None:
            result += (passthrough,)

        return result


class AugmentationTransform(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        hparams: AugmenterParameters,
        color_picker: RandomColorPicker
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: tuple[int, int],
        mode: str
    ) -> str:
        pass


class DrawRandomCircle(AugmentationTransform):
    pass
    def __init__(
        self,
        hparams: AugmenterParameters,
        color_picker: RandomColorPicker
    ) -> None:
        super(AugmentationTransform, self).__init__()
        self.center_direction_angle_distrib = \
            hparams.circle_center_direction_angle_distrib
        if all(len(self.center_direction_angle_distrib) != i for i in (2, 4)):
            raise ValueError("Invalid distribution parameters.")
        if len(self.center_direction_angle_distrib) == 4:
            self.angle_distrib_fun = truncnorm_in_sample_space
        if len(self.center_direction_angle_distrib) == 2:
            self.angle_distrib_fun = random.uniform

        self.center_dist_distrib = hparams.circle_center_dist_distrib

        self.diameter_distrib = hparams.circle_diameter_distrib

        self.color_picker = color_picker
        self.hues = hparams.circle_hues

        self.max_num_transforms = hparams.circle_max_num
        if hparams.circle_max_num < 1:
            raise ValueError(
                f"Expected circle_max_num > 0, got {hparams.circle_max_num}"
            )

    def forward(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: tuple[int, int],
        mode: str
    ) -> str:
        color, color_name = self.color_picker.get_color(self.hues, mode=mode)
        num_circles = random.randint(1, self.max_num_transforms)
        for _ in range(num_circles):
            # Get distance from image center (0, 0) to circle center.
            # dist = sqrt(dist_x ** 2 + dist_y ** 2)
            # dist_<x|y> = 0.5 * image_size[x|y] * <random fraction>
            dist = math.sqrt(sum(
                (
                    0.5 *
                    image_size[i] *
                    truncnorm_in_sample_space(*self.center_dist_distrib)
                ) ** 2
                for i in range(2)
            ))

            # Get angle with direction to circle center from (0, 0).
            angle = self.angle_distrib_fun(*self.center_direction_angle_distrib)
            angle *= math.pi / 180

            # Compute center position.
            c_x = int(dist * math.cos(angle) + image_size[0] // 2)
            c_y = int(dist * math.sin(angle) + image_size[1] // 2)

            # Diameter is a fraction of the smallest of image spatial sizes.
            d = int(
                min(image_size) *
                truncnorm_in_sample_space(*self.diameter_distrib)
            )

            bbox_x0 = c_x - d // 2
            bbox_y0 = c_y - d // 2
            bbox = [
                (bbox_x0, bbox_y0),
                (bbox_x0 + d, bbox_y0 + d)
            ]

            draw.ellipse(bbox, fill=color)
        return (
            f"with {num_circles} {color_name} "
            f"{'circles' if num_circles > 1 else 'circle'}"
        )


class DrawRandomLine(AugmentationTransform):
    def __init__(
        self,
        hparams: AugmenterParameters,
        color_picker: RandomColorPicker
    ) -> None:
        super(AugmentationTransform, self).__init__()
        self.perpendicular_angle_distrib = \
            hparams.line_perpendicular_angle_distrib
        if all(len(self.perpendicular_angle_distrib) != i for i in (2, 4)):
            raise ValueError('Invalid distribution parameters.')
        if len(self.perpendicular_angle_distrib) == 4:
            self.angle_distrib_fun = truncnorm_in_sample_space
        if len(self.perpendicular_angle_distrib) == 2:
            self.angle_distrib_fun = random.uniform

        self.center_dist_distrib = hparams.line_center_dist_distrib

        self.color_picker = color_picker
        self.hues = hparams.line_hues

        self.max_num_transforms = hparams.line_max_num
        if hparams.line_max_num < 1:
            raise ValueError(
                f"Expected line_max_num > 0, got {hparams.line_max_num}"
            )

    def forward(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: tuple[int, int],
        mode: str
    ) -> str:
        color, color_name = self.color_picker.get_color(self.hues, mode=mode)
        num_lines = random.randint(1, self.max_num_transforms)
        for _ in range(num_lines):
            # Get length on perpendicular from image center (0, 0) to the line.
            # dist = sqrt(dist_x ** 2 + dist_y ** 2)
            # dist_<x|y> = 0.5 * image_size[x|y] * <random fraction>
            dist = math.sqrt(sum(
                (
                    0.5 *
                    image_size[i] *
                    truncnorm_in_sample_space(*self.center_dist_distrib)
                ) ** 2
                for i in range(2)
            ))

            # Get angle with perpendicular from (0, 0).
            angle = self.angle_distrib_fun(*self.perpendicular_angle_distrib)
            angle *= math.pi / 180

            # Params for the line of form: y = a * x + b.
            a = math.tan(angle - math.pi / 2)
            b = dist * math.sin(angle)

            # Find all incidences with image sides that are inside the image.
            # Edge cases:
            #   1. Single incidence at the vertex.
            #      This never happens,
            #      would require dist = IMG_SIZE[0] * sqrt(2) // 2.
            #      Our dist <= IMG_SIZE // 2.
            #   2. Incidence on entire edge of the image.
            #      This almost certainly can't happen since even if the angle
            #      would perfectly align, division by 0 prevention should place
            #      the incidence point infinitely far away, invalidating it.
            #   3. Incidence with three or four sides (one or two vertices).
            valid_incidences = []
            def is_valid_incidence(coord, axis):
                return (
                    (-image_size[axis] // 2 <= coord) and
                    (coord <= image_size[axis] // 2)
                )

            for side_x, side_y in (
                (image_size[0] // 2, None),
                (None, image_size[1] // 2),
                (-image_size[0] // 2, None),
                (None, -image_size[1] // 2),
            ):
                if side_x is None:
                    assert side_y is not None
                    y = side_y
                    x = (y - b) / (a + 1e-5)
                    if not is_valid_incidence(x, 0):
                        continue
                else:
                    assert side_x is not None
                    x = side_x
                    y = a * x + b
                    if not is_valid_incidence(y, 1):
                        continue
                valid_incidences.append(
                    (int(x + image_size[0] // 2),
                     int(y + image_size[1] // 2))
                )

            # Process edge case 3.
            if len(valid_incidences) < 2:
                raise RuntimeError(
                    "Line and image have less than two "
                    "valid points of incidence, "
                    "check distance distribution parameters."
                )
            if len(valid_incidences) > 2:
                unique_incidences = {}
                for x, y in valid_incidences:
                    found_close = False
                    for kx, ky in unique_incidences.keys():
                        if not (
                            math.isclose(x, kx, abs_tol=3) and
                            math.isclose(y, ky, abs_tol=3)
                        ):
                            continue
                        unique_incidences[(kx, ky)].append((x, y))
                        found_close = True
                        break
                    if found_close:
                        continue
                    unique_incidences[(x, y)] = [(x, y)]
                xy = []
                for point_list in unique_incidences.values():
                    xy = [
                        (
                            sum(point[i] for point in point_list) \
                                / len(point_list)
                            for i in range(2)
                        )
                    ]
                assert len(valid_incidences) == 2
            else:
                xy = valid_incidences

            draw.line(xy, fill=color, width=2)
        return (
            f"with {num_lines} {color_name} "
            f"{'lines' if num_lines > 1 else 'line'}"
        )


class DecorateImage(nn.Module):
    def __init__(
        self,
        hparams: AugmenterParameters,
        transforms: list[AugmentationTransform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.max_num_transforms = hparams.max_num_transforms

    def _inner_forward(
        self,
        src: Image.Image,
        label_addon: str,
        suffix: str=''
    ) -> tuple[Image.Image, str]:
        draw = ImageDraw.Draw(src)
        num_transforms = random.randint(0, self.max_num_transforms)
        if num_transforms == 0:
            return src, label_addon
        transforms_doing = random.choices(self.transforms, k=num_transforms)
        label_mods = []
        for transform in transforms_doing:
            label_mod = transform(draw, src.size, src.mode)

            label_mods.append(f"{label_mod}{suffix}")

        label_addon = ' '.join([label_addon, *label_mods])
        return src, label_addon

    def forward(
        self,
        inp: tuple[Image.Image, str]
    ) -> tuple[Image.Image, str]:
        src, label_addon = inp
        return self._inner_forward(src, label_addon, ' over it')


class DecorateImageWithPassthrough(DecorateImage):
    def forward(
        self,
        inp: tuple[Image.Image, str, Image.Image]
    ) -> tuple[Image.Image, str, Image.Image]:
        src, label_addon = inp[:2]
        return (*self._inner_forward(src, label_addon), inp[2])


class CompositeMNISTImage(nn.Module):
    def forward(
        self,
        inp: tuple[Image.Image, str, Image.Image]
    ) -> tuple[Image.Image, str]:
        src, label_addon, mnist_image_L = inp

        # White numbers look bad on coloured backgrounds, so paint them black.
        number_source_sheet = Image.new(src.mode, src.size)
        dst = Image.composite(
            src,
            number_source_sheet,
            ImageOps.invert(mnist_image_L)
        )
        return dst, label_addon


class PILnAuxToTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pil_to_tensor = tv.transforms.PILToTensor()

    def forward(self, inp):
        src, label_addon = inp
        dst = self.pil_to_tensor(src)
        return (dst, label_addon)


class ConcatVertical(nn.Module):
    def forward(self, src: tuple[Image.Image, ...]):
        if len(set(img.mode for img in src)) > 1:
            raise ValueError("All input images' modes must be the same.")
        dst = Image.new(
            src[0].mode, (sum(img.width for img in src), src[0].height))

        cum_width = 0
        for img in src:
            dst.paste(img, (cum_width, 0))
            cum_width += img.width
        return dst
