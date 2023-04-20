from typing import Optional, Tuple
import random
import math
from PIL import Image, ImageDraw, ImageOps

from torch import nn
import torchvision as tv

from config import HyperParameters
from color import RandomColorPicker
from utils import truncnorm_intAB


class CreateBackground(nn.Module):
    def __init__(
        self,
        hparams: HyperParameters,
        color_picker: RandomColorPicker
    ) -> None:
        super().__init__()
        self.mode = hparams.IMG_MODE
        self.image_size = hparams.IMG_SIZE
        self.color_picker = color_picker

        self.hues = hparams.background_hues

    def forward(self, passthrough: Optional[Image.Image]=None):
        size = (
            passthrough.size
            if passthrough is not None
            else self.image_size
        )
        mode = self.mode or (
            passthrough.mode if passthrough is not None else None
        ) or 'RGB'

        color, name = self.color_picker.get_color(self.hues, mode)
        img = Image.new(mode, size, color)

        label = f"on {name} background"

        result = (img, label)
        if passthrough is not None:
            result += (passthrough,)

        return result


class DrawRandomCircle(nn.Module):
    # TODO: Assumes square images right now, maybe generalise to rectangles.
    def __init__(
        self,
        hparams: HyperParameters,
        color_picker: RandomColorPicker
    ) -> None:
        super().__init__()
        self.center_direction_angle_distrib = \
            hparams.circle_center_direction_angle_distrib
        if all(len(self.center_direction_angle_distrib) != i for i in (2, 4)):
            raise ValueError('Invalid distribution parameters.')
        self.center_dist_distrib = hparams.circle_center_dist_distrib

        self.diameter_distrib = hparams.circle_diameter_distrib

        self.color_picker = color_picker
        self.hues = hparams.circle_hues

        # TODO: This is ugly, change tranform signature maybe?
        self.image_size = hparams.IMG_SIZE

    def forward(self, draw: ImageDraw.ImageDraw, mode, num_circles=1):
        color, color_name = self.color_picker.get_color(self.hues, mode=mode)
        for _ in range(num_circles):
            # Get angle with direction to circle center
            # from (1, 0) on unit circle.
            if len(self.center_direction_angle_distrib) == 4:
                angle = truncnorm_intAB(*self.center_direction_angle_distrib)
            if len(self.center_direction_angle_distrib) == 2:
                angle = random.uniform(*self.center_direction_angle_distrib)
            angle *= math.pi / 180

            dist = truncnorm_intAB(*self.center_dist_distrib)

            # Compute center position.
            c_x = dist * math.cos(angle) + self.image_size[0] // 2
            c_y = dist * math.sin(angle) + self.image_size[1] // 2

            d = truncnorm_intAB(*self.diameter_distrib)

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


class DrawRandomLine(nn.Module):
    # TODO: Assumes square images right now, maybe generalise to rectangles.
    # Change random generation to do random direction and distance from center.
    def __init__(
        self,
        hparams: HyperParameters,
        color_picker: RandomColorPicker
    ) -> None:
        super().__init__()
        self.perpendicular_angle_distrib = \
            hparams.line_perpendicular_angle_distrib
        if all(len(self.perpendicular_angle_distrib) != i for i in (2, 4)):
            raise ValueError('Invalid distribution parameters.')
        self.center_dist_distrib = hparams.line_center_dist_distrib

        self.color_picker = color_picker
        self.hues = hparams.line_hues

        # TODO: This is ugly, change tranform signature maybe?
        self.image_size = hparams.IMG_SIZE

    def forward(self, draw: ImageDraw.ImageDraw, mode, num_lines=1):
        color, color_name = self.color_picker.get_color(self.hues, mode=mode)
        for _ in range(num_lines):
            # Get angle with perpendicular from (1, 0) on unit circle.
            if len(self.perpendicular_angle_distrib) == 4:
                angle = truncnorm_intAB(*self.perpendicular_angle_distrib)
            if len(self.perpendicular_angle_distrib) == 2:
                angle = random.uniform(*self.perpendicular_angle_distrib)
            angle *= math.pi / 180

            # Get angle
            dist = truncnorm_intAB(*self.center_dist_distrib)

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
                    (-self.image_size[axis] // 2 <= coord) and
                    (coord <= self.image_size[axis] // 2)
                )

            for side_x, side_y in (
                (self.image_size[0] // 2, None),
                (None, self.image_size[1] // 2),
                (-self.image_size[0] // 2, None),
                (None, -self.image_size[1] // 2),
            ):
                if side_x is None:
                    y = side_y
                    x = (y - b) / (a + 1e-5)
                    if not is_valid_incidence(x, 0):
                        continue
                else:
                    x = side_x
                    y = a * x + b
                    if not is_valid_incidence(y, 1):
                        continue
                valid_incidences.append(
                    (int(x + self.image_size[0] // 2),
                     int(y + self.image_size[1] // 2))
                )

            # Process edge case 3.
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
        hparams: HyperParameters,
        transforms
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.max_num_transforms = hparams.max_num_transforms

    def forward(self, inp):
        if len(inp) == 3:
            src, label_addon, passthrough = inp
            suffix = ''
        else:
            src, label_addon = inp
            suffix = ' over it'

        draw = ImageDraw.Draw(src)
        num_transforms = random.randint(0, self.max_num_transforms)
        if num_transforms == 0:
            return inp
        transforms_doing = random.choices(self.transforms, k=num_transforms)
        label_mods = []
        for transform in transforms_doing:
            if isinstance(transform, tuple):
                num = random.randint(1, transform[1])
                label_mod = transform[0](draw, src.mode, num)
            else:
                label_mod = transform(draw, src.mode)

            label_mods.append(f"{label_mod}{suffix}")

        label_addon = ' '.join([label_addon, *label_mods])

        if len(inp) == 3:
            return (src, label_addon, passthrough)
        else:
            return (src, label_addon)


class CompositeMNISTImage(nn.Module):
    def forward(self, inp):
        src, label_addon, mnist_image_L = inp

        # If we're making coloured images, convert MNIST image to RGB.
        # White numbers look bad on coloured backgrounds, invert them.
        if src.mode == 'RGB':
            mnist_image = ImageOps.invert(mnist_image_L).convert('RGB')
        else:
            mnist_image = mnist_image_L

        dst = Image.composite(src, mnist_image, ImageOps.invert(mnist_image_L))
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
    def forward(self, src: Tuple[Image.Image, ...]):
        if len(set(img.mode for img in src)) > 1:
            raise ValueError("All input images' modes must be the same.")
        dst = Image.new(
            src[0].mode, (sum(img.width for img in src), src[0].height))

        cum_width = 0
        for img in src:
            dst.paste(img, (cum_width, 0))
            cum_width += img.width
        return dst
