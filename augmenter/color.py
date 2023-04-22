import random
from scipy.stats import truncnorm
from PIL import ImageColor

from config import HyperParameters
from utils import truncnorm_intAB


class RandomColorPicker:
    def __init__(self, hparams: HyperParameters) -> None:
        self.greyscale_distribution = hparams.greyscale_distrib
        self.hue_ranges = hparams.hue_ranges
        self.hue_distribution = hparams.hue_distrib
        self.saturation_distribution = hparams.saturation_distrib
        self.lightness_distribution = hparams.lightness_distrib

    def get_color(
        self,
        allowed_hues: set[str] | None=None,
        mode:str ='RGB'
    ) -> tuple[
        tuple[int, int, int] | int,
        str
    ]:
        if mode == '1':
            color = random.getrandbits(1)
            name = 'white' if color else 'black'
            return color, name
        if mode == 'L':
            name = random.choice(list(self.greyscale_distribution.keys()))
            color = int(255 * truncnorm.rvs(*self.greyscale_distribution[name]))
            return color, name

        name = random.choice(
            list(self.hue_ranges.keys())
            if allowed_hues is None
            else list(allowed_hues)
        )
        hue = truncnorm_intAB(*self.hue_ranges[name], *self.hue_distribution)
        if hue < 0:
            hue += 360
        saturation = truncnorm.rvs(*self.saturation_distribution)
        lightness = truncnorm.rvs(*self.lightness_distribution)

        color = self.hsl_to_rgb(hue, saturation, lightness)
        if mode == 'RGB':
            return color[:3], name
        else:
            ink_string = f"rgb{color}"
            color = ImageColor.getcolor(ink_string, mode)
            # For compatibility with Greyscale modes
            # that are not denoted with '1' or 'L'.
            if len(color) == 2:
                color = color[0]
            else:
                color = color[:3]
            return color, name

    @staticmethod
    def hsl_to_rgb(h, s, l):
        ink_string = f"hsl({h}, {100 * s}%, {100 * l}%)"
        return ImageColor.getrgb(ink_string)
