from .._utils import parse_package_by_napari_version
parse_package_by_napari_version(__file__, __package__, globals())
import os
from packaging import version
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from .._utils import NAPARI_VERSION

if NAPARI_VERSION >= version.parse("0.4.17"):
    from napari.resources._icons import write_colorized_svgs, _theme_path
    from napari.settings import get_settings

    cube_icon_path = os.path.join(__location__, "cube.svg").replace("\\", "/")
    settings = get_settings()
    theme_name = settings.appearance.theme
    out = _theme_path(theme_name)
    write_colorized_svgs(
        out,
        svg_paths=[cube_icon_path],
        colors=[(theme_name, 'icon')],
        opacities=(0.5, 1),
        theme_override={'warning': 'warning', 'logo_silhouette': 'background'},
    )

__all__ = ["cube_style_path"]
