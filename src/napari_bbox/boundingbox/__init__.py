from .._utils import parse_package_by_napari_version
parse_package_by_napari_version(__file__, __package__, globals())

__all__ = ["BoundingBoxLayer", "register_layer_control", "register_layer_visual"]
