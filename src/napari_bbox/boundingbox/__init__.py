import os
import napari
from packaging import version
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

packages = os.listdir(__location__)
napari_versions = [version.parse(package.replace("napari_", "").replace("_", ".")) for package in packages if package.startswith("napari_")]
for napari_version in reversed(sorted(napari_versions)):
    package = "napari_%s" % str(napari_version).replace(".", "_")
    if not os.path.exists(os.path.join(__location__, package)):
        continue
    version_str = package.replace("napari_", "")
    cur_version = version.parse(version_str.replace("_", "."))
    if version.parse(napari.__version__) < cur_version:
        continue
    module = __import__("", fromlist=[f"{package}"], level=1, globals={"__package__": __package__})
    for attr in getattr(getattr(module, f"{package}"), "__all__"):
        globals()[attr] = getattr(getattr(module, f"{package}"), attr)
    break
__all__ = ["BoundingBoxLayer", "register_layer_control", "register_layer_visual"]
