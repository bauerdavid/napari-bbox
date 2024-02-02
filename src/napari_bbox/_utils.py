import os
from packaging import version
import napari
NAPARI_VERSION = version.parse(napari.__version__)


def parse_package_by_napari_version(file_path, package, _globals):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(file_path)))
    packages = os.listdir(__location__)
    napari_versions = [version.parse(package.replace("napari_", "").replace("_", ".")) for package in packages if
                       package.startswith("napari_")]
    for napari_version in reversed(sorted(napari_versions)):
        subpackage = "napari_%s" % str(napari_version).replace(".", "_")
        if not os.path.exists(os.path.join(__location__, subpackage)):
            continue
        version_str = subpackage.replace("napari_", "")
        cur_version = version.parse(version_str.replace("_", "."))
        if NAPARI_VERSION < cur_version:
            continue
        module = __import__("", fromlist=[f"{subpackage}"], level=1, globals={"__package__": package})
        for attr in getattr(getattr(module, f"{subpackage}"), "__all__"):
            _globals[attr] = getattr(getattr(module, f"{subpackage}"), attr)
        break

__all__ = ["parse_package_by_napari_version", "NAPARI_VERSION"]
