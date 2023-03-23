"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
import numpy as np
import itertools


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        if not all(p.endswith(".csv") for p in path):
            return None
    elif not path.endswith(".csv"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return read_bbox


def read_bbox(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.loadtxt(_path, delimiter=",") for _path in paths]
    bounding_box_corner_list = [np.reshape(data, (len(data), 2, -1)) for data in arrays]
    mask = np.asarray(list(itertools.product((False, True), repeat=bounding_box_corner_list[0].shape[-1])))
    bounding_boxes = [np.asarray([np.where(mask, bbc[1], bbc[0]) for bbc in bounding_box_corners])
                      for bounding_box_corners in bounding_box_corner_list]
    # stack arrays into single array

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "bounding_boxes"  # optional, default is "image"
    return [(data, add_kwargs, layer_type) for data in bounding_boxes]
