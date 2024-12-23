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
    # Handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    # Load all files into arrays
    arrays = [np.loadtxt(_path, delimiter=",") for _path in paths]

    bounding_box_corner_list = []
    for data in arrays:
        # Ensure data is at least 2D
        data = np.atleast_2d(data)
        
        # Determine the number of bounding boxes
        num_bboxes = data.shape[0]
        
        # Each bounding box should have 2 corners, so total columns should be even
        if data.shape[1] % 2 != 0:
            raise ValueError(f"Data in {_path} has an unexpected number of columns: {data.shape[1]}")
        
        # Determine the number of spatial dimensions
        num_dimensions = data.shape[1] // 2
        
        # Reshape to (num_bboxes, 2, num_dimensions)
        reshaped = data.reshape(num_bboxes, 2, num_dimensions)
        bounding_box_corner_list.append(reshaped)

    # Create a mask for all possible corner combinations (optional, based on your original code)
    # Adjust this part based on how you intend to use the bounding boxes
    # For simplicity, we'll skip this unless necessary

    # Stack arrays into a single array if multiple paths are provided
    if len(bounding_box_corner_list) > 1:
        bounding_box_corner_list = np.concatenate(bounding_box_corner_list, axis=0)
    else:
        bounding_box_corner_list = bounding_box_corner_list[0]

    # Optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "bounding_boxes"  # optional, default is "image"

    return [(bounding_box_corner_list, add_kwargs, layer_type)]
