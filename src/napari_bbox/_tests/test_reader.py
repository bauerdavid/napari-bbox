import numpy as np

from .._reader import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """Write bounding box corners to csv and read them back."""

    # each row holds the min and max corner of one 3D bounding box
    my_test_file = str(tmp_path / "myfile.csv")
    corners = np.array([[0, 5, 5, 3, 20, 20], [1, 2, 3, 4, 8, 9]], dtype=float)
    np.savetxt(my_test_file, corners, delimiter=",")

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 3
    assert layer_data_tuple[2] == "boundingboxlayer"

    # make sure it's the same as it started
    np.testing.assert_allclose(corners.reshape(2, 2, 3), layer_data_tuple[0])


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
