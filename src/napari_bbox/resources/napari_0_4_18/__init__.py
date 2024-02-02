import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
cube_style_path = os.path.join(__location__, "cube_button.qss")
__all__ = ["cube_style_path"]
