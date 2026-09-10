"""Dependency-free constants shared by CNN training and ONNX inference."""

CLASS_NAMES = ("mug", "apple", "orange", "can", "cereal_box")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_SIZE = 320
OUTPUT_STRIDE = 8
