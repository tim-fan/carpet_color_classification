import importlib.resources as pkg_resources
from .. import CarpetColorClassifier
from . import data
import json
import cv2

def test_carpet_color_classifier():
    # load a classifier and run over four test images
    
    input_images = [
        'light_blue.png',
        'dark_blue.png',
        'black.png',
        'beige.png',
    ]

    expected_classes = [
        'LIGHT_BLUE',
        'DARK_BLUE',
        'BLACK',
        'BEIGE',
    ]

    # load classifier
    with pkg_resources.path(data, "gmm_params.json") as model_params_path:
        classifier = CarpetColorClassifier(model_params_path)

    # invoke classifier on four test images
    for input_image, expected_class in zip(input_images, expected_classes):

        with pkg_resources.path(data, input_image) as img_path:
            frame = cv2.imread(str(img_path))

        _, actual_class = classifier.classify(frame)
        assert actual_class == expected_class
    
    