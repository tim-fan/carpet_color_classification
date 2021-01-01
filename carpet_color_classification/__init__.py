"""carpet_color_classification."""

import json
from typing import Tuple
import numpy as np
import pandas as pd
import cv2
from sklearn.mixture import GaussianMixture

class CarpetColorClassifier():
    """
    Carpet color classifer
    Distinguishes between the four carpet colors in my office, given
    input images from a floor-facing camera.
    Initialises the classifier from a given json file, containing
    GMM parameters. See notebooks/train_classifier.ipynb for
    overview of how the GMM is trained
    """

    def __init__(self, gmm_param_file:str):
        #prepare classifier based on given gmm params
        with open(gmm_param_file) as param_file:
            model_params = json.load(param_file)

        self.gmm = GaussianMixture(n_components = len(model_params["gmm_means"]), covariance_type='full')
        self.gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(model_params["gmm_covariances"]))
        self.gmm.weights_ = np.array(model_params["gmm_weights"])
        self.gmm.means_ = np.array(model_params["gmm_means"])
        self.gmm.covariances_ = np.array(model_params["gmm_covariances"])

        self.class_index_to_class_name = {int(k):v for k,v in model_params["class_index_to_class_name"].items()}
        self.unclassified_score_threshold = model_params["unclassified_score_threshold"]

    def classify(self, image:np.ndarray) -> Tuple[int, str]:
        """
        given a cv2.image, return color classifcation result.
        Classification is returned as a tuple:
        ( class index, class name) e.g. (2, "LIGHT_BLUE")
        """

        r,g,b = _average_rgb_in_frame(image)
        h,s,v = _rgb_to_hsv(r,g,b)
        hsv_frame = pd.DataFrame([dict(h=h,s=s,v=v)])
        class_index = self.gmm.predict(hsv_frame)[0]
        score = self.gmm.score_samples(hsv_frame)[0]
        
        # apply score filtering
        if score < self.unclassified_score_threshold:
            class_index = 4
        
        print(self.class_index_to_class_name)
        class_name = self.class_index_to_class_name[class_index]
        return (class_index, class_name)

def _average_rgb_in_frame(frame):
    b,g,r = np.mean(frame, axis=(0,1))
    return (r,g,b)

def _rgb_to_hsv(r,g,b):
    rgb_array = np.uint8([[[int(r), int(g), int(b)]]])
    hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)[0][0]
    return (hsv_array[0], hsv_array[1], hsv_array[2])