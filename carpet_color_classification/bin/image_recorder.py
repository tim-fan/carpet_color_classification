"""
image_recorder.

Saves images from webcam to disk

Usage:
    image_recorder <output_directory> [--device=<index>]

Options:
    --device=<index>    Index of video device for capture [default: 0]
"""

# webcam access based on:
#   https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam
#   https://stackoverflow.com/a/19086507/5718370

import cv2
import os
from docopt import docopt

class VideoCaptureWrapper(object):
    def __init__(self, *args, **kwargs):
        self.vid_stream = cv2.VideoCapture(*args, **kwargs)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.vid_stream.release()

def main():

    arguments = docopt(__doc__, version='1.0')
    capture_device = int(arguments['--device'])
    save_dir = arguments['<output_directory>']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_count = 0

    with VideoCaptureWrapper(capture_device) as cap:    
        # Check if the webcam is opened correctly
        if not cap.vid_stream.isOpened():
            raise IOError(f"Cannot open webcam device {capture_device}")
        while(1):
            _,frame = cap.vid_stream.read()
            cv2.imshow('Input',frame)

            image_count += 1
            
            filename = f"{save_dir}/image_{str(image_count).zfill(6)}.png"
            print(f"Saving {filename}")
            cv2.imwrite(filename, frame)

            # exit on 'esc'
            if cv2.waitKey(5)==27:
                cv2.waitKey()
                break

    # cap = cv2.VideoCapture(0)



    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #     cv2.imshow('Input', frame)

    #     c = cv2.waitKey(1)
    #     if c == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
