import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import base64
import numpy as np
import io
import cv2

# https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

landmarker_path = "hand_landmarker.task"

def base64_to_rgb(image_bytes):
    """
    if "," in encoded_string:
        encoded_string = encoded_string.split(",", 1)[1]
    

    decoded_bytes = base64.b64decode(encoded_string)
    """
    
    # convert to np.uint8 buffer
    numpy_arr = np.frombuffer(image_bytes, dtype = np.uint8)

    bgr = cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def hand_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_path),
        running_mode=VisionRunningMode.IMAGE)
    
    return vision.HandLandmarker.create_from_options(options)

landmarker = hand_landmarker()

def bounding_box(image_bytes):
    # convert base 64 encoded string to rgb
    rgb = base64_to_rgb(image_bytes)

    # load the converted image
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)

    # run the task - perform hand landmark detection
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        print("Error")
        return None, rgb 
    
    landmarks = result.hand_landmarks[0]

    h, w, c = rgb.shape

    x_min, x_max = w, 0
    y_min, y_max = h, 0


    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_max = max(x, x_max)
        x_min = min(x, x_min)
        y_max = max(y, y_max)
        y_min = min(y, y_min)

    # adding padding

    padding = 35
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, w)
    y_max = min(y_max + padding, h)


    return (x_min, y_min, x_max, y_max), rgb

'''
# usage:

bbox, rgb_image = bounding_box(encoded_string)
if bbox is not None:
    x1, y1, x2, y2 = bbox
    cropped_image = rgb_image[y1:y2, x1:x2]



if __name__ == "__main__":
    with open("hand.jpg", "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
    encoded_string = encoded_bytes.decode("utf-8")

    bbox, rgb_image = bounding_box(encoded_string)
    if bbox is None:
        print("No hand detected in test image.")
        exit(0)

    x1, y1, x2, y2 = bbox
    print("Bounding box:", bbox)

    #save the iamge
    cropped_image = rgb_image[y1:y2, x1:x2]
    cv2.imwrite("cropped_hand.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
'''