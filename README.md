# HaGRID Hand Gesture Classification + Rock Paper Scissors App


## Project Structure

```plaintext
├── templates/                # HTML templates
│   └── index.html
├── app.py                    # Flask Rock Paper Scissors application
├── boundingbox.py            # MediaPipe Cropping for getting Hand Bounding Box from user input images
├── cropped_hand.jpg          # sample cropped hand output from boundingbox.py
├── hand_landmarker.task      # needed for MediaPipe
├── mobilenet_train.py        # code for training via mobilenetv3
├── mobilenetv3.pkl           # trained mobilenetv3 model
├── script.py                 # code for getting cropped images from hagrid annotations
├── training_resnet.py        # code for training via resnet18
```

### Running the website
`python ./app.py`
