# HaGRID Hand Gesture Classification + Rock Paper Scissors App


## Project Structure

```plaintext
├── templates/                # HTML templates
│   └── index.html
├── app.py                    # Flask Rock Paper Scissors application
├── boundingbox.py            # MediaPipe Cropping for getting Hand Bounding Box from user input images
├── cropped_hand.jpg          # sample cropped hand output from boundingbox.py
├── hand_landmarker.task      # needed for MediaPipe
├── mobilenetv3.pkl           # trained mobilenetv3 model
