from flask import Flask, render_template, request
import base64
import boundingbox
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import random

app = Flask(__name__)
#load our trained model
model = models.mobilenet_v3_large(pretrained=False)
model.classifier[3] = nn.Linear(1280, 6)
model.load_state_dict(torch.load("mobilenetv3.pkl", map_location="cpu"))
model.eval()

#THIS MIGHT BE WRONG ORDER!!!! FIX
CLASS_NAMES = ["fist", "no_gesture", "peace", "peace_inverted", "stop", "stop_inverted"] 

transform = T.Compose([
    T.ToTensor()
])

#add padding to cropped hand, call this before passing into model
def make_square(img):
    h, w, _ = img.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    return cv2.copyMakeBorder(img, top, bottom, left, right,cv2.BORDER_CONSTANT, value=[0, 0, 0])

@app.route('/')
def index():
    return render_template('index.html')

#this route is called by frontend after pressing submit
@app.route('/upload', methods=['POST'])
def upload():
    data = request.json['image']
    header, encoded = data.split(",", 1)
    image_bytes = base64.b64decode(encoded) #use this var as input (might need further processing)
    print("Received image of size:", len(image_bytes), "bytes")

    #GET BOUNDING BOX AND CROP HAND (jary)
    bbox, rgb_image = boundingbox.bounding_box(image_bytes)
    if bbox is None:
        print("No hand detected.")
    else:
        x1, y1, x2, y2 = bbox
        print("Bounding box:", bbox)
        cropped_image = rgb_image[y1:y2, x1:x2]
        # saving locally for testing purposes - remove as needed
        cv2.imwrite("cropped_hand.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        #now, convert cropped image back to base64 so we can send it back
        _, buffer = cv2.imencode('.png', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        cropped_base64 = base64.b64encode(buffer).decode('utf-8')


    #USE OUR TRAINED MODEL TO CLASSIFY THE CROPPED IMAGE (bobby)

    prediction = None

    if bbox is not None:
        #pad the image to 224x224 with helper
        padded = make_square(cropped_image)
        padded = cv2.resize(padded, (224, 224))

        #convert to tensor
        padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = transform(padded_rgb).unsqueeze(0)

        #try passing through model
        with torch.no_grad():
            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            prediction = CLASS_NAMES[pred_idx]

        print("Predicted Class:", prediction)

    #ONCE CLASS HAS BEEN DETERMINED, ADD LOGIC TO SEE IF USER WON AGAINST COMPUTERS RANDOM GUESS (bobby)


        #simplify the gesture to euither rock, paper or scissors
        if prediction in ["peace", "peace_inverted"]:
            player_move = "scissors"
        elif prediction == "fist":
            player_move = "rock"
        elif prediction in ["stop", "stop_inverted"]:
            player_move = "paper"
        elif prediction == "no_gesture":
            player_move = "no_gesture"
        else:
            player_move = "error"

        print("Player Move:", player_move)

        #now, the server opponent will randomly choose either rock paper or scissors
        print("Server is randomly choosing a move...")
        computer_move = random.choice(["rock", "paper", "scissors"])
        print("Computer Move:", computer_move)

        #logic to determine the winner
        if player_move in ["rock", "paper", "scissors"]:
            if player_move == computer_move:
                result = "Tie"
            elif (player_move == "rock" and computer_move == "scissors") or (player_move == "paper" and computer_move == "rock") or (player_move == "scissors" and computer_move == "paper"):
                result = "Player Wins"
            else:
                result = "Computer Wins"
        else:
            result = "Unknown Move"

        print("Game Result:", result)

    #RETURN WHAT CLASS HAND WAS IDENTIFIED AS, AS WELL AS IF USER (bobbty)

    return {
        "status": "success",
        "cropped_image": f"data:image/png;base64,{cropped_base64}" if cropped_base64 else None,
        "prediction": prediction,
        "player_move": player_move.capitalize(),
        "computer_move": computer_move.capitalize(),
        "result": result
    }


if __name__ == '__main__':
    app.run(debug=True)
