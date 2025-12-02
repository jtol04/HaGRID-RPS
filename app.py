from flask import Flask, render_template, request
import base64
import boundingbox
import cv2

app = Flask(__name__)

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


    #RETURN IMAGE OF THE CROPPED HAND TO FRONTEND (bobby)

    #USE OUR TRAINED MODEL TO CLASSIFY THE CROPPED IMAGE (bobby)

    #ONCE CLASS HAS BEEN DETERMINED, ADD LOGIC TO SEE IF USER WON AGAINST COMPUTERS RANDOM GUESS (bobby)

    #RETURN WHAT CLASS HAND WAS IDENTIFIED AS, AS WELL AS IF USER (bobbty)

    return {
        "status": "success",
        "cropped_image": f"data:image/png;base64,{cropped_base64}" if cropped_base64 else None
    }

if __name__ == '__main__':
    app.run(debug=True)
