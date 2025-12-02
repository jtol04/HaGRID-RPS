from flask import Flask, render_template, request
import base64

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

    #RETURN IMAGE OF THE CROPPED HAND TO FRONTEND (bobby)

    #USE OUR TRAINED MODEL TO CLASSIFY THE CROPPED IMAGE (bobby)

    #ONCE CLASS HAS BEEN DETERMINED, ADD LOGIC TO SEE IF USER WON AGAINST COMPUTERS RANDOM GUESS (bobby)

    #RETURN WHAT CLASS HAND WAS IDENTIFIED AS, AS WELL AS IF USER (bobbty)

    return {"status": "success"}

if __name__ == '__main__':
    app.run(debug=True)
