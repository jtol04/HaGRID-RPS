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
    print("data is: ", data)
    header, encoded = data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    print("image_bytes is:", image_bytes)
    #do processing here
    print("Received image of size:", len(image_bytes), "bytes")

    return {"status": "success"}

if __name__ == '__main__':
    app.run(debug=True)
