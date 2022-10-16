

from flask import Flask
from flask import request
import numpy as np
from keras.models import load_model
from keras.utils import load_img
from flask import jsonify



app = Flask(__name__)
 



saved_model = load_model("mymodelvgg16.h5")


def response(path,text,output):
    return {"filename":path,"class":text,"output":output}

@app.route("/predict",methods=['POST'])
def predict():
    image = request.files['file']
    # print(image.read())
    # image.stream()
    path = "req.jpg"
    image.save(path)
    img = load_img(path, target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    output = saved_model.predict(img).tolist()
    print(output)

    return jsonify(response(image.filename,"Captured",output)) if output[0][0] > output[0][1] else jsonify(response(image.filename,"Scanned",output))


        
 
if __name__ == "__main__":
    app.run(debug=True,port=8000)