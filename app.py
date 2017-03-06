import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify
from flask import flash
from flask import redirect

app = Flask(__name__)
model = ResNet50(weights='imagenet')


@app.route('/', methods=['GET'])
def get():
    return '''
    
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=number name=count value=5>
        <input type=submit value="Predict">
    </form>
    '''

@app.route('/', methods=['POST'])
def post():
    count = request.form.get('count', 5)
    file = request.files['file']
    image = Image.open(file.stream)

    image.thumbnail((1000, 224), Image.ANTIALIAS)
    image = image.resize((224, 224))

    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float64)
    image = preprocess_input(image)
    predictions = model.predict(image)
    results = []
    for prediction in decode_predictions(predictions, top=int(count))[0]:
        label_id, label_text, confidence = prediction
        results.append(dict(label_id=label_id, label_text=label_text,
                            confidence=float(confidence)))
    r = sorted(results, key=lambda x: x.get('confidence'), reverse=True)
    return jsonify(r)

app.run(host='0.0.0.0', port=5000)