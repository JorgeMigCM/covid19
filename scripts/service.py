# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com
#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

#Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

#Import Keras
from keras.preprocessing import image

#Import python files
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename
from model_loader import cargarModelo

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

app = Flask(__name__)
CORS(app)
global loaded_model, graph
loaded_model, graph = cargarModelo()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main_page():
	return 'Servicios Activados'

@app.route('/model/covid19/', methods=['GET','POST'])
def default():
    data = {"success": False}
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            image_to_predict = image.load_img(filename, target_size=(224, 224))
            test_image = image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255

            with graph.as_default():
            	result = loaded_model.predict(test_image)[0][0]

            	prediction = 1 if (result >= 0.5) else 0
            	CLASSES = ['Pulmones Normales', 'Covid19 positivo']

            	ClassPred = CLASSES[prediction]
            	ClassProb = result
            	
            	print("Pedicción:", ClassPred)
            	print("Prob: {:.2%}".format(ClassProb))

            	#Results as Json
            	data["predictions"] = []
            	r = {"Diagnostico": ClassPred, "score": float(ClassProb)}
            	data["% Prediccion"].append(r)


    return jsonify(data)

app.run(host='0.0.0.0',port=port, threaded=False)
