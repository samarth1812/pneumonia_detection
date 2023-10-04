from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread, imshow
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)
model = load_model('my_model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['JPG' ,'jpg', 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    # Input an image for classification
      # Replace with the path to your input image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = cv2.resize(img, (224, 224))  # Resize the image to match the model's input size (224x224)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]

# Reshape the image to match the model's input shape
    input_image = img.reshape(1, 224, 224, 1)
    
    
    return input_image




@app.route('/predict',methods=['GET','POST'])
def predict():
      
      ref={0: 'Normal',
      1: 'Pneumonia Affected'
     }
      
      if request.method == 'POST':
          file = request.files['file']
          
          if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)

                img = read_image(file_path)

                # Make the prediction
                answer = model.predict(img)

                # Get the class label with the highest probability
                d = ref[np.argmax(answer)]

                # Calculate the probability of the class label
                probability = round(np.max(answer) * 100, 2)

                return render_template('predict.html', disease=d, prob=probability, user_image=file_path)

              
          else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)