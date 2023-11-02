# For starting server and handling API requests
from flask import Flask,flash,redirect,render_template, request,url_for    

# For storing and Checking up files 
import os         
from werkzeug.utils import secure_filename       

# Extracting necessary information from the image                                                                   
from PIL import Image

# For performing necessary operations on data
import numpy as np

# Undumping the Machine Learning Model
import pickle

UPLOAD_FOLDER = 'static/uploads'                                                        # Providing path of storing the uploaded files
ALLOWED_EXTENSIONS ={'jpg','jpeg'}                                                      # Providing info about the file extensions that are allowed

# Creating our flask app
app = Flask(__name__)

# Undumping our flask model
model = pickle.load(open('cnn_model.pkl','rb'))

# Providing configurations
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'

# Creating a function that will check the whether file extension is allowed or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
# Defining the root route 
@app.route('/')
def index():                                
    flash("📝Format: jpg or jpeg 🔒Category: AIRPLANE ✈️, AUTOMOBILE 🚗, BIRD 🕊️, CAT 🐈, DEER 🦌, DOG 🐕, FROG 🐸, HORSE 🐎, SHIP 🚢, TRUCK 🚚")                                                   
    return redirect(url_for('home'))                                                     # Redirecting it to home route

# Defining the home route                                                                           
@app.route('/home')
def home():
    return render_template('index.html',prediction_text = "Submit to see the magic 🪄")  # Sending an html file in response

# Defining the postimg route for both GET and POST method 
@app.route('/postimg',methods = ['GET','POST'])
def predict():
    
    # If request method is route
    if request.method=='POST':
        
        # If request contains image
        if 'Image' in request.files:
            
            file  =  request.files['Image']                                             # Extracting the image object from the req object
            
            # If file is not uploaded
            if file.filename =='' :
                flash("OOPS!😆, You forgot to upload image")                            # Flashing the msg that file is not uploaded 
                return redirect(url_for('home'))                                        # Redirecting it to home page again 
            
            # If file is there and file extension is correct 
            if file and allowed_file(file.filename):
                
                filename = secure_filename(file.filename)                               # Extracting the file name
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))          # Uploading the file to the upload folder
                
                # Resizing the file as our model takes an image with feature 32*32 pixels and have 3 channels
                input_file = request.files.get('Image')                                 # Extracting the image
                im = Image.open(input_file)                                             # Opening the image
                n_img= im.resize((32,32))                                               # Resizing the image
                
                # Extracting the value of features for our image
                img_test = np.array(n_img)                                              # Extracting the features in the form of a 3D array 
                img_test = img_test.astype('float64')                                   # Converting the values into float for more precision
                img_test /= 255.0                                                       # Normalizing the value of Pixels
                
                # Converting our array to higher dimensions as our model take data with 4 dimensions where one axis is for batch 
                img_test = img_test[np.newaxis,...]                                     # Adding the dimensions
                
                # These are the possible labels for our output
                classes = ["AIRPLANE ✈️","AUTOMOBILE 🚗","BIRD 🕊️","CAT 🐈","DEER 🦌","DOG 🐕","FROG 🐸","HORSE 🐎","SHIP 🚢","TRUCK 🚚"]
                
                # Predicting the output as we have already done with the necessary operations                                     
                y_pred = model.predict(img_test)
                
                # Finding the class with high priority as our model will return the probabilities for each class
                y_classes = [np.argmax(element) for element in y_pred]
                
                #Finding the label for the output class with high priority
                output = classes[y_classes[0]]
                
                # Flashing and rendering the file 
                flash("Here's, my Prediction ⬇️")
                return render_template('index.html',prediction_text = output + "?")   
            
        # If file format does not match
        flash('⚠️Image should be in jpg or jpeg format')
        return redirect(url_for('home'))
    
    # If request method is GET 
    return redirect(url_for('home'))

# Starting the server 
if __name__=='__main__':
    app.run(debug=True)
