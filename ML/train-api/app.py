import tensorflow as tf
import numpy as np
import os
import tempfile
import shutil
import io
from flask import Flask, jsonify, request
from auth import auth
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sklearn.ensemble import IsolationForest
from joblib import dump

app = Flask('__main__')
project_id = "checkcapstone"
bucket_name = "face_storage_capstone"
model_bucket_name = "model_face_bucket"
credentials = service_account.Credentials.from_service_account_file("checkcapstone-17084f1a691c.json")
storage_client = storage.Client.from_service_account_json("checkcapstone-17084f1a691c.json")
connector = Connector(credentials=credentials)
ImgDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
bucket = storage_client.bucket(bucket_name)
model_bucket = storage_client.bucket(model_bucket_name)

app.config['MODEL_FILE'] = 'models/'
predictor = tf.keras.models.load_model(app.config['MODEL_FILE'], compile=False, safe_mode=False)

def getconn():

    getconn = connector.connect(
        "checkcapstone:us-central1:api-cc",
        "pymysql",
        user="root",
        password="password123456789",
        db="capstone"
    )
    return getconn

engine = create_engine("mysql+pymysql://",creator=getconn,)
Session = sessionmaker(bind=engine)


def train_val_generators(TRAINING_DIR):

    train_datagen = ImgDataGenerator(
                    rescale=1./255,
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    brightness_range=[0.25, 1.3],
                    rotation_range=20,
                    horizontal_flip=True,
                    fill_mode="nearest",
                    )

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=5,
                                                        class_mode="binary",
                                                        target_size=(160, 160))

    return train_generator

def sklearn_model(xtrain, ytrain):
   if_clf = IsolationForest(contamination=0.024, max_features=1.0, max_samples=1.0, random_state = 3, n_estimators=200)
   if_clf.fit(xtrain,ytrain)
   return if_clf


@app.route("/")
def index():
    return jsonify({ "status": {
          "code": 200,
          "message": "request success, face detected"
          }, 
          
          "data": None
        }),200

@app.route("/modeler", methods=["POST"])
@auth.login_required()
def model_generator():
    if request.method == "POST":
        session = Session()
        queue = session.execute(text("SELECT user_id,image_path FROM capstone.verification_statue WHERE `status` = 0")).first()
        session.close()
        temp_folder = os.path.join(tempfile.gettempdir(), f"image_model_temp")
        image_model = os.path.join(temp_folder,"user")

        if type(queue) != type(None):
            user_id = queue[0]
            buffer = io.BytesIO()
            list = bucket.list_blobs(prefix=queue[1])

            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            os.makedirs(temp_folder)
            os.makedirs(image_model)
            x=0
            for blob in list:
                x=x+1
                if blob.name != queue[1]:
                    blob.download_to_filename(image_model + "/picture"+str(x)+".jpeg")
                
            dataset = train_val_generators(temp_folder)
            x_train = []
            for x in range(40):
                vector, label = dataset.next()
                face_pattern = predictor.predict(vector)
                x_train.append(face_pattern[0])
                x_train.append(face_pattern[1])
                x_train.append(face_pattern[2])
                x_train.append(face_pattern[3])
            
            x_train = np.array(x_train)
            y_train = np.ones(np.size(x_train))
            verif_model = sklearn_model(x_train, y_train)
            dump(verif_model, buffer)
            byte_im = buffer.getvalue()
            blob = model_bucket.blob(str(user_id)+".joblib")
            blob.upload_from_string(byte_im,content_type="application/octet-stream")

            shutil.rmtree(temp_folder)

            session = Session()
            user_data = {'status':1 ,'user_id':user_id, 'model_path':str(user_id)+".joblib"}
            insert_query = text("UPDATE capstone.verification_statue SET `status`= :status, `model_path`= :model_path  WHERE user_id=:user_id;")
            session.execute(insert_query, user_data)
            session.commit() 
            session.close()

            return jsonify({ "status": {
                "code": 200,
                "message": "model train success"
                }, 
                
                "data": None
            }),200
        
        else:
            return jsonify({ "status": {
                "code": 200,
                "message": "no queue, wait and repeat action"
                }, 
                
                "data": None
            }),200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))