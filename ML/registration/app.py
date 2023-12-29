import dlib
import numpy as np
import io
import os
from flask import Flask, jsonify, request
from auth import auth

from werkzeug.utils import secure_filename
from google.cloud import storage
from PIL import Image, ImageOps, ImageEnhance

from google.oauth2 import service_account
from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text



app = Flask('__main__')
# configuration
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])
project_id = "checkcapstone"
bucket_name = "face_storage_capstone"

# Create a storage client
storage_client = storage.Client.from_service_account_json("checkcapstone-17084f1a691c.json")
bucket = storage_client.bucket(bucket_name)
# Create dlib
detector = dlib.get_frontal_face_detector()
# Create Sql connection
credentials = service_account.Credentials.from_service_account_file("checkcapstone-17084f1a691c.json")
connector = Connector(credentials=credentials)
def get_connector():
    getconn = connector.connect(
        "checkcapstone:us-central1:api-cc",
        "pymysql",
        user="root",
        password="password123456789",
        db="capstone"
    )
    return getconn

engine = create_engine("mysql+pymysql://",creator=get_connector)
Session = sessionmaker(bind=engine)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def convert_and_trim_bb(image, rect):
  startX = 0
  startY = 0
  w = 0
  h = 0
  if len(rect) > 0:
    rect = rect[0]
    # extract the starting and ending (x, y)-coordinates of the bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
	  # return our bounding box coordinates

  return (startX, startY, w, h)

def face_detector(image):
  image = Image.open(image)
  image_array = np.array(image)
  faces = detector(image_array)
  crop = convert_and_trim_bb(image_array, faces)

  if sum(crop) > 1 :
    
    cropped_image = image.crop((crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3]))
    resize_image = cropped_image.resize((160,160))
    enhancer = ImageEnhance.Sharpness(resize_image)
    photo = enhancer.enhance(2)

    return photo
  
  else:
    return None

@app.route("/")
def index():
    return jsonify({
       "status": {
          "code": 200,
          "message": "Success fetching the API"
       },
       "data": None
    }), 200


@app.route("/upload", methods=["POST"])
@auth.login_required()
def uploader_face():
    if request.method == "POST":
      user_id = request.form.get('user_id')
      image_path = user_id + "/"
      session = Session()
      data = session.execute(text("SELECT user_id FROM capstone.verification_statue WHERE user_id = '" + user_id + "'"))
      
      if len(data.all()) == 0:
        session.close()
        image = request.files["face_photo"]

        if image and allowed_file(image.filename):
          buffer = io.BytesIO()
          filename = secure_filename(image.filename)
          face = face_detector(image)

          if type(face) == type(None):
             return jsonify({ "status": {
              "code": 406,
              "message": "request success, face undetected",
            }, 
              "data": 0
          }),406
            
          else:
            #insert id user
            user_data = {'user_id': user_id, 'image_path': image_path, 'status':2 }
            session = Session()
            insert_query = text("INSERT INTO capstone.verification_statue (user_id, image_path, status) VALUES (:user_id, :image_path, :status)")
            session.execute(insert_query, user_data)
            session.commit() 
            session.close()
            #create empty folder
            blob = bucket.blob(image_path)
            blob.upload_from_string('')
            #prepare upload
            blob = bucket.blob(image_path + filename)
            face.save(buffer, format='JPEG')
            byte_im = buffer.getvalue()

            try:
            # Upload the image data to the blob
              blob.upload_from_string(byte_im, content_type="image/jpeg")
              return jsonify({ "status": {
                "code": 200,
                "message": "request success, face detected",
              }, 
                "data": 1
              }),200
            
            except Exception as e:
              return jsonify({ "status": {
                "code": 500,
                "message": f"{e}",
              }, 
                "data": 0
              }),500
        
      else:
        session.close()
        image = request.files["face_photo"]
        if image and allowed_file(image.filename):
          buffer = io.BytesIO()
          filename = secure_filename(image.filename)
          face = face_detector(image)
          
          if type(face) == type(None):
             return jsonify({ "status": {
              "code": 406,
              "message": "request success, face undetected",
            }, 
              "data": 0
          }),406
            
          else:
            blob = bucket.blob(image_path + filename)
            face.save(buffer, format='JPEG')
            byte_im = buffer.getvalue()

            try:
            # Upload the image data to the blob
              blob.upload_from_string(byte_im, content_type="image/jpeg")
              return jsonify({ "status": {
                "code": 200,
                "message": "request success, face detected",
              }, 
                "data": 1
              }),200
            
            except Exception as e:
              return jsonify({ "status": {
                "code": 500,
                "message": f"{e}",
              }, 
                "data": 0
              }),500
          
        else:
          session.close()
          return jsonify({
            "status": {
                "code": 422,
                "message": "request success, wrong file extension"
            },
            "data": None
          }),422

@app.route("/confirm", methods=["POST"])
@auth.login_required()
def confirm_upload():
  user_id = request.form.get('user_id')
  session = Session()
  user_data = {'status':0 ,'user_id':user_id}
  select_query = text("SELECT status FROM capstone.verification_statue WHERE user_id=:user_id")
  query_result = session.execute(select_query, user_data).first()
  session.close()
  if type(query_result) == type(None):
    return jsonify({
      "status": {
        "code": 404,
        "message": "user id not found"
      },
      "data": f"{query_result}",
      }),404
  
  else:
    if query_result[0] == 2:
      session = Session()
      insert_query = text("UPDATE capstone.verification_statue SET `status` = 0 WHERE user_id=:user_id")
      session.execute(insert_query, user_data)
      session.commit()
      select_query = text("SELECT status FROM capstone.verification_statue WHERE user_id=:user_id")
      query_result = session.execute(select_query, user_data).first()
      session.close()

      if query_result[0] == 0:
        confirmation_status = 1
        return jsonify({
          "status": {
              "code": 200,
              "message": "request success, upload confirmed"
          },
          "data": confirmation_status,
          "confirmation_status" : 1,
          "user_id":f"{user_id}"
        }),200
      
      else:
        confirmation_status = 0
        return jsonify({
          "status": {
              "code": 500,
              "message": "request success, upload failed, photo face not confirmed",
              
          },
          "data": confirmation_status,
          "confirmation_status" : 0,
          "user_id":f"{user_id}"
        }),500

    else:
      return jsonify({
      "status": {
        "code": 200,
        "message": "upload has been confirmed"
      },
      "data": f"{query_result}",
      }),200

@app.route("/check", methods=["POST"])
@auth.login_required()
def check_status():
  user_id = request.form.get('user_id')
  session = Session()
  user_data = {'user_id':user_id}
  insert_query = text("SELECT status,user_id FROM capstone.verification_statue WHERE user_id = :user_id")
  query_result = session.execute(insert_query, user_data).first()
  session.close()
  if len(query_result) > 0: 
    return jsonify({
              "status": {
                  "code": 200,
                  "message": "request success, data returned"

              },
              "data": f"{query_result}",
              "account_status": query_result[0]
            }),200
  else: 
    return jsonify({
              "status": {
                  "code": 404,
                  "message": "user id not found"

              },
              "data": f"{query_result}",
              "account_status": None
            }),404


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))