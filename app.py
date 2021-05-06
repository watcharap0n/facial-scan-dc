from flask import Flask, request, flash, redirect, jsonify
from face_recognition_image import Recognition
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

@app.route('/face_rec', methods=['POST'])
def face_rec():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        image = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image)
        detected = Recognition(image)
        prediction = detected.face_recognition_DLIB()
        faces = prediction['face']
        img_name = prediction['img_name']
        host = request.host
        url = f'{host}/{UPLOAD_FOLDER}/{img_name}'
        print(url)
    return jsonify({'face': faces, 'url': url, 'status': 'success'})



if __name__ == '__main__':
    app.run(port=8080, debug=True)
    