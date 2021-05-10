from flask import Flask, request, flash, redirect, jsonify
from face_recognition_image import Recognition
import os


app = Flask(__name__)
app.secret_key = 'watcharaponweeraborirak'
UPLOAD_FOLDER = 'static/uploads'


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.route('/face_rec', methods=['POST'])
def face_rec():
    if 'file' not in request.files:
        flash('No file part')
        return jsonify({'info': 'key form-data Invalid'})
    file = request.files['file']
    if file:
        image = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image)
        detected = Recognition(image)
        prediction = detected.face_recognition_DLIB()
        faces = prediction['face']
        img_name = prediction['img_name']
        process_time = prediction['time']
        host = request.host
        url = f'http://{host}/static/prediction/{img_name}.png'
        print(url)
        return jsonify({
            'face': faces, 
            'unknown': prediction['unknown'], 
            'peoples': prediction['peoples'],
            'url': url, 'status': 'success',
            'process_time': round(process_time, 2)
        })


if __name__ == '__main__':
    app.run(port=8080, debug=True)
    