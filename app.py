from flask import Flask, request, flash, redirect, jsonify, url_for, session, make_response
from face_recognition_image import Recognition
import os
import logging
import requests
import threading
import json
import shutil

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


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def remove_file():
    path_cropped = 'static/cropped'
    path_predict = 'static/prediction'
    if os.path.exists(path_predict):
        for fn in os.listdir(path_predict):
            path = os.path.join(path_predict, fn)
            print(path)
            os.remove(path)
    if os.path.exists(path_cropped):
        shutil.rmtree(path_cropped)
        os.mkdir(path_cropped)
    else:
        print('the no exist file.')


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
        path_prediction = '/static/prediction'
        path_prediction = os.path.join(path_prediction, f'{img_name}.png')
        url = f'http://{host}{path_prediction}'
        out = jsonify({
            'face': faces,
            'unknown': prediction['unknown'],
            'peoples': prediction['peoples'],
            'url': url, 'status': 'success',
            'process_time': round(process_time, 2)
        })
        out.set_cookie('path_timer', path_prediction)
        set_time = 60.0 * 60.0 * 5.0
        timer = threading.Timer(set_time, remove_file)
        timer.start()
        return out


@app.route('/api/get_folder_cropped')
def folder_cropped():
    dirs = 'static/cropped'
    path = os.listdir(dirs)
    return jsonify({'folder-date': path})


@app.route('/api/get_cropped')
def person_cropped():
    lst = []
    host = request.host
    folder = request.args['folder']
    dirs = 'static/cropped'
    path = os.path.join(dirs, folder)
    if os.path.exists(path):
        fn = os.listdir(path)
        for i in fn:
            fn = os.path.join(f'http://{host}/{dirs}/{folder}', i)
            lst.append(fn)
        return jsonify({f'folder-{folder}': lst})


@app.route('/api/get_prediction')
def person_predict():
    lst = []
    host = request.host
    dirs = 'static/prediction'
    path = os.listdir(dirs)
    for fn in path:
        r = os.path.join(f'http://{host}/{dirs}', fn)
        lst.append(r)
    return jsonify({'url': lst}, 'img-name', path)


@app.route('/api/delete_prediction/<string:path>')
def delete_predict(path):
    dirs = 'static/prediction'
    img = os.path.join(dirs, path)
    if os.path.exists(img):
        os.remove(img)
        return jsonify({'status': 'success deleted'})
    raise InvalidUsage('no such image exist', status_code=400)


@app.route('/api/delete_cropped')
def delete_cropped():
    path = request.args['folder']
    dirs = 'static/cropped'
    folder = os.path.join(dirs, path)
    if os.path.exists(folder):
        print('ok')
        for fn in os.listdir(folder):
            fn = os.path.join(folder, fn)
            os.remove(fn)
        os.rmdir(folder)
        return jsonify({'status': 'success deleted'})
    raise InvalidUsage('no such file directory', status_code=400)


@app.route('/api/check_list_trainset')
def trainset():
    trainset = os.listdir('datasets/labels')
    return jsonify({'trainset': trainset, 'total': len(trainset)})


if __name__ == '__main__':
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    app.run(port=8080, debug=True)
