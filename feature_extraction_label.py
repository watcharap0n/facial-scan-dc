import pickle
import time
import cv2
import dlib
import os
from os import listdir

path = 'datasets/crop_train'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model_image/shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('model_image/dlib_face_recognition_resnet_model_v1.dat')

FACE_DETS = []
FACE_NAME = []
time_avg = []
stff = time.time()
for fn in listdir(path):
    if not fn.startswith('.'):
        path_folder = os.path.join(path, fn)
        path_enter = os.path.join(path_folder, 'label')
        for i in listdir(path_folder):
            if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG'):
                path_label = os.path.join(path_folder, i)
                img = cv2.imread(path_label, cv2.COLOR_BGR2RGB)
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc = model.compute_face_descriptor(img, shape, 1)
                    FACE_DETS.append(face_desc)
                    FACE_NAME.append(fn)
                    sec = (time.time() - stff)
                    time_avg.append(sec)
                    print('Done... {} {:.2f} '.format(path_label, sec))
avg = time_avg[-1] / len(time_avg)
print('avg: {} sec '.format(str(round(avg, 2))))
pickle.dump((FACE_DETS, FACE_NAME), open('train_datasets.pk', 'wb'))
