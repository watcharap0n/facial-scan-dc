import cv2
from os import listdir
import os
import dlib
import time
import pickle


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model_image/shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('model_image/dlib_face_recognition_resnet_model_v1.dat')

path = 'datasets/crop_train'
# path = 'datasets/crop_train'
img_pixel = 128
scale = 0.5


FACE_DETS = []
FACE_NAME = []
time_avg = []
stff = time.time()

for fn in listdir(path):
    if not fn.startswith('.'):
        path_fn = os.path.join(path, fn)
        path_enter = os.path.join(path_fn, 'label')
        if not path_enter.startswith('.'):
            os.makedirs(path_enter, exist_ok=True)
        for i in listdir(path_fn):
            path_img = os.path.join(path_fn, i)
            if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG'):
                image_color = cv2.imread(path_img)
                image = cv2.resize(image_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_scale, 1)
                print(f'folder {fn}, det {dets}')
                # print(path_img)
                for k, d in enumerate(dets):
                    x, y = d.left(), d.top()
                    w, h = d.right(), d.bottom()
                    cropped_image = image[y:h, x:w]  # cropped_image
                    image = cv2.resize(cropped_image, (img_pixel, img_pixel), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(path_enter, i), image)
        # print('finishing cropped...')
        for e in listdir(path_enter):
            if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG'):
                path_label = os.path.join(path_enter, e)
                # print(path_label)
                img = cv2.imread(path_label, cv2.COLOR_BGR2RGB)
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc = model.compute_face_descriptor(img, shape, 1)
                    FACE_DETS.append(face_desc)
                    FACE_NAME.append(fn)
                    sec = (time.time() - stff)
                    time_avg.append(sec)
avg = time_avg[-1] / len(time_avg)
# print('avg: {} sec '.format(str(round(avg, 2))))
pickle.dump((FACE_DETS, FACE_NAME), open('trainingset_dc.pk', 'wb'))

            