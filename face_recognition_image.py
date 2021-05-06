import numpy as np
import cv2
import dlib
from imutils import face_utils
import pickle
import os
import time
import uuid
import datetime

class ModelImage:
    FACE_DESC, FACE_NAME = pickle.load(open('train_datasets.pk', 'rb'))
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('model_image/shape_predictor_68_face_landmarks.dat')
    model = dlib.face_recognition_model_v1('model_image/dlib_face_recognition_resnet_model_v1.dat')


class Recognition(ModelImage):
    def __init__(self, file_name):
        self.file_name = file_name

    def face_recognition_DLIB(self):
        path_mkdir = 'static/cropped'
        d1 = str(datetime.datetime.now())
        path_join = os.path.join(path_mkdir, d1)
        os.mkdir(path_join)
        names = []
        image = cv2.imread(self.file_name, cv2.IMREAD_UNCHANGED)
        time_start = time.time()
        scale = 0.5
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = self.detector(gray_scale, 1)
        for d in detected:
            xy = d.left(), d.top()
            wh = d.right(), d.bottom()
            shape = self.sp(gray_scale, d)
            face_desc_first = self.model.compute_face_descriptor(image, shape, 1)
            shape_circle = face_utils.shape_to_np(shape)
            d = []
            for face_desc in self.FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc_first)))
            d = np.array(d)
            idx = np.argmin(d)
            if d[idx] <= 0.34:
                name = self.FACE_NAME[idx]
                name = str(name)
                print(d[idx], name)
                percent = 1.0 - d[idx]
                percent = percent * 100
                cv2.putText(image, f'{name}', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(image, f'{round(percent, 1)}%', (xy[0] + 40, xy[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, xy, wh, (0, 255, 0), 2)
                names.append(name)
            elif d[idx] <= 0.39:
                name = self.FACE_NAME[idx]
                name = str(name)
                print(d[idx], name)
                percent = 1.0 - d[idx]
                percent = percent * 100
                cv2.putText(image, f'{name}', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(image, f'{round(percent, 1)}%', (xy[0] + 40, xy[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(image, xy, wh, (0, 255, 255), 2)
                # for (x, y) in shape_circle:
                #     cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                names.append(name)
            else:
                percent = 1.0 - d[idx]
                percent = percent * 100
                cv2.putText(image, f'{round(percent, 1)}%', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.rectangle(image, xy, wh, (0, 0, 255), 2)
                # for (x, y) in shape_circle:
                #     cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            write_image = image[xy[1] - 40:wh[1], xy[0]:wh[0]]
            cropped_name = uuid.uuid4().hex
            cv2.imwrite(filename=f'{path_join}/{cropped_name}.png', img=write_image)
        img_name = uuid.uuid4().hex
        cv2.imwrite(filename=f'static/prediction/{img_name}.png', img=image)
        endtime = time.time() - time_start
        print(round(endtime, 2))
        print(names)
        result = {
            'face': names,
            'img_name': img_name,
            'time': endtime
        }
        return result

# face = Recognition('test_images/team_mango2018.jpg')
# face.face_recognition_DLIB()
