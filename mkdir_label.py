import os
from os import listdir
import cv2


path = 'datasets/new_labels'


for fn in listdir(path):
    if not fn.startswith('.'):
        path_folder = os.path.join(path, fn)
        path_enter = os.path.join(path_folder, 'label')
        if not path_enter.startswith('.'):
            os.makedirs(path_enter, exist_ok=True)
            for i in listdir(path_folder):
                if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG'):
                    path_img = os.path.join(path_folder, i)
                    image = cv2.imread(path_img)
                    img = os.path.join(path_enter, i)
                    cv2.imwrite(img, image)
            
    