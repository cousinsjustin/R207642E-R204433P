from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from glob import glob
import numpy as np
import json
import cv2
import sys
import os

class ObjectDetection(object):

    def __init__(self):
        self._objects = []

    def video_to_frames(self, video):
        print('splitting to frames...')
        frame_name = 'C:/Users/Cousins/Desktop/ins/static/frames'
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 0
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                cv2.imwrite(frame_name + str(count) + '.jpg', frame)
                print(frame)
            else:
                break
            count = count + 1
        vidcap.release()
        cv2.destroyAllWindows()
        print('Done splitting')

    inceptionV3 = load_model('C:/Users/Cousins/Desktop/ins/vgg16Model.h5')
    def detect(self):
        print('feeding frames to inceptionV3...')
        for frame in self.get_frames():
            image = load_img(frame, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            y_pred = inceptionV3.predict(image)
            label = decode_predictions(y_pred)
            self._objects.append(label[0][1][1])
        print('Done feeding')
        objects_file = 'C:/Users/Cousins/Desktop/ins/detected_objects.txt'
        with open(objects_file, 'w') as f:
            f.write(json.dumps(self._objects))

    def get_frames(self):
        frames_arr = glob("C:/Users/Cousins/Desktop/ins/static/frames/*.jpg")
        return frames_arr

    def get_objects(self):
        return self._objects

    def search_objects(self, _object):
        print('searching...')
        objects_file = 'C:/Users/Cousins/Desktop/ins/detected_objects.txt'
        with open(objects_file, 'r') as objects_file:
            objects = list(json.loads(objects_file.read()))
        search_results = []
        if _object in set(objects):
            for index in range(len(objects)):
                if _object.__eq__(objects[index]):
                    img_url = self.get_frames()[index].split('/')[-1]
                    search_results.append(img_url)
        else:
            return 'Object does not found'
        return search_results

    def read_objects(self):
        objects_file = 'C:/Users/Cousins/Desktop/ins/detected_objects.txt'
        with open(objects_file, 'r') as objects_file:
            objects = set(json.loads(objects_file.read()))
        return objects
