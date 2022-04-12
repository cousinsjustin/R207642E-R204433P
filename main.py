
import io 

#import upload_video as v

#import split_video as sv

#import classify as sst

import cv2 as cv

import tempfile
import streamlit as st

import split_video as ss

import os

import numpy as np

import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# import split_video as sv
from os import listdir
from os.path import isfile, join


#import cv2
#import numpy as np
#import os
from keras.applications.vgg16 import preprocess_input

def split_videos_to_frames(path:str):
    
    cap = cv.VideoCapture(path)
    
    if cap.isOpened():
        
        print("Video Opened")
        
    try:
        if not os.path.exists('frames'):
            
            os.makedirs('frames',exist_ok=True)
            
    except OSError:
        
        print ('Error: Creating directory of data')
        
    currentFrame = 0
    
    while(cap.isOpened()):    
        
            status, frame = cap.read()
            
            name = './frames/frame' + str(currentFrame) + '.jpg'
            
            print ('Creating...' + name)
            
            if status:
                
                cv.imwrite(name, frame)
                
            currentFrame += 1
            
            if currentFrame == 9:
                
                break
            
    cap.release()
    

    cv.destroyAllWindows()

def upload_video(file):
    
    if file is not None:
        
        mace = io.BytesIO(file.read())   
        
        temporary_location = ".datasimple5.mp4"
        
        with open(temporary_location, 'wb') as out:  
            
            out.write(mace.read())  
    
        out.close()
        
        ss.split_videos_to_frames(temporary_location)
    



    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    upload_video(uploaded_file)

classifications = sst.classifyImages()

#function to search in frames
def searchInFrames(object_):
    
    indeces = []
    
    if object_ in classifications:
        st.write(" object has been found")
  
    else:
        st.write("object is not in the frames!")
        
  #input is our varible to store our input      
input = st.text_input("object to search: ")

if st.button('Search'): 
    
    frames =[]
    
    detected_paths = []
    
    searchInFrames(input)
    
    st.write("")
    


def classifyImages():
    model = VGG16()
    from keras.applications.vgg16 import decode_predictions
    classify = []
    frames = [ join('.\\frames', f) for f in listdir('.\\frames') if isfile(join('.\\frames', f)) ]
    for i in range(len(frames)):    

        image = load_img(frames[i], target_size=(224, 224)) 
        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        img_pred = model.predict(image)
        label = decode_predictions(img_pred) 

        label = label[0][0]
        result =  label[1]

        classify.append(result)
    print(classify)
    
    return classify
if __name__ == "__main__":
    classifyImages()


#imports
# import io 

# import cv2 as cv

# import tempfile
# import streamlit as st







