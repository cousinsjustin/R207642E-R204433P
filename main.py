
import io 

#import upload_video as v

#import split_video as sv

#import classify as sst

import cv2 as cv

import tempfile
import streamlit as st

#import split_video as ss

import os

import numpy as np

import pandas as pd

#import cv2
#import numpy as np
#import os
from keras.applications.vgg16 import preprocess_input

def split_videos_to_frames(path:str):
    
    cap = cv2.VideoCapture(path)
    
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
                
                cv2.imwrite(name, frame)
                
            currentFrame += 1
            
            if currentFrame == 9:
                
                break
            
    cap.release()
    

    cv2.destroyAllWindows()

def upload_video(file):
    
    if file is not None:
        
        mace = io.BytesIO(file.read())   
        
        temporary_location = ".datasimple5.mp4"
        
        with open(temporary_location, 'wb') as out:  
            
            out.write(mace.read())  
    
        out.close()
        
        ss.split_videos_to_frames(temporary_location)
    





uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

v.upload_video(uploaded_file)

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


#imports
# import io 

# import cv2 as cv

# import tempfile
# import streamlit as st







