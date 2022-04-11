import cv2
import numpy as np
import os
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
    
