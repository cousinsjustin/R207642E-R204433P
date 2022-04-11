import streamlit as st

import cv2 as cv

import tempfile

import split_video as ss

import os

import io

import numpy as np

import pandas as pd

def upload_video(file):
    
    if file is not None:
        
        mace = io.BytesIO(file.read())   
        
        temporary_location = ".datasimple5.mp4"
        
        with open(temporary_location, 'wb') as out:  
            
            out.write(mace.read())  
    
        out.close()
        
        ss.split_videos_to_frames(temporary_location)
    


        