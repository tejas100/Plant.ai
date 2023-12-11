from django.shortcuts import render
from django.http import HttpResponse, request
from django.core.files.storage import FileSystemStorage
import string, random
import numpy as np
import os, io
import re
import matplotlib.pyplot as plt
from django.test import TestCase
import cv2

from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.path.join(BASE_DIR)


def index(request):
     context = {}
     if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        print(name)
        f = fs.url(name)   #Context
        print(f)

        # Start OF Main

        # ----------------- Directories ------------------- 
        letters = string.ascii_lowercase
        str1 = ( ''.join(random.choice(letters) for i in range(5)) )   
        denoicedir = '/denoice'+str1+'.png' 
        sharpdir = '/sharp'+str1+'.png'                   #Context --Geayscale
        contdir = '/cont'+str1+'.png'                 #Context --Noise
        morphdir = '/morph'+str1+'.png'             #Context --Morph
        mcontdir = '/mcont'+str1+'.png'             #Context --Morph Contrast 
        flatdir = '/flat'+str1+'.png'             #Context --Flat Contrast    
        
        image = cv2.imread('/Users/tejasbk/Documents/code/AI/plantai'+f)  #Context
        print(image.dtype)

        # Reduce noice 
        dst = cv2.fastNlMeansDenoisingColored(image, None, 8, 6, 10, 10)
        cv2.imwrite('/Users/tejasbk/Documents/code/AI/plantai/media/denoice'+denoicedir,dst)

      
        #Gauusian kernel for sharpening
        image = cv2.imread('/Users/tejasbk/Documents/code/AI/plantai/media/denoice'+denoicedir)
        gaussian_blur = cv2.GaussianBlur (image, (7,7), 2)
        # Sharpening using addweighted ()
        sharpened1 = cv2.addWeighted (image, 1.5, gaussian_blur, -0.5, 0)
        # sharpened1 = cv2.addWeighted (image, 3.5, gaussian_blur, -2.5, 0)
        cv2.imwrite('/Users/tejasbk/Documents/code/AI/plantai/media/sharp'+sharpdir,sharpened1)


        # Import model

        # classnames = ['Apple_Healthy', 'Apple_Unhealthy', 'Strawberry_Healthy', 'Strawberry_Unhealthy', 'Tomato_Healthy', 'Tomato_Unhealthy']

        model = load_model('/Users/tejasbk/Documents/code/plantDiseaseDetection/plantDiseaseDetection/model.h5')

        # Pass the preprocessed image
        # img_pil = image.load_img('/Users/tejasbk/Documents/code/AI/plantai/media/sharp'+sharpdir, target_size=(256, 256))
        # name = img_pil.filename
        # img = np.array(img_pil)

        img_array = cv2.imread('/Users/tejasbk/Documents/code/AI/plantai'+f)
        img_array = cv2.resize(img_array, (256, 256))  # Resize to match the target size
        # name = '/Users/tejasbk/Documents/code/AI/plantai/media/sharp'+sharpdir


        x = np.expand_dims(img_array, axis=0)

        output = model.predict(x)
        # output = output.astype(int).sum(axis=1) - 1
        # output = classnames[np.argmax(output[0])]
        # print(classnames[np.argmax(output[0])])
        # plt.imshow(img)
        print(output)

        context['result'] = output

        # Passing middle file to user
        context['orig'] = ".."+f
        context['denoice'] = "../media/denoice"+denoicedir
        context['sharp'] = "../media/sharp"+sharpdir
     
        return render(request,'showr.html',context)

     return render(request, "index.html")
