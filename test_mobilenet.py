import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

noGPU = True

if noGPU:
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"
else:
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  print("Num GPUs Available: ", len(physical_devices))
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

#mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet')
mobile = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

#hardware:
#reolink RLC 410 5MP    ~$50 x4
#netgear PoE switch     $167
#cat 6  2x25 feet       $22 
#cat 6 2x 75 feet       $46
#raspberry pi 4. 4GB    $65
#usb accelerator        $70

#

import time
#background subtraction


#maximum amount of saved storage in the directory = 1024*1024*1024*1024
#save annotated images to some directory
#save images before and after as a mp4 file in higher quality
#manage 2 folders - low quality and high quality

#for retrieval
  #live playback
  #recorded data


  
#notification setup:
# pip install telegram-send
# telegram-send --configure
# insert token:
#     2092290746:AAGqWQGHEBVWmJdvWfjqjWZ1dRa3UQ8XwiM

import telegram_send
##code here
#current = time.time()
#if FrontDoorTimeout and (current - FrontDoorLastTimeout) > 300:
# FrontDoorTimeout = False
#
#if FrontDoorMotionDetected and not FrontDoorTimeout:
# telegram_send.send(messages=["Motion Detected at the front door"])
# FrontDoorTimeout = True
# FrontDoorLastTimeout = time.time() 

#passed by reference, remove timeouts that expired
def ClearTimeouts(TimeOuts):
  for key, data in TimeOuts.items():
    toRemove = []
    for key2, expiration in  data.items():
      if time.time() > expiration:
        toRemove.append(key2)
    for r in toRemove:
      print("\n\n\n\n\n\nTimeout cleared\n\n\n\n\n\n\n\n")
      TimeOuts[key].pop(key2)

def IsInTimeOut(TimeOuts, cameraNum, label):
  for key, data in TimeOuts.items():
    if str(cameraNum+1) == key:
      for d, t in data.items():
        if d == label:
          return True
  return False
      
import cv2
from reolinkapi import Camera

#adding profile to Camera, and RTSP client allows us to use the 640x480 stream as input
#this significantly cuts down on CPU usage and increases FPS
#1.2GB used when res_net is running
#i5 7600k cpu
#25% at 7.5 fps
#70% at 15 fps
#
#maybe the pi 4 can run resnet50 at 4 fps without issues
#

def blocking():
    c1 = Camera("192.168.0.234", "admin", "sterlingarcher", profile="sub")
#    c2 = Camera("192.168.0.234", "admin", "sterlingarcher")
#    c3 = Camera("192.168.0.234", "admin", "sterlingarcher")
#    c4 = Camera("192.168.0.234", "admin", "sterlingarcher")
#    c5 = Camera("192.168.0.234", "admin", "sterlingarcher")
#    c6 = Camera("192.168.0.234", "admin", "sterlingarcher")

    # stream in this case is a generator returning an image (in mat format)
    stream1 = c1.open_video_stream()
#    stream2 = c2.open_video_stream()
#    stream3 = c3.open_video_stream()
#    stream4 = c4.open_video_stream()
#    stream5 = c5.open_video_stream()
#    stream6 = c6.open_video_stream()

    #populate the TimeOuts as they are identified
    #populated with TimeOuts["Camera1"]["Person"] = LastTimeSeen
    TimeOuts = {"1":{"jersey" : time.time()+5}}
    
    
    # or using a for loop
    count = 0
    targets = ["jersey", "screwdriver"]
#    for img in zip(stream1,stream2,stream3,stream4,stream5,stream6):
#       
     
#    for img1,img2,img3,img4,img5,img6 in zip(stream1,stream2,stream3,stream4,stream5,stream6):
    for img1 in stream1:
        count += 1
#        cv2.imshow("name", maintain_aspect_ratio_resize(img, width=600))

#        img1 = cv2.resize(img1, (448,448))
#        cv2.imshow("name", img1)
       
        #remove expired TimeOuts
        ClearTimeouts(TimeOuts)
        
        if count%2 == 0:
          print(img1.shape)
          img1 = cv2.resize(img1, (448,448))
          img2 = img1
#          img3 = cv2.resize(img3, (448,448))
#          img4 = cv2.resize(img4, (448,448))
#          img5 = cv2.resize(img5, (448,448))
#          img6 = cv2.resize(img6, (448,448))
          imgTest = np.array([img1[112:336,112:336],
                            img2[112:336,112:336]])
 #                           img3[112:336,112:336],
 #                           img4[112:336,112:336],
 #                           img5[112:336,112:336],
 #                           img6[112:336,112:336]])
          predictions = mobile.predict(imgTest)
          results = imagenet_utils.decode_predictions(predictions)
          
          #focus only on camera1
          
          FrontDoorMotionDetected = False
          for count, imgResult in enumerate(results):
            #top result only right now
            for r in imgResult[:1]:
              print(r[0], r[1], r[2])
              for target in targets:
                if not IsInTimeOut(TimeOuts, count, target) and r[1] == target and r[2] > 0.7:
                  cam = str(count+1)
                  telegram_send.send(messages=["Person detected on Camera #" + cam])
                  if cam not in TimeOuts.keys():
                    TimeOuts[cam] = {}
                  TimeOuts[cam][target] = time.time()+30

        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


# Call the methods. Either Blocking (using generator) or Non-Blocking using threads
# non_blocking()
blocking()