import numpy as np
import time
import cv2
import telebot
from reolinkapi import Camera
from ncnn.model_zoo import get_model

#hardware:
#RLC 410 5MP    $40 x4
#netgear PoE switch     $50
#300 foot reel          $36
#cat 5 end caps         $10
#raspberry pi 4. 4GB    $65

import configparser



def ClearTimeouts(TimeOuts):
  for key, data in TimeOuts.items():
    toRemove = []
    for key2, expiration in  data.items():
      if time.time() > expiration:
        toRemove.append(key2)
    for r in toRemove:
      print("\n\nTimeout on Camera",key,"for",r," cleared\n\n")
      TimeOuts[key].pop(r)

def IsInTimeOut(TimeOuts, cameraNum, label):
  for key, data in TimeOuts.items():
    if str(cameraNum+1) == key:
      for d, t in data.items():
        if d == label:
          return True
  return False


#adding profile to Camera, and RTSP client allows us to use the 640x480 stream as input
def non_blocking(IPList, pictureMode, TimeoutLength, MotionSensitivity, MinimumObjectSize, targets):
    #populate main variables
    num = len(IPList)
    camImg = [None]*num
    frameCount = [0]*num

    #need a class so the object can hold the callback function
    class callWrapper:
        def __init__(self, id):
            self.id = id

        def inner_callback(self, img):
            nonlocal camImg, frameCount
            frameCount[self.id] += 1
            camImg[self.id] = img

    t = []
    c = []
    bgsub = []
    for count, ip in enumerate(IPList):
        print(ip)
        c.append(Camera(ip[0], ip[1], ip[2], profile="sub"))
        ic = callWrapper(count)
        t.append(c[count].open_video_stream(callback=ic.inner_callback))
        bgsub.append(cv2.createBackgroundSubtractorMOG2())

    TimeOuts = {}
    for i in range(num):
        TimeOuts[str(i+1)] = {}

    thresh = {}
    for target in targets:
        thresh[target] = 0.75

    lastFrame = [0]*num

#    net = get_model("nanodet", target_size=320, nms_threshold=0.5, use_gpu=False)
    #this is slower than nanodet but necessary if we are going to be using night vision images
    net = get_model("mobilenetv2_ssdlite", target_size=320, num_threads=4, use_gpu=False)

    while True:
        for count, curT in enumerate(t):
             if not curT.is_alive():
                 print("Attempting to reboot camera", str(count+1))
                 ip = IPList[count]
                 c[count] = Camera(ip[0], ip[1], ip[2], profile="sub")
                 ic = callWrapper(count)
                 t[count] = c[count].open_video_stream(callback=ic.inner_callback)

        #remove expired TimeOuts
        ClearTimeouts(TimeOuts)

        indexesToCheck = []
        for i in range(num):
            if frameCount[i] != lastFrame[i]:
                indexesToCheck.append(i)
                lastFrame[i] = frameCount[i]

        if len(indexesToCheck) == 0:
            time.sleep(0.001)
            continue

        #for every updated image
        for index in indexesToCheck:
            #grab and check background subtraction for motion
            curImage = camImg[index]
            resized = cv2.resize(curImage[50:-50, 50:-50], (curImage.shape[0]//2, curImage.shape[1]//2))
            mask = bgsub[index].apply(resized)

            #this could be tune-able
            if np.count_nonzero(mask) > MotionSensitivity:
                objects = net(curImage)
                for o in objects:
                    if "class_names" in dir(net) and "label" in dir(o):
                        for target in targets:
                            if net.class_names[int(o.label)] == target and o.prob > thresh[target]:
                                 if IsInTimeOut(TimeOuts, index, target):
                                     print("Camera", index+1, "found", target, "but is in timeout")
                                     continue

                                 if o.rect.w*o.rect.h < MinimumObjectSize:
                                     print("Camera", index+1, "found", target, "but is too small")
                                     continue

                                 telegram.send_message(groupID, target + " detected on Camera" + str(index+1))

                                 if pictureMode:
                                     #draw rectangle
                                     cv2.rectangle(curImage, (int(o.rect.x), int(o.rect.y)), (int(o.rect.x + o.rect.w), int(o.rect.y + o.rect.h)))
                                     #prep image
                                     is_success, im_buf_arr = cv2.imencode(".png", curImage)
                                     byte_im = im_buf_arr.tobytes()
                                     #send image
                                     telegram.send_photo(groupID, photo=byte_im)
                                     #add timeout
                                     TimeOuts[str(index+1)][target] = time.time()+TimeoutLength




parser = configparser.ConfigParser()
parser.read("setupReal.conf")
token = parser.get("setup","TOKEN")
groupID = parser.get("setup","GROUP_ID")
IPAddresses = parser.get("setup", "IP_ADDRESS")
Usernames = parser.get("setup", "USERNAMES")
Passwords = parser.get("setup", "PASSWORDS")
telegram = telebot.TeleBot(token)

TimeoutLength = parser.get("params", "Timeout")
MotionSensitivity = parser.get("params", "Sensitivity")
MinimumObjectSize = parser.get("params", "MinimumSize")
pictureMode = parser.get("params", "SendPictures")
Targets = parser.get("params", "Targets")
IPList=[]

def prepArray(inArray):
    Outarray = []
    for ip in inArray[1:-1].split(","):
        Outarray.append(ip.strip().replace("\"", "").replace("\'", ""))
    return Outarray

IPAddresses = prepArray(IPAddresses)
Usernames = prepArray(Usernames)
Passwords = prepArray(Passwords)
Targets = prepArray(Targets)

#restructure the IP/user/pass
for IP, user, password in zip(IPAddresses, Usernames, Passwords):
    IPList.append([IP, user, password])

non_blocking(IPList, pictureMode, int(TimeoutLength), int(MotionSensitivity), int(MinimumObjectSize), Targets)
