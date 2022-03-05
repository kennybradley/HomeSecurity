import numpy as np
import time
import cv2
import telebot
from reolinkapi import Camera
from ncnn.model_zoo import get_model
import configparser
import datetime
import pytz
import os

#clear any timeout specified for a time that has already passed
def ClearTimeouts(TimeOuts):
  for key, data in TimeOuts.items():
    toRemove = []
    for key2, expiration in  data.items():
      if time.time() > expiration:
        toRemove.append(key2)
    for r in toRemove:
      print("\n\nTimeout on Camera",key,"for",r," cleared\n\n")
      TimeOuts[key].pop(r)

#check to see if the given label for the given camera is current in timeout
def IsInTimeOut(TimeOuts, cameraNum, label):
  for key, data in TimeOuts.items():
    if str(cameraNum+1) == key:
      for d, t in data.items():
        if d == label:
          return True
  return False

#Allow for a dummy class that returns false on is_alive
#in case the script gets stalled, this will allow it to reboot
class Dummy:
    def is_alive():
      print(time.time(), " forcing a reconnection")
      return False

#this is the main processing loop for the image detection pipeline
def runMainLoop(IPList, pictureMode, TimeoutLength, MotionSensitivity, MinimumObjectSize, targets):
    #populate local variables that hold the image Array and current frame counts
    num = len(IPList)
    camImg = [None]*num
    frameCount = [0]*num

    #Need a class so the object can hold the callback function
    #We need the callback function to pass into the video stream
    class callWrapper:
        #set the camera ID
        def __init__(self, id):
            self.id = id

        #update the nonlocal image array and frame count
        def inner_callback(self, img):
            nonlocal camImg, frameCount
            if img is None:
                print("No image found")
                return
            frameCount[self.id] += 1
            camImg[self.id] = img


    #arrays containing the stream, the camera object and the background substraction
    #each of these will be indexed by the camera number
    t = []
    c = []
    bgsub = []
    for count, ip in enumerate(IPList):
        c.append(Camera(ip[0], ip[1], ip[2], profile="sub"))
        ic = callWrapper(count)
        t.append(c[count].open_video_stream(callback=ic.inner_callback))
        bgsub.append(cv2.bgsegm.createBackgroundSubtractorMOG())#CNT())# MOG())

    #Establish the map holding the timeout data
    #make an entry for each camera
    TimeOuts = {}
    for i in range(num):
        TimeOuts[str(i+1)] = {}

    #the thresholds for the targets may need some adjustment or be broken up
    #   into an array parameter so we can have a different value for each camera
    thresh = {}
    for target in targets:
        thresh[target] = 0.8

    lastFrame = [0]*num

    #nanodet is a faster model and could reasonably work for 16 cameras but it doesn't work
    # well with black and white frames which is what we get at night from the IR
#    net = get_model("nanodet", target_size=320, nms_threshold=0.5, use_gpu=False)
    #this is slower than nanodet but necessary if we are going to be using night vision images
    net = get_model("mobilenetv2_ssdlite", target_size=320, num_threads=4, use_gpu=False)

    reconnectTimeout = 15
    reconnect = [0]*num

    lastAttempt = time.time()
    deadOn = False
    
    #main loop
    while True:
        #if any of the cameras have disconnected, attempt to reconnect
        for count, curT in enumerate(t):
             #if the connection is not alive and we aren't on a reconnect timeout
             if not curT.is_alive() and time.time() > reconnect[count]:
                 #reboot the camera by replacing the camera and stream objects
                 print("Attempting to reboot camera", str(count+1))
                 ip = IPList[count]
                 c[count] = Camera(ip[0], ip[1], ip[2], profile="sub")
                 ic = callWrapper(count)
                 t[count] = c[count].open_video_stream(callback=ic.inner_callback)
                 reconnect[count] = time.time() + reconnectTimeout

        #remove any expired TimeOuts
        ClearTimeouts(TimeOuts)

        #check which images have updated
        indexesToCheck = []
        for i in range(num):
            if frameCount[i] != lastFrame[i]:
                indexesToCheck.append(i)
                lastFrame[i] = frameCount[i]

        #if it has been more than 100 seconds since a frame came in
        #  assume that this is dead and force it to reconnect to the cameras
        if len(indexesToCheck) == 0 and (time.time()-lastAttempt) > 100:
            print("100 seconds since a frame was seen, reboot all camera feeds")
            for curT in t:
                #Dummy objects will return is_alive as false forcing a reconnect attempt
                curT = Dummy()
            continue

        #if no images have updated, wait for 25ms and check again
        #   adding this decreased the CPU required by a lot
        #With 4 cameras at 10 fps waiting 0.025s should be the minimum wait
        if len(indexesToCheck) == 0:
            time.sleep(0.025)
            continue

        lastAttempt = time.time()

        #for every updated image
        for index in indexesToCheck:
            #grab and check background subtraction for motion
            curImage = camImg[index]
            #crop the borders and shrink the image for faster bgsub
            resized = cv2.resize(curImage[50:-50, 50:-50], (curImage.shape[0]//2, curImage.shape[1]//2))
            mask = bgsub[index].apply(resized)

            nonzero = np.count_nonzero(mask)
            #motion sensitivity is tune-able
            if nonzero > MotionSensitivity:
                #if motion is high enough, do object detection
                objects = net(curImage)
                #debugging information in case something goes wrong
                print(index, nonzero, datetime.datetime.fromtimestamp(time.time(), pytz.timezone("America/Los_Angeles")))
                #for each detected object check the scores against the target thresholds
                for o in objects:
                    if "class_names" in dir(net) and "label" in dir(o):
                        for target in targets:
                            #check to see if they match the object and are above the detection threshold
                            if net.class_names[int(o.label)] == target and o.prob > thresh[target]:
                                #if the camera is in timeout there is no need to report anything
                                if IsInTimeOut(TimeOuts, index, target):
                                    print("Camera", index+1, "found", target, "but is in timeout")
                                    continue

                                 #if the object is too small don't bother reporting
                                if o.rect.w*o.rect.h < MinimumObjectSize:
                                    print("Camera", index+1, "found", target, "but is too small")
                                    continue

                                 #this is a new detection of an object, report it through telegram
                                try:
                                    telegram.send_message(groupID, target + " detected on Camera" + str(index+1))
                                except Exception as e:
                                    print("Error with telegram send_message", e.description)

                                #if we are reporting pictures, send the picture
                                if pictureMode:
                                    #draw rectangle
                                    cv2.rectangle(curImage, (int(o.rect.x), int(o.rect.y)), (int(o.rect.x + o.rect.w), int(o.rect.y + o.rect.h)), [255,0,0], 3)
                                    #prep image
                                    is_success, im_buf_arr = cv2.imencode(".png", curImage)
                                    byte_im = im_buf_arr.tobytes()
                                    #send image
                                    try:
                                        telegram.send_photo(groupID, photo=byte_im)
                                    except Exception as e:
                                        telegram.send_message(groupID, "Error sending photo:" + e.description)
                                    #add timeout
                                    TimeOuts[str(index+1)][target] = time.time()+TimeoutLength
#end of runMainLoop

dirname = os.path.dirname(__file__)
setup_conf = os.path.join(dirname, 'setup.conf')

#read setup.conf and prep the data so it can be passed into runMainLoop
parser = configparser.ConfigParser()
parser.read(setup_conf)
token = parser.get("setup","TOKEN")
groupID = parser.get("setup","GROUP_ID")
IPAddresses = parser.get("setup", "IP_ADDRESS")
Usernames = parser.get("setup", "USERNAMES")
Passwords = parser.get("setup", "PASSWORDS")
telegram = telebot.TeleBot(token)

TimeoutLength = parser.get("params", "Timeout")
MotionSensitivity = parser.get("params", "Sensitivity")
MinimumObjectSize = parser.get("params", "MinimumSize")
pictureMode = parser.getboolean("params", "SendPictures")
Targets = parser.get("params", "Targets")
IPList=[]

#Need to clean the data in case the user added quotes or spaces
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

runMainLoop(IPList, pictureMode, int(TimeoutLength), int(MotionSensitivity), int(MinimumObjectSize), Targets)
