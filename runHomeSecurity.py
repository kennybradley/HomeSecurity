import numpy as np
import time
import cv2
import telebot
from reolinkapi import Camera
import configparser
import datetime
import pytz
import random
import traceback
from ncnn.utils.objects import Detect_Object, Rect
from ultralytics import YOLO

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
    def is_alive(self):
      print(time.time(), " forcing a reconnection")
      return False

class YOLO11NCNNWrapper:
    def __init__(self, model_folder="yolo11n_person_ncnn_model", class_names=None):
        import os
        import numpy as np
        import torch

        def is_ncnn_compatible(folder):
            param_path = os.path.join(folder, "model.ncnn.param")
            if not os.path.exists(param_path):
                return False
            try:
                with open(param_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "MatMul" in content:
                    print(f"Bypassing NCNN model '{folder}': contains unsupported 'MatMul' layer.")
                    return False
                return True
            except Exception:
                return False

        self.model = None
        # Attempt loading NCNN model folders if available and verified compatible
        for target in [model_folder, "yolo11n_ncnn_model"]:
            if os.path.exists(target) and is_ncnn_compatible(target):
                try:
                    candidate = YOLO(target, task="detect")
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = candidate(dummy, imgsz=640, verbose=False)
                    self.model = candidate
                    print(f"Loaded NCNN FP16 model from '{target}'")
                    break
                except Exception as e:
                    print(f"NCNN inference failed on '{target}' ({e}).")

        if self.model is None:
            print("Using optimized PyTorch INT8 model ('yolo11n.pt')...")
            base_model = YOLO("yolo11n.pt")
            try:
                base_model.model = torch.quantization.quantize_dynamic(
                    base_model.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Applied PyTorch dynamic INT8 quantization for CPU performance.")
            except Exception as q_err:
                print(f"Dynamic INT8 quantization skipped ({q_err}). Using standard PyTorch model.")
            self.model = base_model

        self.class_names = class_names if class_names is not None else self.model.names

    def __call__(self, img):
        results = self.model(img, imgsz=640, verbose=False)[0]
        
        objects = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            prob = float(box.conf[0].item())
            
            x_center, y_center, w, h = box.xywh[0].tolist()
            x = x_center - (w / 2.0)
            y = y_center - (h / 2.0)
            
            if isinstance(self.class_names, dict):
                label_name = self.class_names.get(cls_id, str(cls_id))
            elif isinstance(self.class_names, (list, tuple)):
                label_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            else:
                label_name = str(cls_id)

            obj = Detect_Object()
            obj.label = label_name
            obj.prob = prob
            obj.rect = Rect(x, y, w, h)

            objects.append(obj)
            
        return objects

def process_objects(objects, targets, thresh, MinimumObjectSize, TimeOuts, index, curImage):
    send = False
    found = ""
    score = 0
    for o in objects:
        if type(o) != Detect_Object:
            continue

        for target in targets:
            #check to see if they match the object and are above the detection threshold
            if o.label == target and o.prob > thresh[target]:

                #if the camera is in timeout there is no need to report anything
                if IsInTimeOut(TimeOuts, index, target):
                    print(f"Camera {index+1} found {target} (prob={o.prob:.2f}) but is in timeout", flush=True)
                    continue

                #if the object is too small don't bother reporting
                if o.rect.w*o.rect.h < MinimumObjectSize:
                    print(f"Camera {index+1} found {target} (prob={o.prob:.2f}) but object area ({int(o.rect.w*o.rect.h)} px) is smaller than minimum size ({MinimumObjectSize} px)", flush=True)
                    continue

                send = True
                found = target
                if o.prob > score:
                    score = o.prob
                if pictureMode:
                    # the coordinates are handled in the original coordinate space 
                    print(f"[Detection Match] Camera {index+1}: {o.label} (prob={o.prob:.2f}) at ({int(o.rect.x)}, {int(o.rect.y)}) to ({int(o.rect.x + o.rect.w)}, {int(o.rect.y + o.rect.h)})", flush=True)
                    cv2.rectangle(curImage, (int(o.rect.x), int(o.rect.y)), (int((o.rect.x + o.rect.w)), int((o.rect.y + o.rect.h))), [0,0,255], 3)

    return send, found, score

def sendAlert(curImage, groupID, found, index, score, TimeOuts, TimeoutLength):
    print(f"[sendAlert] Sending alert to Telegram for Camera {index+1}: '{found}' (score={score:.3f})...", flush=True)
    #if we are reporting pictures, send the picture
    try:
        telegram.send_message(groupID, found  + " found on camera " + str(index+1) + " with prob " + str(score)[:5])
    except Exception as e:
        print(f"[sendAlert] Failure sending text on Camera {index+1}: {type(e).__name__}: {e}", flush=True)

    if pictureMode:
        if curImage is None:
            print(f"[sendAlert] Error on Camera {index+1}: curImage is None, cannot encode or send photo.", flush=True)
            return

        #prep image
        is_success, im_buf_arr = cv2.imencode(".jpg", curImage, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not is_success or im_buf_arr is None:
            shape_info = curImage.shape if hasattr(curImage, "shape") else "unknown"
            print(f"[sendAlert] Failed to encode image to JPG on Camera {index+1} (image shape: {shape_info})", flush=True)
            try:
                telegram.send_message(groupID, f"Error encoding photo on camera {index+1}")
            except Exception as e:
                print(f"[sendAlert] Failure sending encode error notification: {e}", flush=True)
            return

        byte_im = im_buf_arr.tobytes()
        img_size_kb = len(byte_im) / 1024.0

        #send image
        try:
            telegram.send_photo(groupID, photo=byte_im)
            TimeOuts[str(index+1)][found] = time.time()+TimeoutLength
            print(f"[sendAlert] Successfully sent photo for Camera {index+1} ({img_size_kb:.1f} KB)", flush=True)
        except Exception as e:
            error_details = str(e)
            if hasattr(e, "description") and e.description:
                error_details = f"{e.description} (code: {getattr(e, 'error_code', 'N/A')})"
            elif hasattr(e, "result_json") and e.result_json:
                error_details = str(e.result_json)

            shape_info = curImage.shape if hasattr(curImage, "shape") else "unknown"
            print(f"[sendAlert] Error sending photo on Camera {index+1} ({img_size_kb:.1f} KB, shape: {shape_info}): {type(e).__name__}: {error_details}", flush=True)
            traceback.print_exc()

            try:
                telegram.send_message(groupID, f"Error sending photo on camera {index+1}: {error_details}")
            except Exception as e2:
                print(f"[sendAlert] Message send failed when reporting photo error: {type(e2).__name__}: {e2}", flush=True)

#this is the main processing loop for the image detection pipeline
def runMainLoop(IPList: list, 
                pictureMode: bool, 
                TimeoutLength: int, 
                MotionSensitivity: list, 
                MinimumObjectSize:int, 
                targets: list):
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
        bgsub.append(cv2.bgsegm.createBackgroundSubtractorCNT(20, True, 1000))

    #Establish the map holding the timeout data
    #make an entry for each camera
    TimeOuts = {}
    for i in range(num):
        TimeOuts[str(i+1)] = {}

    #the thresholds for the targets may need some adjustment or be broken up
    #   into an array parameter so we can have a different value for each camera
    thresh = {}
    for target in targets:
        thresh[target] = 0.7

    lastFrame = [0]*num

    #nanodet is a faster model and could reasonably work for 16 cameras but it doesn't work
    # with black and white frames which is what we get at night from the IR
#    net = get_model("nanodet", target_size=320, nms_threshold=0.5, use_gpu=False)
    #this is slower than nanodet but necessary if we are going to be using night vision images
#    net = get_model("mobilenetv2_ssdlite", target_size=320, num_threads=4, use_gpu=False)
#    net = get_model("yolov7_tiny", num_threads=4, use_gpu=False, use_strides=[16,32])
    net = YOLO11NCNNWrapper(model_folder="yolo11n_person_ncnn_model")
    #second order classifier to avoid FP
#    net2 = get_model("mobilenetv2_ssdlite", target_size=320, num_threads=4, use_gpu=False)

    reconnectTimeout = 15
    reconnect = [0]*num

    lastAttempt = time.time()
    deadOn = False

    last_nonzero = [0]*num
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
            for curT in range(len(t)):
                #Dummy objects will return is_alive as false forcing a reconnect attempt
                t[curT] = Dummy()
            continue

        #if no images have updated, wait for 25ms and check again
        #   adding this decreased the CPU required by a lot
        #With 4 cameras at 10 fps waiting 0.025s should be the minimum wait
        if len(indexesToCheck) == 0:
            time.sleep(0.05)
            continue

        lastAttempt = time.time()

        #this may be sped up by putting all of the indexes in a batch and processing at once
        detect_on_index = []


        # this loop handles background subtraction and determines if we should send the image to the model
        for index in indexesToCheck:
            #grab and check background subtraction for motion
            curImage = camImg[index]
            #crop the borders and shrink the image for faster bgsub
            if index == 0:
                resized = cv2.resize(curImage[80:, 35:-50], (0, 0), fx=0.5, fy=0.5)
            elif index == 1:
                resized = cv2.resize(curImage[180:, :400], (0, 0), fx=0.5, fy=0.5)
            else: #special case to avoid trees, I could add a custom box to the configuration
                resized = cv2.resize(curImage[50:-50, 50:-50], (0, 0), fx=0.5, fy=0.5)

#            from PIL import Image
#            im = Image.fromarray(resized)
#            im.save("temp" + str(index) + ".png")

            resized2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            mask = bgsub[index].apply(resized2)
            nonzero = np.count_nonzero(mask)

            # During intialization the background subtraction is still being established, don't detect objects
            if frameCount[index] < 100:
                continue

            # if motion is high enough, do object detection
            if nonzero > int(MotionSensitivity[index]) and nonzero > 150 + last_nonzero[index]:
                detect_on_index.append(index)
                print(index, nonzero, IPList[index], last_nonzero[index])

                if index == 0:
                    # skip the far left and right of the images, avoid the street
                    cropped = curImage[80:, 35:-50, :].copy()
                elif index == 1: #special case to avoid trees, I could add a custom box to the configuration
                    cropped = curImage[180:, :400, :].copy()
                else: 
                    cropped = curImage[50:-50, 50:-50,:].copy()

                start = time.time()
                objects = net(cropped)
                stop = time.time()
                print(stop-start, " seconds")

                send, target, score = process_objects(objects, targets, thresh, MinimumObjectSize, TimeOuts, index, cropped)

                if send:
                    sendAlert(cropped, groupID, target, index, score, TimeOuts, TimeoutLength)

            last_nonzero[index] = nonzero
#end of runMainLoop


#read setup.conf and prep the data so it can be passed into runMainLoop
parser = configparser.ConfigParser()
parser.read("setup.conf")
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
MotionSensitivity = prepArray(MotionSensitivity)

#restructure the IP/user/pass
for IP, user, password in zip(IPAddresses, Usernames, Passwords):
    IPList.append([IP, user, password])


runMainLoop(IPList, pictureMode, int(TimeoutLength), MotionSensitivity, int(MinimumObjectSize), Targets)
