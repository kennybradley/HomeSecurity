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

#any reolink camera, as many as you want, will run slower after 4
#https://www.pishop.us/product/sata-hard-drive-to-usb-adapter/
#https://www.pishop.us/product/raspberry-pi-4-model-b-2gb/
#any ssd
#16+GB microSD

#prompt password
#prompt telegram info


##connect SSD
#sudo mkfs -t ext4 /dev/sda
#sudo mkdir -p /mnt/ftpServer
#sudo mount -t auto /dev/sda /mnt/ftpServer/
#sudo chmod 777 /mnt/ftpServer
#sudo cp /etc/fstab /etc/fstab.backup

##set the hdd to mount on reboot
#UUID=`lsblk -o NAME,UUID | awk '{split($0,a,"sda"); print a[2]}' | xargs`
#echo "UUID="$UUID" /mnt/ftpServer ext4 0 0 0" > append.txt
#sudo cat /etc/fstab append.txt > /etc/fstab2
#sudo mv /etc/fstab2 /etc/fstab


##get ftp
#sudo apt-get install vsftpd

##configure ftp
#echo -e '\nwrite_enable=YES\nlocal_umask=022\nchroot_local_user=YES\nuser_sub_token='$USER'\nlocal_root=/mnt/ftpServer' >> append.txt
#sudo cat /etc/vsftpd.conf append.txt > vsftpd.conf
#sudo mv vsftpd.conf /etc/vsftpd.conf

#sudo service vsftpd restart
#pip install pyTelegramBotAPI
#pip install ncnn
#pip install reolinkapi
#pip install opencv-python

#sudo apt-get install telegram-cli

#need to write out token.txt
#need to write out group.txt

#need to auto reboot the script if it fails


tokenFile = open("token.txt", "r").read()[:-1]
groupID = open("group.txt", "r").read()[:-1]
telegram = telebot.TeleBot(tokenFile)

#print(tokenFile, groupID)

TimeoutLength = 10
MinimumBackgroundDiff = 2000

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
def non_blocking(IPList, pictureMode=True):
    #populate main variables
    num = len(IPList)
    camImg = [None]*num
    frameCount = [0]*num

    #need a class so the object can hold the callback function
    class callWrapper:
        def __init__(self, id)
            self.id = id

        def inner_callback(self, img):
            nonlocal camImg, frameCount
            frameCount[self.id] += 1
            camImg[self.id] = img

    t = []
    c = []
    bgsub = []
    for count, ip in enumerate(IPList):
        c.append(Camera(ip[0], ip[1], ip[2], profile="sub"))
        ic = callWrapper(count)
        t.append(c[count].open_video_stream(callback=ic.inner_callback))
        bgsub.append(cv2.createBackgroundSubtractorMOG2())

    TimeOuts = {}
    for i in range(num):
        TimeOuts[str(i+1)] = {}

    targets = ["person", "cat", "dog", "cake"]
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
            if np.count_nonzero(mask) > MinimumBackgroundDiff:
                objects = net(curImage)
                for o in objects:
                    if "class_names" in dir(net) and "label" in dir(o):
                        for target in targets:
                            if net.class_names[int(o.label)] == target and o.prob > thresh[target]:
                                 if IsInTimeOut(TimeOuts, index, target):
                                     print("Camera", index+1, "found", target, "but is in timeout")
                                     continue

                                 telegram.send_message(groupID, target + " detected on Camera" + str(index+1))

                                 if pictureMode:
                                     #draw rectangle
                                     cv2.rectangle(curImage, (int(o.rect.x), int(o.rect.y)), (int(o.rect.x + o.rect.w), int>
                                     #prep image
                                     is_success, im_buf_arr = cv2.imencode(".png", curImage)
                                     byte_im = im_buf_arr.tobytes()
                                     #send image
                                     telegram.send_photo(groupID, photo=byte_im)
                                     #add timeout
                                     TimeOuts[str(index+1)][target] = time.time()+TimeoutLength

def non_blocking_single(IP, user, password):
    non_blocking([[IP, user, password]])
