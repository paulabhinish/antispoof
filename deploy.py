
import os
import cv2
import numpy as np
import argparse
import warnings
import time
from random import randint, randrange

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

import pytesseract

pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#check_number= randrange(100, 1000)
check_number=420
tstepauthetication=False

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True
def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    #image = cv2.imread(image_name)
    result = check_image(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    #thresh = cv2.adaptiveThreshold(cvuint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    thresh=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    textpytessetract = pytesseract.image_to_string(blurred,config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789')

    print("input text "+ textpytessetract)
    
    if tstepauthetication:
            color = (0, 0, 255)
            cv2.putText(image,"write "+str(check_number)+" and show to camera",(10, 10),
            cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)
            if (label == 1 and (textpytessetract.strip()==int(check_number))):
                print("*************************** here ")
                print("Image '{}' is Real Face. Score: {:.2f}.")
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (0, 255, 0)
                cv2.putText(
                image,
                "Text found on the image is "+textpytessetract,
                (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)
                cv2.putText(
                image,
                "VERIFIED",
                (400, 400),
                cv2.FONT_HERSHEY_COMPLEX, 10*image.shape[0]/1024, color)
                
            else:
                print("Image '{}' is Fake Face. Score: {:.2f}.") 
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
                print("Prediction cost {:.2f} s".format(test_speed))
                print("result_text")

            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)
        
        
         
    else:
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.")
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (0, 255, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.")
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        print("result_text")

        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)

    #format_ = os.path.splitext(image_name)[-1]
    #result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imshow("webcamtested" , image)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!
while True: 
    
    # Capture frame-by-frame 
    ret, frame1 = cap.read()
    frame = frame1[0:480, 140:500]
    #cv2.imshow('WebCam', frame) 
    test(frame,"./resources/anti_spoof_models",0)
      
    # wait for the key and come out of the loop 
    if cv2.waitKey(1) == ord('q'): 
        break
# Discussed below 
cap.release() 
cv2.destroyAllWindows() 
