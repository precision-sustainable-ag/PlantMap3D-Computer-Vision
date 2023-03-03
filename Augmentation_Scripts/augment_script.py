import cv2
import os
from pathlib import Path
import random
import albumentations as A
import numpy as np


def split(image,desired_shape):
    #Currently assumes that image shape is larger than desired_shape
    out = np.zeros((4,desired_shape[0],desired_shape[1],3))
    out[0] = image[0:desired_shape[0],0:desired_shape[1]]
    out[1] = image[0:desired_shape[0],-desired_shape[1]:]
    out[2] = image[-desired_shape[0]:,0:desired_shape[1]]
    out[3] = image[-desired_shape[0]:,-desired_shape[1]:]
    return out

def blur(image):
    out = [np.zeros(image.shape,dtype=np.uint8),""]
    process = A.GaussianBlur(p=1.0)
    out[0]=process(image=image)["image"]
    out[1]="g"
    return out


project_dir = "/Users/daniel/NCSU-Mini/PSA/CV_Photo_Augment_Proj/"
input_dir = project_dir + "input/"
output_dir = project_dir +"test_run/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

photo = cv2.imread(input_dir+"00000.jpg")
photo = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
out_imgs = split(photo,(1024,1024))
counter = 0
letters = ["a","b","c","d"]
for i in out_imgs:
    [img,code] = blur(i)
    cv2.imwrite(output_dir+"00001"+code+letters[counter]+".jpg",cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
    counter += 1