import cv2
import numpy as np
import math
import time

def trape_wave(pixel,coords,pixel_row,buffer,min,phase):
  out = np.zeros((3),dtype=np.uint8)
  in_row_pos = ((coords[1]+phase) % pixel_row) + 1
  if (in_row_pos > (buffer/2)) and (in_row_pos < (pixel_row/2-(buffer/2))):
    out = pixel
  elif (in_row_pos-(pixel_row/2) > (buffer/2)) and (in_row_pos < (pixel_row-(buffer/2))):
    out = np.add(pixel,pixel*min)
  else:
    if (in_row_pos<=buffer):
      out = np.rint((pixel*(float((in_row_pos)/(buffer/2))))).astype(np.uint8)
    elif (((in_row_pos) <= ((pixel_row/2)+buffer
                            )) and ((in_row_pos) >= ((pixel_row/2)-buffer))):
      slope = float(min/buffer)
      out = np.rint(np.add(pixel,pixel*(float(slope*(in_row_pos-((pixel_row/2)-buffer)))))).astype(np.uint8)
      holder = 5+5
    else:
      out = np.rint((pixel*(float((((in_row_pos-1)-pixel_row)+buffer/2)/buffer)))).astype(np.uint8)
  return out


def conversion(image):
  out = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
  row_width = 200
  buffer = 10
  for i in np.ndindex(image.shape[:2]):
    multiplier = abs(math.cos((1/row_width)*2*math.pi*i[1]))
    factor_sub = 0.67*image[i]*multiplier
    out[i] = trape_wave(image[i],i,row_width,buffer,-0.67,0)
  return out

in_im = cv2.imread("/Users/daniel/NCSU-Mini/PSA/CV_Photo_Augment_Proj/input/00000.jpg")
in_im = cv2.cvtColor(in_im,cv2.COLOR_BGR2RGB)
in_im = cv2.resize(in_im,(512,512))
#in_im = in_im[512:1024,512:1024]
out_im = conversion(in_im)
cv2.imshow("Augmented",cv2.cvtColor(out_im,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)