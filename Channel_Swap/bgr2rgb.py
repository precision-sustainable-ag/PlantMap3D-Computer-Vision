import cv2
import numpy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f",nargs=1)
args = parser.parse_args()
file_path = None
if args.f:
    file_path = os.path.abspath(args.f[0])
if not file_path:
    file_path = os.path.dirname(__file__)
input_path = os.path.join(file_path,"input")
if not os.path.exists(input_path):
    raise FileExistsError("\"input\" dir not found")
output_path = os.path.join(file_path,"output")
if not os.path.exists(output_path):
    os.mkdir(output_path)

files_to_convert = os.listdir(input_path)
for file in files_to_convert:
    if os.path.splitext(file)[1] in [".png",".jpeg"]:
        image = cv2.imread(os.path.join(input_path,file))
        if image is not None:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_path,file),image)
        else:
            print(f"{file} does not seem to be a valid image.")
    else:
        print(f"{file} does not seem to be a valid image.")

