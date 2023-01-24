import time
import depthai as dai
import numpy as np
import os
import pathlib
import cv2

#This section detects the directory that encapsulates the script.
script_path = str(pathlib.Path(__file__).parent.resolve())
#The layout of folders is assumed to be preconfigured relative to the current parent directory. 
photo_in_dir = script_path + "/input/validation_safe/images/"
print(script_path,photo_in_dir)
photos_in = os.listdir(photo_in_dir)
photos_in.sort()
photos_out_dir = script_path+"/input/validation_safe/test/"

def out_path(name):
    name_split = name.split(".")
    return name_split[0]

try:
    os.mkdir(photos_out_dir)
except OSError as error:
    if "exist" in str(error):
        pass
    else:
        print(error)

#This function came from the DepthAI experiments GitHub. 
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

nn_shape = 512

#Blob path is manually changed to whatever model is being tested. This is arbitrary and temporary.
nn_path = pathlib.Path(script_path + "/input/models/small2021_4_8shaves_only_dt_and_is.blob")

pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

print("Creating neural network node...")
node = pipeline.create(dai.node.NeuralNetwork)
node.setBlobPath(nn_path)
node.input.setQueueSize(1)
node.input.setBlocking(False)
out = pipeline.create(dai.node.XLinkOut)
out.setStreamName("output")
node.out.link(out.input)
print("Done")
print("Creating input channel...")
instream = pipeline.create(dai.node.XLinkIn)
instream.setStreamName("input")
instream.out.link(node.input)
print("Done")


with dai.Device(pipeline) as device:
    dev_in = device.getInputQueue("input")
    dev_out = device.getOutputQueue("output",1,False)
    for photo in photos_in:
        path = (photo_in_dir+photo)
        data = cv2.imread(path)
        frame = dai.ImgFrame()
        frame.setData(to_planar(data,(512,512)))
        file_num = out_path(photo)
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setWidth(512)
        frame.setHeight(512)
        cv2.imshow("input",(frame.getCvFrame()))
        dev_in.send(frame)
        data_o = dev_out.get()
        print(data_o.getData())
        #if data_o is not None:
        #    data_out = dai.ImgFrame()
        #    data_out.setData(to_planar(data_o,(512,512)))
        #    data_out.setWidth(512)
        #    data_out.setHeight(512)
        #    frame_out = data_out.getCvFrame()
        #    cv2.imshow("Output",frame_out)
        time.sleep(0.5)
        data.release()
        





        
        
