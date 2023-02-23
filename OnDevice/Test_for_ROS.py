import time
import depthai as dai
import numpy as np
import os
import argparse
import pathlib
import cv2
import matplotlib.pyplot as plt

#Added an CLI argument to allow switching between USB and POE testing
parser = argparse.ArgumentParser(
    prog = "Test for ROS",
    description= "Provides interface with camera pipeline and generates 3 numpy arrays")

parser.add_argument("-u", "-usb",action="store_true")
is_usb_cam = parser.parse_args()

def enable_camera(device,shape):
    depth_out = device.getOutputQueue("depth",1,False)
    segmentation_queue = device.getOutputQueue("seg out",1,False)
    rgb_queue = device.getOutputQueue("rgb",1,False)
    while True:
        segmentation_labels = segmentation_queue.get()
        
        segmentation_labels = np.array(segmentation_labels.getFirstLayerInt32()).reshape(shape[0],shape[1])
        
        out = [None,None,None]
        rgb_data = rgb_queue.get()
        rgb_img = np.array(rgb_data.getCvFrame())
        depth_data = depth_out.get()
        depth_img = np.array(depth_data.getCvFrame())
        depth_img = (depth_img * (255/depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        #layers = data_o.getAllLayers()
        out[0] = rgb_img #shape is (height, width, 3)
        out[1] = segmentation_labels #shape is (height, width, 1)
        out[2] = depth_img #shape is default

        #Output for Testing
        cv2.imshow("RGB",rgb_img)

        if cv2.waitKey(1) == ord('q'):
            break
	
        # cv2.imshow("Segmentation",nn_img)
        # cv2.imshow("Depth",depth_img)
        # time.sleep(5)

#This section detects the directory that encapsulates the script.
script_path = str(pathlib.Path(__file__).parent.resolve())
#The layout of folders is assumed to be preconfigured relative to the current parent directory. 

nn_shape = (512,512)

#Blob path is manually changed to whatever model is being tested. This is arbitrary and temporary.
nn_path = pathlib.Path(script_path + "/input/models/small2021_4_8shaves_only_dt_and_is.blob")

pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

print("Creating color channel...")
#instream = pipeline.create(dai.node.XLinkIn)
#instream.setStreamName("input")
#instream.out.link(node.input)
RGB_Node = pipeline.createColorCamera()
RGB_Node.setPreviewSize(nn_shape[0], nn_shape[1])
RGB_Node.setInterleaved(False)
RGB_Node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
RGB_Node.setBoardSocket(dai.CameraBoardSocket.RGB)

RGB_Out=pipeline.create(dai.node.XLinkOut)
RGB_Out.setStreamName("rgb")
RGB_Out.input.setBlocking(False)
RGB_Out.input.setQueueSize(1)
RGB_Node.preview.link(RGB_Out.input)
print("Done")

print("Creating neural network node...")
nn_node = pipeline.create(dai.node.NeuralNetwork)
nn_node.setBlobPath(nn_path)
nn_node.input.setQueueSize(1)
nn_node.input.setBlocking(False)

RGB_Node.preview.link(nn_node.input)

seg_out = pipeline.create(dai.node.XLinkOut)
seg_out.setStreamName("seg out")
nn_node.out.link(seg_out.input)
print("Done")

print("Creating depth channel...")
monoL = pipeline.create(dai.node.MonoCamera)
monoR = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

depth_out = pipeline.create(dai.node.XLinkOut)
depth_out.setStreamName("depth")
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
monoL.out.link(depth.left)
monoR.out.link(depth.right)
depth.depth.link(depth_out.input)
print("Done")

ip = "169.254.54.205"
cam = dai.DeviceInfo(ip)
if is_usb_cam:
    with dai.Device(pipeline) as device:
        enable_camera(device,nn_shape)
else:
    with dai.Device(pipeline,cam) as device:
        enable_camera(device,nn_shape)
        
        