import time
import depthai as dai
import numpy as np
import os
import pathlib
import cv2
import matplotlib.pyplot as plt


#This section detects the directory that encapsulates the script.
script_path = str(pathlib.Path(__file__).parent.resolve())
#The layout of folders is assumed to be preconfigured relative to the current parent directory. 

#This function came from the DepthAI experiments GitHub. 
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

nn_shape = (1024,1024)

#Blob path is manually changed to whatever model is being tested. This is arbitrary and temporary.
nn_path = pathlib.Path(script_path + "/input/models/large1024_2021_4_8shaves_only_dt_and_is.blob")

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
print("Creating color channel...")
#instream = pipeline.create(dai.node.XLinkIn)
#instream.setStreamName("input")
#instream.out.link(node.input)
colorCam = pipeline.createColorCamera()
colorCam.setPreviewSize(nn_shape[0], nn_shape[1])
colorCam.setInterleaved(False)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.preview.link(node.input)
CamOut=pipeline.create(dai.node.XLinkOut)
CamOut.setStreamName("rgb")
colorCam.preview.ling(CamOut.input)
print("Done")
print("Creating depth channel...")
monoL = pipeline.create(dai.node.MonoCamera)
monoR = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
dout = pipeline.create(dai.node.XLinkOut)
dout.setStreamName("depth")
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
monoL.out.link(depth.left)
monoR.out.link(depth.right)
depth.disparity.link(dout.input)
print("Done")

#Intentionally overwrites the stats file. Copy it if you want to save a certain run's data
output_stats = open((photos_out_dir+"stats.csv"),"w")

with dai.Device(pipeline) as device:
    depth_out = device.getOutputQueue("depth",1,False)
    nn_out = device.getOutputQueue("output",1,False)
    cam_out = device.getOutputQueue("rgb",1,False)
    while True:
    	nn_data = dai.ImgFrame()
    	while nn_data is None:
    		nn_data = nn_out.get()
    		time.sleep(0.1)
    	
    	nn_data = np.array(nn_data.getFirstLayerInt32()).reshape(nn_shape[0],nn_shape[1])
    	soil = np.array([40, 86, 166])
    	clover = np.array([28, 26, 228])
    	broadleaf = np.array([184, 126, 155])
    	grass = np.array([0, 127, 255])
    
    	nn_img = np.zeros((nn_shape[0],nn_shape[1],3),dtype=np.int32)
    	nn_img[nn_data==0] = soil
    	nn_img[nn_data==1] = clover
    	nn_img[nn_data==2] = broadleaf
    	nn_img[nn_data==3] = grass
    	
    	out = np.zeros((3,nn_shape[0],nn_shape[1],3]),dtype=np.int32)
    	cam_data = cam_out.get()
    	cam_img = np.array(cam_data.getFirstLayerInt32()).reshape(nn_shape[0],nn_shape[1])
    	depth_data = depth_out.get()
    	depth_img = np.array(depth_data.getFirstLayerInt32()).reshape(nn_shape[0],nn_shape[1])
    	#layers = data_o.getAllLayers()
    	out[0] = cam_img
    	out[1] = nn_img
    	out[2] = depth_img
    	
    	#Output for Testing
    	cv2.imshow("RGB",cam_img)
    	cv2.imshow("Segmentation",nn_img)
    	cv2.imshow("Depth",depth_img)
    	
        
        



