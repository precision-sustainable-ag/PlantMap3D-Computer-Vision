import time
import depthai as dai
import numpy as np
import os
import argparse
import pathlib
import cv2
import matplotlib.pyplot as plt

class Establish_Pipeline:
    def __init__(self,usb,ip,shape,current_path):
        self.communication_type = usb
        self.script_path = current_path

        self.nn_shape = shape

        self.nn_path = pathlib.Path(self.script_path + "/input/models/small2021_4_8shaves_only_dt_and_is.blob")

        self.pipeline = dai.Pipeline()

        self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)
        
        print("Creating color camera node...")
        self.RGB_Node = self.pipeline.createColorCamera()
        self.RGB_Node.setPreviewSize(self.nn_shape[0], self.nn_shape[1])
        self.RGB_Node.setInterleaved(False)
        self.RGB_Node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.RGB_Node.setBoardSocket(dai.CameraBoardSocket.RGB)

        self.RGB_Out=self.pipeline.create(dai.node.XLinkOut)
        self.RGB_Out.setStreamName("rgb")
        self.RGB_Out.input.setBlocking(False)
        self.RGB_Out.input.setQueueSize(1)
        self.RGB_Node.preview.link(self.RGB_Out.input)
        print("Done")

        print("Creating neural network node...")
        self.nn_node = self.pipeline.create(dai.node.NeuralNetwork)
        self.nn_node.setBlobPath(self.nn_path)
        self.nn_node.input.setQueueSize(1)
        self.nn_node.input.setBlocking(False)

        self.RGB_Node.preview.link(self.nn_node.input)

        self.seg_out = self.pipeline.create(dai.node.XLinkOut)
        self.seg_out.setStreamName("seg out")
        self.nn_node.out.link(self.seg_out.input)
        print("Done")

        print("Creating depth channel...")
        self.monoL = self.pipeline.create(dai.node.MonoCamera)
        self.monoR = self.pipeline.create(dai.node.MonoCamera)
        self.depth = self.pipeline.create(dai.node.StereoDepth)

        self.depth_out = self.pipeline.create(dai.node.XLinkOut)
        self.depth_out.setStreamName("depth")
        self.monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        self.monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)

        self.monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        self.monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.monoL.out.link(self.depth.left)
        self.monoR.out.link(self.depth.right)
        self.depth.depth.link(self.depth_out.input)
        print("Done")

        if not self.communication_type:
            self.ip = ip
            self.cam = dai.DeviceInfo(ip)
            with dai.Device(self.pipeline,self.cam) as self.device:
                self.enable_camera()
        else:
            with dai.Device(self.pipeline) as self.device:
                self.enable_camera()


    def enable_camera(self):
        depth_out = self.device.getOutputQueue("depth",1,False)
        segmentation_queue = self.device.getOutputQueue("seg out",1,False)
        rgb_queue = self.device.getOutputQueue("rgb",1,False)
        while True:
            segmentation_labels = segmentation_queue.get()
        
            segmentation_labels = np.array(segmentation_labels.getFirstLayerInt32()).reshape(self.nn_shape[0],self.nn_shape[1])
        
            self.out = [None,None,None]
            rgb_data = rgb_queue.get()
            rgb_img = np.array(rgb_data.getCvFrame())
            depth_data = depth_out.get()
            depth_img = np.array(depth_data.getCvFrame())
            depth_img = (depth_img * (255/self.depth.initialConfig.getMaxDisparity())).astype(np.uint8)
            self.out[0] = rgb_img #shape is (height, width, 3)
            self.out[1] = segmentation_labels #shape is (height, width, 1)
            self.out[2] = depth_img #shape is default

            #Output for Testing
            cv2.imshow("RGB",rgb_img)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    
    is_usb_cam = True

    path = str(pathlib.Path(__file__).parent.resolve())
    neural_network_shape = (512,512)
    seg_pipeline = Establish_Pipeline(is_usb_cam,"169.254.54.205",neural_network_shape,path)
    seg_pipeline.enable_camera()