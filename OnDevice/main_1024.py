import time
import depthai as dai
import numpy as np
import os
import pathlib
import cv2
import pickle
import matplotlib.pyplot as plt

typemod = "large1024"

#This section detects the directory that encapsulates the script.
script_path = str(pathlib.Path(__file__).parent.resolve())
#The layout of folders is assumed to be preconfigured relative to the current parent directory. 
photo_in_dir = script_path + "/input/validation_safe/images/"
print(script_path,photo_in_dir)
photos_in = os.listdir(photo_in_dir)
photos_in.sort()
photos_out_dir = script_path+"/input/validation_safe/"+typemod+"/"
label_dir = script_path+"/input/validation_safe/labels/"

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

nn_shape = 1024

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
print("Creating input channel...")
instream = pipeline.create(dai.node.XLinkIn)
instream.setStreamName("input")
instream.out.link(node.input)
print("Done")


predictions = []
stats = []

#Intentionally overwrites the stats file. Copy it if you want to save a certain run's data
output_stats = open((photos_out_dir+"stats.csv"),"w")

with dai.Device(pipeline) as device:
    dev_in = device.getInputQueue("input")
    dev_out = device.getOutputQueue("output",1,False)
    for photo in photos_in:
        if not (".jpg" in photo):
            continue
        start_time = time.time()
        path = (photo_in_dir+photo)
        data = cv2.imread(path)
        frame = dai.ImgFrame()
        frame.setData(to_planar(data,(nn_shape,nn_shape)))
        file_num = out_path(photo)
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setWidth(nn_shape)
        frame.setHeight(nn_shape)
        #cv2.imshow("input",(frame.getCvFrame()))
        dev_in.send(frame)
        data_o = dev_out.get()
        #layers = data_o.getAllLayers()
        lay1 = np.array(data_o.getFirstLayerInt32()).reshape(nn_shape,nn_shape)
        #print(lay1.shape)
        #print(lay1)
        #if data_o is not None:
        #    data_out = dai.ImgFrame()
        #    data_out.setData(to_planar(data_o,(512,512)))
        #    data_out.setWidth(512)
        #    data_out.setHeight(512)
        #    frame_out = data_out.getCvFrame()
        #    cv2.imshow("Output",frame_out)
        end_time = time.time()
        time.sleep(0.5)
        time_taken = end_time - start_time
        print("Time taken to process {:s}: {:.4f} seconds".format(photo,time_taken))
        stats.append(time_taken)
        #data.release()
        predictions.append(lay1)
        #cv2.imwrite(, lay1)
        #wierd issue with for loop not exiting properly, so forced a break
        #also allowed me to set it to stop prematurely to speed testing
        if photo == "0038.jpg":
            break

def get_jaccard_score(ground_truth, predicted, num_classes=4):
    #sklearn is deprecated, but scikit-learn seems to include it well
    from sklearn.metrics import jaccard_score

    scores = []
    for i in range(num_classes):
        class_gt = (ground_truth == i).astype(int)
        class_pred = (predicted == i).astype(int)
        score = jaccard_score(class_gt.ravel(), class_pred.ravel(), average='macro')
        scores.append(score)

    # print("Jaccard similarity scores for each class: ", scores)
    # print("Mean Jaccard Score: ",sum(scores)/4)
    return sum(scores)/4

labels = os.listdir(label_dir)
labels = list(filter(lambda item: ".png" in item, labels))
labels.sort()

def get_ground_truth(label_dir, labels, i):
    label = cv2.imread(label_dir + labels[i])
    if label.shape[1] != nn_shape:
        label = cv2.resize(label,(nn_shape,nn_shape), interpolation = cv2.INTER_AREA)

    # the three channels
    soil = np.array([40, 86, 166])
    clover = np.array([28, 26, 228])
    broadleaf = np.array([184, 126, 155])
    grass = np.array([0, 127, 255])

    img = label

    label_seg = np.zeros((img.shape[:2]), dtype=np.int32) #No numpy.int assumed int32
    label_seg[(img==soil).all(axis=2)] = 0
    label_seg[(img==clover).all(axis=2)] = 1
    label_seg[(img==broadleaf).all(axis=2)] = 2
    label_seg[(img==grass).all(axis=2)] = 3

    return label_seg

scores = []

def save_out(pred, i):
    output_file_name = (photos_out_dir + (str(i+1).zfill(4))+".png")
    soil = np.array([40, 86, 166])
    clover = np.array([28, 26, 228])
    broadleaf = np.array([184, 126, 155])
    grass = np.array([0, 127, 255])
    
    img = np.zeros((nn_shape,nn_shape,3),dtype=np.int32)
    img[pred==0] = soil
    img[pred==1] = clover
    img[pred==2] = broadleaf
    img[pred==3] = grass
    #print(img.shape)
    #img[(pred==0).all(axis=1)] = soil
    #img[(pred==1).all(axis=1)] = clover
    #img[(pred==2).all(axis=1)] = broadleaf
    #img[(pred==3).all(axis=1)] = grass
    
    cv2.imwrite(output_file_name, img)

for i in range(len(predictions)):
    ground_truth = get_ground_truth(label_dir, labels, i)
    predicted = predictions[i]
    #Put score into var to save individual scores to CSV
    holder = get_jaccard_score(ground_truth,predicted)
    scores.append(holder)
    save_out(predicted,i)
    output_stats.write(str(i+1).zfill(4)+","+str(stats[i])+","+str(holder)+",")
output_stats.close()
print("Jaccard Score on the validation dataset is :", sum(scores)/len(scores))
        

