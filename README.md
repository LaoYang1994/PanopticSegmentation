# PanopticSegmentation
This repo provides the code to combine the semantic and instance segmentation predictions for panoptic segmentation two-channel-pngs output. Compared with the [official version](https://github.com/cocodataset/panopticapi), it tries to solve occlusion problems and takes some object relationships into account, according to the method of the **third place winner** in the COCO2018 panoptic segmentation competition. A detailed description can be found [here](http://presentations.cocodataset.org/ECCV18/COCO18-Panoptic-PKU_360.pdf). 

## Requirements
+ Anaconda3 (we highly recommend！)
+ pycocotools

## File Organization
I advise you to organize you file in the following ways.

```
PanopticSegmentation
├─files
  ├─annotations
  |─detresults
```

You can form the file organization by the tools **generate_file_organization.py**

```
cd 'this repo'
python tools/generate_file_organization.py
```
  
+ annotations: store annotation files like panoptic_val2017.json and instances_val2017.json
+ detresults: store instance segmentation result which is stored in a json file
+ images: store the images
+ models: store the semantic model. In **cal_panoptic.py**, we use a random matrix to replace the semantic prediction matrix. Actually, you should get the matrix by making a inference with your own semantic model.

## Usage
+ validation
```
python cal_panoptic.py
```
+ test
```
python cal_panoptic.py -t
```
