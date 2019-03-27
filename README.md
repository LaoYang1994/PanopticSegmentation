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

You can form the file organization by the tools **generate_file_organization.py**. It will generate the file structure automatically.

```
cd REPO_ROOT_DIR
python tools/generate_file_organization.py
```
  
+ annotations: store annotation files like panoptic_val2017.json and instances_val2017.json
+ detresults: store instance segmentation result which is stored in a json file

## Usage
If you just want to generate the two-channel-pngs, just run as following.
+ validation
```
python tools/cal_panoptic.py
```
+ test (-t for test-dev)
```
python tools/cal_panoptic.py -t
```

If you want to evaluate the result after generating two-channel-pngs automatically, just need to add '-e'. If the two-channel-pngs have been generated and you only want to evaluate, then just add '-sc'.

**If you have any question, just leave an issue!**
