# PanopticSegmentation
This code is used to fuse the semantic segmentation result and instance segmentation result, which generates two-channel-pngs. We won the **third place** in COCO2018 panoptic segmentation.

## Requirements
+ Anaconda3
+ pycocotools

## File Organization
I advise you to organize you file in the following ways.

```
PanopticSegmentation
├─files
  ├─annotations
  ├─detresults
  ├─images
  │  ├─test2017
  │  └─val2017
  └─models
```
  
+ annotations: store annotation files like panoptic_val2017.json and instances_val2017.json
+ detresults: store instance segmentation result which is stored in a json file
+ images: store the images
+ models: store the semantic model.
++ In **cal_panoptic.py**, we use a random matrix to replace the semantic prediction matrix. Actually, you should get the matrix by inference with your semantic model.
