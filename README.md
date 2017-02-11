# pre3D
For 3D bounding box prediction for Kitti Dataset

## Net parameters
* rcnn/symbol/symbol_3dbox.py	sym for 3D Bbox estimation
* rcnn/symbol/proposal_3dbox.py Operators for MSCNN

## Run time
* example/env 		python script
* example/shell		shell script

Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

Value | Name      | Description
------|-----------|------------------------------------------------------------------------------------------------------------------------------
1     | type      | Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare' 
1     | truncated | Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
1     | occluded  | Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
1     | alpha     |   Observation angle of object, ranging [-pi..pi]
4     | bbox      |   2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
3     | dimensions|   3D object dimensions: height, width, length (in meters)
3     | location  |   3D object location x,y,z in camera coordinates (in meters)
1     | rotation_y|   Rotation ry around Y-axis in camera coordinates [-pi..pi]
1     | score     |   Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.

## Usage
* Clear the Cache:    ***sh clear.sh***
* Generate data file: ***sh gen_data.sh***
* Train Model:        ***sh example/shell/train_3dbox.sh***
* Test Result:        ***sh example/shell/test_3dbox.sh***

## Training step
Firstly, train the 3dbox net with the initialization from vgg nets

## IMDB data
* Data classes ('__background__', 'car', 'pedestrian', 'cyclist')
* roidb list ['boxes', 'gt_classes', 'gt_ry', 'gt_alpha', 'gt_dims', 'gt_poses', 'gt_overlaps','max_classes', 'max_overlaps', 'flipped']
* Note: in the AnchorLoader step, the generated gt_boxes is in 5-dims, (x1, y1, x2, y2, cls)
* The coordinate in imdb is follows as ,
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)

## Mean value for car, people, bicycle
car mean is      [1.63512059e+00   1.67750928e+00   4.17339054e+00]
people mean is   [1.72657895e+00   6.49309211e-01   8.21940789e-01]
bicycle mean is  [1.74532787e+00   6.06147541e-01   1.76270492e+00]


