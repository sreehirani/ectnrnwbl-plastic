
# The Garbage Sorting System



# About the Hardware

There are three main parts/components to the system: a trash chute, a robotic arm, and a linear actuator.

## 1. The trash chute
[TO-DO]

## 2. The robotic arm component
This system comprises of the [Yahboom's DOFBOT-Pi kit](https://category.yahboom.net/collections/rp-robotics/products/dofbot-pi). 

The kit comes with a Rasperry Pi, a robotic arm, and a USB camera. The rest of the components in the kit can also be found [here](https://category.yahboom.net/collections/rp-robotics/products/dofbot-pi).


## 3. The linear actuator
[TO-DO]


# About the Software

Ecotone's garbage sorting code is a modification of the toy garbage sorting code that was provided for the DOFBOT-PI. More on that over [here](http://www.yahboom.net/study/Dofbot-Pi) (Under 10.AI Vision > 10.8 Garbage sorting).

The toy garbage code only works on the wooden blocks that came with the kit. It uses a variation of the YOLOv4-Tiny model targeted at a limited dataset (images printed on the wooden blocks).

Ecotone's code also uses a pre-trained model of YOLOv4-Tiny, which is accessible [here](https://github.com/bubbliiiing/yolov4-tiny-tf2). The instructions there were not in English, so I used this [reference](https://github.com/qqwweee/keras-yolo3/) for guidance. This [post](https://www.v7labs.com/blog/yolo-object-detection) could also be helpful for understanding how YOLO works. 

The folder containing the code can be found on the Raspberry Pi at `~/ecotone_garbage_system`. All the required folder and scripts are in that folder, namely:

 * `garbage_sorting_notebok.ipynb`: It is the entry point. This is what is run. It presents an interface for (1) further calibration of the robotic arm and (2) visualizing object detection in real time. 
 * `dofbot_config.py`: this script handles any required calibration of the robotic arm joints and the camera
 * `garbage_identify.py`: this script handles the object detection
 * `garbage_grap_move.py`: this script handles the motion of the robotic arm (e.g. moving recyclable items to the right side of the arm)


**Notes:**
 * Overall, [DOFBOT-PI docs](http://www.yahboom.net/study/Dofbot-Pi) is the main reference for updating the code for controlling the motion of the robotic arm.
 * There is a system image provided at [DOFBOT-PI docs](http://www.yahboom.net/study/Dofbot-Pi) (under the Downloads section). After flashing the Pi with this Linux image, Tensorflow and OpenCV will be installed. So that was really helpful since it usually is a pain to install those two libraries from scratch on Raspberry Pis.
 * The documentation at *10.8 Garbage sorting* on the [DOFBOT-PI docs](http://www.yahboom.net/study/Dofbot-Pi) is the main reference for creating or updating the garbage detection code.


# Instructions on setting up the system:

A document detailing this process can be found [here](https://docs.google.com/document/d/1h5AJnbZxFPD6MtEedEg6OkWBqdly3Io5qe2fJULOBHk/edit?usp=sharing).


# Future work:

 * Test out different models: [Mask R-CNN](https://github.com/matterport/Mask_RCNN), or other versions of [YOLO](https://github.com/ultralytics/yolov5).
 * Collect and use our own pictures for training
 * Or, use available open [datasets](https://github.com/AgaMiko/waste-datasets-review#summary).