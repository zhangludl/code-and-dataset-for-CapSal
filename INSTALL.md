# Mask R-CNN for Object Detection and Segmentation

This instruction indicates the installation steps of MASKRCNN used in the CapSal model. Please refer to the original [repository](https://github.com/matterport/Mask_RCNN.git) if you have any questions.

The repository includes:
* Source code of CapSal based on ResNet101
* Training code on COCO-CapSal
* Pre-trained weights for CapSal



# Getting Started
The codes required in the CapSal model are stored in the `CapSal`. To begin with, you should first install the requirements for the MaskRCNN benchmark.

## Requirements
Python 2.7 , TensorFlow 1.4.1, Keras 2.1.4 and other common packages listed in `requirements.txt`.

### MS COCO Requirements:
To train or test on COCO-CapSal, you'll also need:
* pycocotools (installation instructions below)
* [COCO-CapSal Dataset]()


If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).


## Installation
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco






