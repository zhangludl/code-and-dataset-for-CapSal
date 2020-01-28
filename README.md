# Code-and-Dataset-for-CapSal
   This project provides the code and datasets for 'CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection', CVPR 2019. [Paper link](https://drive.google.com/open?id=1JcZMHBXEX-7AR1P010OXg_wCCC5HukeZ)
    
  
   Our code is implemented based on the Mask RCNN in Tensorflow and Keras. You can first install the maskrcnn according to the [instruction](https://github.com/matterport/Mask_RCNN.git) or `INSTALL.md`. 
# COCO-CapSal Dataset
   The COCO-CapSal dataset provides the saliency ground truth as well as the image captions for each image. It contains 5265 images for training and 1459 ones for validation. The annotations can be downloaded at [BaiduYun](https://pan.baidu.com/s/1iU8A-RII7rvOG9KHz5Dysg) or [GoogleDrive](https://drive.google.com/open?id=1d04vkomA2sT2cUAst9CJYYHwTwNkSg2p). The folder 'capsal' contains the images, ground truth maps as well as the caprions (json file) of both training and validation sets. 
# Evaluation
For testing the CapSal model, first download the trained model at [BaiduYun](https://pan.baidu.com/s/1dQwQ5AdJqBfSSgZPUNR_gg) or [Google](https://drive.google.com/drive/folders/1d04vkomA2sT2cUAst9CJYYHwTwNkSg2p?usp=sharing)
) and put it under the `./model`. Run `test_capsal.py` to obtain the saliency maps of different datasets. 
The saliency map is avaliable at [Google](https://drive.google.com/open?id=1d04vkomA2sT2cUAst9CJYYHwTwNkSg2p) or [BaiduYun](https://pan.baidu.com/s/1LtlK3ZH8adZCEi8n0ys9BA).
# Train
Run 'train.py'.
# Citation
        @InProceedings{Zhang_2019_CVPR,
                author = {Zhang, Lu and Zhang, Jianming and Lin, Zhe and Lu, Huchuan and He, You},
                title = {CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection},
                booktitle = CVPR,
                year = {2019}}
