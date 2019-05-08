## Instruction
  Download the COCO-CapSal dataset from [BaiduYun](https://pan.baidu.com/s/1iU8A-RII7rvOG9KHz5Dysg) or [Google](). We provide the images, ground truth, image captions and instance masks.
* `train_img_gt.zip`: images and GT of training set.
* `val_img_gt.zip`: images and GT for validation set.
* `train.zip`: image captions and instance masks for training set.
* `val.zip`: image captions and instance masks for validation set.

  `train.zip` contains `vocabulary.csv` (dictionary of captions) and `train.npy` (captions and instance masks). 
  `val.zip` contains `caption_gt.json` (gt captions for evaluation) and `val.npy` (captions and instance masks). 
  `train/val.npy` contains `image_id` (image index), `image_name` (image name), `gt` (GT saliency map), `masks` (instance masks), `b_box` (bounding box), `captionw` (captions).
  For training, first run `preprocessing.py` to preprocess the caption data.
