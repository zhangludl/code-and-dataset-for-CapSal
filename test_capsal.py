import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io

from capsal.config import Config
from capsal import utils
from capsal import model_new10_upcap11 as modellib
from capsal.eval_cap import COCOEvalCap
import json

os.environ["CUDA_VISIBLE_DEVICES"]='1'
from capsal.vocabulary import Vocabulary
import skimage.transform
# import skimage
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class SaliencyConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "saliency"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 5265 // IMAGES_PER_GPU#25256 5265
    VALIDATION_STEPS = 100 // IMAGES_PER_GPU
    TRAIN_ROIS_PER_IMAGE = 200
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
    DETECTION_MIN_CONFIDENCE = 0.8
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.

    #
    # # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #
    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32
    #
    # # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100
    #
    # # use small validation steps since the epoch is small

class SaliencyDataset(utils.Dataset):
    def load_sal(self, subset):
        """Load the saliency dataset for train or validation.
        dataset_dir: The root directory of the saliency dataset..
        subset: train or val.
        """
        # Add classes
        self.add_class("saliency", 1, "foreground")
        if subset == 'train':
            sal_dataset = np.load('./data/train.npy',encoding='latin1')
        else:
            sal_dataset = np.load('./data/val.npy',encoding='latin1')
            self.sal_data = sal_dataset

        for sal_info in sal_dataset:

            image_id = int(sal_info['image_id'])
            image_name = sal_info['image_name']
            masks = sal_info['masks'].astype(np.int32)
            gt = sal_info['gt'].astype(np.float32)
            if subset == 'train':
                caption = sal_info['caption'].astype(np.int32)

                caption_mask = sal_info['caption_mask'].astype(np.float32)
            # b_box = float(sal_info['b_box'])
            if subset == 'train':
                dataset_dir = './data/train_img_gt/image/'
                self.add_image("saliency", image_id=image_id, path=os.path.join(dataset_dir, image_name),
                               mask=masks, image_name=image_name, gt=gt, caption=caption, caption_mask=caption_mask)
            else:
                dataset_dir = './data/val_img_gt/image/'
                self.add_image("saliency", image_id=image_id, path=os.path.join(dataset_dir, image_name),
                               mask=masks, image_name=image_name, gt=gt)
            #
    def load_mask(self,image_id):
        info = self.image_info[image_id]
        gt = info['gt']
        # getmask
        mask = info['mask']
        return mask, np.ones([mask.shape[-1]],dtype=np.int32)
    def load_caption(self,image_id):
        info = self.image_info[image_id]
        caption = info['caption']
        caption_mask = info['caption_mask']
        # caption = np.zeros((2,15))
        return caption, caption_mask
    def image_reference(self,image_id):
        #':return the path og the image'
        info = self.image_info[image_id]
        if info["source"] == "saliency":
            return info['id']
        else:
            super(self.__class__).image_reference(self, image_id)
def load_img_list(dataset):

    if dataset == 'coco':
        path = '/home/zhanglu/Mask_RCNN/val/val'
    elif dataset == 'HKU-IS':
        path = './dataset/HKU-IS/HKU-IS_Image'
    elif dataset == 'PASCAL-S':
        path = './dataset/pascal-s/PASCAL_S-Image'
    elif dataset == 'DUT':
        path = './dataset/DUTS-TR/DUTS/DUT-test/DUT-test-Image'
    elif dataset == 'THUS':
        path = './dataset/THUR/THUR-Image'
    elif dataset == 'SOC':
        path = './dataset/SOC6K_Release/'

    imgs = os.listdir(path)

    return path, imgs
def predict2(model):
    datasets = ['coco']#'coco','PASCAL-S','SOC','ECSSD','DUT','THUS','HKU-IS'
    for dataset in datasets:
        print(dataset)
        path, imgs = load_img_list(dataset)

        save_dir = './result'
        save_dir1 = save_dir + '/result1'+'_'+dataset +'/'
        if not os.path.exists(save_dir1):
            os.mkdir(save_dir1)
        save_dir2 = save_dir + '/result_pixel1'+'_'+dataset +'/'
        if not os.path.exists(save_dir2):
            os.mkdir(save_dir2)
        save_dir3 = save_dir + '/combine1'+'_'+dataset +'/'
        if not os.path.exists(save_dir3):
            os.mkdir(save_dir3)
        save_dir4 = save_dir + '/caption' + '_' + dataset + '/'
        if not os.path.exists(save_dir4):
            os.mkdir(save_dir4)
        idx = 0

        for f_img in imgs:
            print(idx)
            image_name = f_img


            image = skimage.io.imread(os.path.join(path, f_img))
                # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                    image = skimage.color.gray2rgb(image)
                # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                    image = image[..., :3]
            if image.shape[0] > 1024 or image.shape[1] > 1024:
                    image = skimage.transform.resize(image,(800,800),preserve_range=1)
                    image = image.astype(np.uint8)
            r = model.detect([image], verbose=0)[0]
                # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                #                             class_names, r['scores'])
            score_masks = r['proposal'].astype(np.float32)
            score_masks = np.squeeze(score_masks)
            pixel_mask = r['pixel'].astype(np.float32)
            combine_mask = r['combine'].astype(np.float32)


            cv2.imwrite(save_dir1 + image_name, score_masks * 255)
            cv2.imwrite(save_dir2 + image_name, pixel_mask * 255)
            cv2.imwrite(save_dir3 + image_name, combine_mask * 255)
            idx = idx +1

def predict(dataset,model,save_dir):
    class_names = ['BG','foreground']
    image_ids = dataset.image_ids
    save_dir = './result'
    save_dir1 = save_dir + '/result/'
    if not os.path.exists(save_dir1):
        os.mkdir(save_dir1)
    save_dir2 = save_dir + '/result_pixel/'
    if not os.path.exists(save_dir2):
        os.mkdir(save_dir2)
    save_dir3 = save_dir + '/combine/'
    if not os.path.exists(save_dir3):
        os.mkdir(save_dir3)
    # save_dir4 = save_dir + '/combine4/'
    # if not os.path.exists(save_dir4):
    #     os.mkdir(save_dir4)
    vocabulary = Vocabulary(5000,
                            './data/vocabulary.csv')
    ids =[]
    caption = {}
    for image_id in image_ids:
        word_out = []
        print(image_id)
        image = dataset.load_image(image_id)

        image_name = dataset.image_info[image_id]['image_name']
        img_name2, ext = os.path.splitext(image_name)
        final = np.zeros((image.shape[0],image.shape[1]))
        final_pro = np.zeros((image.shape[0], image.shape[1]))
        final_combine = np.zeros((image.shape[0], image.shape[1]))
        id = dataset.image_info[image_id]['id']
        ids.append(id)



        r = model.detect([image], verbose=0)[0]

        cap_id = np.squeeze(r['word']).astype(np.int)
        word = vocabulary.get_sentence(cap_id)
        word_out.append(word.replace('.',''))
        caption[id] = word_out

        score_masks = r['proposal'].astype(np.float32)
        score_masks = np.squeeze(score_masks)
        out_name = save_dir1 + img_name2 + '.jpg'
        cv2.imwrite(out_name, score_masks * 255)
        pixel_mask = r['pixel'].astype(np.float32)
        out_name = save_dir2 + img_name2 + '.jpg'
        cv2.imwrite(out_name, pixel_mask * 255)
        combine_mask = r['combine'].astype(np.float32)
        out_name = save_dir3 + img_name2 + '.jpg'
        cv2.imwrite(out_name, combine_mask * 255)
    caption_gt = json.load(open('./data/caption_gt.json'), encoding='utf-8')
    ceval = COCOEvalCap(caption_gt, caption)
    ceval.evaluate(ids)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("--command",
                        default='evaluate', required=False,
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default='',
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')

    parser.add_argument('--model', required=False,
                        default='/home/zhanglu/Mask_RCNN_new/logs/saliency20181122T1118/mask_rcnn_saliency_0020.h5',#',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = SaliencyConfig()
    else:
        class InferenceConfig(SaliencyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.8
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)

    if args.model.lower() == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

    else:
        model.load_weights(model_path, by_name= True)

   
    # Validation dataset
    dataset_val = SaliencyDataset()
    dataset_val.load_sal('val')
    dataset_val.prepare()
    print("Running COCO evaluation on {} images.".format(1459))
    predict2(model)
    
