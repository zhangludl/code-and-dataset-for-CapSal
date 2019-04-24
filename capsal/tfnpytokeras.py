import numpy as np
import pandas as pd
from mrcnn.vocabulary import Vocabulary
model_dir = '/home/zhanglu/image_captioning/models/332999.npy'
data_dict = np.load(model_dir).item()
new_dict = {}
keras_indices_order = np.argsort([0,2,1,3])
for k, v in data_dict.items():
    if 'optimizer' not in k:
        if 'lstm' in k and 'kernel' in k:
            i = v[:,:512]
            j = v[:,512:1024]
            c = v[:,1024:1536]
            o = v[:,1536:2048]
            new_v = np.concatenate([i,c,j,o],axis=1)
            kernel = new_v[:1024,:]
            recurrent_kernel = new_v[1024:,:]
            new_k = 'gcap_lstm/kernel:0'
            new_k2 = 'gcap_lstm/recurrent_kernel:0'
            new_dict[new_k]=kernel
            new_dict[new_k2] = recurrent_kernel
        elif 'lstm' in k and 'bias' in k:
            i = v[ :512]
            j = v[512:1024]
            c = v[1024:1536]
            o = v[1536:2048]
            new_v = np.concatenate([i, c, j, o], axis=0)
            new_k = 'gcap_lstm/bias:0'
            new_dict[new_k] = new_v
        elif 'embedding' in k:
            new_k = 'gcap_embedding/embeddings:0'
            new_dict[new_k] = v
        elif 'attend' in k:
            new_k = k.replace('attend/','gcap_attend_')
            new_dict[new_k] = v
        elif 'initialize' in k:
            new_k = k.replace('initialize/', 'gcap_initialize_')
            new_dict[new_k] = v
        elif 'decode' in k:
            new_k = k.replace('decode/', 'gcap_decode_')
            new_dict[new_k] = v
        elif 'down_imagefeature' in k:
            new_k = k.replace('down_imagefeature', 'gcap_down_imagefeature')
            new_dict[new_k] = v
np.save('keras_caption2.npy',new_dict)
# import os
# import sys
# import random
# import math
# import cv2
# import numpy as np
# import skimage.io
# import matplotlib
# import matplotlib.pyplot as plt
# import pickle
# import time
# import utils
# import scipy.io as scio
# import json
# from mrcnn.vocabulary import Vocabulary
# def load_coco_data(data_path='/home/zhanglu/Mask_RCNN/train/train', split='train'):
#     data_path = os.path.join(data_path, split)
#     start_t = time.time()
#     data = {}
#
#     # data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' %split))
#     with open(os.path.join(data_path, '%s.file.names.pkl' % split), 'rb') as f:
#         data['file_names'] = pickle.load(f)
#     with open(os.path.join(data_path, '%s.captions.pkl' % split), 'rb') as f:
#         data['captions'] = pickle.load(f)
#     with open(os.path.join(data_path, '%s.annotations.pkl' % split), 'rb') as f:
#         data['annotations'] = pickle.load(f)
#     with open(os.path.join(data_path, '%s.image.idxs.pkl' % split), 'rb') as f:
#         data['image_idxs'] = pickle.load(f)
#
#     if split == 'train':
#         with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
#             data['word_to_idx'] = pickle.load(f)
#
#     for k, v in data.iteritems():
#         if type(v) == np.ndarray:
#             print k, type(v), v.shape, v.dtype
#         else:
#             print k, type(v), len(v)
#     end_t = time.time()
#     print "Elapse time: %.2f" % (end_t - start_t)
#     return data
# def load_training_sample(lines):
#
#
#
#
#     files = []
#     labels = []
#     sals = []
#     for line in lines:
#         # labels.append('/home/zhanglu/Documents/dataset/DUTS-TR/DUTS-TR-Mask01-extend/%s' % line.replace('.jpg', '.png'))
#         # files.append('/home/zhanglu/Documents/dataset/DUTS-TR/DUTS-TR-Image-extend/%s' % line)
#         #
#         labels.append('/home/zhanglu/Mask_RCNN/val/gt/%s' % line.replace('.jpg', '.png'))
#         files.append('/home/zhanglu/Mask_RCNN/val/val/%s' % line)
#         # sals.append('/home/zhanglu/Documents/dataset/DUTS-TR/contour-extend/%s' % line.replace('.jpg','.png'))
#     return files, labels
# def save_pickle(data, path):
#     with open(path, 'wb') as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#         print ('Saved %s..' %path)
# data = load_coco_data(data_path='/home/zhanglu/Mask_RCNN/pixel-prediction/data', split='train')
# word_to_idx = data['word_to_idx']
# n_examples = data['captions'].shape[0]
# n_iters_per_epoch = int(np.ceil(float(n_examples)))
# captions = data['captions']
# image_idxs = data['image_idxs']
# annotations = data['annotations']
# cap_sentence = annotations['caption']
# # vocabulary = Vocabulary(5000,
# #                             '/home/zhanglu/image_captioning/vocabulary.csv')
# # capidxs = []
# # for i, capp in enumerate(cap_sentence):
# #     idx = vocabulary.process_sentence(capp)
# #     capidxs.append(idx)
# img_dir = data['file_names']
# num = image_idxs.shape[0]
# num = img_dir.shape[0]
#
# # embedding_index = {}
# # f = open('/home/zhanglu/Downloads/glove.6B/glove.6B.300d.txt')
# # a = 0
# # for line in f:
# #     values = line.split()
# #     word = values[0]
# #     coefs = np.asarray(values[1:],dtype='float32')
# #     embedding_index[word] = coefs
# #
# # num_words = 6277#6951
# # embedding_matrix = np.zeros((6277,300))
# # for word, i in word_to_idx.items():
# #     embedding_vector = embedding_index.get(word)
# #     if embedding_vector is not None:
# #         embedding_matrix[i] = embedding_vector
# # save_pickle(embedding_matrix, 'embedding_matrix2.pkl')
# files,labels = load_training_sample(img_dir)
# in_dir = '/home/zhanglu/Mask_RCNN/seg_fet_val'
# in_dir2 = '/home/zhanglu/Mask_RCNN/train/cls_masks_true'
# save_dir = '/home/zhanglu/Mask_RCNN/proposal-prediction/Result/coco/gtptoposa'
# gt_store = {};
# vocabulary = Vocabulary(5000,
#                             '/home/zhanglu/image_captioning/vocabulary.csv')
# word = vocabulary.get_sentence([0])
# data = np.load('/home/zhanglu/image_captioning/train/data.npy').item()
# word_idxs = data['word_idxs']
# masks = data['masks']
# annotations = pd.read_csv('/home/zhanglu/image_captioning/train/anns.csv')
# captions = annotations['caption'].values
# image_ids = annotations['image_id'].values
# image_files = annotations['image_file'].values
# feature_files = annotations['feature_file'].values
# result=[]
# a = np.load('/home/zhanglu/Mask_RCNN/train/train_data_upcap.npy')
# aaa = 0
# for i in range(num):
#
#     # img_idx = image_idxs[i]
#     # file_names = files[img_idx]
#     # img_name = img_dir[img_idx]
#     # out_dir1 = os.path.join(in_dir, img_name).replace('.jpg', '.npy')
#     # out_dir2 = os.path.join(in_dir2, img_name).replace('.jpg', '.mat')
#     #
#     # cap = captions[i,:]
#     # gt = scio.loadmat(out_dir2)
#     #
#     # gt_mask = gt['new_mask']
#     #
#     # gt_ids = np.squeeze(gt['id'])
#     # if len(gt_mask.shape) > 2:
#     #     gt_map = np.max(gt_mask, axis=2)
#     #     bb = utils.extract_bboxes(gt_mask).astype(np.float32)
#     # else:
#     #     gt_map = gt_mask
#     #     gt_mask1 = np.reshape(gt_mask, [gt_mask.shape[0], gt_mask.shape[1], 1])
#     #     bb = utils.extract_bboxes(gt_mask1).astype(np.float32)
#     #     gt_mask = gt_mask1
#     # result.append({'image_id': gt_ids,
#     #                'masks': gt_mask,
#     #                'gt': gt_map,
#     #                'b_box': bb,
#     #                'caption':cap,
#     #                'image_name': img_name})
#     print(i)
#     ### old
#     file_names = files[i]
#     img_name = img_dir[i]
#     out_dir1 = os.path.join(in_dir, img_name).replace('.jpg', '.npy')
#     out_dir2 = os.path.join(in_dir2, img_name).replace('.jpg', '.mat')
#
#     gt = scio.loadmat(out_dir2)
#
#     gt_mask = gt['new_mask']
#
#     gt_ids = np.squeeze(gt['id'])
#     cap_ids = np.where(image_ids == gt_ids)
#     caption = captions[cap_ids]
#     cap = word_idxs[cap_ids,:]
#     cap_mask = masks[cap_ids,:]
#
#     if cap.shape[1] !=0:
#         aaa = aaa + cap.shape[1]
#         if len(gt_mask.shape) > 2:
#             gt_map = np.max(gt_mask, axis=2)
#             bb = utils.extract_bboxes(gt_mask).astype(np.float32)
#         else:
#             gt_map = gt_mask
#             gt_mask1 = np.reshape(gt_mask, [gt_mask.shape[0], gt_mask.shape[1], 1])
#             bb = utils.extract_bboxes(gt_mask1).astype(np.float32)
#             gt_mask = gt_mask1
#         result.append({'image_id': gt_ids,
#                        'masks': gt_mask,
#                        'gt': gt_map,
#                        'b_box': bb,
#                        'caption':cap,
#                        'caption_mask':cap_mask,
#                        'image_name': img_name})
#         print(i)
#     else:
#         print(i)
#
# np.save('/home/zhanglu/Mask_RCNN/train/train_data_upcap2.npy', result)
#
#
#
#
#
#     # if not os.path.exists(out_dir1):
#         # if img_name !=b:
#     # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# #     image = skimage.io.imread(file_names)
# # # Run detection
# #     if len(image.shape) <3:
# #         image = np.stack((image,image,image),axis=2)
#
#
