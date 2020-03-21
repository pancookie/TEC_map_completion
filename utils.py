# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import pandas as pd
from pandas import Series,DataFrame, np
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
#
import cv2 as cv
import scipy.sparse
import pyamg
#
from time import gmtime, strftime
import _pickle as pickle

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def deal_TEC_mask_(path, n1, n2):
    f = open(path, 'rb')
    imfiles = pickle.load(f).astype(np.float32).reshape(n1, n2)
    f.close()
    idx = np.argwhere(imfiles == 9999.)

    mask = np.ones([n1, n2])

    for idx_ in idx:
        mask[idx_[0], idx_[1]] = 0

    mask = mask.reshape(n1, n2, 1)
    
    return mask

def deal_TEC_mask_complete(path, n1, n2):
    imfiles = np.loadtxt(path, delimiter='\n').astype(np.float32).reshape(n1, n2)
    idx = np.argwhere(imfiles == 9999.)

    mask = np.ones([n1, n2])

    for idx_ in idx:
        mask[idx_[0], idx_[1]] = 0

    mask = mask.reshape(n1, n2, 1)

    return mask


def deal_TEC_mask(path,n1,n2): # mask the file
    imfiles=np.loadtxt(path) /80. -1. 
    data=np.reshape(imfiles,(n1,n2,1))

def deal_TEC(path, w, h): # was used to fill the gaps, batch-wisely
    # the normalization could be improved?
    f = open(path, 'rb')
    data = pickle.load(f).astype(np.float32).reshape(w, h, 1)
    f.close()
    return data / 100.0 - 1.

def deal_TEC_complete(path, w, h): # was used to fill the gaps, batch-wisely
    data = np.loadtxt(path, delimiter='\n').astype(np.float32).reshape(w, h, 1)
    return data / 100.0 - 1.

def save_TEC(path,content):
    f = open(path,'wb')
    f.write(content)
    f.close()

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def prepare_mask(mask): # mask is 2-D binary matrix
    if type(mask[0][0]) is np.ndarray: # when the mask is a 3D, eg. h, w, cnls
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask # 2D

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # preprocess the target and source into right dimensions
    if len(img_target.shape) > 3:
        img_target = img_target.reshape(img_target.shape[-3], img_target.shape[-2], img_target.shape[-1])

    if len(img_source.shape) > 3:
        img_source = img_source.reshape(img_source.shape[-3], img_source.shape[-2], img_source.shape[-1])
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0), # 0
            max(-offset[1], 0), # 0
            min(img_target.shape[0]-offset[0], img_source.shape[0]), # 64 e.g.
            min(img_target.shape[1]-offset[1], img_source.shape[1])) # 64
    region_target = ( # 0, 0, 64, 64
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1]) # 64, 64

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False # binarize 0 and 1 to False and True
    img_mask[img_mask!=False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil') # Identity matrix, diagonal
    ### ? ###
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1] # from 0 to 4095
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    ### ? ###

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] # 64, 64, 1
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]# 64, 64, 1
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        # x[x>255] = 255
        # x[x<0] = 0
        x = np.array(x, img_target.dtype)

        ### blend without margins
        img_target = img_target.reshape(64, 64)
        img_source = img_source.reshape(64, 64)

        img_target[img_target == 9999.] = 0.
        img_target = img_target + img_mask * img_source

        img_target[5:(64-5), 5:(64-5)] = x[5:(64-5), 5:(64-5)] 

        ###
    return img_target

def blend_kai(img_target, img_source, img_mask, offset=(0, 0)):
    # preprocess the target and source into right dimensions
    if len(img_target.shape) > 3:
        img_target = img_target.reshape(img_target.shape[-3], img_target.shape[-2], img_target.shape[-1])

    if len(img_source.shape) > 3:
        img_source = img_source.reshape(img_source.shape[-3], img_source.shape[-2], img_source.shape[-1])
    # compute regions to be blended
# =============================================================================
#     region_source = (
#             max(-offset[0], 0), # 0
#             max(-offset[1], 0), # 0
#             min(img_target.shape[0]-offset[0], img_source.shape[0]), # 64 e.g.
#             min(img_target.shape[1]-offset[1], img_source.shape[1])) # 64
#     region_target = ( # 0, 0, 64, 64
#             max(offset[0], 0),
#             max(offset[1], 0),
#             min(img_target.shape[0], img_source.shape[0]+offset[0]),
#             min(img_target.shape[1], img_source.shape[1]+offset[1]))
# =============================================================================
    region_source = (0, 0, 64, 74)
    region_target = (0, 0, 64, 74)
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1]) # 64, 64

    img_mask = prepare_mask(img_mask)
    ####### extend west and east margins
    img_target_ext = np.zeros([64, 74, 1]).astype(type(img_target[0, 0, 0]))
    img_target_ext[:, 0:5] = img_target[:, 59:64]
    img_target_ext[:, 5:69] = img_target
    img_target_ext[:, 69:74] = img_target[:, 0:5]

    img_mask_ext = np.zeros([64, 74]).astype(np.int8)
    img_mask_ext[:, 0:5] = img_mask[:, 59:64]
    img_mask_ext[:, 5:69] = img_mask
    img_mask_ext[:, 69:74] = img_mask[:, 0:5]

    img_source_ext = np.zeros([64, 74, 1]).astype(type(img_source[0, 0, 0]))
    img_source_ext[:, 0:5] = img_source[:, 59:64]
    img_source_ext[:, 5:69] = img_source
    img_source_ext[:, 69:74] = img_source[:, 0:5]
    #######

    # clip and normalize mask image
    # img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]] # 0:64
    # img_mask = prepare_mask(img_mask)
    # img_mask[img_mask==0] = False # binarize 0 and 1 to False and True
    # img_mask[img_mask!=False] = True
    #####
    img_mask_ext = prepare_mask(img_mask_ext)
    img_mask_ext[img_mask_ext==0] = False
    img_mask_ext[img_mask_ext!=False] = True


    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil') # Identity matrix, diagonal
    ### ? ###
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask_ext[y,x]: ####
                index = x+y*region_size[1] # from 0 to 4095
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    ### ? ###

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask_ext.shape)

    # for each layer (ex. RGB)
    # import pdb; pdb.set_trace()
    for num_layer in range(img_target_ext.shape[2]):
        # get subimages
        t = img_target_ext[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] # 0:64, 0:64, 0
        s = img_source_ext[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]# 0:64, 0:64, 0
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask_ext[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size) # 64, 74
        # x[x>255] = 255
        # x[x<0] = 0
        x = np.array(x, img_target.dtype)

# =============================================================================
#         ### blend without margins
#         img_target = img_target.reshape(64, 64)
#         img_source = img_source.reshape(64, 64)
#
#         img_target[img_target == 9999.] = 0.
#         img_target = img_target + img_mask * img_source
#
#         img_target[5:(64-5), 5:(64-5)] = x[5:(64-5), 5:(64-5)]
# =============================================================================
        img_target_ext[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target_ext[:, 5:69].reshape(64, -1)
