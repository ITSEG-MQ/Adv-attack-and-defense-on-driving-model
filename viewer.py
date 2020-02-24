'''
Results video generator Udacity Challenge 2
Original By: Comma.ai Revd: Chris Gundling
'''

from __future__ import print_function

import argparse
import sys
import numpy as np
# import h5py
import json
import pandas as pd
from os import path
import time
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
# import matplotlib.backends.backend_agg as agg
import pylab
from model import BaseCNN, Vgg16, Nvidia
import torch
from advGAN.models import Generator

# from data import min_max_scaler


from skimage import transform as tf

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
        [114.86050156739812, 60.83953551083698],
        [129.74572757609468, 50.48459567870026],
        [132.98164627363735, 46.38576532847949],
        [301.0336906326895, 98.16046448916306],
        [238.25686790036065, 62.56535881619311],
        [227.2547443287154, 56.30924933427718],
        [209.13359962247614, 46.817221154818526],
        [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
        [25.275895776451954, 1.42189132706374],
        [36.062291434927694, 1.6376192402332563],
        [40.376849698318004, 1.42189132706374],
        [11.900765159942026, -2.1376192402332563],
        [22.25570499207874, -2.1376192402332563],
        [26.785991168638553, -2.029755283648498],
        [37.033067044190524, -2.029755283648498],
        [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
  p1, p2 = tform3_img((x, y))[0]
  return p2, p1

# ***** functions to draw lines *****


def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[int(row)-sz:int(row)+sz, int(col)-sz:int(col)+sz] = color



def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****


def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014  # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
  curvature = angle_steers_rad / \
      (steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * \
      np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 255, 0)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)


def draw(img, y_pred, y_true=None):
    img = imresize(img, size=(160, 320))
    red = (255, 0, 0)
    # blue = (0, 255, 0)

    speed_ms = 5  # log['speed'][i]  
    if y_true:
      draw_path_on(img, speed_ms, y_true/5.0)
    draw_path_on(img, speed_ms, y_pred/5.0, red)
    return img



  # noise_u = np.load(model_name + '_universal_attack_noise.npy')
  # noise_u = torch.from_numpy(noise_u).type(torch.FloatTensor).to(device)

  # advGAN_generator = Generator(3,3, model_name).to(device)
  # advGAN_uni_generator = Generator(3,3, model_name).to(device)

  # advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))    
  # advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
  # noise_seed = np.load(model_name + '_noise_seed.npy')
  # noise_a = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))

# ***** main loop *****
if __name__ == "__main__":
  pass
  # exp1_fig()
    # parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('--dataset', type=str,
    #                     help='dataset folder with csv and image folders')
    # parser.add_argument('--camera', type=str, default='center',
    #                     help='camera to use, default is center')
    # parser.add_argument('--resized-image-width',
    #                     default='320', type=int, help='image resizing')
    # parser.add_argument('--resized-image-height',
    #                     default='160', type=int, help='image resizing')
    # parser.add_argument('--prediction',
    #                     type=str, help='prediction result csv file path', default='result.csv')
    # parser.add_argument('--output',
    #                     type=str, help='image output path')
    # args = parser.parse_args()

    # dataset_path = args.dataset
    # image_size = (args.resized_image_height, args.resized_image_width)
    # camera = args.camera


    # df_test = pd.read_csv(args.prediction, usecols=[
    #                       'frame_id', 'steering_angle'], index_col=None)
    # timestamps = df_test['frame_id'].values
    # df_truth = pd.read_csv('ch2_final_eval.csv', usecols=[
    #                        'frame_id', 'steering_angle'], index_col=None)
    # yhat = df_test['steering_angle'].values
    # list_truth = df_truth.frame_id.values
    # list_test = df_test.frame_id.values
    # timestamps = [i for i in list_truth if i in list_test]
    # timestamps.sort()
    # y = np.array([df_truth['steering_angle'][df_truth['frame_id'] == i].values for i in timestamps]).reshape(-1,)

    # # 对测试集进行归一化（引用data里面的scaler）
    # # 估计在调用会出错
    # # y_scaled = min_max_scaler.transform(y.reshape(-1,1)).reshape(len(y),)
    # # y = y_scaled 

    # rmse = np.sqrt(np.mean((yhat-y)**2))
    # print("RMSE:%s" % rmse)

    
    # fig_rmse = plt.figure()
    # l1 = plt.plot(y, c='blue', label='truth')
    # l2 = plt.plot(yhat, c='red', linestyle='--', label='prediction')
    # plt.title('RMSE:' + str(rmse))
    # plt.legend(loc=2)
    # fig_rmse.savefig(path.join(args.output, 'rmse.jpg'))
    # fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    # ax = fig.gca()
    # ax.tick_params(axis='x', labelsize=8)
    # ax.tick_params(axis='y', labelsize=8)

    # line1, = ax.plot([], [], 'b.-', label='Human')
    # line2, = ax.plot([], [], 'r.-', label='Model')
    # A = []
    # B = []
    # ax.legend(loc='upper left', fontsize=8)

    # red = (255, 0, 0)
    # blue = (0, 0, 255)

    # speed_ms = 5  # log['speed'][i]

    # # Run through all images
    # limits = 1
    # for i in timestamps:
    #     if limits > 0:
    #       img = imread(path.join(dataset_path, 'center', str(i) + '.jpg'))

    #       img = imresize(img, size=(160, 320))

    #       predicted_steers = df_test['steering_angle'][df_test['frame_id'] == i].values
    #       actual_steers = df_truth['steering_angle'][df_truth['frame_id'] == i].values
    #       print(predicted_steers, actual_steers)
    #       draw_path_on(img, speed_ms, actual_steers/5.0)
    #       draw_path_on(img, speed_ms, predicted_steers/5.0, red)
    #       # plt.imshow(img)
    #       # plt.show()
    #       A.append(predicted_steers)
    #       B.append(actual_steers)
    #       line1.set_ydata(A)
    #       line1.set_xdata(range(len(A)))
    #       line2.set_ydata(B)
    #       line2.set_xdata(range(len(B)))
    #       ax.relim()
    #       ax.autoscale_view()

    #       print ("save picture: %s" % str(i))
    #       # print(img.shape)
    #       imsave(path.join(args.output, '%s.jpg' % (str(i))), img)
    #       limits -= 1

