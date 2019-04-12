#########################################################################
# MODEL PARAMETERS														#
#########################################################################

import os

# select a running mode
mode_debug = 1
# version sellection
v_model = 2
# batch size
b_s = 4 # be exactly divided by the total number of training as well validation image
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# number of rows of learned features
shape_r_f = 30
# number of cols of learned features
shape_c_f = 40
# number of epochs
nb_epoch = 1
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16
# the height of the inputting images
img_H = 240
# the width of the inputting images
img_W = 320
# path of pretrained model_1
pth_pm_1 = os.path.join('pretrained_models', 'resnet50-19c8e357.pth')
# path of saved parameters
log_dir = os.path.join('Results')

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of raw images
Imgs = os.path.join('Data', 'Imgs')
# path of raw maps
Maps = os.path.join('Data', 'Maps')
# path of raw fixs
Fixs = os.path.join('Data', 'Fixs')
# the percentage of training dataset
train_per = 0.8
# the percentage of validation dataset
valid_per = 0.1
# the percentage of testing dataset
test_per = 0.1
# path of training images
imgs_train_path = os.path.join('Data', 'imgs_train')
# path of training maps
maps_train_path = os.path.join('Data', 'maps_train')
# path of training fixation maps
fixs_train_path = os.path.join('Data', 'fixs_train')
# number of training images
#nb_imgs_train = 0
# path of validation images
imgs_val_path = os.path.join('Data', 'imgs_val')
# path of validation maps
maps_val_path = os.path.join('Data', 'maps_val')
# path of validation fixation maps
fixs_val_path = os.path.join('Data', 'fixs_val')
# number of validation images
# nb_imgs_val = 0
# path of testing images
imgs_test_path = os.path.join('Data', 'imgs_test')
# path of testing maps
maps_test_path = os.path.join('Data', 'maps_test')
#path of testing fixation maps
fixs_test_path = os.path.join('Data', 'fixs_test')
# number of testing images
# nb_imgs_test = 0
# need to split the data ?
split_data = 0
#need to generate the corresponding txt ?
generate_txt = 0
# path of txt recording training images
imgs_train_txt_path = os.path.join('Data', 'imgs_train.txt')
# path of txt recording training maps
maps_train_txt_path = os.path.join('Data', 'maps_train.txt')
# path of txt recording training fixs
fixs_train_txt_path = os.path.join('Data', 'fixs_train.txt')
# path of txt recording validation images
imgs_val_txt_path = os.path.join('Data', 'imgs_val.txt')
# path of txt recording validation maps
maps_val_txt_path = os.path.join('Data', 'maps_val.txt')
# path of txt recording validation fixs
fixs_val_txt_path = os.path.join('Data', 'fixs_val.txt')
# path of txt recording testing images
imgs_test_txt_path = os.path.join('Data', 'imgs_test.txt')
# path of txt recording testing maps
maps_test_txt_path = os.path.join('Data', 'maps_test.txt')
# path of txt recording testing fixs
fixs_test_txt_path = os.path.join('Data', 'fixs_test.txt')
# how many images are needed to compute the mean and std
CNum = 160
# computed mean of the training images
NormMean_imgs = [0.50893384, 0.4930997, 0.46955067]
# computed std of the training images
NormStd_imgs = [0.2652982, 0.26586023, 0.27988392]
# need to compute the mean and std ?
compute_ms = 0
# the coefficient of KL-DIV
scal_KLD = 10
# the coefficient of CC
scal_CC = -2
# set default of the epsilon for DC
epsilon_DC = 0.001
# the coefficient of NSS
scal_NSS = -1
# initialize the learning rate
lr_init = 0.001

# feature_channel = 512
# upsampling factor = 240/30 = 8


