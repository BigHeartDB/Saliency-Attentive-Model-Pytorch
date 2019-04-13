#------------------------------------- step 0 : Input needed packages ---------------------------------------
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.models import resnet50
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from utilities import MyDataset
from config import *
import glob
import random
import shutil
import cv2
from models import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------ step 1 : Data Pre-processing ----------------------------------------
# split the raw data...
def data_spliting():
    # generate the traning, validation and testing dataset...
    for root_imgs, dirs_imgs, files_imgs in os.walk(Imgs):
        for sDir in dirs_imgs:
            imgs_list = glob.glob(os.path.join(root_imgs, sDir)+'/*.jpg')
            maps_list = []
            fixs_list = []
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            # find the corresponding maps and fixs within the current class
            for imgs_index in range(imgs_num):
                str_cur_img = imgs_list[imgs_index]
                str_cur_map = str_cur_img[0:5] + 'Maps' + str_cur_img[9:]
                str_cur_01, str_cur_02 = str_cur_img.split('.')
                str_cur_fix = str_cur_img[0:5] + 'Fixs' + str_cur_01[9:] + '.mat'
                maps_list.append(str_cur_map)
                fixs_list.append(str_cur_fix)

            train_point = int(imgs_num * train_per)
            valid_point = int(imgs_num * (train_per + valid_per))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir_imgs = imgs_train_path + '_' + sDir + '/'
                    out_dir_maps = maps_train_path + '_' + sDir + '/'
                    out_dir_fixs = fixs_train_path + '_' + sDir + '/'
                elif i < valid_point:
                    out_dir_imgs = imgs_val_path + '_' + sDir + '/'
                    out_dir_maps = maps_val_path + '_' + sDir + '/'
                    out_dir_fixs = fixs_val_path + '_' + sDir + '/'
                else:
                    out_dir_imgs = imgs_test_path + '_' + sDir + '/'
                    out_dir_maps = maps_test_path + '_' + sDir + '/'
                    out_dir_fixs = fixs_test_path + '_' + sDir + '/'

                if not os.path.exists(out_dir_imgs):
                    os.makedirs(out_dir_imgs)
                if not os.path.exists(out_dir_maps):
                    os.makedirs(out_dir_maps)
                if not os.path.exists(out_dir_fixs):
                    os.makedirs(out_dir_fixs)
                out_path_imgs = out_dir_imgs + os.path.split(imgs_list[i])[-1]
                out_path_maps = out_dir_maps + os.path.split(maps_list[i])[-1]
                out_path_fixs = out_dir_fixs + os.path.split(fixs_list[i])[-1]
                shutil.copy(imgs_list[i], out_path_imgs)
                shutil.copy(maps_list[i], out_path_maps)
                shutil.copy(fixs_list[i], out_path_fixs)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point-train_point, imgs_num-valid_point))

# generate the txt...
def txt_generating(txt_path, img_dir, type_str):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # obtain the files' name from the img_dir
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if not img_list[i].endswith(type_str):
                    continue
                # label = img_list[i].split('_')[0]
                img_path = os.path.join(i_dir, img_list[i])
                # line = img_path + ' ' + label + '\n'
                line = img_path + '\n'
                f.write(line)
    f.close()
    print('Done !')

# compute the mean and std of the images from the training dataset
def data_norm(CNum):
    img_h, img_w = img_H, img_W
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    with open(imgs_train_txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for i in range(CNum):
            img_path = lines[i].rstrip().split()[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_h, img_w))
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print(i)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

# data pre-processing
def data_preprocessing():
    normTransform = transforms.Normalize(NormMean_imgs, NormStd_imgs)
    imgsTransform = transforms.Compose([
        transforms.Resize([img_H, img_W]),
        transforms.
        transforms.ToTensor(), # 0-255 automatically transformed to 0-1
        normTransform
    ])
    mapsTransform = transforms.Compose([
        transforms.Resize([shape_r_out, shape_c_out]),
        transforms.ToTensor()
    ])
    fixsTransform = transforms.Compose([
        transforms.Resize([shape_r_out, shape_c_out]),
        transforms.ToTensor()
    ])

    train_data = MyDataset(imgs_train_txt_path, maps_train_txt_path, fixs_train_txt_path,
                           transform_img=imgsTransform, transform_map=mapsTransform, transform_fix=fixsTransform)
    val_data = MyDataset(imgs_val_txt_path, maps_val_txt_path, fixs_val_txt_path,
                         transform_img=imgsTransform, transform_map=mapsTransform, transform_fix=fixsTransform)

    return train_data, val_data


# ------------------------------------ step 2 : Net Defining ------------------------------------------------
class ZHANGYiNet_REPRO_1(nn.Module):

    def __init__(self):
        super(ZHANGYiNet_REPRO_1, self).__init__()

        # debug
        if v_model == 1: # the input size for official resnet50 must be 224 * 224
            # load the official pretrained ResNet50
            dcn = resnet50()
            dcn.load_state_dict(torch.load(pth_pm_1))
            self.dcn = dcn

        if v_model == 2:

            # define the dilated ResNet50 (with output_channel=512)
            self.dcn = MyDRN()

            # define the attentive convLSTM
            self.attentiveLSTM = MyAttentiveLSTM(nb_features_in=512, nb_features_out=512,
                                                 nb_features_att=512, nb_rows=3, nb_cols=3)

            # define the learnable gaussian priors
            self.gaussian_priors = MyPriors()

            # define the final convolutional neural network
            self.endconv = nn.Conv2d(in_channels=512, out_channels=1,
                kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.upsampling = nn.UpsamplingBilinear2d([shape_r_out, shape_c_out])
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # the dilated convlutional network based on ResNet 50
        x = self.dcn(x)

        # the convLSTM model
        x = self.attentiveLSTM(x)

        # the learnable prior block
        x = self.gaussian_priors(x)

        # the non local neural block

        # the final convolutional neural network
        x = self.endconv(x)
        x = self.relu(x)
        x = self.upsampling(x)
        x = self.sigmoid(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# ------------------------------------ step 3 : Optimizer and LOSS ----------------------------------------------------
if __name__ == '__main__':

    # call the proposed model
    net = ZHANGYiNet_REPRO_1()
    net.initialize_weights()
    # net.cuda()

    # build a Kullback-Leibler divergence
    criterion_KLD = nn.KLDivLoss()
    # criterion_KLD.cuda()

    # build a dice coefficient
    criterion_DC = MyDiceCoef()
    # criterion_DC.cuda()

    # build a correlation coefficient
    criterion_CC = MyCorrCoef()
    # criterion_CC.cuda()

    # build a normalized scanpath saliency
    criterion_NSS = MyNormScanSali()
    # criterion_NSS.cuda()

    # define the optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)

    # define the scheduler for learning rate decreasing
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)


# ------------------------------------ step 4 : model training ----------------------------------------------
    # initialize the writer
    writer = SummaryWriter(log_dir=log_dir, comment='reproducing_01')

    # generate the training, validation and testing dataset
    if split_data:
        data_spliting()

    #generate the corresponding txt of images, maps and fixs
    if generate_txt:
        # training dataset
        txt_generating(imgs_train_txt_path, imgs_train_path, 'jpg')
        txt_generating(maps_train_txt_path, maps_train_path, 'jpg')
        txt_generating(fixs_train_txt_path, fixs_train_path, 'mat')
        # validation dataset
        txt_generating(imgs_val_txt_path, imgs_val_path, 'jpg')
        txt_generating(maps_val_txt_path, maps_val_path, 'jpg')
        txt_generating(fixs_val_txt_path, fixs_val_path, 'mat')
        # testing dataset
        txt_generating(imgs_test_txt_path, imgs_test_path, 'jpg')
        txt_generating(maps_test_txt_path, maps_test_path, 'jpg')
        txt_generating(fixs_test_txt_path, fixs_test_path, 'mat')

    # compute the mean and std of training images
    if compute_ms:
        data_norm(CNum)

    # data pre-processing
    train_data, val_data = data_preprocessing()

    # load the training and validation dataset
    train_loader = DataLoader(dataset=train_data, batch_size=b_s, shuffle=True)
    valid_loader = DataLoader(dataset=val_data, batch_size=b_s, shuffle=True)

    # the main loop
    for epoch in range(nb_epoch):
        loss_sigma = 0.0
        correct = 0.0
        total = 0.0
        correct_val = 0.0
        total_val = 0.0
        scheduler.step()

        for i, data in enumerate(train_loader):

            inputs, maps, fixs = data
            inputs, maps, fixs = Variable(inputs), Variable(maps), Variable(fixs)

            #debug (check the validation of images and maps)
            # plt.figure
            # show_map = maps[0,0,:,:]
            # show_map.numpy()
            # plt.subplot(1,2,1)
            # plt.imshow(show_map, cmap='gray')
            # show_img = inputs[0,:,:,:]
            # show_img = show_img.transpose(0,2)
            # show_img = show_img.transpose(0,1)
            # show_img.numpy()
            # plt.subplot(1,2,2)
            # plt.imshow(show_img)
            # plt.show()

            if epoch == 0:
                if i == 0:
                    writer.add_graph(net, inputs)

            # forward
            optimizer.zero_grad()
            outputs = net(inputs)

            # compute loss ("2" is an experiential default)
            loss = scal_KLD * criterion_KLD(outputs, maps) + scal_CC * criterion_CC(outputs, maps)\
                   + scal_NSS * criterion_NSS(outputs, fixs) + 2

            # compute metric
            metric = criterion_DC(outputs, maps)

            # backward
            loss.backward()
            optimizer.step()

            # statistics for predicted information with varied LOSS
            correct += metric.item()
            total += 1
            loss_sigma += loss.item()

            if i % 1 == 0:
                loss_avg = loss_sigma / 1
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, nb_epoch, i + 1, len(train_loader), loss_avg, correct/total))

                # record training loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # record learning rate
                writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
                # record accuracy
                writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

            # model visualization
            if i % 19 == 0:

                # visualize the inputs, maps and outputs
                show_inputs = make_grid(inputs)
                show_maps = make_grid(maps)
                show_outputs = make_grid(outputs)
                writer.add_image('Input_group', show_inputs)
                writer.add_image('Map_group', show_maps)
                writer.add_image('Out_group', show_outputs)

                # visualize the important feature maps (512*30*40) between the drn and convLSTM
                x_show = inputs
                for net_part_name, net_part_layer in net._modules.items():
                    if net_part_name == 'dcn':
                        x_show = net_part_layer(x_show)
                    else:
                        break
                x_show_fb = x_show[0]
                x_show_fb.unsqueeze_(1)
                show_feature = make_grid(x_show_fb) # just show the features of the first batch
                writer.add_image('Feature_group', show_feature)


        # record grads and weights
        for name, layer in net.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        if epoch % 1 == 0:
            loss_sigma = 0.0
            net.eval()
            for i, data in enumerate(valid_loader):

                images, maps, fixs = data
                images, maps, fixs = Variable(images), Variable(maps), Variable(fixs)

                # forward
                outputs = net(images)
                outputs.detach_()

                # compute loss ("2" is an experiential default)
                loss = scal_KLD * criterion_KLD(outputs, maps) + scal_CC * criterion_CC(outputs, maps) \
                       + scal_NSS * criterion_NSS(outputs, fixs) + 2

                # compute metric
                metric = criterion_DC(outputs, maps)

                loss_sigma += loss.item()
                correct_val += metric
                total_val += 1

                print("Validation: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                      format(epoch + 1, nb_epoch, i + 1, len(valid_loader), loss_sigma, correct_val / total_val))

                # record validation loss
                writer.add_scalars('Loss_group', {'valid_loss': loss_sigma}, epoch)
                # record validation accuracy
                writer.add_scalars('Accuracy_group', {'valid_acc': correct_val / total_val}, epoch)

    print('finished training !')


# ------------------------------------ step 5 : model saving ------------------------------------------------

    net_save_path = os.path.join(log_dir, 'net_params.pkl')
    torch.save(net.state_dict(), net_save_path)

    # the end
    print('job done !')

    # debug (to check the model's graph)
    # for i, data in enumerate(train_loader):
    #     inputs, maps = data
    #     inputs, maps = Variable(inputs), Variable(maps)
    #     if i == 0:
    #         with writer:
    #             writer.add_graph(net, inputs)
    #         print('successful debugging')
    #
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #
    #     loss = -1 * (scal_KLD * criterion_KLD(outputs, maps) + scal_CC * criterion_CC(outputs, maps))
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     break