
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from config import nb_timestep, epsilon_DC, b_s, shape_r_out, shape_c_out, \
    shape_r_f, shape_c_f, nb_gaussian

#-----------------------------------------------------------------------------------------------------------------------
# define the main frame of the dilated ResNet 50
class MyDRN(nn.Module):

    def __init__(self):
        super(MyDRN, self).__init__()

        # conv_1
        self.zeropad = nn.ZeroPad2d(padding=3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                 padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # to save the computing memory: inplace=True
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv_2
        self.myConvBlock1 = MyConvBlock(kernel_size=3, filters=[64, 64, 256], stride=1, in_channel=64,
                                        dilation=1, padding=1)
        self.myIdentityBlock1 = MyIdentityBlock(kernel_size=3, filters=[64, 64, 256], in_channel=256,
                                                dilation=1, padding=1)
        self.myIdentityBlock2 = MyIdentityBlock(kernel_size=3, filters=[64, 64, 256], in_channel=256,
                                                dilation=1, padding=1)

        # conv_3
        self.myConvBlock2 = MyConvBlock(kernel_size=3, filters=[128, 128, 512], stride=2, in_channel=256,
                                        dilation=1, padding=1)
        self.myIdentityBlock3 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)
        self.myIdentityBlock4 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)
        self.myIdentityBlock5 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)

        # conv_4
        self.myConvBlock3 = MyConvBlock(kernel_size=3, filters=[256, 256, 1024], stride=1, in_channel=512,
                                        dilation=2, padding=2)
        self.myIdentityBlock6 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock7 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock8 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock9 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock10 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                 dilation=2, padding=2)

        # conv_5
        self.myConvBlock4 = MyConvBlock(kernel_size=3, filters=[512, 512, 2048], stride=1, in_channel=1024,
                                        dilation=4, padding=4)
        self.myIdentityBlock11 = MyIdentityBlock(kernel_size=3, filters=[512, 512, 2048], in_channel=2048,
                                                 dilation=4, padding=4)
        self.myIdentityBlock12 = MyIdentityBlock(kernel_size=3, filters=[512, 512, 2048], in_channel=2048,
                                                 dilation=4, padding=4)

        # conv_feat
        self.convfeat = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False)

    def forward(self, x):

        # layer_1
        x = self.zeropad(x) # [8,3,240,320] -> [8,3,246,326]
        x = self.conv1(x) # [8,3,246,326] -> [8,64,120,160]
        x = self.bn1(x) # size equal
        x = self.relu(x) # size equal
        x = self.maxpool1(x) # [8,64,120,160] -> [8,64,60,80]

        # layer_2
        x = self.myConvBlock1(x)  # size equal
        x = self.myIdentityBlock1(x) # size equal
        x = self.myIdentityBlock2(x) # size equal

        # layer_3
        x = self.myConvBlock2(x) # [8,64,60,80] -> [8,512,30,40]
        x = self.myIdentityBlock3(x) # size equal
        x = self.myIdentityBlock4(x) # size equal
        x = self.myIdentityBlock5(x) # size equal

        # layer_4
        x = self.myConvBlock3(x) # [8,512,30,40] -> [8,1024,30,40]
        x = self.myIdentityBlock6(x) # size equal
        x = self.myIdentityBlock7(x) # size equal
        x = self.myIdentityBlock8(x) # size equal
        x = self.myIdentityBlock9(x) # size equal
        x = self.myIdentityBlock10(x) # size equal

        # layer_5
        x = self.myConvBlock4(x) # [8,1024,30,40] -> [8,2048,30,40]
        x = self.myIdentityBlock11(x) # size equal
        x = self.myIdentityBlock12(x) # size equal

        # layer_output
        x = self.convfeat(x) # [8,2048,30,40] -> [8,512,30,40]

        return x


#-----------------------------------------------------------------------------------------------------------------------
# deine the accessories for the proposed DRN frame
class MyIdentityBlock(nn.Module):

    def __init__(self,kernel_size, filters, in_channel, dilation, padding):
        super(MyIdentityBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter1, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_filter2)
        self.conv3 = nn.Conv2d(in_channels=nb_filter2, out_channels=nb_filter3, kernel_size=1, stride=1,
                            padding=0, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_filter3)

    def forward(self, x):

        Input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = Input+x
        x = self.relu(x)

        return x


class MyConvBlock(nn.Module):

    def __init__(self, kernel_size, filters, stride, in_channel, dilation, padding):
        super(MyConvBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter1, kernel_size=1, stride=stride,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_filter2)
        self.conv3 = nn.Conv2d(in_channels=nb_filter2, out_channels=nb_filter3, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_filter3)
        self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter3, kernel_size=1, stride=stride,
                                  padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):

        y = self.shortcut(x)
        y = self.bn3(y)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = y+x
        x = self.relu(x)

        return x


#----------------------------------------------------------------------------------------------------------------------
# define the main frame of the attentive convLSTM
class MyAttentiveLSTM(nn.Module):

    def __init__(self, nb_features_in, nb_features_out, nb_features_att, nb_rows, nb_cols):
        super(MyAttentiveLSTM, self).__init__()

        # define the fundamantal parameters
        self.nb_features_in = nb_features_in
        self.nb_features_out = nb_features_out
        self.nb_features_att = nb_features_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

        # define convs
        self.W_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=self.nb_features_att,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_a = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_att,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.V_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=1,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=False)

        self.W_i = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_i = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_f = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_f = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_c = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_c = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_o = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_o = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        # define activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # define number of temporal steps
        self.nb_ts = nb_timestep

    def forward(self, x):

        # gain the current cell memory and hidden state
        h_curr = x
        c_curr = x

        for i in range(self.nb_ts):

            # the attentive model
            my_Z = self.V_a(self.tanh(self.W_a(h_curr) + self.U_a(x)))
            my_A = self.softmax(my_Z)
            AM_cL = my_A * x

            # the convLSTM model
            my_I = self.sigmoid(self.W_i(AM_cL) + self.U_i(h_curr))
            my_F = self.sigmoid(self.W_f(AM_cL) + self.U_f(h_curr))
            my_O = self.sigmoid(self.W_o(AM_cL) + self.U_o(h_curr))
            my_G = self.tanh(self.W_c(AM_cL) + self.U_c(h_curr))
            c_next = my_G * my_I +  my_F * c_curr
            h_next = self.tanh(c_next) * my_O

            c_curr = c_next
            h_curr = h_next

        return h_curr


# define the non local neural network block
class MyNonLocBlock(nn.Module):
    pass


# define the learnable priors
class MyPriors(nn.Module):

    def __init__(self, gp):
        super(MyPriors, self).__init__()
        self.conv5_5 = nn.Conv2d(in_channels=(512+nb_gaussian), out_channels=512, kernel_size=5,
                               stride=1, padding=8, dilation=4, groups=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.gp = gp

    def forward(self, x):

        # cancate the generated gaussian priors with the original inputs
        gp_seq = []
        for c in range(b_s):
            gp_seq.append(self.gp)
        gp_seq = torch.cat(gp_seq, 0)
        x = torch.cat([gp_seq, x], 1)

        # the conv 5*5 (to learn the feature of center bias from generated gaussian kernels)
        x = self.conv5_5(x)
        x = self.relu(x)

        return x

def generate_gaussian_prior():

    # initialize the gaussian feature clip
    gp = torch.zeros(nb_gaussian, shape_r_f, shape_c_f)

    # initialize the gaussian distribution along x and y in each of the feature  map
    f_r = torch.zeros(shape_r_f, nb_gaussian)
    f_c = torch.zeros(shape_c_f, nb_gaussian)

    # lock the generated parameters
    random.seed(666)

    # randomly initialize the delta and mean for the gaussian prior
    delta_x = torch.zeros(nb_gaussian, 1)
    delta_y = torch.zeros(nb_gaussian, 1)
    for r in range(nb_gaussian):
        delta_x[r, 0] = random.uniform(0.1, 0.9)
        delta_y[r, 0] = random.uniform(0.2, 0.8)
    sigma_x = delta_x * (0.3 * ((shape_r_f - 1) * 0.5 - 1) + 0.8)
    sigma_y = delta_y * (0.3 * ((shape_c_f - 1) * 0.5 - 1) + 0.8)

    # randomly initialize gaussian distribution along X and Y in each of the feature map
    for gr in range(nb_gaussian):
        gaussian_cur = cv2.getGaussianKernel(shape_r_f, sigma_x[gr, 0])
        gaussian_cur = torch.from_numpy(gaussian_cur)
        f_r[:, gr] = gaussian_cur[:, 0]
    for gc in range(nb_gaussian):
        gaussian_cur = cv2.getGaussianKernel(shape_c_f, sigma_y[gc, 0])
        gaussian_cur = torch.from_numpy(gaussian_cur)
        f_c[:, gc] = gaussian_cur[:, 0]

    # generate each of the gaussian map
    for j in range(nb_gaussian):
        for m in range(shape_r_f):
            for n in range(shape_c_f):
                gp[j, m, n] = f_r[m, j] * f_c[n, j]

    gp.unsqueeze_(0)

                # another way to gain the gp
                # gp[j,m,n] = 1 / (2 * np.pi * delta_x[j,0] * delta_y[j,0]) \
                #     * torch.exp((-1) * ((f_r[m,j]-mean_x[j,0])*(f_r[m,j]-mean_x[j,0])/(2*delta_x[j,0]*delta_x[j,0])
                #                 +(f_c[n,j]-mean_y[j,0])*(f_c[n,j]-mean_y[j,0])/(2*delta_y[j,0]*delta_y[j,0])))

                # debug
                # plt.figure
                # show_01 = gp[0,:,:]
                # show_01.numpy()
                # plt.subplot(1,4,1)
                # plt.imshow(show_01, cmap='gray')
                # show_02 = gp[5,:,:]
                # show_02.numpy()
                # plt.subplot(1,4,2)
                # plt.imshow(show_02, cmap='gray')
                # show_03 = gp[9,:,:]
                # show_03.numpy()
                # plt.subplot(1,4,3)
                # plt.imshow(show_03, cmap='gray')
                # show_04 = gp[13,:,:]
                # show_04.numpy()
                # plt.subplot(1,4,4)
                # plt.imshow(show_04, cmap='gray')
                # plt.show()

    return gp


# ---------------------------------------------------------------------------------------------------------------------
# define the dice coefficient
class MyDiceCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyDiceCoef, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1) # [8,1,240.320] -> [8, 76800]
        target = target.view(target.size(0), -1) # [8,1,240.320] -> [8, 76800]

        DC = []

        for i in range(b_s):
            DC.append((2 * torch.sum(input[i] * target[i]) + epsilon_DC) / \
             (torch.sum(input[i]) + torch.sum(target[i]) + epsilon_DC))
            DC[i].unsqueeze_(0)

        DC = torch.cat(DC, 0)
        DC = torch.mean(DC)

        return DC

# define the correlation coefficient
class MyCorrCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyCorrCoef, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        CC = []

        for i in range(b_s):
            im = input[i] - torch.mean(input[i])
            tm = target[i] - torch.mean(target[i])

            CC.append(torch.sum(im * tm) / (torch.sqrt(torch.sum(im ** 2))
                                            * torch.sqrt(torch.sum(tm ** 2))))
            CC[i].unsqueeze_(0)

        CC = torch.cat(CC,0)
        CC = torch.mean(CC)

        return CC

class MyNormScanSali(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyNormScanSali, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        NSS = []
        target_logic = torch.zeros(b_s, shape_r_out * shape_c_out )

        for i in range(b_s):

            # normalize the predicted maps
            input_norm = (input[i] - torch.mean(input[i])) / torch.std(input[i])

            # compute the logic matrix of fixs
            for m in range(shape_r_out * shape_c_out):
                if target[i,m] != 0:
                    target_logic[i,m] = 1

            NSS.append(torch.mean(torch.mul(input_norm, target_logic[i])))
            NSS[i].unsqueeze_(0)

        NSS = torch.cat(NSS, 0)
        NSS = torch.mean(NSS)

        return NSS