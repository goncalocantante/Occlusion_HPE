import numpy as np
import cv2
import os
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import scipy.io as sio
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import utils
import random

class Dataset_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, image_dir, mat_dir,transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.image_dir = image_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        #self.xi_dir = xi_dir

        #filename_list = get_list_from_filenames(filename_path)
        self.mat_dir = mat_dir

        #self.X_train = filename_list
        #self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(image_dir)
    

    def __getitem__(self, index):
        img = Image.open(self.image_dir[index])
        img = img.convert(self.image_mode)
        mat_path = self.mat_dir[index]
        #xi_path = self.xi_dir[index]
        #xi = utils.get_xi_from_mat(xi_path)
        #Xi = torch.FloatTensor(xi)

        #mat = sio.loadmat(self.xi_dir)
        #Xi_values = mat['Xi_values']
        #Xi = Xi_values[index][0]
        #Xi = torch.FloatTensor(Xi)
        # Crop the face loosely
        #pt2d = utils.get_pt2d_from_mat(mat_path)
        #x_min = min(pt2d[0,:])
        #y_min = min(pt2d[1,:])
        #x_max = max(pt2d[0,:])
        #y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        #k = np.random.random_sample() * 0.2 + 0.2
        #x_min -= 0.6 * k * abs(x_max - x_min)
        #y_min -= 2 * k * abs(y_max - y_min)
        #x_max += 0.6 * k * abs(x_max - x_min)
        #y_max += 0.6 * k * abs(y_max - y_min)
        #img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        #image = img

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        #rnd = np.random.random_sample()
        #if rnd < 0.5:
        #    yaw = -yaw
        #    roll = -roll
        #    img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        #rnd = np.random.random_sample()
        #if rnd < 0.05:
        #    img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        #transformations = transforms.Compose([transforms.Resize((240,240)), transforms.ToTensor()])

        if self.transform is not None:
            img = self.transform(img)

        #image = transformations(image)
        #image = image.resize((240,240))
        #image = np.array(image)
        

        return img, labels, cont_labels #, Xi #, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length


class Dataset_AFLW2000(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, image_dir, mat_dir, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.image_dir = image_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        #filename_list = get_list_from_filenames(filename_path)
        self.mat_dir = mat_dir

        #self.X_train = filename_list
        #self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(image_dir)
    

    def __getitem__(self, index):
        img = Image.open(self.image_dir[index])
        img = img.convert(self.image_mode)
        mat_path = self.mat_dir[index]
        # Crop the face loosely
        #pt2d = utils.get_pt2d_from_mat(mat_path)
        #x_min = min(pt2d[0,:])
        #y_min = min(pt2d[1,:])
        #x_max = max(pt2d[0,:])
        #y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        #k = np.random.random_sample() * 0.2 + 0.2
        #x_min -= 0.6 * k * abs(x_max - x_min)
        #y_min -= 2 * k * abs(y_max - y_min)
        #x_max += 0.6 * k * abs(x_max - x_min)
        #y_max += 0.6 * k * abs(y_max - y_min)
        #img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        #image = img

        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        #transformations = transforms.Compose([transforms.Resize((240,240)), transforms.ToTensor()])

        if self.transform is not None:
            img = self.transform(img)

        #image = transformations(image)
        #image = image.resize((240,240))
        #image = np.array(image)
        

        return img, labels, cont_labels #, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class Dataset_BIWI(Dataset):
    def __init__(self, image_dir, pose_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.image_dir = image_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        #filename_list = get_list_from_filenames(filename_path)
        self.pose_path = pose_path
        #self.X_train = filename_list
        #self.y_train = filename_list
        self.image_mode = image_mode
        #self.length = len(filename_list)
        self.length = len(pose_path)

    def __getitem__(self, index):
        #img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
        #img = img.convert(self.image_mode)

        img = Image.open(self.image_dir[index])
        img = img.convert(self.image_mode)
        pose = self.pose_path[index]

        #y_train_list = self.y_train[index].split('/')
        #bbox_path = os.path.join(self.data_dir, y_train_list[0] + '/dockerface-' + y_train_list[-1] + '_rgb' + self.annot_ext)

        # Load pose in degrees
        pose_annot = open(pose, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)

        R = np.array(R)
        T = R[3,:]
        R = R[:3,:]
        pose_annot.close()

        R = np.transpose(R)

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        # Loosely crop face
        #k = 0.35
        #x_min -= 0.6 * k * abs(x_max - x_min)
        #y_min -= k * abs(y_max - y_min)
        #x_max += 0.6 * k * abs(x_max - x_min)
        #y_max += 0.6 * k * abs(y_max - y_min)
        #img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        #cont_labels_translation = torch.FloatTensor([t1, t2, t3])
        #cont_labels = torch.tensor([yaw, pitch, roll], dtype=torch.float) #VER SE ISTO FUNCIONA

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels #self.X_train[index], cont_labels_translation

    def __len__(self):
        # 15,667
        return self.length
