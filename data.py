import numpy as np 
import pandas as pd 
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import os
import random
from scipy import ndimage
import cv2
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import viewer

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class AdvDataset(Dataset):
    def __init__(self, root_dir, hmb_list, transform=None, type_='train'):
        self.root_dir = root_dir
        self.hmb_list = hmb_list  
        self.type = type_
        self.data = self.get_data()
        self.transform = transform
    
    def get_data(self):
        data_info = []
        if self.type == 'train':
            for hmb in self.hmb_list:
                df = pd.read_csv(hmb + '.csv')
                start_name = int(df['image_name'][0][:-4])
                if 'opt_attack' in self.root_dir:
                    name_list = [str(start_name + i*64) + '.npy' for i in range(len(df))]
                else:
                    name_list = [str(start_name + i) + '.npy' for i in range(len(df))]
                df['image_name'] = name_list
                data_info.append(df)
            return pd.concat(data_info)

        elif self.type == 'test':
            df = pd.read_csv('ch2_final_eval.csv')
            start_name = int(df['frame_id'][0])
            name_list = [str(start_name + i) + '.npy' for i in range(len(df))]
            df['frame_id'] = name_list
            # df['frame_id'] = df['frame_id'].apply(str)
            return df

    def __len__(self):
        # return len(self.images)
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        # image_name = str(self.images.iloc[idx, 4][7:])

        if self.type == 'train':
            image_name = self.data.iloc[idx, 0]
            steer = self.data.iloc[idx, 1]
            hmb = self.data.iloc[idx, 2]
            # # hmb = self.images.iloc[idx, 1]
            img_address = os.path.join(self.root_dir, hmb, image_name)
            # image = io.imread(img_address)
            image = np.load(img_address)
            # print(steer)

            sample = {'image': image, 'steer': np.array([steer])}
        else:
            image_name = self.data.iloc[idx, 0]
            steer = self.data.iloc[idx, 1]            
            img_address = os.path.join(self.root_dir, self.hmb_list[0], 'npy', image_name)
            image = np.load(img_address)
            sample = {'image': image, 'steer': np.array([steer])}            
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class UdacityDataset(Dataset):
    
    def __init__(self, root_dir, hmb_list, transform=None, type_='train'):
        self.root_dir = root_dir
        self.hmb_list = hmb_list
        self.type = type_
        self.data = self.get_data()
        # self.images = self.create_image_list()
        # print(len(self.images))
        # self.steering = self.create_steering_list()
        # self.filter_images()
        # print(len(self.images))

        # self.images, self.steering = self.create_image_and_steering_list()
        self.transform = transform
    


    def create_image_list(self):
        image_csv_list = []
        for hmb in self.hmb_list:
            df = pd.read_csv(os.path.join(self.root_dir, hmb, 'camera.csv'))
            df = df[df['filename'].str.contains('center')]
            df['frame_id'] = df['frame_id'].apply(str)
            df['HMB'] = hmb.upper()
            image_csv_list.append(df)

        return pd.concat(image_csv_list) 
    
    def create_steering_list(self):
        steering_csv_list = []
        if self.type == 'train':
            for hmb in self.hmb_list:
                df = pd.read_csv(os.path.join(self.root_dir, hmb, 'steering.csv'))
                df['timestamp'] = df['timestamp'].apply(str)
                steering_csv_list.append(df)

            return pd.concat(steering_csv_list)
        elif self.type == 'test':
            df = pd.read_csv('ch2_final_eval.csv')
            df['frame_id'] = df['frame_id'].apply(str)
            return df

    def get_data(self):
        data_info = []
        if self.type == 'train':
            for hmb in self.hmb_list:
                df = pd.read_csv(hmb + '.csv')
                data_info.append(df)
            return pd.concat(data_info)
            
        elif self.type == 'test':
            df = pd.read_csv('ch2_final_eval.csv')
            df['frame_id'] = df['frame_id'].apply(str)
            return df

    def create_image_and_steering_list(self):
        image_csv_list = []
        for hmb in self.hmb_list:
            df = pd.read_csv(os.path.join(self.root_dir, hmb, 'camera.csv'))
            df = df[df['filename'].str.contains('center')]
            df['frame_id'] = df['frame_id'].apply(str)
            df['HMB'] = hmb.upper()
            image_csv_list.append(df)
        image_list = pd.concat(image_csv_list)    

        steering_csv_list = []
        for hmb in self.hmb_list:
            df = pd.read_csv(os.path.join(self.root_dir, hmb, 'steering.csv'))
            df['timestamp'] = df['timestamp'].apply(str)
            steering_csv_list.append(df) 
        
        steer_list = pd.concat(steering_csv_list)
        # new_image_list = pd.DataFrame({'image_name':[], 'HMB':[]})


        return image_list, steer_list

    def __len__(self):
        # return len(self.images)
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        # image_name = str(self.images.iloc[idx, 4][7:])

        if self.type == 'train':
            image_name = self.data.iloc[idx, 0]
            steer = self.data.iloc[idx, 1]
            hmb = self.data.iloc[idx, 2]
            # # hmb = self.images.iloc[idx, 1]
            img_address = os.path.join(self.root_dir, hmb, 'center', image_name)
            image = io.imread(img_address)
            # print(steer)

            sample = {'image': image, 'steer': np.array([steer])}
        elif self.type == 'test':
            image_name = self.data.iloc[idx, 0] + '.jpg'
            steer = self.data.iloc[idx, 1]            
            img_address = os.path.join(self.root_dir, self.hmb_list[0], 'center', image_name)
            image = io.imread(img_address)
            sample = {'image': image, 'steer': np.array([steer])}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        image = sample['image']
        image = image[200:,:]
        image = cv2.resize(image, self.output_size)
        return {'image': image, 'steer': sample['steer']}

class Preprocess(object):

    def __init__(self, model_name='baseline'):
        self.model_name = model_name
    
    def __call__(self, sample):
        image = sample['image']
        # if self.model_name == 'baseline':
        image = image / 255. 
        # elif self.model_name == 'nvidia':
        #     image = image / 255. - 0.5
        
        return {'image': image, 'steer': sample['steer']}

class RandFlip(object):
    
    def __call__(self, sample):
        if random.randint(0, 1) == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['steer'] = - sample['steer']
        
        return sample

class RandBrightness(object):

    def __call__(self, sample):
        img = sample['image']
        if random.randint(0, 3) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = 1 + np.random.uniform(-0.2, 0.2)
            img[:, :, 2] = img[:, :, 2] * brightness
            img[:, :, 2][img[:, :, 2] > 255] = 255
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return {'image': img, 'steer': sample['steer']}
        
class RandRotation(object):
    def __call__(self, sample):
        rotate = random.uniform(-1, 1)
        sample['image'] = ndimage.rotate(sample['image'], rotate, reshape=False)
        return sample

class RandRotateView(object):
    def __call__(self, sample):
        if random.randint(0, 3) == 1 and abs(sample['steer']) < 0.3:
            rotate = random.uniform(-30, 30)
            delta_angle = rotate / 60
            sample['image'] = ndimage.rotate(sample['image'], rotate, reshape=False)
            sample['steer'] = sample['steer'] + delta_angle
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, steer = sample['image'], sample['steer']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'steer': torch.from_numpy(steer)}






if __name__ == "__main__":
    # test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    # test_dataset = UdacityDataset('C:\\Users\\45040206\\Documents\\udacity-data', ['testing'], test_composed, 'test')
    # sample = test_dataset[10]
    # print(sample['image'].size(), sample['steer'].size())
    composed = transforms.Compose([Rescale((128, 128)),  RandRotation(), Preprocess('baseline')])
    dataset = UdacityDataset('/media/dylan/Program/cg23-dataset', ['HMB1','HMB6'], transform=composed)
    print(len(dataset))
    dataloader = DataLoader(dataset, 4, False)
    sample = dataset[2000]
    print(sample['steer'])
    # ax = plt.subplot(1, 3, 1)
    # plt.imshow()
    # ax = plt.subplot(1, 3, 2)
    # plt.imshow(ndimage.rotate(sample['image'], 80, reshape=False))
    # ax = plt.subplot(1, 3, 3)
    # plt.imshow(ndimage.rotate(sample['image'], -80, reshape=False))
    fig = plt.figure()
    plt.imshow(cv2.resize(viewer.draw(ndimage.rotate(sample['image'], 30, reshape=False), sample['steer'], 0.5), (128, 128)))
    plt.show()
    # fig = plt.figure()
    # for i in range(4):
    #     j = random.randint(0, len(dataset))
    #     sample = dataset[j]
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(sample['image'])
    #     print(sample['steer'])
    #     if i == 3:
    #         plt.show()
    #         break