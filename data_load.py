import torch
import numpy as np
import os, math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import *

num_classes = 8
'''
csv 파일을 불러오고, 전처리하는 코드
'''

IMAGE_MAX = 751
TARGET_INDICES = 8

SHEAR_X_MEAN = -5.739277102506654e-05
SHEAR_X_VAR = 6.038777341161147e-06

SHEAR_Y_MEAN = 9.821450035845327e-05
SHEAR_Y_VAR = 4.545456837905018e-06

class ImageDataset(Dataset):
    def __init__(self, path='./data/train'):
        img = []
        sheer = []
        label = []

        for lab in os.listdir(path):
            for file_name in os.listdir(path + '/' + lab):
                # 파일 확장자가 .csv인 경우만 처리
                if file_name.endswith('.csv'):
                    file_path = os.path.join(path + '/' + lab, file_name)

                    sum_of_frame = 0
                    count = 0

                    f = open(file_path, 'r')
                    lines = f.readlines()
                    lines = [lst for lst in lines if lst != '\n' or lst != []]

                    for index in range(len(lines)):
                        lines[index] = lines[index].split('\n')[0]
                        lines[index] = lines[index].split(',')
                        lines[index] = [item for item in lines[index] if item.strip()]

                        frame = int(lines[index][0])
                        sum_of_frame += int(lines[index][0])
                        count += 1

                        img.append(list(map(int, lines[index][1:1 + 80 * frame])))
                        sheer.append(list(map(float, lines[index][1 + 80 * frame:])))
                        label.append(int(lab))
        # 이미지
        for index in range(0, len(img)):
            img[index] = np.array(img[index])  # 리스트를 NumPy 배열로 변환
            img[index] = img[index].reshape(int(len(img[index]) / 80), 80)

        # label
        label = torch.Tensor(label)
        self.label = F.one_hot(label.to(torch.int64), num_classes=num_classes)

        # sheer
        shear_temp = []
        for index in range(0, len(sheer)):
            temp = np.array(sheer[index]) # 일단 재가공하고자 하는 shear값만 미리 저장해 두고
            temp = temp.reshape(int(len(temp) / 3), 3) # 이 shear 값을 (frame, 3)의 크기로 만듦
            temp = np.delete(temp, 2, axis=1) # 세 번째 열 삭제 - z값 버림
            temp = self.interpolate_sheer_frames(temp, len(img[index])) # shear의 frame 길이를 img의 frame 길이만큼 늘림
            shear_temp.append(temp)

        self.img, self.shear, self.label = self.break_images_smaller(img, shear_temp, self.label, TARGET_INDICES, 2) # img와 shear를 8 frame으로 쪼갬
        self.img = torch.Tensor(self.img)
        self.img = self.img.reshape(len(self.img), TARGET_INDICES, 10, 8)
        self.preprocess() # img를 16*16사이즈 이미지로 resize & 0-1의 값으로 정규화

        self.shear = torch.Tensor(self.shear)

        self.shear[:, 0] -= SHEAR_X_MEAN
        self.shear[:, 0] /= math.sqrt(SHEAR_X_VAR)
        self.shear[:, 1] -= SHEAR_Y_MEAN
        self.shear[:, 1] /= math.sqrt(SHEAR_Y_VAR)

        # shear 정규화
        #self.sheer[:, :, 0] -= SHEAR_X_MEAN
        #self.sheer[:, :, 0] /= math.sqrt(SHEAR_X_VAR)
        #self.sheer[:, :, 1] -= SHEAR_Y_MEAN
        #self.sheer[:, :, 1] /= math.sqrt(SHEAR_Y_VAR)

        #self.sheer = torch.nan_to_num(self.sheer) # Nan값 없애기

        ##

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.img[index], self.label[index], self.shear[index]

    def preprocess(self):
        '''
        8*10 image를 16*16으로 늘림
        '''
        self.img = self.img.float()
        self.img /= IMAGE_MAX
        self.img = F.interpolate(self.img, size=(16,16), mode='bilinear')
        self.img = self.img.reshape(len(self.img), TARGET_INDICES, 1, 16, 16)

    def interpolate_sheer_frames(self, video, target_frame_count):
        """
        (원본 프레임 수, 이미지 크기)의 영상 데이터를
        (목표 프레임 수, 이미지 크기)로 선형 보간하는 함수.

        Args:
            video (list or np.ndarray): 원본 영상 데이터, shape (original_frame_count, image_size).
            target_frame_count (int): 목표 프레임 수 (기본값: 50).

        Returns:
            np.ndarray: 보간된 영상 데이터, shape (target_frame_count, image_size).
        """
        original_frame_count, image_size = video.shape

        # 원본 프레임 인덱스와 목표 프레임 인덱스 생성
        original_indices = np.linspace(0, original_frame_count - 1, num=original_frame_count)
        target_indices = np.linspace(0, original_frame_count - 1, num=target_frame_count)

        interpolated_video = np.zeros((target_frame_count, image_size))
        for i in range(image_size):
            # 각 이미지의 픽셀(세로 방향)을 개별적으로 보간
            interpolated_video[:, i] = np.interp(target_indices, original_indices, video[:, i])

        return interpolated_video

    def break_images_smaller(self, original_img, original_shear, original_label, target_indices, stride):
        '''
        frame을 target_indices로 쪼갬 (image, shear 모두 적용됨)

        만약 원본 프레임이 30 frame이고, 이를 8 frame씩 쪼갠다면
        (30 - 8) + 1 만큼으로 쪼개짐

        (original_indices - target_indices) + 1
        '''
        img = [] # temporary image list
        shear = []
        label = []
        for index in range(len(original_img)):
            frame = stride # 첫 번째 거 제외
            while frame <= len(original_img[index]) - target_indices:
                img.append(original_img[index][frame : frame + target_indices])
                temp_shear = [0, 0]
                temp_index = frame
                for i in range(stride):
                    temp_shear += (original_shear[index][temp_index] - original_shear[index][temp_index-1])
                    temp_index -= 1
                shear.append(temp_shear)

                label.append(original_label[index])
                frame += stride

        return img, shear, label


    def average_of_sheer_frames(self, sheer):
        '''
        shear 값을 이미지 프레임 수에 맞게 보간함
        shear가 8 frame이고 이미지가 30 frame이라면, shear를 30 frame에 맞게 보간
        '''
        video = np.array(sheer)  # 리스트를 NumPy 배열로 변환
        video = video.reshape(int(len(video) / 3), 3)

        averaged_sheer = []
        for i in range(3):
            # 각 이미지의 픽셀(세로 방향)을 개별적으로 보간
            averaged_sheer.append(np.mean(video[:, i]))

        averaged_sheer = np.array(averaged_sheer)
        return averaged_sheer

def get_na_df(df):
    '''
    데이터 프레임에 Nan이 있는지 확인 (디버깅용)
    return: true / false boolean 값
    '''
    na_df = df.isnan().sum()
    na_cols = na_df[na_df > 0].index
    return df[df.isna().any(axis=1)][na_cols]

ImageDataset()