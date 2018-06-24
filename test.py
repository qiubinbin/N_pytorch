import pandas as pd
import csv
import os
from skimage import io,transform
from PIL import Image
import matplotlib.pyplot as plt


# # print(file1[0,0])
# print(os.path.join('qiu','bin'))
#
# file2=open('faces/face_landmarks.csv')
# print(list(csv.reader(file2))[0][0])
# import matplotlib.pyplot as plt
# from PIL import  Image
# f=Image.open('faces/britney-bald.jpg')
# plt.imshow(f)
# plt.show()
file2 = pd.read_csv('faces/face_landmarks.csv')
file1=io.imread('faces/'+file2.iloc[0,0])
# print(file1.shape)
# print(file1.transpose(2,0,1).shape)
# print(file2.iloc[0,1],file2.iloc[0,2])
plt.imshow(transform.resize(file1,(500,500),mode='constant'))
# plt.scatter(file2.iloc[0,1],file2.iloc[0,2])
#
plt.show()
