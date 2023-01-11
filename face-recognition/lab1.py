import scipy.io
import os
from os import listdir
from os.path import join, isfile
import csv
import cv2
from typing import NamedTuple

class Image(NamedTuple):
    image: int
    csv_num: int

file_2 = open('/home/anastasia/Έγγραφα/biometrics/lab1/caltech/caltech_labels.csv')

type(file_2)

csvreader = csv.reader(file_2)

header = []
header = next(csvreader)
header

times = 0
value = header

#i save in the values array the values of the csv file of the images that appear more that 20 times in our file 
rows = []
values = []
for row in csvreader:
    rows.append(row)
    if value == row:
        times = times + 1
    else:   
        if times >= 20:
            values.append(value)
        times = 1
        value = row
rows

print(values)
print('\n')
for x in values:
    print(x)

file = open('/home/anastasia/Έγγραφα/biometrics/lab1/caltech/caltech_labels.csv')

type(file)

csvreader = csv.reader(file)

header = []
header = next(csvreader)
header

path = '/home/anastasia/Έγγραφα/biometrics/lab1/caltech/'
rows = []
i = 0

for file in listdir('/home/anastasia/Έγγραφα/biometrics/lab1/caltech'):
    print(file)
    filename = os.fsdecode(file)
    if(filename[0] != 'i'):
        continue
    else:
        num = filename[6:10]
        for row2 in csvreader:
            for index in values:
                if index == row2:
                    image = path + file
                    image = cv2.imread(image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('Grayscale', gray)
                    cv2.waitKey(0) 
            
                    cv2.destroyAllWindows()  
                    break  

file_2.close()

mat = scipy.io.loadmat('/home/anastasia/Έγγραφα/biometrics/lab1/caltech/ImageData.mat')

train_image = []
train_labels = []
test_images = []
test_labels = []


               
