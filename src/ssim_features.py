import os

import jpeglib
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

version_list = ['6b', '7', '8', '8a', '8b', '8c', '8d', '9', '9a', '9b', '9c', '9d', '9e', '9f', 'turbo120', 'turbo130',
                'turbo140', 'turbo150', 'turbo200', 'turbo210', 'mozjpeg101', 'mozjpeg201', 'mozjpeg300', 'mozjpeg403']
c0 = ['6b', 'turbo120', 'turbo130', 'turbo140', 'turbo150', 'turbo200', 'turbo210', 'mozjpeg101', 'mozjpeg201']
c1 = ['7', '8', '8a', '8b', '8c', '8d', '9', '9a', '9b', '9c', '9d']
c2 = ['9e', '9f']
c3 = ['mozjpeg300', 'mozjpeg403']
bsp = ['6b', '7', '9e', 'mozjpeg300']

def compare(img1, img2):
    img1_r = img1[:, :, 0]
    img1_g = img1[:, :, 1]
    img1_b = img1[:, :, 2]

    img2_r = img2[:, :, 0]
    img2_g = img2[:, :, 1]
    img2_b = img2[:, :, 2]

    (score_r, diff_r) = structural_similarity(img1_r, img2_r, full=True)
    (score_g, diff_g) = structural_similarity(img1_g, img2_g, full=True)
    (score_b, diff_b) = structural_similarity(img1_b, img2_b, full=True)

    image_diff_r = (1 - score_r) * 100
    image_diff_g = (1 - score_g) * 100
    image_diff_b = (1 - score_b) * 100
    image_diff_avg = (image_diff_r + image_diff_g + image_diff_b) / 3
    return image_diff_avg

def dataframe2arff(df, filepath):
    with open(filepath, "w") as arff_file:
        arff_file.write("@RELATION Name_Of_Data\n\n")
        for value in df.columns.values:
            if value == "LABEL":
                arff_file.write("@ATTRIBUTE {} ".format(value) + '{')
                i = len(df[value].unique())
                for class_label in df[value].unique():
                    i -= 1
                    if i == 0:
                        arff_file.write("{}".format(class_label) + '}' + "\n\n")
                    else:
                        arff_file.write("{},".format(class_label))
            else:
                val_type = "NUMERIC"
                if df[value].dtype == 'O':
                    val_type = "string"
                arff_file.write("@ATTRIBUTE {} {}\n".format(value, val_type))
        arff_file.write("@DATA\n")
        arff_file.write(df.to_csv(header=False, index=False))

#path = "/home/dsiegel/Pictures/Logitech_Brio210500/"
#path = "/mnt/c/pitsec-jpeg-fingerprinting/data/compressed"
path = "data/compressed"
#temp_path = "/mnt/c/pitsec-jpeg-fingerprinting/temp.jpeg"
temp_path = "temp.jpeg"
#temp_path1 = "/mnt/c/pitsec-jpeg-fingerprinting/temp1.jpeg"
temp_path1 = "temp1.jpeg"
results = []
columns = ['diff_C0', 'diff_C1', 'diff_C2', 'diff_C3', 'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3', 'file', 'LABEL']
for file in tqdm.tqdm(os.listdir(path)[:5]):
    im = Image.open(os.path.join(path, file))
    for version in bsp:
        temp_list = []
        if version in c0:
            file_cluster = "C0"
        elif version in c1:
            file_cluster = "C1"
        elif version in c2:
            file_cluster = "C2"
        elif version in c3:
            file_cluster = "C3"
        else:
            file_cluster = version
        with jpeglib.version(version):
            b = np.asarray(im)
            c = jpeglib.from_spatial(b)
            c.write_spatial(temp_path)
        before = cv2.imread(temp_path)
        for version2 in bsp:
            with jpeglib.version(version2):
                a = Image.open(temp_path)
                # a = Image.open("data/alaska/00001.tif")
                b = np.asarray(a)
                c = jpeglib.from_spatial(b)
                c.write_spatial(temp_path1)
            after = cv2.imread(temp_path1)
            image_diff_avg = compare(before, after)
            temp_list.append(image_diff_avg)
        lowest = min(temp_list)
        highest = max(temp_list)
        for i in range(0,4):
            temp_list.append((temp_list[i] - lowest) / (highest - lowest))
        temp_list.append(file)
        temp_list.append(file_cluster)
        results.append(temp_list)

df = pd.DataFrame(results, columns=columns)
#df.to_csv("/mnt/c/pitsec-jpeg-fingerprinting/test_output/output_addLayer.csv")

df.to_csv("test_output/output_addLayer.csv", index=False)

#dataframe2arff(df, "/mnt/c/pitsec-jpeg-fingerprinting/test_output/data_addLayer.arff")

dataframe2arff(df, "test_output/data_addLayer.arff")