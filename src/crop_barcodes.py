#crop images

import os
import pandas as pd
import cv2

DATA_FOLDER = './data'

df = pd.read_csv("./data/annotations.tsv", delimiter='\t')
for i in range(len(df)):
    image = cv2.imread(os.path.normpath(os.path.join(DATA_FOLDER, 'images_ini/', f"{df['filename'][i].split('/')[1]}")))[..., ::-1]
    x1 = int(df['x_from'][i])
    y1 = int(df['y_from'][i])
    x2 = int(df['x_from'][i]) + int(df['width'][i])
    y2 = int(df['y_from'][i]) + int(df['height'][i])
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, 2)
    cv2.imwrite(os.path.normpath((os.path.join(DATA_FOLDER, 'images/',  f"{df['filename'][i].split('/')[1]}"))), crop)