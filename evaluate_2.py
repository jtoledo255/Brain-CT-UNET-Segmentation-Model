from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from pyimagesearch import config
from scipy.spatial.distance import directed_hausdorff
import cv2
import math
import sys
from skimage import metrics
import matplotlib.pyplot as plt
import pandas as pd
from pydicom import dcmread
def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def group_cases_np(groundtruth_path,prediction_path):
    cases = []
    for i in os.listdir(groundtruth_path):
        name_split = i.split('_')
        if name_split[0] not in cases:
            cases.append(name_split[0])
    ground_dir = os.listdir(groundtruth_path)
    ground_dir.sort()

    prediction_dir = os.listdir(prediction_path)
    prediction_dir.sort()
    test_ls = []
    another_empty_stack = []
    for j in cases:
        empty_stack = []
        for i in ground_dir:
            if j in i:
                gtMask = Image.open(groundtruth_path+i).convert('1')
                np_gtMask = np.array(gtMask)
                empty_stack.append(np_gtMask)

        test_ls.append(np.stack(empty_stack))
    another_empty_stack.append(test_ls)
    test_ls = []
    prediction_empty_stack = []
    for j in cases:
        empty_stack = []
        for i in ground_dir:
            if j in i:
                prediction_Mask = Image.open(prediction_path+i).convert('1')
                np_prediction_Mask = np.array(prediction_Mask)
                empty_stack.append(np_prediction_Mask)

        test_ls.append(np.stack(empty_stack))
    prediction_empty_stack.append(test_ls)
    return another_empty_stack, prediction_empty_stack

def threeddice(gt_vol,pred_vol):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_vol == 1, gt_vol == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_vol == 0, gt_vol == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_vol == 1, gt_vol == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_vol == 0, gt_vol == 1))
    DSC = 2*TP/(2*TP + FP + FN)

    return DSC

def get_coordinates(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def HD(coord_1,coord_2):
    #loop through the coordinates

    min_dist_ls = []
    for i in coord_1:
        for j in i: #j is the coordinate in set 1

            dist_ls = sys.maxsize
            for k in coord_2:
                for l in k: # l is the coordinate in set 2
                    dist = np.absolute(math.dist(j[0],l[0]))
                    if dist < dist_ls:
                        dist_ls = dist
            min_dist_ls.append(dist_ls)

    min_dist_ls_2 = []
    for i in coord_2:
        for j in i: #j is the coordinate in set 2
            dist_ls = sys.maxsize
            for k in coord_1:
                for l in k: # l is the coordinate in set 1
                    dist = np.absolute(math.dist(j[0],l[0]))
                    if dist < dist_ls:
                        dist_ls = dist
            min_dist_ls_2.append(dist_ls)
    max_dist_1 = np.max(min_dist_ls)
    max_dist_2 = np.max(min_dist_ls_2)

    haus = np.maximum(max_dist_1,max_dist_2)
    return haus
def get_coordinates2(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list_coord = []
    for i in contours:
        for j in i:
            list_coord.append(j[0])

    return list_coord

def HD2(coord_1,coord_2):
    #loop through the coordinates

    min_dist_ls = []

    for j in coord_1: #j is the coordinate in set 1

        dist_ls = sys.maxsize

        for l in coord_2: # l is the coordinate in set 2
            dist = np.absolute(math.dist(j,l))
            if dist < dist_ls:
                dist_ls = dist
        min_dist_ls.append(dist_ls)

    min_dist_ls_2 = []

    for j in coord_2: #j is the coordinate in set 2
        dist_ls = sys.maxsize
        for l in coord_1: # l is the coordinate in set 1
            dist = np.absolute(math.dist(j,l))
            if dist < dist_ls:
                dist_ls = dist
        min_dist_ls_2.append(dist_ls)
    max_dist_1 = np.max(min_dist_ls)
    max_dist_2 = np.max(min_dist_ls_2)

    haus = np.maximum(max_dist_1,max_dist_2)
    return haus
def findpixeldistance(case_id):
    
    if 'GSW' in case_id:
        data = pd.read_csv('/nfs/kitbag/data1/jtoledo/tbi_project/image_directory_PBI.csv')
        #anted_data = data[data['angio']==False]
        image_directory  = data[data['image_path'].str.contains(case_id)]
        if case_id == 'GSW-0055':
            ds = dcmread(os.path.join(image_directory['image_path'].tolist()[0],'00001_8aa5e72491bd5b79.dcm'))
        else:
            ds = dcmread(os.path.join(image_directory['image_path'].tolist()[0],os.listdir(image_directory['image_path'].tolist()[0])[0]))
        pixel_dis = ds.PixelSpacing[0]

    else:
        data = pd.read_csv('/nfs/kitbag/data1/jtoledo/tbi_project/image_directory_test2.csv')
        image_directory  = data[data['image_path'].str.contains(case_id)]
        ds = dcmread(os.path.join(image_directory['image_path'].tolist()[0],os.listdir(image_directory['image_path'].tolist()[0])[0]))
        pixel_dis = ds.PixelSpacing[0]

    return pixel_dis
test = '2modelc_PBI'


pred_path = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_prediction_images_'+test+'/'
pred_dir = os.listdir(pred_path)
pred_dir.sort()
#if j < 1:
gt_path = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_gt_images_'+test+'/'

gt_dir = os.listdir(gt_path)
gt_dir.sort()
gt_group,prediction_group = group_cases_np(pred_path,gt_path)

dice_3d = []
for i in range(len(prediction_group[0])):
    gt_vol = gt_group[0][i]
    prediction_vol = prediction_group[0][i]

    dsc = threeddice(gt_vol,prediction_vol)
    dice_3d.append(dsc)
dice = []
slice_num = []
haus_dis = []
haus_dis_mine = []
count = 0
case = 'test'
for i in range(len(gt_dir)):
    name_split = pred_dir[i].split('_')

    if case != name_split[0]:
        pixel_dis = findpixeldistance(name_split[0])
        case = name_split[0]
        # print(case)
        # print(pixel_dis)
    
    # if 'GSW' not in case:
    #     continue

    
    predMask = Image.open(pred_path+pred_dir[i]).convert('L')
    np_predMask = np.array(predMask)

    gtMask = Image.open(gt_path+gt_dir[i]).convert('L')
    np_gtMask = np.array(gtMask)

    if np_gtMask.max() == 0 and np_predMask.max() > 0:
        dice.append(0)
        #continue
    elif np_predMask.max() == 0 and np_gtMask.max() == 0:
        dice.append(1)
    else:

        dice.append(dice_metric(np.ma.make_mask(np_predMask),np.ma.make_mask(np_gtMask)))
        pred_coord = get_coordinates2(np_predMask)
        gt_coord = get_coordinates2(np_gtMask)

        try:
            #haus_dis_mine.append(HD2(pred_coord,gt_coord)*pixel_dis)

            haus_dis_mine.append((max(directed_hausdorff(pred_coord,gt_coord)[0], directed_hausdorff(gt_coord, pred_coord)[0]))*pixel_dis)
        except:
            print('error evaluating image', gt_path+gt_dir[i],'and',pred_path+pred_dir[i])
            print(i)
            print(j)


        slice_num.append(count)
    count = count + 1


savepath = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+config.DATE+'/'+config.ANATOMY+'/'+test+'final_results2.csv'
# Create an array with NaNs
dice_3d_expanded = np.full(154, np.nan)
dice_3d_expanded[:6] = dice_3d  # Fill only the first 6 rows

df = pd.DataFrame(np.column_stack([dice, dice_3d_expanded, haus_dis_mine]), 
                  columns=['2D DSC', '3D DSC', 'HD'])

df.to_csv(savepath)

print('DICE')
print(np.mean(dice))
print(np.std(dice))
print('3D Dice')
print(np.mean(dice_3d))
print(np.std(dice_3d))
print('HD')
print(np.mean(haus_dis_mine))
print(np.std(haus_dis_mine))
