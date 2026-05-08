# USAGE
# python train.py
# import the necessary packages
from sched import scheduler
from pyimagesearch.dataset25 import SegmentationDataset
from pyimagesearch.model2 import UNet2
#from pyimagesearch import config_2 as config
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import random
from torchsummary import summary
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import pickle
# load the image and mask filepaths in a sorted manner
#imagePaths = sorted(list(paths.list_images(config.IMAGE_DIR)))
def find_list_dir(array):
    master = []
    for j in array:
        fold_images = []
        for base in j:
            for (dirpath, dirnames, filenames) in os.walk(base):
                for filename in filenames:
                    fold_images.append(base+'/'+filename)
        master.append(sorted(fold_images))
    return master

def split_data(image_dir,mask_dir, random_seed,blur):
    # image_case_list = os.listdir(image_dir)
    # mask_case_list = os.listdir(mask_dir)
    # overlap_cases = [i for i in image_case_list if i in mask_case_list]
    #cases_test, cases_remain = train_test_split(overlap_cases,train_size = .2, random_state = random_seed)

    with open("testing_cases_TBI","rb") as fp:
        cases_test = pickle.load(fp)
    
    with open("training_cases_TBI","rb") as fp:
        cases_train_ls = pickle.load(fp)
    
    with open("validation_cases_TBI","rb") as fp:
        cases_val_ls = pickle.load(fp)
    # kf = KFold(n_splits = 8)
    # data_split = kf.split(cases_remain)


    # cases_train_ls = []
    #cases_val_ls = []
    # for i, (train_index,val_index) in enumerate(data_split):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={val_index}")
    #     cases_train =  [cases_remain[index] for index in train_index]
    #     cases_val = [cases_remain[index] for index in val_index]
    #     cases_train_ls.append(cases_train)
    #     cases_val_ls.append(cases_val)

    if blur == True:
        image_dir = image_dir.rstrip('/')
        image_dir = image_dir+'_blurredb/'
    
   
    train_images_path = [[image_dir + s for s in j] for j in cases_train_ls]
    train_mask_path = [[mask_dir + s for s in j] for j in cases_train_ls]

    test_images_path = [image_dir + s for s in cases_test]
    test_mask_path = [mask_dir + s for s in cases_test]

    val_images_path = [[image_dir + s for s in j] for j in cases_val_ls]
    val_mask_path = [[mask_dir + s for s in j] for j in cases_val_ls]

    #now we list individual images for training the neural network

    

    testImages= []
    testMasks = []

    trainImages = find_list_dir(train_images_path)
    trainMasks =  find_list_dir(train_mask_path)

    validationImages =  find_list_dir(val_images_path)
    validationMasks = find_list_dir(val_mask_path)
    
    for i in range(len(test_images_path)):
        testImages_list = sorted(list(paths.list_images(test_images_path[i])))
        testMasks_list = sorted(list(paths.list_images(test_mask_path[i])))
        for j in range(len(testImages_list)):
            testImages.append(testImages_list[j])
            testMasks.append(testMasks_list[j])

    return trainImages,testImages,validationImages,trainMasks,testMasks,validationMasks

def reset_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()
        print('Weights Reset')
#imagePaths = sorted(imagePaths)
#maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

## partition the data into training and testing splits using 85% of
## the data for training and the remaining 15% for testing
#split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)
## unpack the data split
#(trainImages, testImages) = split[:2]
#(trainMasks, testMasks) = split[2:]
trainImages,testImages,validationImages,trainMasks,testMasks,validationMasks = split_data(config.IMAGE_DATASET_PATH,config.MASK_DATASET_PATH,42,False)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
#print("[INFO] saving testing image paths...")
#f = open(config.TEST_PATHS, "w")
#f.write("\n".join(testImages))
#f.close()
#print("[INFO] saving validation image paths...")
#f = open(config.VAL_PATHS, "w")
#f.write("\n".join(validationImages))
#f.close()
test_data = pd.DataFrame(testImages)
#test_data = test_data.transpose()
train_data = pd.DataFrame(trainImages)
train_data = train_data.transpose()
validation_data = pd.DataFrame(validationImages)
validation_data = validation_data.transpose()

validation_data.to_csv(config.VAL_PATHS)
train_data.to_csv(config.TRAIN_PATHS)
test_data.to_csv(config.TEST_PATHS)



# define transformations
transforms = transforms.Compose([transforms.ToPILImage(), #check if this converts to 3 channels
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# initialize our UNet model
unet = UNet2(num_classes=1).to(config.DEVICE)
# create the train and test datasets
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    

# for i in range(len(trainMasks)):
#     for j in range(len(trainImages[i])):
#         mask_image = find_between(trainMasks[i][j],'/','.binary')
#         mask_image = mask_image.split('/')
#         mask_image = mask_image[-1]
#         train_image = find_between(trainImages[i][j],'/','.png')
#         train_image = train_image.split('/')
#         train_image = train_image[-1]
        
#         if train_image!= mask_image:
#             print('Fold '+ str(i) + ' image '+ str(j) +' are not match')


for k in range(8):
    
    trainDS = SegmentationDataset(imagePaths=trainImages[k], maskPaths=trainMasks[k],
        transforms=transforms)
    valDS = SegmentationDataset(imagePaths=validationImages[k], maskPaths=validationMasks[k],
        transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the validation set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=96)
    valLoader = DataLoader(valDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=96)

    
    # train_features, train_labels = next(iter(trainLoader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[7].squeeze()
    # label = train_labels[7].squeeze()
    # f, axarr = plt.subplots(1,2)

    # axarr[0].imshow(img, cmap="gray")
    # axarr[1].imshow(label)
    # plt.show()
    # print(f"Label: {label}")
    # quit()
    # initialize our UNet model
    #unet = UNet(encChannels=(1,16,32,64)).to(config.DEVICE)
    
    
    unet.apply(reset_weights)


    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)
    scheduler = ExponentialLR(opt, gamma=0.9)
    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    valSteps = len(valDS) // config.BATCH_SIZE
    # initialize a dictionary to store training history
    H = {"train_loss": [], "val_loss": []}

    # loop over epochs
    print("[INFO] training the network...",'fold:',str(k))
    loss_start = 100
    loss_track = []
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in valLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalValLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        
        #update LR 

        scheduler.step()
        
        if avgValLoss.cpu().detach().numpy() < loss_start:
            loss_start = avgValLoss.cpu().detach().numpy()
            # serialize the model to disk
            torch.save(unet, config.MODEL_PATH+str(k)+'_.pth')
        
        loss_track.append(loss_start)

        
        




        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Val loss: {:.4f}".format(
            avgTrainLoss, avgValLoss))
        print('Lowest Validation Loss: ', str(loss_start), 'and current learning rate: ',str(opt.param_groups[0]["lr"]))
        if (e+1) % 10 == 0:
            loss_track_np = np.array(loss_track)
            diff_loss_track = np.diff(loss_track_np)
            if np.sum(diff_loss_track) == 0:
                print('Training stopping early')
                break
            loss_track = []
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH+str(k)+'.png')
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH+str(k)+'_.pth')
