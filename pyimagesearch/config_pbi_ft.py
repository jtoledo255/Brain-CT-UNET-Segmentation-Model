# import the necessary packages
import torch
import os
# base path of the dataset
ANATOMY = 'inner_skull'
DATE = '20240314'
DATASET_PATH = os.path.join("dataset", "train")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = '/nfs/kitbag/data1/jtoledo/tbi_project/image_data/'
MASK_DATASET_PATH = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/Masks_PBI/'+ANATOMY+'_combined/'
#since images and masks are in different locations we need to locate corresponding masks and image
mask_file_names = os.listdir(MASK_DATASET_PATH)

IMAGE_DIR=[IMAGE_DATASET_PATH + s for s in mask_file_names]

#TRAINING_MASK_DATASET_PATH = '/home/jtoledo/Documents/tbi_project/segementations/train_masks/'+ANATOMY+'/'
# define the test split
TEST_SPLIT = 0.25
VAL_SPLIT = 0.25
TRAIN_SPLIT = 0.5
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 8
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+DATE+'/'+ANATOMY+"/output"
# define the path to the output serialized model, model training
# plot, and testing image paths
#MODEL_PATH = os.path.join(BASE_OUTPUT, "inner_skull_net")
MODEL_PATH = os.path.join(BASE_OUTPUT, "inner_skull_net_PBIFT")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot_finetunePBI"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths_finetunePBI.csv"])
VAL_PATHS = os.path.sep.join([BASE_OUTPUT, "validation_paths_finetunePBI.csv"])
TRAIN_PATHS = os.path.sep.join([BASE_OUTPUT, "train_paths_finetunePBI.csv"])