from pyimagesearch import config_2  as config
#from pyimagesearch import config_pbi_ft as config
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff
import cv2
import math
from pydicom import dcmread

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage, cmap = 'bone')
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.savefig(pdf, format = 'pdf')
def get_25_image(imagePath):
    
	imagePath_split = imagePath.split('/')
	current_case = imagePath_split[-2]
	current_slice = imagePath_split[-1]
	current_slice = current_slice.split('.')
	current_slice = current_slice[0]
	# load the image from disk, swap its channels from BGR to RGB,
	# and read the associated mask from disk in grayscale mode
	dir_25 = '/nfs/kitbag/data1/jtoledo/tbi_project/image_data_25_combined/'
	images_25 = os.listdir(dir_25+current_case)
	images_25.sort()
	stack_images = []
	for i in images_25:
		if current_slice in i:
			#print('slice found')
			image = cv2.imread(dir_25+current_case+'/'+i,cv2.IMREAD_GRAYSCALE)
			image = image.astype("float32") / 255
			image = cv2.resize(image, (512, 512))
			convert_tensor = transforms.ToTensor()
			image = convert_tensor(image)

			stack_images.append(image)
		
				
	stack_slices = torch.stack(stack_images,dim=1)
	return stack_slices
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
		#image = cv2.imread(imagePath)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (512, 512))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split('/')
		filename = filename[-1].split('.')
		# print(filename)
		filename = imagePath.split(os.path.sep)[-2]+'/'+filename[0]+'.binary.png'
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		# print(filename)
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_HEIGHT))
        		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.expand_dims(image, 2)
		image = np.transpose(image, (2, 0, 1))

		image = np.expand_dims(image, 0)

		image = torch.from_numpy(image).to(config.DEVICE)
		   
		image_25= get_25_image(imagePath)
        
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image_25.to(config.DEVICE)).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		
		#print(predMask.size())
		# prepare a plot for visualization
		#prepare_plot(orig, gtMask, predMask)

		return orig, gtMask, predMask
	
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

def get_coordinates2(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list_coord = []
    for i in contours:
        for j in i:
            list_coord.append(j[0])

    return list_coord

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
test_path = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+config.DATE+'/'+config.ANATOMY+'/output/test_paths_finetunePBI.csv'
#imagePaths = open(test_path).read().strip().split("\n")
imagePaths = pd.read_csv(test_path)
imagePaths = imagePaths['0'].values.tolist()
imagePaths.sort()

#imagePaths = np.random.choice(imagePaths, size=14)

experiment = 'modelc_PBI'
if not os.path.isdir('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_prediction_images_2'+experiment+'/'):
    os.mkdir('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_prediction_images_2'+experiment+'/')
if not os.path.isdir('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_gt_images_2'+experiment+'/'):
    os.mkdir('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_gt_images_2'+experiment+'/')

dice = []
hausdorff = []
subject_ID = []
case = 'test'
for path in imagePaths:

    
    
	
    # iterate over the randomly selected test image paths
    #pdf = PdfPages('evalulation/'+config.DATE+'/'+config.ANATOMY+"/output_TBI_"+experiment+str(i)+".pdf")
    stack_images = []
    count = 0
    for i in range(8):
        # make predictions and visualize the results
        #print("[INFO] load up model...")
        unet = torch.load(config.MODEL_PATH+str(i)+'_.pth').to(config.DEVICE)
        orig,gtMask,predMask=make_predictions(unet, path)

		
        stack_images.append(predMask)
        name_split = path.split('/')
        case_id = name_split[-2]
        slice_id = name_split[-1]
        segmentation_name = case_id+'_'+slice_id
        write_a = False
            

        predMask_single = (predMask > config.THRESHOLD) 
        predMask_single = predMask_single
        predMask_single = predMask_single.astype(np.uint8)
        
        if write_a:
            print(np.max(predMask_single))
            
            cv2.imwrite('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_prediction_images_2'+experiment+'/'+segmentation_name+str(i)+'.png',predMask_single*255)
                
    prediction_stack = np.stack(stack_images)
    predMask = np.mean(prediction_stack, axis = 0)
    
    
    predMask = (predMask > config.THRESHOLD) * 255
    predMask = predMask.astype(np.uint8)
    # print(prediction_stack.shape)
    # print(predMask.shape)

    name_split = path.split('/')
    image_id = name_split[-1]
    name_split = name_split[-2].split('_')
    
    
    if case != name_split[0]:
        pixel_dis = findpixeldistance(name_split[0])
        case = name_split[0]

    np_gtMask = np.array(gtMask)
    np_predMask = np.array(predMask)
    pred_coord = get_coordinates2(np_predMask)
    gt_coord = get_coordinates2(np_gtMask)

    # if 'GSW'  in case:
    #      continue
    
    
    if np_gtMask.max() == 0:
        continue
    elif np_gtMask.max() > 0 and np_predMask.max() == 0:
        continue
    else:
        dice.append(dice_metric(np.ma.make_mask(np_predMask),np.ma.make_mask(np_gtMask)))
        subject_ID.append(segmentation_name)
  
        hd = (max(directed_hausdorff(pred_coord,gt_coord)[0], directed_hausdorff(gt_coord, pred_coord)[0]))
        hausdorff.append(np.multiply(hd,pixel_dis))

       
        
   

    write = True

    if write:
        print('Averaged printed')
        cv2.imwrite('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_prediction_images_2'+experiment+'/'+segmentation_name+'.png',predMask)
        cv2.imwrite('evalulation/'+config.DATE+'/'+config.ANATOMY+'/model_gt_images_2'+experiment+'/'+segmentation_name+'.png',gtMask)

savepath = '/nfs/kitbag/data1/jtoledo/tbi_project/segementations/evalulation/'+config.DATE+'/'+config.ANATOMY+'/'+experiment+'final_results_model_2.csv'


df = pd.DataFrame(np.column_stack([subject_ID,dice,hausdorff]),columns = ['image_id','2D DSC','HD'])
df.to_csv(savepath)
print(len(dice))
print(np.mean(dice))
print(np.mean(hausdorff))
