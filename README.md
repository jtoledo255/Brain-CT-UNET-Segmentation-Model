# Brain-CT-UNET-Segmentation-Model
UNET based segmentation model trained using both non-penetrating and penetrating traumatic brain injury head CT scans. Model uses a cross validation ensemble approach with 2.5D inputs.

# Data Preperation
Imaging data is loaded from HCT scans that have been preprocessed into .png file format. DICOM files were read, rescaled to HU, and a final brain window was applied to the scan. Input for the model is a 2.5D stack, with 7 total sequential slices, centered on the slice of interest. 
