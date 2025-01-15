import numpy as np
import skimage
import nibabel as nib
import cv2
from utils import calculate_confusion_matrix, get_bounding_boxes
from skimage import measure
import argparse


def main(args):
  MARS_atlas = args.MARS_ATLAS
  Test_Maskfiles = args.GroundTruth_Mask_files
  Test_Atlasfiles = args.Registered_MARS_files
  SAM_unthresh_files = args.SAM_output_files
  mni_atlas = nib.load(MARS_atlas).get_fdata()

  possible_values = np.unique(np.round(mni_atlas))
  possible_values = possible_values[1:]
  # Threshold for white matter, frontal, Internal capsule and External capsule = 0.5
  # Threshold for Basal Ganglia, Temporal, Cerebellum, Parietal, Insular = 0.55
  # Threshold for Thalamus, Hippocampus, Brain stem, Occipital = 0.65

  thresholds = [0.65, 1, 0.55, 0.65, 0.5, 0.5, 0.65, 0.5,0.5,0.55,0.55, 0.65, 0.55]
  thresholds = [threshold+0.1  for threshold in thresholds] #threhold incremented by 0.1 for VALDO
  for i in range(0,len(Test_Maskfiles)):
    grd = nib.load(Test_Maskfiles[i]).get_fdata()
    mask = nib.load(SAM_unthresh_files[i]).get_fdata()
    mask[:,:,0.8*mask.shape[2]:] =0
    mask_unthresh = mask
    mask = (mask>=0.5)*1
    kernel = np.ones((3,3), np.uint8)
    mask = np.asarray(mask).astype(np.uint8) 
    mask1 =mask
    kernel = np.ones((1,1), np.uint8)
    mask1 = cv2.erode(mask,kernel)
    conf_matrix = calculate_confusion_matrix(np.asarray(grd), np.asarray(mask))
    total_conf_matrix_ori+=conf_matrix
    pred_boxes = get_bounding_boxes(mask)
    print("Volume:",i, "No:", len(pred_boxes))
    atlas_values = nib.load(Test_Atlasfiles[i]).get_fdata()
    if (len(pred_boxes))>0:
        print(mask.shape,  atlas_values.shape)
        for bbox in pred_boxes:
            min_x, min_y,min_z, max_x, max_y, max_z= bbox
            centroid_x, centroid_y, centroid_z = np.round((min_x+max_x)/2).astype(int), np.round((min_y+max_y)/2).astype(int), np.round((min_z+max_z)/2).astype(int)
            centroid_mask_val = mask_unthresh[np.round(centroid_x).astype(int), np.round(centroid_y).astype(int), np.round(centroid_z).astype(int)]
            region_value = atlas_values[np.round((min_x+max_x)/2).astype(int), np.round((min_y+max_y)/2).astype(int), np.round((min_z+max_z)/2).astype(int)]
            index = np.where(possible_values == np.round(region_value))[0][0]
            if (centroid_mask_val <= thresholds[index]):
                labelled_array,numc = measure.label((mask>0)*1, return_num=True,connectivity=3)
                target_comp_label = labelled_array[np.round(centroid_x).astype(int), np.round(centroid_y).astype(int), np.round(centroid_z).astype(int)]
                target_comp_mask =labelled_array==target_comp_label
                mask[target_comp_mask]=0
    mask = np.asarray((mask>0)*1).astype('float')
    conf_matrix = calculate_confusion_matrix(np.asarray(grd), np.asarray(mask))
    print(conf_matrix)
    total_conf_matrix+=conf_matrix


if __name__ == "__main__":
    # MARS region atlas in MNI space is taken from
    # https://github.com/v-sundaresan/microbleed-size-counting 
    parser = argparse.ArgumentParser(description="Run segmentation model on test data")
    parser.add_argument("--MARS_ATLAS", type=str, default="cuda:1", help="MARS Atlas in MNI")
    parser.add_argument("--SAM_output_files", type=str, nargs='+',required=True, help="List of unthresholded SAM output files ")
    parser.add_argument("--Registered_MARS_files", type=str,nargs='+', required=True, help="List of all MARS maps registerd to test files")
    parser.add_argument("--GroundTruth Mask files", type=str, nargs='+', required=True, help="List of all ground truth mask files")
    args = parser.parse_args()
    main(args)