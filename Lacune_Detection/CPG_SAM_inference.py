import numpy as np
import glob
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
import skimage
import cv2
from segment_anything import sam_model_registry, SamPredictor
from u2net_cl import U2NETP
from skimage.measure import label
from utils import cut_zeros1d, tight_crop_data, determine_dice_metric, get_bounding_boxes, calculate_confusion_matrix, box_intersection, helper_resize
import argparse
from scipy.ndimage import binary_fill_holes
from skimage import measure
from sklearn.metrics import confusion_matrix
from skimage.measure import regionprops, label
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from skimage.io import imread
from scipy.ndimage import zoom
from skimage.transform import resize





class axis3D_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, file_paths,  mask_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.mask_paths = mask_paths
        self.transform = transform # This is where you can add augmentations

    def __getitem__(self, index):
        axial_mask=[]
        volume = nib.load(self.file_paths[index])
        vol = volume.get_fdata()
        pix_dim =volume.header['pixdim']
        mask = nib.load(self.mask_paths[index]).get_fdata()
        untouched_mask = mask
        vol,  mask, crop_params = tight_crop_data(vol,  mask)
        cropped_shape = np.shape(vol)
        data=[]
        data_mmax=[]
        vol_sqrt = np.sqrt(vol)
        
        perc_99_val = np.percentile(vol_sqrt,99.5)
        vol_filt= vol_sqrt[vol_sqrt<perc_99_val]
        min_int = np.min(vol_filt)
        max_int = np.max(vol_filt)
          

        for i in range(0, (np.shape(vol))[2]):
          slice=[]
          ax_slice = vol[:,:,i]
          ax_slice= resize(ax_slice,(256,256),order=1)
          ax_slice = np.asarray(ax_slice)
          ax_slice1 = ax_slice
          mean_int = np.mean(ax_slice)
          std_int = np.std(ax_slice)
          ax_slice= (ax_slice- mean_int) / (std_int+1e-6) 
                 
          ax_slice1= np.sqrt(ax_slice1)

          

          ax_slice1= (ax_slice1- min_int) / (max_int - min_int+1e-6) 
          data_mmax.append(ax_slice1)

          ax_mask = mask[:,:,i]
          ax_mask= resize(ax_mask,(256,256),order=0)
          axial_mask.append(ax_mask)
          slice.append(ax_slice)
          slice= np.asarray(slice)
          data.append(slice)
        
      
        data=np.asarray(data)
        axial_mask =np.asarray(axial_mask)
        data_mmax= np.asarray(data_mmax)
        data = torch.from_numpy(data).float()  
        data_mmax = torch.from_numpy(data_mmax).float()  
        axial_mask = torch.from_numpy(axial_mask).float()           

        return data, data_mmax, axial_mask, cropped_shape, crop_params, untouched_mask, pix_dim

    def __len__(self):
        return len(self.file_paths)
    


def main(args):
    model_type = "vit_h"
    device = args.device
    sam_checkpoint = args.sam_checkpoint
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    root_test = args.root_test
    volume_test = os.path.join(root_test, "**", "*FLAIR.nii.gz")  #Search Pattern
    mask_test = os.path.join(root_test, "**", "mask.nii.gz")

    Test_files= glob.glob(volume_test, recursive=True)
    Test_Maskfiles= glob.glob(mask_test, recursive=True)

    Test_files= natsorted(Test_files)
    Test_Maskfiles= natsorted(Test_Maskfiles)


    test_dataset = axis3D_dataset(Test_files, Test_Maskfiles)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = U2NETP(in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device=device)
    model.eval()

    total_conf_matrix = np.zeros((2, 2), dtype=int)
    threshold =0.5
  

    from skimage.measure import label
    for i, pack in enumerate(test_dataloader, start=1):
        print("Volume No:",i)     
        images, images_mmax, gt, cropped_shape, crop_params,untouched_mask,pix_dim = pack
        gt = np.asarray(gt, np.float32)
        images = torch.squeeze(images,dim=1)
        images = torch.squeeze(images,dim=0) 
        images_mmax = torch.squeeze(images_mmax,dim=0)
        images_mmax= np.asarray(images_mmax)
        images_mmax[images_mmax<0.2]=0
        gt= np.squeeze(gt,axis=0)
        result=[]

        # Predictions of CPG on axial slices
        for j in range (0, (np.shape(images))[0]):
                img = images[j:j+1]
                grd_tr = gt[j:j+1]
                img= img.to(device=device)
                model= model.to(device=device)
                res5= model(img)
                res5 = res5.sigmoid().data.cpu().numpy().squeeze()
                result.append(res5)

        images = torch.squeeze(images,dim=1)
        result = np.asarray(result)
        kernel = np.ones((3,3), np.uint8)
        result_unthresh = np.asarray(result).astype('float32')
        result = np.asarray(result>0.5).astype(np.uint8)
        result = cv2.dilate(result,kernel)        
        gt = np.asarray(gt)
        images = np.asarray(images)
        conf_matrix = calculate_confusion_matrix(np.asarray(gt), np.asarray(result))     
        print(conf_matrix)
        bboxes = get_bounding_boxes((result>0.5)*1)

        for bbox in bboxes:
             #  Inspect the Coronal slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[2]
             slice = (images_mmax[:,:,index])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[1], Centroid[0]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
            
             sum_value = 350 / np.round(pix_dim[0,1],1)
             if(first_score>0.75):
                # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]     
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                        
                # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.15)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                filled_mask = cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                # Check for ellipticity and major/minor axis lengths
                else:
                  fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                  for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if ( props[0].axis_major_length/minor_length>=5  or props[0].axis_major_length<2):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0

            



        result = np.asarray(result)
        bboxes = get_bounding_boxes((result>0.5)*1)
        for bbox in bboxes:
             #  Inspect the Sagittal slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[1]
             slice = (images_mmax[:,index,:])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[2], Centroid[0]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label, multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
             if(first_score>0.75):
                # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value):    
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                        
                # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.15)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                filled_mask = cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                # Check for ellipticity and major/minor axis lengths
                else:
                     fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                     for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if (props[0].axis_major_length/minor_length>=5  or props[0].axis_major_length<2):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0


        result = np.asarray(result)
        bboxes = get_bounding_boxes((result>0.5)*1)
        for bbox in bboxes:
             #  Inspect the Axial slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[0]
             slice = (images_mmax[index,:,:])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[2], Centroid[1]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
             if(first_score>0.75):
               # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value): 
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0

                # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.15)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                filled_mask = cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask   
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                # Check for ellipticity and major/minor axis lengths
                else:
                     fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                     for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if (props[0].axis_major_length>20 or props[0].axis_major_length/minor_length>=5 ):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0
                                                               

        result = (result>0.5)*1
        result = np.asarray(result).astype('float32')
        gt = np.asarray(gt)
        images = np.asarray(images)
        result = np.transpose(result, (1,2,0))
        result_unthresh = np.transpose(result_unthresh, (1,2,0))
        images = np.transpose(images, (1,2,0))
        gt = np.transpose(gt, (1,2,0))
        

        result_mask = np.asarray((result>0.5)*1)
        result_unthresh = result_unthresh*result_mask
        untouched_mask = np.asarray(torch.squeeze(untouched_mask, dim=0))
        act_images, act_output, act_output_unthresh, act_label =helper_resize(images, result,result_unthresh, untouched_mask, cropped_shape,crop_params)

        # Remove any 3D candidate predictions that has maximum axial diameter less than 3mm
        result = (act_output_unthresh>0)*1
        result = np.asarray(result).astype('float32')
        kernel = np.ones((2,2), np.uint8)
        result = cv2.erode(result,kernel)
        gt = (act_label>0.5)*1
        pred_boxes = get_bounding_boxes(result>0.5)
        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
        for pred_box in pred_boxes:
         unique_2 = False
         min_x, min_y, min_z, max_x, max_y, max_z = pred_box
         max_dia =0
         max_depth =0
         for n in range(min_z, max_z):
          crop_pred= result[min_x:max_x, min_y:max_y, np.round(n).astype(int)]
          crop_pred = [crop_pred>threshold]
          crop_pred_bool = np.asarray(crop_pred).astype(bool)
          if (np.unique(crop_pred_bool).shape[0] >1):
            unique_2 = True
            labeled_mask = label(crop_pred_bool)
            props = regionprops(labeled_mask[0])
            new_dia = (props[0].axis_major_length)/pix_dim[0,1]
            if new_dia>max_dia:
                max_dia = new_dia
                max_depth =n


         if max_dia <= 3/np.round(pix_dim[0,1],1) and unique_2 == True:
                  labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                  Centroid= np.trunc([(min_x + max_x)/2 , (min_y + max_y)/2, (min_z + max_z)/2]).astype(int)
                  if (result[Centroid[0],Centroid[1], Centroid[2]]==0 and (np.sum(result[min_x:max_x,min_y:max_y,max_depth]))>1):
                      Centroid[0] =np.where(result[min_x:max_x,min_y:max_y,max_depth]==1)[0][0]+min_x
                      Centroid[1] =np.where(result[min_x:max_x,min_y:max_y,max_depth]==1)[1][0]+min_y
                      Centroid[2] = max_depth
                  target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                  target_comp_mask =labelled_array==target_comp_label
                  result[target_comp_mask]=0
                  act_output_unthresh[target_comp_mask]=0
        
        #  Remove any 3D candidate that touches the outer brain region as sulcus
        result = np.asarray(result).astype('float32')
        result_mask = np.asarray((result>0.5)*1)
        result_unthresh = act_output_unthresh*result_mask
        act_output = np.array(result_unthresh>0).astype(np.uint8)
        act_label = np.asarray((gt>0.5)*1).astype(np.uint8)     
        result_mask = np.asarray(result_mask).astype('float32')

        img = nib.load(Test_files[i-1]).get_fdata()
        min_int = np.min(img)
        max_int = np.max(img)
        act_images = img
        act_images= (act_images- min_int) / (max_int - min_int+1e-6) 
        brain_mask = np.asarray((act_images>0.15)*1).astype(np.uint8)              
        mask_data = brain_mask.astype(bool)
        filled_mask = binary_fill_holes(mask_data)
        filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
        labels = label(act_output>0.5)
        unique_labels = np.unique(labels)[1:]  


        new_act_output = np.zeros_like(act_output)
        for label1 in unique_labels:
          mask = (labels == label1)
          roi = filled_mask*mask
          sulcus = np.logical_xor(roi, mask)
          if((np.sum(sulcus))>1):
           new_act_output[mask] = 0
           result_unthresh[mask]=0
          else:
           new_act_output[mask] = result_unthresh[mask]  
        conf_matrix = calculate_confusion_matrix(np.asarray(act_label), np.asarray(result_unthresh>0))
        print("After SAM:", conf_matrix)
        total_conf_matrix += conf_matrix
        ori_vol_affine = nib.load(Test_files[i-1]).affine
        predict_nii=nib.Nifti1Image(result_unthresh,affine=ori_vol_affine)
        filename = f"predicted{i}.nii.gz"
        full_file_path = os.path.join(args.folder_path, filename)
        nib.save(predict_nii, full_file_path)
    print(total_conf_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation model on test data")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--root_test", type=str, required=True, help="Root directory for training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save predictions")
    args = parser.parse_args()
    main(args)