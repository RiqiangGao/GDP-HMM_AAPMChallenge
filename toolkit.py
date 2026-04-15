'''
This script is adapted from the toolkit of below CVPR paper. 
If you find the functions in this script are helpful to you (for the challenge and beyond), please kindly cite the original paper: 

Riqiang Gao, Bin Lou, Zhoubing Xu, Dorin Comaniciu, and Ali Kamen. 
"Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy." 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Disclaimer: This is for research purpose only. This is not part of the any existing Siemens Healthineers product.

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt 
import torch
from scipy import ndimage

import SimpleITK as sitk

import os


from monai.transforms import (
    Compose,
    Resized,
    RandFlipd, 
    RandRotated,
    SpatialPadd, 
    SpatialCropd,
    RandSpatialCropd
)


'''
Content: 

Section 1: Geometries, including angle_plate and beam_plate creation
Section 2: DVHs and Visualization
Section 3: for pytorch data loader

'''



#---------------------------------------------------------------------------------#
#--------------------------------- Section 1: Geometries--------------------------#
#---------------------------------------------------------------------------------#

def PlotGantry(bg_img, angles, x, y, length, width = 4):

    angles = [angle - 90 for angle in angles]
    
    points = [(int(x + length * np.cos(np.pi / 180 * float(angle))), int(y + length * np.sin(np.pi / 180 * float(angle)))) for angle in angles]
    for point in points:
        cv2.line(bg_img, point, (x, y), (1), width)
    return bg_img

def interpolate_point_on_line(x1, y1, z1, x2, y2, z2, y_c):
    """
    Returns the coordinates of point C on the line segment from (x1, y1, z1) to (x2, y2, z2)
    with the specified y-coordinate y_c.
    """
    # Calculate the ratio of y_c relative to the total y-distance between points A and B

    if y2 == y1:
        y2 += 1
    ratio = (y_c - y1) / (y2 - y1)
    
    # Use linear interpolation to find the corresponding x and z coordinates
    x_c = x1 + ratio * (x2 - x1)
    z_c = z1 + ratio * (z2 - z1)
    
    return int(x_c), int(y_c), int(z_c)


def interpolate_line(x1, y1, z1, x2, y2, z2, y_c):
    """
    Returns all the coordinates along the line segment from (x1, y1, z1) to (x2, y2, z2).
    """
    # Calculate the distance between the points

    x_c, y_c, z_c = interpolate_point_on_line(x1, y1, z1, x2, y2, z2, y_c)

    length = max(abs(x_c - x1), abs(y_c - y1), abs(z_c - z1))
    
    # Generate linearly spaced coordinates between the points
    x_coords = np.linspace(x1, x_c, length + 1)
    y_coords = np.linspace(y1, y_c, length + 1)
    z_coords = np.linspace(z1, z_c, length + 1)
    
    # Round coordinates to integers
    x_coords = np.round(x_coords).astype(int)
    y_coords = np.round(y_coords).astype(int)
    z_coords = np.round(z_coords).astype(int)
    
    # Combine coordinates into tuples
    coordinates = [(x, y, z) for x, y, z in zip(x_coords, y_coords, z_coords)]
    
    return coordinates


def get_source_from_angle(isocenter, angle, space):
    angle = angle - 90 
    
    points =  isocenter[0], int(isocenter[1] + 1000 * np.sin(np.pi / 180 * float(angle)) / space[1]), int(isocenter[2] + 1000 * np.cos(np.pi / 180 * float(angle)) / space[2])
    return points

def get_nonzero_coordinates(binary_mask):
    """
    Returns the coordinates of non-zero values in a binary mask.
    """
    nonzero_coords = np.transpose(np.nonzero(binary_mask))
    return [tuple(coord) for coord in nonzero_coords]

def surface_coordinates(binary_mask):
    nonzero_coords = get_nonzero_coordinates(binary_mask)
    surface_coords = []
    for coord in nonzero_coords:
        x, y, z = coord
        if (binary_mask[x-1, y, z] == 0 or
            binary_mask[x+1, y, z] == 0 or
            binary_mask[x, y-1, z] == 0 or
            binary_mask[x, y+1, z] == 0 or
            binary_mask[x, y, z-1] == 0 or
            binary_mask[x, y, z+1] == 0):
            surface_coords.append((x, y, z))  
    return surface_coords

def get_per_beamplate(PTV_mask, isocenter, space, gantry_angle, with_distance = True):
    
    # PTV_mask (z, x, y), sicenter = (z, x, y), space = (z, x, y), gantry_angle = float

    source = get_source_from_angle(isocenter, gantry_angle, space)
    #print ('source', source)


    surface_coords = surface_coordinates(PTV_mask)
    #print (time.time() - start)

    all_points = []
    for point in surface_coords:
        if source[1] > point[1]:
            y_c = 0
        else:
            y_c = PTV_mask.shape[1] - 1
        path = interpolate_line(source[0], source[1], source[2], point[0], point[1], point[2], y_c = y_c)
        all_points.extend(path)
    #print (time.time() - start)

    beam_plate = np.zeros_like(PTV_mask).astype(np.uint8)
    for item in set(all_points):
        if item[0] >= 0 and item[0] < PTV_mask.shape[0] and item[1] >= 0 and item[1] < PTV_mask.shape[1] and item[2] >= 0 and item[2] < PTV_mask.shape[2]:
            beam_plate[item[0], item[1], item[2]] = 1
    #print (time.time() - start)
    beam_plate = ndimage.binary_dilation(beam_plate, structure=np.ones((4,4,4))) #.astype(PTV_mask.dtype)
    beam_plate = ndimage.binary_erosion(beam_plate, structure=np.ones((3,3,3))) #.astype(PTV_mask.dtype)

    #print (time.time() - start)

    if with_distance:
        x_indices = np.arange(beam_plate.shape[0])
        y_indices = np.arange(beam_plate.shape[1])
        z_indices = np.arange(beam_plate.shape[2])

        x_coords, y_coords, z_coords = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')

        # Compute distances using broadcasting
        distances = (x_coords - source[0])**2 + (y_coords - source[1])**2 + (z_coords - source[2])**2

        r_dis = ((source[0] - isocenter[0]) ** 2 + (source[1] - isocenter[1]) ** 2 + (source[2] - isocenter[2]) ** 2) / distances 
        beam_plate = beam_plate * r_dis
        #print (time.time() - start)

    return beam_plate

def get_allbeam_plate(PTV_mask, isocenter, space, angles, with_distance = True, isVMAT = False, skip_angle = True):
    all_beam_plate = np.zeros_like(PTV_mask).astype('float')
    for i, angle in enumerate(angles):
        if isVMAT and skip_angle and i % 2 == 1: # this is for speeding up the computation for VMAT beam_plate. User may or may not change the parameters (skip_angle) if separately preparing beam_plate
            continue
        all_beam_plate += get_per_beamplate(PTV_mask, isocenter, space, angle, with_distance = with_distance)
    return all_beam_plate


#---------------------------------------------------------------------------------#
#--------------------- Section 2: DVHs and Visualization -------------------------#
#---------------------------------------------------------------------------------#


def getDVH(dose_arr, mask, binsize=0.1, dmax=None):
    '''
    Calculate DVH per Region of Interest(ROI)
    dose_arr: dose array
    mask: mask of GTV/OAR
    binsize: bin size of the histogram, default 0.1
    dmax: maximum dose value for DVH calculation, using max(dose_arr)*110% if not given
    return: values of dose and DVH
    '''
    dosevalues = dose_arr[mask>0]
    
    if dmax is None:
        dmax = np.amax(dosevalues)*1.0
    hist, bin_edges = np.histogram(dosevalues, bins=np.arange(0,dmax,binsize))
    DVH = np.append(1, 1 - np.cumsum(hist)/len(dosevalues)) * 100
    return bin_edges, DVH

def NPZ2DVH(roi_dict, needed_mask, ref_ptv_name = None, ref_dose = 70, bin_size = 4, with_plt = True, save_plt_path = None):
    '''
    Calculate DVH per plan. 
    roi_dict: the data dictionary loaded from NPZ file
    needed_mask: the roi name list wanted to plot in one figure
    ref_ptv_name: if ref_ptv_name is not None, the dose will be scaled to match D97 (3 percentile dose value of ref ptv) to prescribed dose (i.e., ref_dose)
    ref_dose: prescribed dose of the reference ptv, only effective when ref_ptv_name is not None
    bin_size: bin size for the histogram
    with_plt: plt the figure or not
    save_plt_path: is not None, the plot figure will be saved to the path
    
    return: dvh_dict: the dictionary of all dvh values
    '''
    
    if roi_dict['dose'].max() > 200:
        dose_arr = roi_dict['dose'] * roi_dict['dose_scale'] 
    
    if dose_arr.max() < 10:
        dose_arr = 80 / dose_arr.max()  * dose_arr

    if ref_ptv_name is not None:
        ptv = roi_dict[ref_ptv_name]

        scale = ref_dose / np.percentile(dose_arr[ptv > 0.5], 3) 
    
        dose_arr = dose_arr * scale

    dvh_dict = {}

    for key in needed_mask:
        if roi_dict[key].max() == 0:
            continue
        #try:
        bin_edges_dose, DVH_dose = getDVH(dose_arr, roi_dict[key], binsize= bin_size, dmax=None)
        dvh_dict[key]  = {'dose': bin_edges_dose.tolist(), 'dvh': DVH_dose.tolist()}

        if with_plt:
            plt.plot(bin_edges_dose, DVH_dose,  label=key)
    if with_plt:
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')  
        plt.legend(bbox_to_anchor=(1.01, 1.01))
        plt.show()
    return dvh_dict

def NormalizeImg(imgVol, maskVol = None):
    '''
        Normalize the data to be reasonable scale.
    '''
    i_min, i_max = np.percentile(imgVol, (0.5,99.5))
    if maskVol is not None:
        if maskVol.sum()>0:
            mask_slice_idx = np.nonzero(np.sum(imgVol, axis=(1,2)))[0]
            bg_slice_idx = [imgVol[i] for i in mask_slice_idx]
            i_min, i_max = np.percentile(bg_slice_idx, (0.5,99.5))

    imgVol_norm = (imgVol - i_min) / (i_max - i_min + 1e-10)
    imgVol_norm[imgVol_norm<0] = 0
    imgVol_norm[imgVol_norm>1] = 1
    imgVol_norm = np.uint8(imgVol_norm * 255)
    return imgVol_norm

def NPZ2DVH_2Dose(roi_dict, needed_mask, additional_dose, ref_ptv_name = None, ref_dose = 70, bin_size = 4, with_plt = True, save_plt_path = None):
    '''
    Similar to NPZ2DVH, but plot the additional dose on the same figure
    '''
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939',
                '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#6b6ecf', '#b5cf6b',
                '#e7ba52', '#d6616b', '#ce6dbd', '#de9ed6', '#ad494a', '#8ca252',
                '#e7969c', '#bd9e39', '#17becf', '#bdbdbd', '#c7e9c0', '#fdbe85']


    if roi_dict['dose'].max() > 200:
        dose_arr = roi_dict['dose'] * roi_dict['dose_scale'] 
    else:
        dose_arr = roi_dict['dose']
    
    if dose_arr.max() < 10:
        dose_arr = 80 / dose_arr.max()  * dose_arr

    if ref_ptv_name is not None:
        ptv = roi_dict[ref_ptv_name]

        scale = ref_dose / np.percentile(dose_arr[ptv > 0.5], 3) 
    
        dose_arr = dose_arr * scale

    dvh_dict = {}

    cnt = 0

    for key in needed_mask:
        if roi_dict[key].max() == 0:
            continue
    
        bin_edges_dose, DVH_dose = getDVH(dose_arr, roi_dict[key], binsize= bin_size, dmax=None)
        #dvh_dict[key]  = {'dose': bin_edges_dose.tolist(), 'dvh': DVH_dose.tolist()}

        bin_edges_add_dose, DVH_add_dose = getDVH(additional_dose, roi_dict[key], binsize= bin_size, dmax=None)
        #dvh_dict[key]  = {'add_dose': bin_edges_add_dose.tolist(), 'dvh': dvh_dict.tolist()}
        
        if with_plt:
            plt.plot(bin_edges_dose, DVH_dose,  label=key, color = colors[cnt],  linewidth=2)
            plt.plot(bin_edges_add_dose, DVH_add_dose,  label=key + '*', linestyle='--', color = colors[cnt],  linewidth=2)

        cnt += 1

    if with_plt:
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')  
        if cnt > 10:
            plt.legend(bbox_to_anchor=(1.01, 1.01), ncol=2)
            print ('two columns')
        else:
            plt.legend(bbox_to_anchor=(1.01, 1.01))
        
        
        if save_plt_path is not None:
            plt.savefig(save_plt_path, bbox_inches='tight')
            plt.close()
    return dvh_dict


def save_screenshot(npz_dict, PTV_name, masked_by_body = True, index = None): 
    
    '''
    save screenshot of the data loaded from npz, this can be used for quality checking
    '''
    
    if index is None:
        index = np.argmax(npz_dict['dose'].sum(axis = 1).sum(axis = 1))
    ptv = (npz_dict[PTV_name] > 0).astype('uint8') * 255
    img = NormalizeImg(npz_dict['img'])
    dose = NormalizeImg(npz_dict['dose'])

    if masked_by_body:

        body_indx = (npz_dict['Body'][index] > 0.5).astype('uint8') 
    else:
        body_indx = 1  

    angle_plate = npz_dict['angle_plate'] / npz_dict['angle_plate'].max() * 255

    beam_plate = npz_dict['beam_plate'] / npz_dict['beam_plate'].max() * 255
    
    img_x, img_y = img.shape[1:]
    save_img = np.zeros((img_x , img_y * 5), dtype = np.uint8)

    save_img[:, :img_y] = img[index] * body_indx
    save_img[:, img_y:img_y * 2] = dose[index] * body_indx
    save_img[:, img_y * 2:img_y * 3] = ptv[index] * body_indx

    save_img[:, img_y * 3:img_y * 4] = angle_plate * body_indx

    save_img[:, img_y * 4:img_y * 5] = beam_plate[index] *  body_indx
    return save_img

def save_mhd(data, root_dir, filename, spacing=None, origin=None, direction=None ):
    '''
    save 3D data to mhd file. mhd file can be opened by itk-snap or some other tools
    '''

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    volume = sitk.GetImageFromArray(data)
    if spacing is not None:
        volume.SetSpacing(spacing)
    if origin is not None:
        volume.SetOrigin(origin)
    if direction is not None:
        volume.SetDirection(direction)
    sitk.WriteImage(volume, root_dir + f'/' + filename + '.mhd')
    return

#---------------------------------------------------------------------------------#
#----------------- Section 3: for pytorch data loader ----------------------------#
#---------------------------------------------------------------------------------#

def combine_oar(tmp_dict, need_list,norm_oar = True, OAR_DICT = None):

    '''
    this function is used to support the data loader. 

    tmp_dict: the dictionary of the data loaded from the npz file
    need_list: the list of the OARs needed to be combined
    norm_oar: if True, the OARs will be normalized to the same scale
    OAR_DICT: the dictionary of the OARs, the key is the name of the OAR, the value is the index of the OAR in the combined
    '''
    
    comb_oar = torch.zeros(tmp_dict['img'].shape)  
    cat_oar = torch.zeros([32] + list(tmp_dict['img'].shape)[1:])
    for key in OAR_DICT.keys():
        
        if key not in need_list:
            continue

        if key in tmp_dict.keys():
            single_oar = tmp_dict[key]
        else:
            single_oar = torch.zeros(tmp_dict['img'].shape)

        cat_oar[OAR_DICT[key]-1: OAR_DICT[key]]  = single_oar

        if norm_oar:
            comb_oar = torch.maximum(comb_oar, single_oar.round() * (1.0 + 4.0 * OAR_DICT[key] / 30))
        else:
            comb_oar = torch.maximum(comb_oar, single_oar.round() * OAR_DICT[key])  # changed in this version

    return comb_oar, cat_oar


def combine_ptv(tmp_dict,  scaled_dose_dict):

    prescribed_dose = []
    
    cat_ptv = torch.zeros([3] + list(tmp_dict['img'].shape)[1:])
    prescribed_dose = [0] * 3

    comb_ptv = torch.zeros(tmp_dict['img'].shape)

    cnt = 0
    for key in scaled_dose_dict.keys():
       
       tmp_ptv = tmp_dict[key] * scaled_dose_dict[key]  
       
       prescribed_dose[cnt] =  scaled_dose_dict[key] 
       
       cat_ptv[cnt] = tmp_ptv 
       comb_ptv = torch.maximum(comb_ptv, tmp_ptv)

       cnt += 1

    # sort the cat_ptv according to the prescribed dose
    paired = [(cat_ptv[i], prescribed_dose[i]) for i in range(len(prescribed_dose))]
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    cat_ptv = torch.stack([x[0] for x in paired_sorted])
    prescribed_dose = [x[1] for x in paired_sorted]
    
    return comb_ptv, prescribed_dose, cat_ptv 

def tr_augmentation(KEYS, in_size, out_size, crop_center):
    return Compose([
        SpatialCropd(keys = KEYS, roi_center = crop_center, roi_size = [int(in_size[0] * 1.2), int(in_size[1] * 1.2), int(in_size[2] * 1.2)], allow_missing_keys = True),
        SpatialPadd(keys = KEYS, spatial_size = [int(in_size[0] * 1.2), int(in_size[1] * 1.2), int(in_size[2] * 1.2)], mode = 'constant', allow_missing_keys = True),
        RandSpatialCropd(keys = KEYS, roi_size = [int(in_size[0] * 0.85), int(in_size[1] * 0.85), int(in_size[2] * 0.85)], max_roi_size = [int(in_size[0] * 1.2), int(in_size[1] * 1.2), int(in_size[2] * 1.2)], random_center = True, random_size = True, allow_missing_keys = True), 
        RandRotated(keys = KEYS, prob=0.8, range_x= 1, range_y = 0.2, range_z = 0.2, allow_missing_keys = True),
        RandFlipd(keys = KEYS, prob = 0.4, spatial_axis = 0, allow_missing_keys = True), 
        RandFlipd(keys = KEYS, prob = 0.4, spatial_axis = 1, allow_missing_keys = True), 
        RandFlipd(keys = KEYS, prob = 0.4, spatial_axis = 2, allow_missing_keys = True), 
        Resized(keys = KEYS, spatial_size = out_size, allow_missing_keys = True), 
    ])

def tt_augmentation(KEYS, in_size, out_size, crop_center):
    return Compose([
        SpatialCropd(keys = KEYS, roi_center = crop_center, roi_size = in_size, allow_missing_keys = True),
        SpatialPadd(keys = KEYS, spatial_size = in_size, mode = 'constant', allow_missing_keys = True),
        Resized(keys = KEYS, spatial_size = out_size, allow_missing_keys = True), 
    ])



