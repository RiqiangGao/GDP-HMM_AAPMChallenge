'''
This one-in-all script has been used to convert DICOM files to npz files for the GDP-HMM challenge. 

Note: Since we have provided the npz files for the challenge, you do not need to run this script. 
But this script can help you understand how npz files are saved, and you can adjust the functions for your research purpose.

If you used this script for your research, please kindly cite the following paper and keep this information in your code:
@article{gao2025automating,
  title={Automating RT Planning at Scale: High Quality Data For AI Training},
  author={Gao, Riqiang and Diallo, Mamadou and Liu, Han and Magliari, Anthony and Sackett, Jonathan and Verbakel, Wilko and Meyers, Sandra and Zarepisheh, Masoud and Mcbeth, Rafe and Arberet, Simon and others},
  journal={arXiv preprint arXiv:2501.11803},
  year={2025}
}

Data source (please read the data usage agreement in HuggingFace repo before using the data): 

    DICOM: https://huggingface.co/datasets/Jungle15/Radiotherapy_HaN_Lung_AIRTP
    npz: https://huggingface.co/datasets/Jungle15/GDP-HMM_Challenge 

Disclaimer: This is for research purpose only. This is not part of the any existing Siemens Healthineers product.
'''

'''
Content: 
    1. ReadDose
    2. GetIMRTAngleList
    3. GetVMATAngleList
    4. ReadCTSeries
    5. IMRTvsVMAT
    6. getResampledImageVolume
    7. resample_img
    8. GetPixelIsocenter
    9. convert_rtstruct_to_binary_mask
    10. get_npz_dict_ref_CT
'''

import SimpleITK as sitk
import pydicom
import numpy as np
from skimage.draw import polygon
from scipy.ndimage import binary_erosion, binary_dilation

def ReadDose(dose_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dose_path)
    dose = reader.Execute()
    return dose

def GetIMRTAngleList(ds):
    angles = []
    for k in range(len(ds[0x300a, 0x00b0].value)):
        
        angles.append(ds[0x300a, 0x00b0].value[k][0x300a, 0x0111].value[0][0x300a, 0x011e].value)
    return angles

def GetVMATAngleList(ds):
    
    for k in range(len(ds[0x300a, 0x00b0].value)):
        if ds[0x300a, 0x00b0].value[k][0x300a, 0x0111].value[0][0x300a, 0x011f].value == 'NONE':
            continue
        angles = []
        for i in range(len(ds[0x300a, 0x00b0].value[k][0x300a, 0x0111].value)):
            angles.append(ds[0x300a, 0x00b0].value[k][0x300a, 0x0111].value[i][0x300a, 0x011e].value)
        #print (len(angles), min(angles), max(angles))
        return angles

def ReadCTSeries(series_fold):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_fold)
    reader.SetFileNames(dicom_names)
    CT = reader.Execute()
    return CT

def IMRTvsVMAT(ds):
    
    '''
    output if the PLAN is VMAT
    '''
    
    GantryRotation = []
    for i in range(len(ds[0x300a, 0x00b0].value)):
        GantryRotation.append(ds[0x300a, 0x00b0].value[i][0x300a, 0x0111].value[0][0x300a, 0x011f].value)
    isVMAT = len(set(['CC', 'CW']) & set(GantryRotation)) > 0
    return isVMAT

def getResampledImageVolume(imageData, referenceImage):
    
    newImageVol = sitk.GetArrayFromImage(imageData)

    imageSize = imageData.GetSize()
    referenceImageSize = referenceImage.GetSize()
    #print ('max before resample', newImageVol.max())
    if imageSize != referenceImageSize:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(referenceImage)
        reSampledImg = resampler.Execute(imageData)
        newImageVol = sitk.GetArrayFromImage(reSampledImg) #.astype(np.int32)
        #print ('max after resample', newImageVol.max())
    return newImageVol


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def GetPixelIsocenter(CT, dcm_plan):
    img_ori = CT.GetOrigin()
    img_direct = CT.GetDirection()
    img_spac = CT.GetSpacing()

    isocenter = dcm_plan[0x300a, 0x00b0].value[0][0x300a, 0x0111].value[0][0x300a, 0x012c].value

    x_ = (isocenter[0] - img_ori[0]) / img_spac[0] * img_direct[0]
    y_ = (isocenter[1] - img_ori[1]) / img_spac[1] * img_direct[4]
    z_ = (isocenter[2] - img_ori[2]) / img_spac[2] * img_direct[8]

    return [int(x_), int(y_), int(z_)]


def convert_rtstruct_to_binary_mask(ds, CT, dose, version = 2, fill_strcuts = []):
    #ds = pydicom.dcmread(rtstruct_file)
    #CT = Sitk.ReadCTSeries(CT_file)
    #dose = Sitk.ReadDose(dose_file)
    structures = ds.StructureSetROISequence
    res_dict = {}
    for i in range(len(ds.ROIContourSequence)):

        try: 
            contour_data = ds.ROIContourSequence[i].ContourSequence
        except:
            continue

        mask = np.zeros(sitk.GetArrayFromImage(CT).shape, dtype=np.uint8)
        cnt = 0
        for contour in contour_data:
            cnt += 1
            
            contour_points = np.array(contour.ContourData).reshape((-1, 3))
            indices = [CT.TransformPhysicalPointToIndex(p) for p in contour_points]
            z_indices = set(idx[2] for idx in indices)
            
            for z in z_indices:
                slice_contour_points = np.array([idx[:2] for idx in indices if idx[2] == z])
                slice_image = mask[z, :, :]
                rr, cc = polygon(slice_contour_points[:, 1], slice_contour_points[:, 0], slice_image.shape)
                if version == 1 or structures[i].ROIName in fill_strcuts:
                    mask[z, rr, cc] = 1  # this version cannot handle ring structures 
                else:
                    mask[z, rr, cc] ^= 1
        
        
        if dose is None:
            res_dict[structures[i].ROIName] = mask
        else:
            binary_mask = sitk.GetImageFromArray(mask)
            binary_mask.CopyInformation(CT)
            mask_arr = getResampledImageVolume(binary_mask, dose).astype('uint8')
            res_dict[structures[i].ROIName] = mask_arr

    return res_dict


def get_npz_dict_ref_CT(sess_path, rs_path, rd_path = None, plan_path = None, img_spac = None, need_rtplan = False, version = 2, fill_strcuts = []):

    CT_ori = ReadCTSeries(sess_path)
    
    if img_spac is not None:
        CT = resample_img(CT_ori, out_spacing = img_spac)

    rs_dcm = pydicom.dcmread(rs_path)
    
    data_dict = convert_rtstruct_to_binary_mask(rs_dcm, CT_ori, CT, version, fill_strcuts=fill_strcuts)

    data_dict['all_mask'] = list(data_dict.keys())

    img_arr = sitk.GetArrayFromImage(CT) #
    data_dict['img'] = img_arr

    if rd_path is not None:
        dose_scale = pydicom.dcmread(rd_path).DoseGridScaling 
        dose = ReadDose(rd_path)
        dose_arr = getResampledImageVolume(dose, CT) # sitk.GetArrayFromImage(dose)
        data_dict['dose'] = dose_arr
        data_dict['dose_scale'] = float(dose_scale)
    #data_dict['contour'] = contour 
    

    if need_rtplan and plan_path is not None: 
        try: 
            dcm_plan = pydicom.dcmread(plan_path)
            isocenter = GetPixelIsocenter(CT, dcm_plan) # x, y, z as in dcm 
            isocenter = [isocenter[2], isocenter[1], isocenter[0]] # z, y, x as in numpy array

            data_dict['isVMAT'] = IMRTvsVMAT(dcm_plan)
            if not data_dict['isVMAT']: # return isVMAT
                angle_list = GetIMRTAngleList(dcm_plan)
            else:
                angle_list = GetVMATAngleList(dcm_plan)

            angle_list = list(set(angle_list))
            
            data_dict['isocenter'] = np.array(isocenter).astype('float32') 
            data_dict['angle_list'] = np.array(angle_list).astype('float32')
        except:
            print ('---- rtplan read error')
            
    data_dict['origin'] = CT.GetOrigin()
    data_dict['spacing'] = CT.GetSpacing()
    data_dict['direction'] = CT.GetDirection()
    data_dict['size'] = CT.GetSize()
    
    return data_dict

def get_coords(arr_3d):
    arr_3d = binary_erosion(arr_3d, structure=np.ones((3,3,3)))
    arr_3d = binary_dilation(arr_3d, structure=np.ones((3,3,3)))
    non_zero_coords = np.array(np.nonzero(arr_3d))
    min_coords = non_zero_coords.min(axis=1)
    max_coords = non_zero_coords.max(axis=1)
    return min_coords, max_coords

def spatial_crop_dict(data_dict, z_len_mm = 224, ref_ptv = 'PTV_Total'):

    
    half_size = int(round(z_len_mm / 2 / data_dict['spacing'][2]))
    isocenter = data_dict['isocenter']

    
    ptv_min_coords, ptv_max_coords = get_coords(data_dict[ref_ptv].copy())
    body_min_coords, body_max_coords = get_coords(data_dict['Body'].copy())

    half_size = int(round(z_len_mm / 2 / data_dict['spacing'][2]))


    start_z = min(isocenter[0] - half_size, ptv_min_coords[0] - 8)
    start_z = max(start_z, 0)

    end_z = max(isocenter[0] + half_size, ptv_max_coords[0] + 8)
    end_z = min(end_z, data_dict['Body'].shape[0])

    start_z, end_z = int(round(start_z)), int(round(end_z))
    
    tmp_body = data_dict['Body'].copy()[start_z:end_z, :, :]
    
    body_min_coords, body_max_coords = get_coords(tmp_body)

    start_y = max(body_min_coords[1] - 3, 0)
    end_y = min(body_max_coords[1] + 3, data_dict['Body'].shape[1])

    start_x = max(body_min_coords[2] - 3, 0)
    end_x = min(body_max_coords[2] + 3, data_dict['Body'].shape[2])
    
    start_x, end_x = int(round(start_x)), int(round(end_x)) 
    start_y, end_y = int(round(start_y)), int(round(end_y))

    cropped_dict = {}

    for key in data_dict.keys():

        if isinstance(data_dict[key], np.ndarray) and len(data_dict[key].shape) == 3:
            cropped_dict[key] = data_dict[key][start_z:end_z, start_y:end_y, start_x:end_x]
        elif isinstance(data_dict[key], np.ndarray) and len(data_dict[key].shape) == 2:
            cropped_dict[key] = data_dict[key][start_y:end_y, start_x:end_x]
        else:
            cropped_dict[key] = data_dict[key]

    # update the origin, direction, size, and voxel_spacing
    cropped_dict['size'] = cropped_dict['Body'].shape

    cropped_dict['origin'] = [data_dict['origin'][0] + start_x * data_dict['spacing'][0],  # x, 
                                data_dict['origin'][1] + start_y * data_dict['spacing'][1], # y, 
                                data_dict['origin'][2] + start_z * data_dict['spacing'][2]] # z

    cropped_dict['isocenter'] = [isocenter[0] - start_z, isocenter[1] - start_y, isocenter[2] - start_x] # at voxel space
    return cropped_dict, [start_z, start_y, start_x], [end_z, end_y, end_x]


if __name__ == "__main__": 

    '''
    The sample patient should be downloaded from the HuggingFace repo: https://huggingface.co/datasets/Jungle15/Radiotherapy_HaN_Lung_AIRTP. 
    '''

    sess_path = '/pct_ids/users/z004b27b/data/GDP-HMM_Challenge/DICOM_huggingface/sample_patient/CT'
    rs_path = '/pct_ids/users/z004b27b/data/GDP-HMM_Challenge/DICOM_huggingface/sample_patient/RTSTRUCT/RS.1.2.276.0.7230010.3.1.4.2836367763.1540.1729193423.833.dcm'
    rd_path = '/pct_ids/users/z004b27b/data/GDP-HMM_Challenge/DICOM_huggingface/sample_patient/RTDOSE/RD.1.2.246.352.71.7.413130124983.224673.20241114190315.dcm'
    plan_path = '/pct_ids/users/z004b27b/data/GDP-HMM_Challenge/DICOM_huggingface/sample_patient/RTPLAN/RP.1.2.246.352.71.5.413130124983.235208.20241114185423.dcm'
    img_spac = [2.5, 2.5, 2]

    fill_strcuts = ['PTV_Total', 'Body'] # you may define any fill_strcuts you want to fill the contours. 'PTV_Total' and 'Body' are the minimum to keep the crop be the same as we provided.


    data_dict = get_npz_dict_ref_CT(sess_path, rs_path, rd_path = rd_path, plan_path = plan_path, img_spac = img_spac, need_rtplan = True, version = 2, fill_strcuts=fill_strcuts)
    crop_dict, starts, ends = spatial_crop_dict(data_dict, z_len_mm = 256, ref_ptv = 'PTV_Total')
    
    np.savez_compressed('local_path/sample_patient.npz', crop_dict)
    print ('---------------------please check the functions in toolkit.py for angle plate and beam plate creation!---------------------------')
    print ('npz saved')

