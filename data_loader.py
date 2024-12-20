from monai.transforms import (
    Compose,
    Resized,
    RandFlipd, 
    RandRotated,
    SpatialPadd, 
    SpatialCropd,
    RandSpatialCropd
)

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb

from scipy import ndimage 

import random


OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

OAR_DICT = {OAR_LIST[i]: (i+1) for i in range(len(OAR_LIST))}


def combine_oar(tmp_dict, need_list,norm_oar = True):
    
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
       
       tmp_ptv = tmp_dict[key] * scaled_dose_dict[key]  #(tmp_detail_dict[tv_name][key]['D98'] + tmp_detail_dict[tv_name][key]['D95'] ) / scale / 2
       
       prescribed_dose[cnt] =  scaled_dose_dict[key] #(tmp_detail_dict[tv_name][key]['D98'] + tmp_detail_dict[tv_name][key]['D95'] ) / scale / 2
       
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
        #ToTensord(keys = KEYS),
    ])


class MyDataset(Dataset):
    
    def __init__(self, cfig, phase):
        '''
        phase: train, validation, or testing 
        
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        
        df = pd.read_csv(cfig['csv_root'])
        
        df = df.loc[df['phase'] == phase]

        self.phase = phase
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        if len(str(PatientID)) < 3:
            PatientID = f"{PatientID:0>3}"

        data_npz = np.load(data_path, allow_pickle=True)

        In_dict = dict(data_npz)['arr_0'].item()

        In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 
        
        ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
        In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] 

        norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict['PTVHighOPT'].astype('bool')], 3) + 1e-5) # D97
        In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
        
        In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)


        isocenter = In_dict['isocenter']

        angle_plate_3D = np.zeros(In_dict['img'].shape)

        z_begin = int(isocenter[0]) - 5
        z_end = int(isocenter[0]) + 5
        z_begin = max(0, z_begin)
        z_end = min(angle_plate_3D.shape[0], z_end)
        D3_plate = np.repeat(In_dict['angle_plate'][np.newaxis, :, :], z_end - z_begin, axis = 0)
        if D3_plate.shape[1] != angle_plate_3D.shape[1] or D3_plate.shape[2] != angle_plate_3D.shape[2]:
            D3_plate = ndimage.zoom(D3_plate, (1, angle_plate_3D.shape[1] / D3_plate.shape[1], angle_plate_3D.shape[2] / D3_plate.shape[2]), order = 0)
        angle_plate_3D[z_begin: z_end] = D3_plate

        In_dict['angle_plate'] = angle_plate_3D
        
        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)
        
        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.aug = tt_augmentation(KEYS, self.cfig['in_size'],  self.cfig['out_size'], isocenter)
            else:
                self.aug = tr_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)
            
        if self.phase in ['val', 'test', 'valid', 'external_test']:
            self.aug = tt_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)


        In_dict = self.aug(In_dict)


        data_dict = dict()  
        try:
            need_list = list(self.pat_obj_dict[ID.split('+')[0]].values())[0] # list(a.values())[0]
        except:
            need_list = OAR_LIST
            print (ID.split('+')[0], ID.split('+')[1], '-------------not in the pat_obj_dict')
        comb_oar, cat_oar  = combine_oar(In_dict, need_list, self.cfig['norm_oar'])

        opt_dose_dict = {}
        dose_dict = {}
        for key in self.scale_dose_Dict[PatientID].keys():
            if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                opt_dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                if key != 'PTV_High':
                    dose_dict[self.scale_dose_Dict[PatientID][key]['StructName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                else:
                    dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                
            
        comb_optptv, prs_opt, cat_optptv  = combine_ptv(In_dict, opt_dose_dict)
        comb_ptv, _, cat_ptv  = combine_ptv(In_dict, dose_dict)

        

        data_dict['angle_plate'] = In_dict['angle_plate']
        data_dict['beam_plate'] = In_dict['beam_plate']

        data_dict['label'] = In_dict['dose'] * In_dict['Body'] 
        data_dict['prompt'] = torch.tensor([In_dict['isVMAT'], len(prs_opt), self.site_list[index], self.cohort_list[index]]).float()
        
        prompt_extend = data_dict['prompt'][None, :, None, None].repeat(1, self.cfig['out_size'][0] // 4, self.cfig['out_size'][1], self.cfig['out_size'][2])
        
        if self.cfig['CatStructures']:
            data_dict['data'] = torch.cat((cat_optptv, cat_ptv, cat_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0)
        else:
            data_dict['data'] = torch.cat((comb_optptv, comb_ptv, comb_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0) # , In_dict['obj_2DGy'], In_dict['obj_2DWei']
        

        data_dict['oar'] = cat_oar
        data_dict['ptv'] = cat_ptv
        data_dict['optptv'] = cat_optptv

        prescribed_dose = [self.scale_dose_Dict[PatientID]['PTV_High']['PDose']]
        if 'PTV_Mid' in self.scale_dose_Dict[PatientID].keys():
            prescribed_dose.append(self.scale_dose_Dict[PatientID]['PTV_Mid']['PDose'])
        else:
            prescribed_dose.append(0)
        if 'PTV_Low' in self.scale_dose_Dict[PatientID].keys():
            prescribed_dose.append(self.scale_dose_Dict[PatientID]['PTV_Low']['PDose'])
        else:
            prescribed_dose.append(0)

        data_dict['prescrbed_dose'] = torch.tensor(prescribed_dose).float()

        del In_dict

        data_dict['id'] = ID

        return data_dict
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
    def train_dataloader(self):
        dataset = MyDataset(self.cfig,  phase='train') 
        return DataLoader(dataset, batch_size=self.cfig['train_bs'],  shuffle=True, num_workers=self.cfig['num_workers'])
    
    def val_dataloader(self):
        dataset = MyDataset(self.cfig, phase='valid') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])
    
    def test_dataloader(self):
        dataset = MyDataset(self.cfig, phase='test') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])

if __name__ == '__main__':

    cfig = {
            'train_bs': 2,
             'val_bs': 2, 
             'num_workers': 0, 
             'csv_root': 'meta_data.csv',
             'scale_dose_dict': 'PTV_DICT.json',
             'pat_obj_dict': 'Pat_Obj_dict.json',
             'down_HU': -1000,
             'up_HU': 1000,
             'denom_norm_HU': 500,
             '3d_scale': 5,
             'in_size': (96, 128, 144), 
             'out_size': (96, 128, 144), 
             'norm_oar': True,
             'CatStructures': False,
             'dose_div_factor': 10 
             }
    
    loaders = GetLoader(cfig)
    train_loader = loaders.train_dataloader()

    for i, data in enumerate(train_loader):
        
        print (data['data'].shape, data['label'].shape, data['beam_plate'].shape, data['prompt'].shape)

        
    
