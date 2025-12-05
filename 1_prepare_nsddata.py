import os
import sys
import numpy as np
import pandas as pd
import random
random.seed(42)
import re
from tqdm import tqdm
import h5py
import json
import scipy.io as spio
import nibabel as nib
from pycocotools.coco import COCO
import cv2
from collections import Counter

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-path", "--path", help="NSD path", default="dataset/NSD/")
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("--min_word_freq", type=int, default=1)
parser.add_argument("--max_len", type=int, default=40)
args = parser.parse_args()
sub=int(args.sub)
path = args.path
min_word_freq = args.min_word_freq
max_len = args.max_len
roi_name = ['OFA','FFA','OWFA','VWFA','OPA','EBA','FBA']
category_to_index = {'person':0, 'vehicle':1, 'outdoor':2, 'animal':3, 'accessory':4, 'sports':5, 
                     'kitchen':6, 'food':7, 'furniture':8, 'electronic':9, 'appliance':10, 'indoor':11}

assert sub in [1,2,5,7]

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def get_image_supercategory(image_id, coco_train, coco_val, category_to_supercategory):
    coco_allimgs_train = coco_train.getImgIds()
    coco_allimgs_val = coco_val.getImgIds()
    if image_id in coco_allimgs_train:
        # image_url = coco_train.loadImgs([image_id])[0]['coco_url']
        # print(image_url)
        annotation_ids = coco_train.getAnnIds(imgIds=image_id)
        annotations = coco_train.loadAnns(annotation_ids)
    elif image_id in coco_allimgs_val:
        # image_url = coco_val.loadImgs([image_id])[0]['coco_url']
        # print(image_url)
        annotation_ids = coco_val.getAnnIds(imgIds=image_id)
        annotations = coco_val.loadAnns(annotation_ids)

    # 统计每个 supercategory 的出现频率
    supercategory_count = {}
    for ann in annotations:
        category_id = ann['category_id']
        supercategory = category_to_supercategory[category_id]
        if supercategory in supercategory_count:
            supercategory_count[supercategory] += 1
        else:
            supercategory_count[supercategory] = 1
    
    # 找出出现频率最高的 supercategory
    most_frequent_supercategory = max(supercategory_count, key=supercategory_count.get)
    return most_frequent_supercategory

if __name__ == "__main__":

    mask_root = path + 'nsddata/ppdata/'

    stim_order_f = path + 'nsddata/experiments/nsd/nsd_expdesign.mat'
    exp_design = loadmat(stim_order_f)

    nsd_stiminfo_file = path+"nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    stiminfo = pd.read_pickle(nsd_stiminfo_file)  # 73000*40

    subject_idx  = exp_design['subjectim']  # 8*10000
    trial_order  = exp_design['masterordering']  # 1*30000  值为1-10000
    cocoId_arr = np.zeros(shape=subject_idx.shape, dtype=int)  # 8*10000
    for j in range(len(subject_idx)):
        cocoId = np.array(stiminfo['cocoId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
        nsdId = np.array(stiminfo['nsdId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
        imageId = subject_idx[j]-1
        for i,k in enumerate(imageId):
            cocoId_arr[j,i] = (cocoId[nsdId==k])[0]
    
    # Selecting ids for training and test data
    sig_train = {}
    sig_train_cocoId = {}
    sig_test = {}
    sig_test_cocoId = {}
    num_trials = 37*750  # 24980 + 2770 = 27750 fMRI trails
    for idx in range(num_trials):
        ''' nsdId as in design csv files'''
        nsdId = subject_idx[sub-1, trial_order[idx] - 1] - 1
        info = stiminfo[stiminfo['nsdId']==nsdId]
        if info['shared1000'].iloc[0]:
            if nsdId not in sig_test:
                sig_test[nsdId] = []
            sig_test[nsdId].append(idx)
            if nsdId not in sig_test_cocoId:
                sig_test_cocoId[nsdId] = []
            sig_test_cocoId[nsdId].append(info['cocoId'].iloc[0])
        else:
            if nsdId not in sig_train:
                sig_train[nsdId] = []
            sig_train[nsdId].append(idx)
            if nsdId not in sig_train_cocoId:
                sig_train_cocoId[nsdId] = []
            sig_train_cocoId[nsdId].append(info['cocoId'].iloc[0])

    train_im_idx = list(sig_train.keys())
    test_im_idx = list(sig_test.keys())

    '''
    save fmri
    '''
    roi_dir = path + 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
    betas_dir = path + 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)
    
    # faces:OFA, FFA
    voxel_floc_faces = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz"%(sub)).get_fdata()
    OFA_mask = (voxel_floc_faces==1)
    FFA_mask = (voxel_floc_faces==2) | (voxel_floc_faces==3)
    # words:OWFA, VWFA
    voxel_floc_words = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz"%(sub)).get_fdata()
    OWFA_mask = (voxel_floc_words==1)
    VWFA_mask = (voxel_floc_words==2) | (voxel_floc_faces==3)

    # places:OPA
    voxel_floc_places = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz"%(sub)).get_fdata()
    OPA_mask = (voxel_floc_places==1)

    # bodies:EBA, FBA
    voxel_floc_bodies = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz"%(sub)).get_fdata()
    EBA_mask = (voxel_floc_bodies==1)
    FBA_mask = (voxel_floc_bodies==2) | (voxel_floc_bodies==3)
    
    roi_mask = np.stack((OFA_mask, FFA_mask, OWFA_mask, VWFA_mask, 
                         OPA_mask, EBA_mask, FBA_mask), axis=0)
    
    roi_vn = {}
    roi_mask_index = {}
    roi_mask_fmri = {}
    for i, name in enumerate(roi_name):
        mask = roi_mask[i]
        print ("%s \t: %d" % (name, np.sum(mask)))
        roi_vn[name] = np.sum(mask)
        roi_mask_index[name] = mask
        roi_mask_fmri[name] = np.zeros((num_trials, np.sum(mask))).astype(np.float32)
    np.savez('dataset/processed_data/subj{:02d}/roi_vn.npz'.format(sub), **roi_vn)
    np.save('dataset/processed_data/subj{:02d}/roi_mask_index.npy'.format(sub), roi_mask_index)    
    
    for i in tqdm(range(37), desc="Processing"):  # 37 sessions
        beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
        beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)  # (81, 104, 83, 750)  each voxel's 750 trials
        for name in roi_name:
            roi_mask = roi_mask_index[name]
            roi_mask_fmri[name][i*750:(i+1)*750] = beta_f[roi_mask].transpose()
        del beta_f
    print('roi mask fmri data are loaded.')

    num_train, num_test = len(train_im_idx), len(test_im_idx)
    for name in roi_name:
        vox_dim = roi_mask_fmri[name].shape[1]
        fmri = roi_mask_fmri[name]
        fmri_array = np.zeros((num_train,vox_dim))
        for i, idx in enumerate(train_im_idx):
            fmri_array[i] = fmri[sorted(sig_train[idx])].mean(0)
        np.save('dataset/processed_data/subj{:02d}/train_roi_mask_{}.npy'.format(sub,name),fmri_array)
        
        fmri_array = np.zeros((num_test,vox_dim))
        for i, idx in enumerate(test_im_idx):
            fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)    
        np.save('dataset/processed_data/subj{:02d}/test_roi_mask_{}.npy'.format(sub,name),fmri_array)
    print("fMRI data is saved.")

    num_train, num_test = len(train_im_idx), len(test_im_idx)
    for name in roi_name:
        vox_dim = roi_mask_fmri[name].shape[1]
        fmri = roi_mask_fmri[name]
        fmri_3_array = []
        for i, idx in enumerate(train_im_idx):
            if len(sig_train[idx]) == 3:
                fmri_3_array.append(fmri[sig_train[idx]])
        fmri_3_array = np.array(fmri_3_array)    
        np.save('dataset/processed_data/subj{:02d}/train_roi_mask_{}_3fmri.npy'.format(sub,name),fmri_3_array)       
    print("3 fMRI data is saved.")
    
    '''
    save images
    '''
    f_stim = h5py.File(path + 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    stim = f_stim['imgBrick'][:]
    print('Stimuli are loaded.')

    num_train, num_test = len(train_im_idx), len(test_im_idx)
    im_dim, im_c = 256, 3
    stim_array = np.zeros((num_train, im_c, im_dim, im_dim))
    for i,idx in enumerate(train_im_idx):
        img = cv2.resize(stim[idx], (im_dim,im_dim))
        img = img.transpose(2,0,1)
        stim_array[i] = img/255.
    np.save('dataset/processed_data/subj{:02d}/train_stim.npy'.format(sub), stim_array)
    stim_array = np.zeros((num_test, im_c, im_dim, im_dim))
    for i,idx in enumerate(test_im_idx):
        img = cv2.resize(stim[idx], (im_dim,im_dim))
        img = img.transpose(2,0,1)
        stim_array[i] = img/255. 
    np.save('dataset/processed_data/subj{:02d}/test_stim.npy'.format(sub), stim_array)
    print('Stimuli data is saved.')
    
    '''
    save captions
    '''
    annots_cur = np.load(path + 'annots/COCO_73k_annots_curated.npy')
    print('Caption are loaded.')

    num_train, num_test = len(train_im_idx), len(test_im_idx)
    word_freq = Counter()
    train_captions = np.empty((num_train,),dtype=annots_cur.dtype)
    for i,idx in enumerate(train_im_idx):
        captions_array = annots_cur[idx,:]
        clean_captions_array = []
        for text in captions_array:
            clean_text = text.replace("\n", ' ')
            clean_text = clean_text.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)  # 去除标点符号
            clean_text = re.sub(r'\d+', ' ', clean_text)  # 去除数字
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # 去掉多余的空格和首尾空格        
            clean_captions_array.append(clean_text)
            clean_tokens = clean_text.split()
            word_freq.update(clean_tokens)
        non_empty_captions = [caption for caption in clean_captions_array if caption.strip() != ""]
        train_captions[i] = random.choice(non_empty_captions)
    # np.save('dataset/processed_data/subj{:02d}/train_cap.npy'.format(sub), train_captions)

    test_captions = np.empty((num_test,),dtype=annots_cur.dtype)
    for i,idx in enumerate(test_im_idx):
        captions_array = annots_cur[idx,:]
        clean_captions_array = []
        for text in captions_array:
            clean_text = text.replace("\n", ' ')
            clean_text = clean_text.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)  # 去除标点符号
            clean_text = re.sub(r'\d+', ' ', clean_text)  # 去除数字
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # 去掉多余的空格和首尾空格        
            clean_captions_array.append(clean_text)
            clean_tokens = clean_text.split()
            word_freq.update(clean_tokens)
        non_empty_captions = [caption for caption in clean_captions_array if caption.strip() != ""]
        test_captions[i] = random.choice(non_empty_captions)
    # np.save('dataset/processed_data/subj{:02d}/test_cap.npy'.format(sub), test_captions)
    # print("Caption data are saved.")

    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    with open(os.path.join('dataset/processed_data/subj' + f"{sub:02d}" + '/WORDMAP.json'), 'w') as j:
        json.dump(word_map, j)


    train_enc_captions = []
    train_caplens = []
    for j, c in enumerate(train_captions):
        # Encode captions
        c = c.split()
        if len(c) > 40:
            enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c[:40]] + [
                    word_map['<end>']] 
            c_len = len(enc_c)
        else: 
            enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
            c_len = len(c) + 2
            
        train_enc_captions.append(enc_c)
        train_caplens.append(c_len)
    # Save encoded captions and their lengths to JSON files
    with open(os.path.join('dataset/processed_data/subj' + f"{sub:02d}" + '/train_captions.json'), 'w') as j:
        json.dump(train_enc_captions, j)

    with open(os.path.join('dataset/processed_data/subj' + f"{sub:02d}" + '/train_caplens.json'), 'w') as j:
        json.dump(train_caplens, j)

        
    test_enc_captions = []
    test_caplens = []
    for j, c in enumerate(test_captions):
        # Encode captions
        c = c.split()
        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
        
        # Find caption lengths
        c_len = len(c) + 2
        test_enc_captions.append(enc_c)
        test_caplens.append(c_len)
    # Save encoded captions and their lengths to JSON files
    with open(os.path.join('dataset/processed_data/subj' + f"{sub:02d}" + '/test_captions.json'), 'w') as j:
        json.dump(test_enc_captions, j)

    with open(os.path.join('dataset/processed_data/subj' + f"{sub:02d}" + '/test_caplens.json'), 'w') as j:
        json.dump(test_caplens, j)

    '''
    save supercategories
    '''
    train_annotation_file = path + 'annots/instances_train2017.json'
    coco_train = COCO(train_annotation_file)
    val_annotation_file = path + 'annots/instances_val2017.json'
    coco_val = COCO(val_annotation_file)

    categories = coco_train.loadCats(coco_train.getCatIds()) 
    category_to_supercategory = {cat['id']: cat['supercategory'] for cat in categories}
    
    supercategories_list = []
    for i,idx in enumerate(sig_train_cocoId.values()):
        supercategories_list.append(get_image_supercategory(idx[0], coco_train, coco_val, category_to_supercategory))
    supercategories = np.array([category_to_index[supercategory] for supercategory in supercategories_list])
    np.save('dataset/processed_data/subj{:02d}/train_supercategories.npy'.format(sub), supercategories)

    supercategories_list = []
    for i,idx in enumerate(list(sig_test_cocoId.values())):
        supercategories_list.append(get_image_supercategory(idx[0], coco_train, coco_val, category_to_supercategory))
    supercategories = np.array([category_to_index[supercategory] for supercategory in supercategories_list])
    np.save('dataset/processed_data/subj{:02d}/test_supercategories.npy'.format(sub), supercategories)
