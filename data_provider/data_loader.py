import os
import numpy as np
import re
import random
import json
import torch
from torch.utils.data import Dataset
from data_provider.uea import normalize_batch_ts
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

class NSDLoader(Dataset):
    def __init__(self, root_path, cap_len, supercategories, flag=None):
        self.root_path = root_path

        self.fmri, self.cate, self.name, self.new_cap, \
        self.new_cap_label, self.mt, self.sample_index = self.load_NSD(self.root_path, cap_len, supercategories, flag=flag)
        
    def load_NSD(self, data_path, cap_len, supercategories, flag=None):
        filenames = []
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()

        roi_names = ['V1d', 'V2d', 'V3d', 'OPA', 'EBA',
                     'V1v', 'V2v', 'V3v', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'FBA']

        filtered_files = [filename for filename in filenames if flag in filename]

        fmri_list = []
        cate_list = []
        name_list = []
        cap_list = []
        for name in roi_names:
            for j in range(len(filtered_files)):
                path = data_path + '/' + filtered_files[j]
                if name in filtered_files[j]: 
                    if 'vs' in filtered_files[j]:
                        data = np.load(path)
                        fmri_list.append(data)
        for j in range(len(filtered_files)):
            path = data_path + '/'  + filtered_files[j]
            if 'supercategories' in filtered_files[j]:
                data = np.load(path)
                cate_list.append(data)
            elif 'name' in filtered_files[j]:
                data = np.load(path)
                name_list.append(data)
            elif 'cap' in filtered_files[j]:
                data = np.load(path)
                cap_list.append(data)

        fmri = np.array(fmri_list).transpose(1,0,2)

        cate = np.expand_dims(cate_list[0], axis=1)

        name = name_list[0]

        # -------------------------------------------------------------------------------
        vocabs_cap = json.load(open(self.root_path + '/vocabs_cap.json', "r", encoding="utf-8"))
        
        new_cap = []
        new_cap_label = []
        new_cap_size = []
        for i in range(len(cap_list[0])):
            text = cap_list[0][i].lower()

            for sc in vocabs_cap['spe_chars']:
                if sc in text:
                    text = text.replace(sc, f' {sc} ')
            text = ' '.join(text.split())

            temp = [vocabs_cap['tokens2id']['<sos>']]
            temp2 = []
            for token in text.split():
                temp.append(vocabs_cap['tokens2id'][token])
                temp2.append(vocabs_cap['tokens2id'][token])
            temp2.append(vocabs_cap['tokens2id']['<eos>'])

            if len(temp) > cap_len:
                new_cap_size.append(cap_len)
                temp = temp[:cap_len]
                temp2 = temp2[:cap_len]
            else:
                new_cap_size.append(len(temp))
                temp += [0] * (cap_len - len(temp))
                temp2 += [0] * (cap_len - len(temp2))

            new_cap.append(temp)
            new_cap_label.append(temp2)
        new_cap = np.array(new_cap)
        new_cap_label = np.array(new_cap_label)
        
        """  Multi-scale tokens"""
        mt = np.zeros([len(fmri), 3])
        mt[:, 0] = len(fmri) * [vocabs_cap['tokens2id']['<ls>']]
        mt[:, 1] = len(fmri) * [vocabs_cap['tokens2id']['<ms>']]
        mt[:, 2] = len(fmri) * [vocabs_cap['tokens2id']['<ss>']]
        mt = np.int32(mt)

        sample_index = np.arange(len(cate))[:,np.newaxis]

        fmri, cate, name, new_cap, new_cap_label, mt, sample_index = shuffle(fmri, cate, name, new_cap, new_cap_label, mt, sample_index, random_state=42)
        return fmri, cate, name, new_cap, new_cap_label, mt, sample_index

    def __getitem__(self, index):
        return torch.from_numpy(self.fmri[index]), torch.from_numpy(self.cate[index]), \
            torch.from_numpy(self.name[index]), torch.from_numpy(self.new_cap[index]), \
            torch.from_numpy(self.new_cap_label[index]), torch.from_numpy(self.mt[index]), \
            torch.from_numpy(self.sample_index[index])

    def __len__(self):
        return len(self.new_cap)
