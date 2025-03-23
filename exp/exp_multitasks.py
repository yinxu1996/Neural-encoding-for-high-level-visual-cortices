from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, auc
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from data_provider.data_loader import (
    NSDLoader,
)

data_dict = {
    "NSD": NSDLoader,
}

warnings.filterwarnings("ignore")

class Exp_Multitasks(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.smoothing_function = SmoothingFunction().method1

    def _build_model(self):
        # model input depends on data
        model = (
            self.model_dict[self.args.model].Transformer_multitasks(self.args).float()
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model
    
    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = self.data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion
    
    def data_provider(self, args, flag):
        Data = data_dict[args.data]

        data_set = Data(
            root_path=args.root_path,
            cap_len=args.cap_len,
            supercategories = args.supercategories,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        return data_set, data_loader
    
    def vali(self, val_data, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_mAP = 0.0
        with torch.no_grad():
            for i, (fmri, cate, name, new_cap, new_cap_label, mt, sample_index) in enumerate(val_loader):
                batch_fmri = fmri.float().to(self.device)
                batch_cate = cate.squeeze().to(self.device)
                batch_name = name.float().to(self.device)
                batch_cap_input = new_cap.to(self.device)
                batch_cap_output = new_cap_label.to(self.device)
                mt = mt.to(self.device)
                
                # mask
                tgt_mask, tgt_padding_mask = self.model.create_mask(batch_cap_input, self.device)

                pre_cate_label, acc_cate, loss_cate, \
                pre_name_label, loss_name, mAP, \
                loss_cap \
                    = self.model(batch_fmri, mt, batch_cate, batch_name, batch_cap_input, 
                                  batch_cap_output, tgt_mask, tgt_padding_mask)
                loss =  loss_cate + loss_name + loss_cap

                val_acc += acc_cate
                val_mAP += mAP

                val_loss += loss

            self.model.train()
            return val_loss / len(val_loader), val_acc / len(val_loader), val_mAP / len(val_loader)
    
    def train(self):
        train_data, train_loader = self._get_data(flag="train")
        val_data, val_loader = self._get_data(flag="val")

        path = ("./checkpoints/" + self.args.task_name + "/" + self.args.model + "/")
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        loss_plt, acc_plt, mAP_plt = [], [], []
        epoch_times = []
        start_time = time.time() 
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = 0.0
            train_acc = 0.0
            train_mAP = 0.0

            self.model.train()
            epoch_start = time.time()
            
            for i, (fmri, cate, name, new_cap, new_cap_label, mt, sample_index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_fmri = fmri.float().to(self.device) 
                batch_cate = cate.squeeze().to(self.device)
                batch_name = name.float().to(self.device)
                batch_cap_input = new_cap.to(self.device)
                batch_cap_output = new_cap_label.to(self.device)
                mt = mt.to(self.device)

                # mask
                tgt_mask, tgt_padding_mask = self.model.create_mask(batch_cap_input, self.device)
                
                pre_cate_label, acc_cate, loss_cate, \
                pre_name_label, loss_name, mAP, \
                loss_cap \
                    = self.model(batch_fmri, mt, batch_cate, batch_name, batch_cap_input, 
                                  batch_cap_output, tgt_mask, tgt_padding_mask)
                loss = loss_cate + loss_name + loss_cap

                train_acc += acc_cate
                train_mAP += mAP

                train_loss += loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                model_optim.step()
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)
            train_mAP = train_mAP / len(train_loader)
            val_loss, val_acc, val_mAP = self.vali(val_data, val_loader)

            loss_plt.append(val_loss.cpu().numpy().item())
            acc_plt.append(val_acc.cpu().numpy().item())
            mAP_plt.append(val_mAP)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}, Steps: {train_steps},\n"
                    f"Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f},\n"
                    f"Train acc: {train_acc:.3f} Train mAP: {train_mAP:.3f} | Val acc: {val_acc:.3f} Val mAP: {val_mAP:.3f},\n"
                )

            early_stopping(
                val_loss,
                self.model,
                path,
                self.args.subject,
                self.args.seed
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        end_time = time.time()
        average_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = end_time - start_time
        print(f"each epoch time: {average_epoch_time}")
        print(f"training all time: {total_training_time}")
        
        best_model_path = path + "{}_seed{}_checkpoint.pth".format(self.args.subject,self.args.seed)

        self.model.load_state_dict(torch.load(best_model_path))
        
        loss_plt = np.array(loss_plt[:-self.args.patience])
        acc_plt = np.array(acc_plt[:-self.args.patience])
        mAP_plt = np.array(mAP_plt[:-self.args.patience])
        return self.model   
    
    def calculate_ap(self, y_true, y_scores):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = auc(recall, precision)
        return ap

    def calculate_ap_all(self, y_true, y_scores, num_classes):
        ap_list = []
        
        for i in range(num_classes):
            ap = self.calculate_ap(y_true[:, i], y_scores[:, i])
            ap_list.append(ap)   
        return ap_list


