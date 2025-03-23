import argparse
import os
import torch
from exp.exp_multitasks import Exp_Multitasks
import random
import numpy as np
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--model", type=str, default='Cogformer'
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="random seed"
    )

    parser.add_argument(
        "--subject", type=str, default='subj01'
    )
    # data loader
    parser.add_argument(
        "--data", type=str, default="NSD", help="dataset type"
    )
    parser.add_argument(
        "--root_path", 
        type=str, 
        default="dataset/NSD/processed_data/",
        help="root path of the data file"
    )  
 
    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1", help="device ids of multiple gpus"
    )

    args = parser.parse_args()
    args.task_name = 'multitasks'
    args.root_path = 'dataset/NSD/processed_data/'+args.subject
    
    fMRI_info = np.load(args.root_path + '/roi_vn.npz')
    fMRI_info = {key: int(fMRI_info[key]) for key in fMRI_info}
    voxel_sel_num = min(list(fMRI_info.values()))

    vocabs_cap = json.load(open(args.root_path + '/vocabs_cap.json', 'r', encoding='utf-8'))

    args.data = 'NSD'
    args.batch_size = 128
    args.enc_in = voxel_sel_num
    args.brain_roi = 14
    args.dec_in = len(vocabs_cap['id2tokens'])
    args.e_model = 256
    args.d_model = 256
    args.d_ff = 512
    args.n_heads = 8
    args.e_layers = 3
    args.d_layers = 3
    args.dropout = 0.3
    args.learning_rate = 0.0001
    args.train_epochs = 300
    args.patience = 10
    args.cap_len = 50  # cap length
    args.supercategories = 12
    args.labels = 80
    args.beam_size = 3

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print("Args in experiment:")
    print(args)
    
    Exp = Exp_Multitasks
    
    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    exp = Exp(args)
    print(
        ">>>>>>>start training: {}, {}, seed{} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(args.model, args.subject, args.seed)
    )
    exp.train()