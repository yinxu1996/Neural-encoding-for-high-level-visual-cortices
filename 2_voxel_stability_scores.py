import os 
import numpy as np 
import argparse
from scipy.ndimage import zoom
from scipy.stats import pearsonr

root_path = 'dataset/processed_data/'

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("--sub", type=int, default=1)
args = parser.parse_args()

sub = args.sub

fMRI_info = np.load(root_path + 'subj%02d/roi_vn.npz'%(sub))
fMRI_info = {key: int(fMRI_info[key]) for key in fMRI_info}
voxel_sel_num = min(list(fMRI_info.values()))

def voxel_selection(data_path):
    filenames = []
    for filename in os.listdir(data_path):
        filenames.append(filename)

    roi_names = list(fMRI_info.keys())

    filtered_train_files = [filename for filename in filenames if 'train' in filename]
    train_X_list = []
    for name in roi_names:
        for j in range(len(filtered_train_files)):
            path = data_path + filtered_train_files[j]
            if name in filtered_train_files[j]: 
                if '3fmri' in filtered_train_files[j]:
                    X_I_Y = np.load(path)
                    X_I_Y = X_I_Y / 300
                    train_X_list.append(X_I_Y)

    # 归一化
    train_X_allvoxels = np.concatenate(train_X_list, axis=2)  # 7581,3,8104
    norm_mean_train = np.mean(train_X_allvoxels, axis=0)
    norm_scale_train = np.std(train_X_allvoxels, axis=0, ddof=1)
    train_3fmri = (train_X_allvoxels - norm_mean_train) / norm_scale_train

    fmri_3 = np.transpose(train_3fmri,(2,1,0))
    stability_scores = []
    for v in range(len(fmri_3)):
        one = fmri_3[v]
        r_AB, _ = pearsonr(one[0], one[1])
        r_AC, _ = pearsonr(one[0], one[2])
        r_BC, _ = pearsonr(one[1], one[2])
        s = (r_AB + r_AC + r_BC) / 3
        stability_scores.append(s)
    stability_scores = np.array(stability_scores)
    np.save(data_path + 'all_voxels_stability_scores.npy', stability_scores)

    maxs_voxel = fmri_3[np.argmax(stability_scores)]
    mins_voxel = fmri_3[np.argmin(stability_scores)]
    np.savez(data_path + 'max_min_stability_voxel.npz', maxs_voxel=maxs_voxel, mins_voxel=mins_voxel)
    
    return stability_scores

def fMRI_saved(stability_scores):
    roi_sel_v_index = {}
    for roi_name, vn in fMRI_info.items():
        sel_v_index = np.argsort(stability_scores[:vn])[-voxel_sel_num:]
        roi_sel_v_index[roi_name] = sel_v_index
        train_path = root_path + 'subj%02d/train_roi_mask_'%(sub) + roi_name + '.npy'
        train_ = np.load(train_path) / 300
        train_fmri_sv = train_[:,sel_v_index]

        test_path = root_path + 'subj%02d/test_roi_mask_'%(sub) + roi_name + '.npy'
        test_ = np.load(test_path) / 300
        test_fmri_sv = test_[:,sel_v_index]
        norm_mean_train = np.mean(train_fmri_sv, axis=0)
        norm_scale_train = np.std(train_fmri_sv, axis=0, ddof=1)
        train_fmri_sv_norm = (train_fmri_sv - norm_mean_train) / norm_scale_train
        test_fmri_sv_norm = (test_fmri_sv - norm_mean_train) / norm_scale_train
        np.save('dataset/processed_data/subj{:02d}/train_roi_mask_{}_vs.npy'.format(sub,roi_name), train_fmri_sv_norm)
        np.save('dataset/processed_data/subj{:02d}/test_roi_mask_{}_vs.npy'.format(sub,roi_name), test_fmri_sv_norm)

        stability_scores = stability_scores[vn:]
    np.save('dataset/processed_data/subj{:02d}/roi_sel_v_index.npy'.format(sub), roi_sel_v_index)

if __name__ == "__main__":
    data_path = root_path + 'subj%02d/'%(args.sub)
    stability_scores = voxel_selection(data_path)
    fMRI_saved(stability_scores)
