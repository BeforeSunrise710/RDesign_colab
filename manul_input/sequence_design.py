from Bio import SeqIO
from Bio.PDB import PDBParser
import numpy as np
import os
import os.path as osp
import subprocess
import torch
import numpy as np
import sys 
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)
# sys.path.append(root_dir)
import json
import argparse
from new_main import Exp
from tqdm import tqdm
from methods.utils import cuda
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F


def process_single_pdb(pdb_file):
    backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    alphabet_set = 'AUCG'

    pdb_name = osp.basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure('', pdb_file)
    coords = {
        'P': [], "O5'": [], "C5'": [], "C4'": [], "C3'": [], "O3'": []
    }


    # models = structure[0]
    for model in structure:
        chain = list(model.get_chains())[0]
        chain_name = chain.id

        seq = ''  
        coords_dict = {atom_name: [np.nan, np.nan, np.nan] for atom_name in backbone_atoms}

        for residue in chain:
            # if residue.id[0] == " ":  
            seq += residue.get_resname()

            for atom in residue:
                if atom.name in backbone_atoms:
                    coords_dict[atom.name] = atom.get_coord()

            list(map(lambda atom_name: coords[atom_name].append(list(coords_dict[atom_name])), backbone_atoms))


        for atom_name in backbone_atoms:
            assert len(seq) == len(coords[atom_name]), f"Length of sequence {len(seq)} and coordinates {len(coords[atom_name])} for {atom_name} do not match."


        bad_chars = set(seq).difference(alphabet_set)
        if len(bad_chars) != 0:
            print('Found bad characters in sequence:', bad_chars)

        break 

    data = {
        'seq': seq,
        'coords': coords,
        'chain_name': chain_name,
        'name': pdb_name
    }

    return data


def featurize_HC(batch):
    """ Pack and pad batch into torch tensors """
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    # print(L_max)
    # L_max = 2000
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)
    
    ss_pair = []
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        names.append(b['name'])
        
        clus[i] = i

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    clus = torch.from_numpy(clus).to(dtype=torch.long)
    return X, S, mask, lengths, clus, ss_pos, ss_pair, names

def eval_sequence(processed_data):
    
    # pre_base_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    # pre_great_pairs = ((0, 1), (1, 0), (2, 3), (3, 2))

    # svpath = '/tancheng/experiments/RDesign/results/ex_rfam_HCGNN_s222/'
    svpath = osp.join(parent_dir,'ckpt_path/')
    config = json.load(open(svpath+'model_param.json','r'))
    args = argparse.Namespace(**config)
    print(args)
    # args.res_dir = './results'
    # print(args)
    exp = Exp(args)
    exp.method.model.load_state_dict(torch.load(svpath+'checkpoint.pth'))
    exp.method.model.eval()


    alphabet = 'AUCG'
    S_preds, S_trues, name_lst, rec_lst = [], [], [], []
    S_preds_lst, S_trues_lst = [], []
    from API.single_file import Single_input

    single_data = Single_input(processed_data)

    f1s = []
    for idx, sample in enumerate(single_data):
        sample = featurize_HC([sample])
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = sample
        X, S, mask, ss_pos = cuda((X, S, mask, ss_pos), device=exp.device)
        logits, gt_S = exp.method.model.sample(X=X, S=S, mask=mask)
        log_probs = F.log_softmax(logits, dim=-1)
        # secondary sharpen
        # ss_pos = ss_pos[mask == 1].long()
        # log_probs = log_probs.clone()
        # log_probs[ss_pos] = log_probs[ss_pos] / exp.args.ss_temp
        S_pred = torch.argmax(log_probs, dim=1)
        

        _, _, f1, _ = precision_recall_fscore_support(S_pred.cpu().numpy().tolist(), gt_S.cpu().numpy().tolist(), average=None)
        f1s.append(f1.mean())

        S_preds += S_pred.cpu().numpy().tolist()
        S_trues += gt_S.cpu().numpy().tolist()

        S_preds_lst.append(''.join([alphabet[a_i] for a_i in S_pred.cpu().numpy().tolist()]))
        S_trues_lst.append(''.join([alphabet[a_i] for a_i in gt_S.cpu().numpy().tolist()]))
        name_lst.extend(names)

        cmp = S_pred.eq(gt_S)
        recovery_ = cmp.float().mean().cpu().numpy()
        rec_lst.append(recovery_)

    _, _, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)

    return name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <pdb_file_path>, using default setting")
        pdb_file_path = osp.join(root_dir, 'RDesign/example/4tux_rna_C.pdb')
    else:
        pdb_file_path = sys.argv[1]   
    processed_data = process_single_pdb(pdb_file_path)
    name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst = eval_sequence(processed_data)
    for name, f1, rec, s_pred, s_true in zip(name_lst, f1s,rec_lst, S_preds_lst, S_trues_lst):
        if 1:
            print("----------Result------------")
            print("PDB_ID:", name)
            print("F1 Score:", f1,"Recovery Score:",rec)
            print("Predicted Sequence:",s_pred)
            print("True Sequence:",s_true)
    
