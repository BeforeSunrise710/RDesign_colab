import os
import os.path as osp
import re
import subprocess
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
import warnings
from multiprocessing import Pool
import random
from datetime import datetime

import os
import numpy as np
import _pickle as cPickle
from tqdm import tqdm

import torch.utils.data as data

warnings.filterwarnings('ignore')

import pickle
rnasolo_path = '/tancheng/data/RNAsolo'

item_lst = []
for item in os.listdir(os.path.join(rnasolo_path, 'cif')):
    item_lst.append(item)
# with open('/tancheng/data/RNAsolo/rnasolo_item.pickle', 'wb') as f:
#     pickle.dump(item_lst, f)

# SAVE THE ITEM LIST IN ORDER
with open(osp.join(rnasolo_path, 'rnasolo_item.pickle'), 'rb') as f:
    item_lst = pickle.load(f)


# FOR TMP CIF FILES FOR EACH CHAIN
header = ['data_pdb\n',
    '#\n',
    'loop_\n',
    '_atom_site.group_PDB\n',
    '_atom_site.id\n',
    '_atom_site.type_symbol\n',
    '_atom_site.label_atom_id\n',
    '_atom_site.label_alt_id\n',
    '_atom_site.label_comp_id\n',
    '_atom_site.label_asym_id\n',
    '_atom_site.label_entity_id\n',
    '_atom_site.label_seq_id\n',
    '_atom_site.pdbx_PDB_ins_code\n',
    '_atom_site.Cartn_x\n',
    '_atom_site.Cartn_y\n',
    '_atom_site.Cartn_z\n',
    '_atom_site.occupancy\n',
    '_atom_site.B_iso_or_equiv\n',
    '_atom_site.pdbx_formal_charge\n',
    '_atom_site.auth_seq_id\n',
    '_atom_site.auth_comp_id\n',
    '_atom_site.auth_asym_id\n',
    '_atom_site.auth_atom_id\n',
    '_atom_site.pdbx_PDB_model_num\n',]

tail = [
    '#'
]

def add_noise(coord, coef, mean=0):
    now = int(datetime.now().timestamp())
    # 使用当前时间作为随机种子
    seed=random.seed(now)
    t = random.random()
    std = 0.01*abs(coord)*coef
    return coord + np.random.normal(mean, std, size=1)


def create_new_cif(pdb_name,coef,id):
    backbone_atoms = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']
    old_cif_path = osp.join(rnasolo_path, 'cif', pdb_name+'.cif')
    old_cif = open(old_cif_path, 'r').readlines()
    parser = MMCIFParser()
    with open(osp.join(rnasolo_path, 'results', pdb_name+'_' +str(id) + '.cif'), 'w') as f:
        f.writelines([line for line in header])
        for c_i, linee in enumerate(old_cif):
            if 'ATOM' in linee:
                # pattern = r"\d+\.\d{3}"
                pattern = r'-?\d+\.\d{3}\s+-?\d+\.\d{3}\s+-?\d+\.\d{3}'
                try:
                    matches = re.findall(pattern, linee)[0].split()
                except:
                    print(linee)
                x, y, z = float(matches[0]), float(matches[1]), float(matches[2])
                x_ = abs(float(add_noise(x,coef)))
                y_ = abs(float(add_noise(y,coef)))
                z_ = abs(float(add_noise(z,coef)))
                rep = [x_, y_,z_]
                for i, match in enumerate(matches):
                    try:
                        linee = linee.replace(match, f"{rep[i]:.3f}")
                    except:
                        print(pdb_name, coef, id)
                f.writelines(linee)

        f.writelines([line for line in tail])
        new_cif_path = osp.join(rnasolo_path,'results', pdb_name+ '_' +str(id) + '.cif' )
    # structure
    parser = MMCIFParser()
    structure = parser.get_structure('', new_cif_path)
    coords = {
        'P': [], 'O5\'': [], 'C5\'': [], 'C4\'': [], 'C3\'': [], 'O3\'': []
    }

    cif_chain_nams = [chain.id for chain in structure.get_chains()]
    cif_seqs = ''
    for chain_name in cif_chain_nams:
        chain = structure[0][chain_name]
        
        for residue in chain:
            coords_dict = {atom_name: [np.nan, np.nan, np.nan] for atom_name in backbone_atoms}
            for atom in residue:
                if atom.name in backbone_atoms:
                    coords_dict[atom.name] = atom.get_coord()
            # list(map(lambda atom_name: coords[atom_name].append(list(atom.get_coord())), backbone_atoms))
            list(map(lambda atom_name: coords[atom_name].append(list(coords_dict[atom_name])), backbone_atoms))

        # check chain seq
        cif_seq = ''.join([residue.resname.strip() for residue in chain])
        cif_seqs = cif_seqs + cif_seq

    assert len(coords['P']) == len(coords['O5\'']) == len(coords['C5\'']) == len(coords['C4\'']) == len(coords['O3\'']) == len(cif_seqs) 
    assert len(cif_seqs) >= 5
    return new_cif_path, old_cif_path, coords


def gengerate_100_cif(cif):
    rms = []
    tms = []
    indx = []
    coord = []
    cif_list=[]
    max_try = 10
    for i in range(1, 101):
        coef = np.random.random()*1.1 + 1e-6
        rmsd = 1.1
        tmscore = 0.7 

        cur_try = 0
        while (float(rmsd) > 1 or float(tmscore) < 0.8) and cur_try < max_try: 
            cur_try += 1
            coef = coef*0.9         
            new, old, coords = create_new_cif(cif,coef,i)
            # TODO check if the condition matches
            try:
                printed_log = subprocess.Popen(["./USalign", new, old], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
                
                rmsd_pattern = r'RMSD=   (\d+\.\d+)' 
                match = re.search(rmsd_pattern, printed_log.decode('utf-8'))
                rmsd = float(match.group(1))

                tm_pattern = r'TM-score= (\d+\.\d+)'
                match = re.search(tm_pattern, printed_log.decode('utf-8'))
                tmscore = float(match.group(1))
            except:
                print(new, old)

        if cur_try >= max_try:
            continue

        rms.append(rmsd)
        tms.append(tmscore)
        indx.append(i)
        coord.append(coords)
    return rms, tms, indx, coord        


import multiprocessing
from multiprocessing import Pool, Manager

num_cpus = multiprocessing.cpu_count()

lst = Manager().list()  
shared_dict = Manager().dict({"min_num": 1e3})
data = cPickle.load(open('/tancheng/data/RNAsolo/train_data.pt','rb'))
data_chunks = np.array_split(data, num_cpus)

def process_chunk(chunk):
  global _min_num, lst
  for entry in tqdm(chunk):
    cif_id = entry['name']
    cur_clus = entry['cluster']
    chain_idxs = entry['chain_idxs']
    seq = entry['seq']
    c_idx = entry['num_chains']
    rms, tms, indx, coords = gengerate_100_cif(cif_id)
    for i in range(len(coords)):
        print(len(coords))
        shared_dict["min_num"] = min(len(coords), shared_dict["min_num"])
        try:
            lst.append(
                {
                'seq': seq,
                'coords': coords[i-1],
                'chain_idxs': chain_idxs,
                'num_chains': c_idx,
                'cluster': cur_clus,
                'name': cif_id,
                'gname': f'{cif_id}_{i}',
                'rmsd':rms[i-1],
                'tm-score':tms[i-1]
                }
            )
        except Exception as e:
            print(i, e)


if __name__ == '__main__':
    
    with Pool(num_cpus) as p:
        p.map(process_chunk, data_chunks)
    # for cur_chunk in data_chunks:
    #     process_chunk(cur_chunk)
    print(shared_dict["min_num"], len(lst))

    cPickle.dump(list(lst), open('/tancheng/experiments/RDesign/data/RNAsolo/train_generated_data.pt', 'wb'))

