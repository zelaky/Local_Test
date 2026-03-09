import os
import numpy as np
import time
import argparse
import pickle
import pandas as pd
import glob
import json
import tqdm
from itertools import product
from tqdm import trange, tqdm

import random

import scipy
import sklearn
from sklearn.model_selection import KFold

from datetime import datetime
from omegaconf import DictConfig, open_dict, OmegaConf, ListConfig

import warnings
import sys

sys.path.append('../')
from src.data_class import matrix_class
from src.ICQF import ICQF
from src.LICQF import LICQF
from src.utils_LICQF import intervention_aware_initialize
from utils.utils_statistics import compute_CI, compute_consensus_LICQF, plot_factor
from utils.utils import get_by_subj, bestmatch
def main():

    #############################
    # load base config
    cfg = OmegaConf.load("/data/NNT/Zoe/Dissertation_2025/LICQF/src/base_config_consensus_20260304.yaml")

    cfg.version = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.dir_output = os.path.join(cfg.dir_output, cfg.dataname, cfg.version)

    print('version : ', cfg.version)
    print('output directory : ', cfg.dir_output)

    #############################
    # random seed for reproduciility
    np.random.seed(cfg.random_state)
    random.seed(cfg.random_state)

    
    #############################
    # create directory for output
    cfg.dir_bootstrap = os.path.join(cfg.dir_output, 'bootstrap')
    cfg.dir_log = os.path.join(cfg.dir_output, 'log')
    
    os.makedirs(cfg.dir_output, exist_ok=True)
    os.makedirs(cfg.dir_bootstrap, exist_ok=True)
    os.makedirs(cfg.dir_log, exist_ok=True)

    
    #############################
    # load data
    # all matrices should be ready as the input argument for LICQF algorithm
    # see load_data.ipynb for details on preparation of all matrices
    M_df = pd.read_csv(cfg.path_M, index_col=0)
    M_raw_df = pd.read_csv(cfg.path_M_raw, index_col=0)
    nan_mask_df = pd.read_csv(cfg.path_nan_mask, index_col=0)
    D_df = pd.read_csv(cfg.path_D, index_col=0)
    agebin_df = pd.read_csv(cfg.path_agebin, index_col=0)

    if cfg.path_C is not None:
        C_df = pd.read_csv(cfg.path_C, index_col=0)
        assert (M_df.index == C_df.index).all()
        C = C_df.values
    else:
        C_df = None
        C = None
    
    subjlist = M_df.index
    assert (M_df.index == M_raw_df.index).all()
    assert (M_df.index == nan_mask_df.index).all()
    assert (M_df.index == D_df.index).all()
    

    M = M_df.values
    M_raw = M_raw_df.values
    nan_mask = nan_mask_df.values
    D = D_df.values
    agebin_indicator = agebin_df.values

    
    # load matrix class
    MF_data = matrix_class(
        M=M,
        M_raw=M_raw,
        nan_mask=nan_mask,
        C=C,
        subjlist=subjlist,
    )
    MF_data.D = D

    # model configuration
    #############################
    
    optimal_dimension = cfg.dimension
    optimal_W_beta = cfg.W_beta
    optimal_Q_beta = cfg.Q_beta

    #############################
    
    # save M, M_raw, nan_mask, C, D for record
    #############################
    
    cfg.path_matrix_class = os.path.join(cfg.dir_output, 'matrix_class.pickle')
    with open( cfg.path_matrix_class, 'wb' ) as file:
        pickle.dump(MF_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    # save it into csv data format for storage
    M_df.to_csv(os.path.join(cfg.dir_output, 'M.csv'), index=True)
    M_raw_df.to_csv(os.path.join(cfg.dir_output, 'M_raw.csv'), index=True)
    nan_mask_df.to_csv(os.path.join(cfg.dir_output, 'nan_mask.csv'), index=True)
    D_df.to_csv(os.path.join(cfg.dir_output, 'D.csv'), index=True)
    agebin_df.to_csv(os.path.join(cfg.dir_output, 'agebin_indicator.csv'), index=True)
    
    if C_df is not None:
        C_df.to_csv(os.path.join(cfg.dir_output, 'C.csv'), index=False)
    
    #############################
    # create swarm path
    cfg.path_swarm = os.path.join(cfg.dir_output, "swarm" + '_' + cfg.version + ".swarm")

    
    #############################
    # save config
    OmegaConf.save(config=cfg, f=os.path.join(cfg.dir_output, 'config.yaml'))

    
    #############################
    # Create swarm
    ### create swarm file
    swarm_file = open(cfg.path_swarm, "w")

    ### start writing swarm file line by line
    for i in tqdm(range(cfg.replicate)):

        dir_i = os.path.join(cfg.dir_bootstrap, 'replicate_{}'.format(i))
        os.makedirs(dir_i, exist_ok=True)

        if cfg.bootstrap:
            _train = random.choices(list(M_df.index), k=len(M_df.index))
        else:
            _train = copy.deepcopy(list(M_df.index))

        path_train_index = os.path.join(dir_i, 'train_index.csv')
        _train_df = pd.DataFrame(_train, columns=['train_index'])
        _train_df.to_csv(path_train_index, index=True)

        # train_matrix = get_by_subj(MF_data, _train)
            

        # run specific data/output path
        # path_bcv_mask = os.path.join(cfg.dir_bcv_mask, "mask-{}-{}.npz".format(r, j))
        # path_stat = os.path.join(cfg.dir_stat, "cv-d{}-bW{}-bQ{}-r{}-j{}.npy".format(hp[0], hp[1], hp[2], r, j))

        # data argument
        args  = ' --path_matrix_class {}'.format(cfg.path_matrix_class)
        args += ' --path_agebin {}'.format( os.path.join(cfg.dir_output, 'agebin_indicator.csv') )
        args += ' --path_train_index {}'.format( path_train_index )

        # model argument
        args += ' --method {}'.format( cfg.method )
        args += ' --n_components {}'.format( cfg.dimension )
        args += ' --W_beta {}'.format( cfg.W_beta )
        args += ' --Q_beta {}'.format( cfg.Q_beta)
        args += ' --regularizer {}'.format( cfg.regularizer )
        args += ' --rho {}'.format(cfg.rho)
        args += ' --W_upperbd {}'.format(cfg.W_upperbd)
        args += ' --W_max {}'.format(cfg.W_max)
        args += ' --Q_upperbd {}'.format(cfg.Q_upperbd)
        args += ' --Q_max {}'.format(cfg.Q_max)
        args += ' --M_upperbd {}'.format(cfg.M_upperbd)
        args += ' --M_max {}'.format(cfg.M_max)
        args += ' --weighted_mask {}'.format(cfg.weighted_mask)
        args += ' --max_iter {}'.format(cfg.max_iter)

        args += ' --random_state {}'.format(int(cfg.random_state) + i)
        args += ' --verbose {}'.format(cfg.verbose)

        # output argument
        args += ' --dir_output {}'.format( dir_i )
        args += ' \n'

        swarm_file.write("cd ./; python ./run_consensus_LICQF.py" + args)
                
    swarm_file.close()

    #############################
    # generate terminal command to run swarm
    
    ### general setting
    swarm_runcode = ' swarm -f '
    swarm_runcode += ' ' + cfg.path_swarm + ' '
    swarm_runcode += ' --logdir ' + cfg.dir_log
    swarm_runcode += ' --noht '

    ### biowulf setting
    swarm_runcode += ' --partition ' + cfg.partition + ' '
    swarm_runcode += ' --gb-per-process ' + cfg.gb_per_process + ' '
    swarm_runcode += ' --time ' + cfg.time + ' '
    swarm_runcode += ' --gres=lscratch:10 '
    
    swarm_runcode += ' -b ' + cfg.batch + ' '

    ### print command
    print('=============Sample swarm command==============')    
    print(swarm_runcode)
    print('===============================================')
    
    if cfg.run_swarm:
        os.system(swarm_runcode)


if __name__ == '__main__':
    main()
