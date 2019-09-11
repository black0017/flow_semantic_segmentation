""" Calculate error metrics for KITTI15. Ground truth and estimation images
should be in folders and with common filenames. """
from __future__ import division
import os
import numpy as np
import sys
sys.path.insert(0, '/')

import flowlib

def evaluate_aee_directories(estimation_folder, ground_truth_folder):
    """ Inputs: The directories with the estimated and the ground truth flows.
        Output: The mean average endpoint error over the valid pixels. """
    estimation_path = os.path.realpath(estimation_folder)
    ground_truth_path = os.path.realpath(ground_truth_folder)
    estfiles = sorted(os.listdir(estimation_path))
    gtfiles = sorted(os.listdir(ground_truth_path))
    estfiles = [estimation_path + '/' + fil for fil in estfiles]
    gtfiles = [ground_truth_path + '/' + fil for fil in gtfiles]
    print(estfiles)
    for i, fil in enumerate (estfiles):
        print(estfiles[i], gtfiles[i])
    # Check filenames
    for i in range(len(estfiles)):
        assert estfiles[i].split('/')[-1] == gtfiles[i].split('/')[-1], \
            "Filenames should be the same"

    error_total = 0
    pixels_total = 0
    for i, est in enumerate(estfiles):
        # 2 KITTI specific lines
        gt, valid = flowlib.read_gen_flow(gtfiles[i])
        flow, valid2 = flowlib.read_gen_flow(est)



        ##########################################################################

        print('--------------------------------------')
        print(flow.shape, gt.shape)
        print(valid.shape, valid2.shape)
        #print( np.max(valid)  )
        print('sums !!!!!!!!!!!!')
        print( np.sum(valid)  ) # everywhere False 
        print( np.sum(valid2) ) # everywhere True 
        
        #print( np.max(valid2)  )
        print('--------------------------------------')

        assert gt.shape == flow.shape, "Shapes should be equal"
        outlier_prct = flowlib.calc_aee(flow, gt, valid2)

        error_total += outlier_prct * np.sum(valid2)
        pixels_total += np.sum(valid2)

    print(error_total, pixels_total)
    return (error_total / pixels_total)


def evaluate_outlier_directories(estimation_folder, ground_truth_folder):
    """ Inputs: The directories with the estimated and the ground truth flows.
        Output: The outlier percentage over the valid pixels. """
    estimation_path = os.path.realpath(estimation_folder)
    ground_truth_path = os.path.realpath(ground_truth_folder)
    estfiles = sorted(os.listdir(estimation_path))
    gtfiles = sorted(os.listdir(ground_truth_path))
    estfiles = [estimation_path + '/' + fil for fil in estfiles]
    gtfiles = [ground_truth_path + '/' + fil for fil in gtfiles]

    # Check filenames
    for i in range(len(estfiles)):
	print(estfiles[i], gtfiles[i])
        assert estfiles[i].split('/')[-1] == gtfiles[i].split('/')[-1], \
            "Filenames should be the same"
        
	
    outliers_total = 0
    pixels_total = 0
    for i, est in enumerate(estfiles):
        gt, valid = flowlib.read_flow_kitti(gtfiles[i])
        flow, valid2 = flowlib.read_flow_kitti(est)
        assert gt.shape == flow.shape, "Shapes should be equal"
        outlier_prct = flowlib.calc_outliers(flow, gt, valid)

        outliers_total += outlier_prct * np.sum(valid)
        pixels_total += np.sum(valid)

    return outliers_total / pixels_total


#outlier = evaluate_outlier_directories('converted', 'finalTest')
#print('outlier', outlier)

aee = evaluate_aee_directories('converted', 'GT_test')
print('aee', aee)

