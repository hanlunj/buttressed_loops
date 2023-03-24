import pyrosetta
import pyrosetta.toolbox.numpy_utils as np_utils
import itertools
from itertools import chain, combinations
import argparse
import sys
import os
import numpy as np
import glob
import random
import time
from sklearn.decomposition import PCA



# ideal values for the peptide bond (taken from the alanine params file)
_peptide_bond_params = {'atom1': 'C', 'atom2': 'N', 'bond_length': 1.328685,
                        'angle1': 116.199993, 'angle2': 121.699997,
                        'torsion': 180.}

_DEBUG = False


def rmsd_2_np_arrays(crds1,
                     crds2):
        #"""Returns RMSD between 2 sets of [nx3] numpy array"""
        #D assert(crds1.shape[1] == 3)
        #D assert(crds1.shape == crds2.shape)

        ##Corrected to account for removal of the COM
        COM1 = np.sum(crds1,axis=0) / crds1.shape[0]
        COM2 = np.sum(crds2,axis=0) / crds2.shape[0]
        crds1-=COM1
        crds2-=COM2
        n_vec = np.shape(crds1)[0]
        correlation_matrix = np.dot(np.transpose(crds1), crds2)
        v, s, w_tr = np.linalg.svd(correlation_matrix)
        is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
        if is_reflection:
                s[-1] = - s[-1]
                v[:,-1] = -v[:,-1]
        E0 = sum(sum(crds1 * crds1)) + sum(sum(crds2 * crds2))
        rmsd_sq = (E0 - 2.0*sum(s)) / float(n_vec)
        rmsd_sq = max([rmsd_sq, 0.0])

        return np.sqrt(rmsd_sq)

def rmsd_by_ndxs_atoms(pose1,
                     init_res1,
                     end_res1,
                     pose2,
                     init_res2,
                     end_res2,
                    target_atoms=["CA","C","O","N"]):

    numRes=(end_res1-init_res1+1)
    coorA=np.zeros(((len(target_atoms)*numRes),3), float)
    coorB=np.zeros(((len(target_atoms)*numRes),3), float)

    counter=0
    for ires in range (init_res1, (end_res1+1)):
        for jatom in target_atoms:
            for dim in range(0,3):
                coorA[counter,dim]=(pose1.residue(ires).xyz(jatom)[dim])
            counter+=1

    counter=0
    for ires in range (init_res2, (end_res2+1)):
        for jatom in target_atoms:
            for dim in range(0,3):
                coorB[counter,dim]=(pose2.residue(ires).xyz(jatom)[dim])
            counter+=1

    #Calculate the RMSD
    rmsdVal = rmsd_2_np_arrays(coorB, coorA)

    return rmsdVal

def bbrmsd_check(p, posepool, loop_s, loop_e, cutoff=0.5):
    rmsd_check = True
    replace_check = -1
    for i in range(len(posepool)):
        if p.size() != posepool[i][0].size():
            continue
        else:
            bbrmsd = rmsd_by_ndxs_atoms(p, loop_s, loop_e, posepool[i][0], loop_s, loop_e)
            if bbrmsd <= cutoff:
                rmsd_check = False
                replace_check = i
                break

    return [rmsd_check, replace_check]


def find_num_repeats_by_rmsd(pose, max_rmsd=0.5, max_num_repeats=6):
    num_repeats_curr = -1
    num_repeats = 2
    while num_repeats < max_num_repeats:
        num_repeats += 1
        #print('\n{}'.format(num_repeats))
        repeat_len = int(pose.size()/num_repeats)
        if pose.size()%repeat_len != 0:
            continue
        repeat1 = pose.clone()
        repeat1.delete_residue_range_slow(repeat_len+1, pose.size())
        repeat2 = pose.clone()
        repeat2.delete_residue_range_slow(2*repeat_len+1, pose.size())
        repeat2.delete_residue_range_slow(1, repeat_len)
        rmsd = rmsd_by_ndxs_atoms(repeat1,1,repeat1.size(),repeat2,1,repeat2.size())
        #print(num_repeats, rmsd)
        if rmsd > max_rmsd:
            continue
        if num_repeats > num_repeats_curr:
            num_repeats_curr = num_repeats

    if num_repeats_curr == -1:
        print('Error: num repeat not found')
        return -1
    
    return num_repeats_curr   

def find_repeatlen_by_rmsd(pose, max_rmsd=0.5, sliding_window=10, min_repeatlen=20):
    '''
        this function fails when working on parametrically generated DHR

        sliding_window: number or res for rmsd calculation (less then repeatlen so genkic loop not to be included)
                        this number might be tricky to optimize

    '''
    for repeatlen in range(min_repeatlen, pose.size()+1):
        rmsd = rmsd_by_ndxs_atoms(pose, 1, sliding_window, pose, 1+repeatlen, sliding_window+repeatlen)
        print(repeatlen, rmsd)
        if rmsd < max_rmsd:
            return repeatlen
    return -1



def configure_peptide_stub_mover(no_of_additions, stub_mode, offset,
                                 res_name='ALA'):

    assert(stub_mode in ['Append', 'Prepend','Insert'])

    psm = pyrosetta.rosetta.protocols.cyclic_peptide.PeptideStubMover()

    residues = [1] * no_of_additions if stub_mode == 'Prepend' else \
        list(range(offset, offset + no_of_additions))
    for anchor_res in residues:

        #if _DEBUG:
        #    print('PeptideStubMover: {}  {}'.format(stub_mode, anchor_res))
        anchor_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(str(anchor_res))
        psm.add_residue(stub_mode, res_name, 0, False, '', 1, anchor_res, anchor_selector, '')

    return psm


def _add_residues_with_conformation(p, n_res, stub_mode, offset, res_type):
    psm = configure_peptide_stub_mover(n_res, stub_mode, offset, res_type)
    psm.apply(p)


def extend_residue(p, n_res, anchor_pos, method='Insert'):
    _add_residues_with_conformation(p, n_res, method, anchor_pos, 'ALA')

def declare_bond(p, res_indices):
    assert(len(res_indices) == 2)
    db = pyrosetta.rosetta.protocols.cyclic_peptide.DeclareBond()
    #if _DEBUG:
    #    print('DeclareBond: {}  {}'.format(res_indices[0], res_indices[-1]))

    db.set(res_indices[0], _peptide_bond_params['atom1'],
           res_indices[1], _peptide_bond_params['atom2'], True)  
    db.apply(p)

def _config_my_task_factory_old(added_residues, distance=6., allowed_aa='ACDEFGHIKLMNPQRSTVWY'):

    number_of_residues = len(added_residues)

    loop_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in added_residues:
        loop_residues.append_index(res_num)

    # only use a subset of (canonical) AAs when designing loop residues
    restrict_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    #restrict_res.aas_to_keep('AVLIFMSTNQYW')
    #restrict_res.aas_to_keep('AP') # bb check
    #restrict_res.aas_to_keep('RKDEQNHSTYW') # polar residues
    restrict_res.aas_to_keep(allowed_aa) 
    allow_design_on_loop_res = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        restrict_res, loop_residues)

    # disable design for all residues outside the loop
    restrict_nbrs_to_repack = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), loop_residues, True)  # flip the selection

    # don't pack residues more than 10. A from the loop
    nbrhood_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        loop_residues, distance, True)
    only_repack_neighborhood = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), nbrhood_sel, True)

    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(allow_design_on_loop_res)
    tf.push_back(only_repack_neighborhood)
    tf.push_back(restrict_nbrs_to_repack)
    return tf

def _config_my_task_factory(added_residues, allowed_aa='ACDEFGHIKLMNPQRSTVWY', include_neighbor_repack=True, neighbor_dist=6.0):

    number_of_residues = len(added_residues)

    loop_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in added_residues:
        loop_residues.append_index(res_num)

    # only use a subset of (canonical) AAs when designing loop residues
    restrict_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    #restrict_res.aas_to_keep('AVLIFMSTNQYW')
    #restrict_res.aas_to_keep('AP') # bb check
    #restrict_res.aas_to_keep('RKDEQNHSTYW') # polar residues
    restrict_res.aas_to_keep(allowed_aa) 
    allow_design_on_loop_res = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        restrict_res, loop_residues)

    nbrhood_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        loop_residues, neighbor_dist, False)
    nbrhood_sel_inclusive = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        loop_residues, neighbor_dist, True)


    # disable design for all residues outside the loop
    restrict_nbrs_to_repack = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), nbrhood_sel, False)  # not flip the selection

    # don't pack residues more than 10. A from the loop
    only_repack_neighborhood = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), nbrhood_sel_inclusive, True)

    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(allow_design_on_loop_res)
    tf.push_back(only_repack_neighborhood)
    tf.push_back(restrict_nbrs_to_repack)
    return tf


def _config_my_task_factory_layer(core_selector, boundary_selector, surface_selector, helix_ncap_selector, 
            core_aa='ACDEFGHIKLMNPQRSTVWY', boundary_aa='ACDEFGHIKLMNPQRSTVWY', surface_aa='ACDEFGHIKLMNPQRSTVWY', helix_ncap_aa='DNST', include_neighbor_repack=True, neighbor_dist=6.0):
    
    core_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    core_res.aas_to_keep(core_aa) 
    core_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(core_res, core_selector)

    boundary_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    boundary_res.aas_to_keep(boundary_aa) 
    boundary_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(boundary_res, boundary_selector)

    surface_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    surface_res.aas_to_keep(surface_aa) 
    surface_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(surface_res, surface_selector)

    core_boundary_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_selector, boundary_selector)
    core_boundary_surface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_selector, surface_selector)

    helix_ncap_res_subset = None
    if helix_ncap_selector != None:
        helix_ncap_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
        helix_ncap_res.aas_to_keep(helix_ncap_aa) 
        helix_ncap_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(helix_ncap_res, helix_ncap_selector)        
        core_boundary_surface_helixcap_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_surface_selector, helix_ncap_selector)
        neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_helixcap_selector, neighbor_dist, False) # not including selector residues
        design_n_neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_helixcap_selector, neighbor_dist, True) # including selector residues
    else:
        neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_selector, neighbor_dist, False) # not including selector residues
        design_n_neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_selector, neighbor_dist, True) # including selector residues

    repack_nbr_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), neighbor_selector)    

    freeze_rest_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), design_n_neighbor_selector, True) # flip the selection

    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(core_res_subset)
    tf.push_back(boundary_res_subset)
    tf.push_back(surface_res_subset)
    if helix_ncap_res_subset != None:
        tf.push_back(helix_ncap_res_subset)        
    tf.push_back(repack_nbr_subset)
    tf.push_back(freeze_rest_subset)
    return tf    



def _config_my_move_map(p, input_obj, include_neighbor=True, neighbor_dist=6.0):
    mm = pyrosetta.MoveMap()
    if isinstance(input_obj, list):
        ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector('{}'.format(','.join([str(x) for x in input_obj])))
    else:
        # input is already a selector
        ind_selector = input_obj
    ind = ind_selector.apply(p)
    mm.set_bb(ind)
    if include_neighbor:
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(ind_selector, neighbor_dist)
        nbr = nbr_selector.apply(p)
        #print(nbr)
        mm.set_chi(nbr)
    else:
        mm.set_chi(ind)

    return mm


def fast_design_rosettacon2018(p, sf, design_reslist=[], allowed_aa='ACDEFGHIKLMNPQRSTVWY', rmsd_check_resids=[], include_neighbor=True, neighbor_dist=6.0):
    pose = p.clone()
    fd = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign()
    fd.set_task_factory(_config_my_task_factory(design_reslist, allowed_aa=allowed_aa, include_neighbor_repack=include_neighbor, neighbor_dist=neighbor_dist))
    fd.set_scorefxn(sf)
    fd.set_movemap(_config_my_move_map(p, design_reslist, include_neighbor=include_neighbor, neighbor_dist=neighbor_dist))

    relaxscript = pyrosetta.rosetta.std.vector_std_string()
    relaxscript.append('repeat {}'.format(pyrosetta.rosetta.basic.options.get_integer_option('relax:default_repeats')))
    relaxscript.append('reference 0.3     3.1     -2.6     -2.55    4.8     -0.5    0.7      4.5     -1.6     4.0     3.9     -1.7     -2.0     -1.5     -1.0     -2.0    -2.0     4.0     9.0     3.7')
    relaxscript.append('ramp_repack_min 0.079 0.01     1.0')
    relaxscript.append('reference 2.2619  4.8148  -1.6204  -1.6058  2.7602  1.0350  1.3406   2.5006  -0.6895  1.9223  2.3633  -0.3009  -4.2787   0.1077   0.0423  -0.4390 -0.7333  3.2371  4.7077  2.3379')
    relaxscript.append('ramp_repack_min 0.295 0.01     0.5')
    relaxscript.append('reference 2.2619  4.5648  -1.6204  -1.6158  2.5602  1.1350  1.2406   2.3006  -0.7895  1.7223  2.1633  -0.3009  -4.3787   0.1077   0.0423  -0.4390 -0.7333  3.1371  4.4077  2.1379')
    relaxscript.append('ramp_repack_min 0.577 0.01     0.0')
    relaxscript.append('reference 2.2619  4.3148  -1.6204  -1.6358  1.9602  1.4350  0.8406   1.8006  -0.8895  1.3223  1.4633  -0.3009  -4.6787  -0.1077  -0.1423  -0.5390 -0.9333  2.7371  3.7077  1.7379')
    relaxscript.append('ramp_repack_min 1     0.00001  0.0')
    relaxscript.append('accept_to_best')
    relaxscript.append('endrepeat')
    #print(relaxscript)
    fd.set_script_from_lines(relaxscript)

    fd.apply(p)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(p, rmsd_check_resids[0], rmsd_check_resids[-1], pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from design: ",rmsd)
    return None

def fast_design_rosettacon2018_layer(p, sf, core_selector, boundary_selector, surface_selector, helix_ncap_selector,
                                    core_aa='ACDEFGHIKLMNPQRSTVWY', boundary_aa='ACDEFGHIKLMNPQRSTVWY', surface_aa='ACDEFGHIKLMNPQRSTVWY', helix_ncap_aa='DNST', rmsd_check_resids=[], include_neighbor=True, neighbor_dist=6.0):

    pose = p.clone()
    fd = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign()
    fd.set_task_factory(_config_my_task_factory_layer(core_selector, boundary_selector, surface_selector, helix_ncap_selector, core_aa, boundary_aa, surface_aa, helix_ncap_aa, include_neighbor_repack=include_neighbor, neighbor_dist=neighbor_dist))
    fd.set_scorefxn(sf)
    core_boundary_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_selector, boundary_selector)
    core_boundary_surface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_selector, surface_selector)
    fd.set_movemap(_config_my_move_map(p, core_boundary_surface_selector, include_neighbor=include_neighbor, neighbor_dist=neighbor_dist))

    relaxscript = pyrosetta.rosetta.std.vector_std_string()
    relaxscript.append('repeat {}'.format(pyrosetta.rosetta.basic.options.get_integer_option('relax:default_repeats')))
    relaxscript.append('reference 0.3     3.1     -2.6     -2.55    4.8     -0.5    0.7      4.5     -1.6     4.0     3.9     -1.7     -2.0     -1.5     -1.0     -2.0    -2.0     4.0     9.0     3.7')
    relaxscript.append('ramp_repack_min 0.079 0.01     1.0')
    relaxscript.append('reference 2.2619  4.8148  -1.6204  -1.6058  2.7602  1.0350  1.3406   2.5006  -0.6895  1.9223  2.3633  -0.3009  -4.2787   0.1077   0.0423  -0.4390 -0.7333  3.2371  4.7077  2.3379')
    relaxscript.append('ramp_repack_min 0.295 0.01     0.5')
    relaxscript.append('reference 2.2619  4.5648  -1.6204  -1.6158  2.5602  1.1350  1.2406   2.3006  -0.7895  1.7223  2.1633  -0.3009  -4.3787   0.1077   0.0423  -0.4390 -0.7333  3.1371  4.4077  2.1379')
    relaxscript.append('ramp_repack_min 0.577 0.01     0.0')
    relaxscript.append('reference 2.2619  4.3148  -1.6204  -1.6358  1.9602  1.4350  0.8406   1.8006  -0.8895  1.3223  1.4633  -0.3009  -4.6787  -0.1077  -0.1423  -0.5390 -0.9333  2.7371  3.7077  1.7379')
    relaxscript.append('ramp_repack_min 1     0.00001  0.0')
    relaxscript.append('accept_to_best')
    relaxscript.append('endrepeat')
    #print(relaxscript)
    fd.set_script_from_lines(relaxscript)

    fd.apply(p)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(p, rmsd_check_resids[0], rmsd_check_resids[-1], pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from design: ",rmsd)
    return None

def fast_design_layer_relaxscript(p, sf, core_selector, boundary_selector, surface_selector, helix_ncap_selector,
                                    core_aa='ACDEFGHIKLMNPQRSTVWY', boundary_aa='ACDEFGHIKLMNPQRSTVWY', surface_aa='ACDEFGHIKLMNPQRSTVWY', helix_ncap_aa='DNST', rmsd_check_resids=[], include_neighbor=True, neighbor_dist=6.0,
                                    my_relaxscript='MonomerDesign2019.txt'):

    pose = p.clone()
    fd = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign()
    fd.set_task_factory(_config_my_task_factory_layer(core_selector, boundary_selector, surface_selector, helix_ncap_selector, core_aa, boundary_aa, surface_aa, helix_ncap_aa, include_neighbor_repack=include_neighbor, neighbor_dist=neighbor_dist))
    fd.set_scorefxn(sf)
    core_boundary_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_selector, boundary_selector)
    core_boundary_surface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_selector, surface_selector)
    fd.set_movemap(_config_my_move_map(p, core_boundary_surface_selector, include_neighbor=include_neighbor, neighbor_dist=neighbor_dist))


    relaxscript = pyrosetta.rosetta.std.vector_std_string()
    relaxscript.append('repeat {}'.format(pyrosetta.rosetta.basic.options.get_integer_option('relax:default_repeats')))
    with open(my_relaxscript,'r') as fin_rs:
        for line in fin_rs.readlines()[1:]:
            relaxscript.append(line.strip())
        #print(relaxscript)
    fd.set_script_from_lines(relaxscript)

    fd.apply(p)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(p, rmsd_check_resids[0], rmsd_check_resids[-1], pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from design: ",rmsd)
    return None



def my_own_2D_numpy_to_rosetta(np_arr):
    ros_container = pyrosetta.rosetta.numeric.xyzMatrix_double_t(0.)
    ros_container.xx = np_arr[0, 0]
    ros_container.xy = np_arr[0, 1]
    ros_container.xz = np_arr[0, 2]

    ros_container.yx = np_arr[1, 0]
    ros_container.yy = np_arr[1, 1]
    ros_container.yz = np_arr[1, 2]

    ros_container.zx = np_arr[2, 0]
    ros_container.zy = np_arr[2, 1]
    ros_container.zz = np_arr[2, 2]
    return ros_container


def my_own_rotate_pose(p, R):
    '''Apply a rotation matrix to all of the coordinates in a Pose.

    Args:
        p (Pose): The Pose instance to manipulate
        R (np.mat): A rotation matrix to apply to the Pose coordinates

    Returns:
        None. The input Pose is manipulated.

    '''
    # t must be an xyzMatrix_Real
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            v = p.residue(i).atom(j).xyz()

            x = R.xx * v.x + R.xy * v.y + R.xz * v.z
            y = R.yx * v.x + R.yy * v.y + R.yz * v.z
            z = R.zx * v.x + R.zy * v.y + R.zz * v.z

            p.residue(i).atom(j).xyz(pyrosetta.rosetta.numeric.xyzVector_double_t(x, y, z))



def my_own_translate_pose(p, t):
    '''Apply a translation to all of the coordinates in a Pose.

    Args:
        p (Pose): The Pose instance to manipulate
        t (np.array): A vector to add to the Pose coordinates

    Returns:
        None. The input Pose is manipulated.

    '''
    # t must be an xyzVector_double_t
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            p.residue(i).atom(j).xyz(p.residue(i).atom(j).xyz() + t)
            
def get_anchor_coordinates_from_pose(p, reslist):
    _bb_atoms = ['N', 'CA', 'C', 'O']
    coords = list()
    for resNo in reslist:
        res = p.residue(resNo)
        # only iterate over relevant atoms
        for i in _bb_atoms:
            coords.append([res.xyz(i).x, res.xyz(i).y, res.xyz(i).z])
    return np.mat(coords)


def numpy_to_rosetta(np_arr):
    '''Pack values in a numpy data structure into the analogous Rosetta
    data structure.

    Args:
        np_arry (np.mat or np.array): One- (1 x 3) or two-dimensional (3 x 3)
            numpy matrix

    Returns:
        The values in the appropriate Rosetta container
            (numeric.xyzVector_double_t or numeric.xyzMatrix_double_t)

    '''
    # start off assuming a 1D array
    dim = 1

    # ensure that we are in 3D space
    if np.shape(np_arr)[1] != 3:
        dim = -999
    elif np.shape(np_arr)[0] == np.shape(np_arr)[1]:
        dim = 2
        
    if dim == 1:
        # handle the 1D case
        ros_container = pyrosetta.rosetta.numeric.xyzVector_double_t(0.)
        ros_container.x = np_arr[0, 0]
        ros_container.y = np_arr[0, 1]
        ros_container.z = np_arr[0, 2]

        return ros_container

    elif dim == 2:
        # handle the 2D case
        ros_container = pyrosetta.rosetta.numeric.xyzMatrix_double_t(0.)
        ros_container.xx = np_arr[0, 0]
        ros_container.xy = np_arr[0, 1]
        ros_container.xz = np_arr[0, 2]

        ros_container.yx = np_arr[1, 0]
        ros_container.yy = np_arr[1, 1]
        ros_container.yz = np_arr[1, 2]

        ros_container.zx = np_arr[2, 0]
        ros_container.zy = np_arr[2, 1]
        ros_container.zz = np_arr[2, 2]

        return ros_container

    # get out of here!
    raise ValueError('Packing {}-dimensional numpy arrays '.format(dim) +
                     'into Rosetta containers is not currently supported')
    
    
def align_pose_to_anchor_coords(p, target_coord, anchor_coord):
    R, t = np_utils.rigid_transform_3D(anchor_coord, target_coord)
    #np_utils.rotate_pose(p, np_utils.numpy_to_rosetta(R)) # JHL, sth wrong w/ dig's pyrosetta: xx() not callable, but xx directly accessible
    my_own_rotate_pose(p, my_own_2D_numpy_to_rosetta(R))  # JHL, so I had to rewrite np->rosetta and rotation function to change xx() to xx
    #np_utils.translate_pose(p, np_utils.numpy_to_rosetta(t.T)) # on mac there's no translate_pose
    my_own_translate_pose(p, numpy_to_rosetta(t.T)) # so i copied the ones for older rosetta codes
    return p

def get_aligned_repeat_pose(input_pose, target_pose, anchor_resid, target_resid):
    ''' align the anchor_resid residue of a copy of input pose
        to the target_resid residue of the input pose
    '''
    target_coord = get_anchor_coordinates_from_pose(target_pose, [target_resid]) 
    anchor_coord = get_anchor_coordinates_from_pose(input_pose, [anchor_resid])
    new_pose = input_pose.clone()
    new_pose_aligned = align_pose_to_anchor_coords(new_pose, target_coord, anchor_coord)
    return new_pose_aligned

def poorman_repeat_propagate(pose, repeatlen=-1, num_repeat=4, overhang=1):
    '''
        does not require reference pose, but requires repeatlen

        overhang: index of residue in the 2nd repeat to be used for alignment; should >= 1
    '''

    def _append_residue(pose, source_pose, resid, source_resid):
        if source_pose.chain(source_resid) != source_pose.chain(source_resid-1):
            pose.append_residue_by_jump(source_pose.residue(source_resid), resid-1)
        else:
            pose.append_residue_by_bond(source_pose.residue(source_resid))
        return pose 
    
    assert(pose.size() >= repeatlen+overhang)


    if repeatlen == -1:
        repeatlen = find_repeatlen_by_rmsd(pose, max_rmsd=0.5, sliding_window=10, min_repeatlen=20)
    
    # build the 1st repeats
    new_pose = pyrosetta.rosetta.core.pose.Pose()
    new_pose.append_residue_by_jump(pose.residue(1), 1)
    for i in range(2, repeatlen+overhang+1):
        new_pose = _append_residue(new_pose, pose, i, i)
    
    # build the rest repeats
    for rep in range(num_repeat-1):
        current_repeat_pose = get_aligned_repeat_pose(pose, new_pose, overhang, (rep+1)*repeatlen+overhang)
        for i in range(overhang+1,repeatlen+overhang+1):
            # avoid padding overhang into last repeat
            if rep == num_repeat-2 and i > repeatlen:
                break
            else:
                new_pose = _append_residue(new_pose, current_repeat_pose, rep*repeatlen+i, i)
          
    return new_pose



#===================================  TJ's repeat propagation ================================>>>>>>

def extract_mer(in_pose, mer_start=1, mer_end=9):
    fragment_pose = pyrosetta.rosetta.core.pose.Pose()
    fragment_pose.append_residue_by_jump(in_pose.residue(mer_start), 1)
    for i in  range(mer_start + 1, mer_end+1):
        fragment_pose.append_residue_by_bond(in_pose.residue(i))
    return fragment_pose

def return_loops(rmsd_dict, offset=0):
    loops = {}
    loop = []
    j = 1
    for key in range(1+offset, len(rmsd_dict)+1+offset):
        if rmsd_dict[key] == 0.0 and len(loop) > 0:
            #print('adding to dict')
            loops[j] = [loop[0], loop[-1]]
            loop = []
            j=j+1    
        elif rmsd_dict[key] == 0.0 and len(loop) == 0:
            #print('continuing')
            continue
        elif rmsd_dict[key] > 0.0:
            loop.append(key)
            #print('appending',loop)
        else:
            print('wtf; how?')
    return loops

def superimpose_pose_on_pose(mod_pose, ref_pose, ref_start, ref_end):
    pyrosetta.rosetta.protocols.grafting.superimpose_overhangs_heavy(ref_pose, mod_pose, False, ref_start+1, ref_end-1, 2, 2)

def copy_phi_psi_omega(mod_pose, ref_pose, numb_repeats_=4, base_repeat=2):
    repeatlen = int(ref_pose.size()/numb_repeats_)
    nrepeat = numb_repeats_
    base_offset = (base_repeat-1)*repeatlen
    for i in range(0,nrepeat):
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            mod_pose.set_phi  ( j+i*repeatlen, ref_pose.phi  (pos) )
            mod_pose.set_psi  ( j+i*repeatlen, ref_pose.psi  (pos) )
            mod_pose.set_omega( j+i*repeatlen, ref_pose.omega(pos) )
            j=j+1

def idealize_to_tolerance(pose, tolerance=0.000001):
    idealize = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
    idealize.atom_pair_constraint_weight(0.01)
    for i in range(1, 10):
        before_pose_copy = pose.clone()
        idealize.apply(pose)
        rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(pose, before_pose_copy)
        #print('idealizing round', i, rmsd)
        if rmsd < tolerance:
            break

def superimpose_by_tm_align(loop_pose, ref_pose):
    tmalign = pyrosetta.rosetta.protocols.hybridization.TMalign()
    tmalign.apply(loop_pose,ref_pose)

def return_loop_start_and_end(loop_pose, ref_pose, nrepeats=4):
    distance = pyrosetta.rosetta.numeric.xyzVector_double_t.distance
    min_ref_distance_dict = {}
    min_loop_distance_dict = {}
    for i in range(1,int(ref_pose.size()+1)):
        for j in range(1,int(loop_pose.size()+1)):
            d = distance(ref_pose.residue(i).xyz('CA'), loop_pose.residue(j).xyz('CA'))
            if i in min_ref_distance_dict:
                if d < min_ref_distance_dict[i]:
                    min_ref_distance_dict[i] = d
            else:
                min_ref_distance_dict[i] = d
            if j in min_loop_distance_dict:
                if d < min_loop_distance_dict[j]:
                    min_loop_distance_dict[j] = d
            else:
                min_loop_distance_dict[j] = d
    return return_loops(min_ref_distance_dict), return_loops(min_loop_distance_dict)

def propogate_loops_idealize_and_symmetrize(loop_pose, ref_pose, num_repeats=4):
    # get repeat length
    '''
    if (ref_pose.size()/num_repeats).is_integer():
        repeat_len = int(ref_pose.size()/num_repeats)
    else:
        print('quitting ; repeat length not divisible by',num_repeats)
        return
    '''
    repeat_len = int(ref_pose.size()/num_repeats)
    # get the start and end of loop1 and loop2 for reference and new loops
    ref_loops, loops = return_loop_start_and_end(loop_pose, ref_pose, nrepeats=num_repeats)
    #print('ref loops:', ref_loops)
    #print(loops)
    #print('new loops',loops)
     
    if len(loops) < 1:
        print('could not find the loop on new pose; something is wrong; ending')
        return
    for i in loops:
        if loops[i][1] - loops[i][0] > 5: # hard coded cutoff, to skip the 0-length loops
            loop1 = extract_mer(loop_pose, loops[i][0], loops[i][1]) 
            

            # Caution!! again hardcoded for the case where the loop is built on the last repeat
            loop1_s_e = [[(ref_loops[i][0]-2)%repeat_len], [(ref_loops[i][-1]+2)%repeat_len]]
            # avoid the index same as repeat_len and got set as 0
            for x in loop1_s_e:
                if x[0] == 0:
                    x[0] = repeat_len


            #print(loops[i][0], loops[i][1])
            #print(loop1_s_e)

            loop1_pose_v = [extract_mer(loop_pose, loops[i][0]-2, loops[i][1]+2) for x in range(num_repeats)]

            break
    if loop1.size() == 0:
        print('loops not found!!')
        sys.exit(1)

    #if _DEBUG:
    #    print('DEBUG:     PROPAGATION: loop1_s_e', loop1_s_e)

    # get superposition for each loop
    # If this is not done well, will cause horrible problems
    ref_loop1_start = loop1_s_e[0][0]
    ref_loop1_end = loop1_s_e[1][0]
    #tmp_index=1
    for tmp_pose in loop1_pose_v:
        #print('superimposing loop1',tmp_index)
        #tmp_index=tmp_index+1
        if ref_loop1_end > ref_pose.size():
            ref_loop1_end = ref_pose.size() # this might cause pose length differences... maybe???
        #print(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
        superimpose_pose_on_pose(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
        ref_loop1_start += repeat_len
        ref_loop1_end += repeat_len
    
        
    ## now we build a new pose by piecing together the reference pose and new superimposed loops
    ## be careful with index, additions, and subtrations
    my_new_pose = pyrosetta.rosetta.core.pose.Pose()
    my_new_pose.append_residue_by_jump(ref_pose.residue(1), 1)
    for i in range(0, num_repeats):
        if i == 0: ### handle the first repeat becasue we had to start the pose by jump ...
            for j in range(2, loop1_s_e[0][0]+repeat_len*i):
                #start of repeat
                if 'Nterm' in ref_pose.residue(j).name():
                    my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                else:
                    my_new_pose.append_residue_by_bond(ref_pose.residue(j)) 
            for j in range(1, loop1_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j)) # loop1
            for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                # second part of repeat
                if 'Nterm' in ref_pose.residue(j).name():
                    my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                else:
                    my_new_pose.append_residue_by_bond(ref_pose.residue(j))
            ##for j in range(1, loop2_pose_v[i].size()+1):
            ##    my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j)) # loop2
        else: ### handle internal repeats and last repeat
            for j in range(repeat_len*i+1, loop1_s_e[0][0]+repeat_len*i):
                if 'Nterm' in ref_pose.residue(j).name():
                    my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                else:
                    my_new_pose.append_residue_by_bond(ref_pose.residue(j))
            for j in range(1, loop1_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j))
            for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                if 'Nterm' in ref_pose.residue(j).name():
                    my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                else:
                    my_new_pose.append_residue_by_bond(ref_pose.residue(j))
            #for j in range(1, loop2_pose_v[i].size()+1):
            #    my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j))
    
    
    
    return my_new_pose 

#<<<<<<=============================  TJ's repeat propagation ======================================

#===================================  TJ's repeat propagation as a mover =============================>>>>>

class repeat_propagate_mover(pyrosetta.rosetta.protocols.moves.Mover):
    '''
        mover version of poorman_repeat_propagate for preselection mover in genKIC
    '''
    def __init__(self, _ref_pose, _num_repeats=4):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.ref_pose = _ref_pose
        self.num_repeats = _num_repeats

    def __str__(self):
        return f'num_repeat: {self.num_repeat}'

    #===================================  TJ's repeat propagation ================================>>>>>>

    def extract_mer(self, in_pose, mer_start=1, mer_end=9):
        fragment_pose = pyrosetta.rosetta.core.pose.Pose()
        fragment_pose.append_residue_by_jump(in_pose.residue(mer_start), 1)
        for i in  range(mer_start + 1, mer_end+1):
            fragment_pose.append_residue_by_bond(in_pose.residue(i))
        return fragment_pose

    def return_loops(self, rmsd_dict, offset=0):
        loops = {}
        loop = []
        j = 1
        for key in range(1+offset, len(rmsd_dict)+1+offset):
            if rmsd_dict[key] == 0.0 and len(loop) > 0:
                #print('adding to dict')
                loops[j] = [loop[0], loop[-1]]
                loop = []
                j=j+1    
            elif rmsd_dict[key] == 0.0 and len(loop) == 0:
                #print('continuing')
                continue
            elif rmsd_dict[key] > 0.0:
                loop.append(key)
                #print('appending',loop)
            else:
                print('wtf; how?')
        return loops

    def superimpose_pose_on_pose(self, mod_pose, ref_pose, ref_start, ref_end):
        pyrosetta.rosetta.protocols.grafting.superimpose_overhangs_heavy(ref_pose, mod_pose, False, ref_start+1, ref_end-1, 2, 2)

    def copy_phi_psi_omega(self, mod_pose, ref_pose, numb_repeats_=4, base_repeat=2):
        repeatlen = int(ref_pose.size()/numb_repeats_)
        nrepeat = numb_repeats_
        base_offset = (base_repeat-1)*repeatlen
        for i in range(0,nrepeat):
            j=1
            while j <= repeatlen:
                pos = j + base_offset
                mod_pose.set_phi  ( j+i*repeatlen, ref_pose.phi  (pos) )
                mod_pose.set_psi  ( j+i*repeatlen, ref_pose.psi  (pos) )
                mod_pose.set_omega( j+i*repeatlen, ref_pose.omega(pos) )
                j=j+1

    def idealize_to_tolerance(self, pose, tolerance=0.000001):
        idealize = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
        idealize.atom_pair_constraint_weight(0.01)
        for i in range(1, 10):
            before_pose_copy = pose.clone()
            idealize.apply(pose)
            rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(pose, before_pose_copy)
            #print('idealizing round', i, rmsd)
            if rmsd < tolerance:
                break

    def return_loop_start_and_end(self, loop_pose, ref_pose, nrepeats=4):
        distance = pyrosetta.rosetta.numeric.xyzVector_double_t.distance
        min_ref_distance_dict = {}
        min_loop_distance_dict = {}
        for i in range(1,int(ref_pose.size()+1)):
            for j in range(1,int(loop_pose.size()+1)):
                d = distance(ref_pose.residue(i).xyz('CA'), loop_pose.residue(j).xyz('CA'))
                if i in min_ref_distance_dict:
                    if d < min_ref_distance_dict[i]:
                        min_ref_distance_dict[i] = d
                else:
                    min_ref_distance_dict[i] = d
                if j in min_loop_distance_dict:
                    if d < min_loop_distance_dict[j]:
                        min_loop_distance_dict[j] = d
                else:
                    min_loop_distance_dict[j] = d
        return self.return_loops(min_ref_distance_dict), self.return_loops(min_loop_distance_dict)

    def propogate_loops_idealize_and_symmetrize(self, loop_pose, ref_pose, num_repeats=4):
        # get repeat length
        '''
        if (ref_pose.size()/num_repeats).is_integer():
            repeat_len = int(ref_pose.size()/num_repeats)
        else:
            print('quitting ; repeat length not divisible by',num_repeats)
            return
        '''
        repeat_len = int(ref_pose.size()/num_repeats)
        # get the start and end of loop1 and loop2 for reference and new loops
        ref_loops, loops = self.return_loop_start_and_end(loop_pose, ref_pose, nrepeats=num_repeats)
        #print('ref loops:', ref_loops)
        #print(loops)
        #print('new loops',loops)
         
        if len(loops) < 1:
            print('could not find the loop on new pose; something is wrong; ending')
            return
        for i in loops:
            if loops[i][1] - loops[i][0] > 5: # hard coded cutoff, to skip the 0-length loops
                loop1 = self.extract_mer(loop_pose, loops[i][0], loops[i][1]) 
                

                # Caution!! again hardcoded for the case where the loop is built on the last repeat
                loop1_s_e = [[(ref_loops[i][0]-2)%repeat_len], [(ref_loops[i][-1]+2)%repeat_len]]
                # avoid the index same as repeat_len and got set as 0
                for x in loop1_s_e:
                    if x[0] == 0:
                        x[0] = repeat_len


                #print(loops[i][0], loops[i][1])
                #print(loop1_s_e)

                loop1_pose_v = [self.extract_mer(loop_pose, loops[i][0]-2, loops[i][1]+2) for x in range(num_repeats)]

                break
        if loop1.size() == 0:
            print('loops not found!!')
            sys.exit(1)

        #if _DEBUG:
        #    print('DEBUG:     PROPAGATION: loop1_s_e', loop1_s_e)

        # get superposition for each loop
        # If this is not done well, will cause horrible problems
        ref_loop1_start = loop1_s_e[0][0]
        ref_loop1_end = loop1_s_e[1][0]
        #tmp_index=1
        for tmp_pose in loop1_pose_v:
            #print('superimposing loop1',tmp_index)
            #tmp_index=tmp_index+1
            if ref_loop1_end > ref_pose.size():
                ref_loop1_end = ref_pose.size() # this might cause pose length differences... maybe???
            #print(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
            self.superimpose_pose_on_pose(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
            ref_loop1_start += repeat_len
            ref_loop1_end += repeat_len
        
        ## now we build a new pose by piecing together the reference pose and new superimposed loops
        ## be careful with index, additions, and subtrations
        ## FOR THIS MOVER version: I have to modify loop_pose isntead of create a new pose!
        loop_pose.delete_residue_range_slow(2, loop_pose.size())
        for i in range(0, num_repeats):
            if i == 0: ### handle the first repeat becasue we had to start the pose by jump ...
                for j in range(2, loop1_s_e[0][0]+repeat_len*i):
                    #start of repeat
                    if 'Nterm' in ref_pose.residue(j).name():
                        loop_pose.append_residue_by_jump(ref_pose.residue(j),loop_pose.size())
                    else:
                        loop_pose.append_residue_by_bond(ref_pose.residue(j)) 
                for j in range(1, loop1_pose_v[i].size()+1):
                    loop_pose.append_residue_by_bond(loop1_pose_v[i].residue(j)) # loop1
                for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                    # second part of repeat
                    if 'Nterm' in ref_pose.residue(j).name():
                        loop_pose.append_residue_by_jump(ref_pose.residue(j),loop_pose.size())
                    else:
                        loop_pose.append_residue_by_bond(ref_pose.residue(j))
                ##for j in range(1, loop2_pose_v[i].size()+1):
                ##    loop_pose.append_residue_by_bond(loop2_pose_v[i].residue(j)) # loop2
            else: ### handle internal repeats and last repeat
                for j in range(repeat_len*i+1, loop1_s_e[0][0]+repeat_len*i):
                    if 'Nterm' in ref_pose.residue(j).name():
                        loop_pose.append_residue_by_jump(ref_pose.residue(j),loop_pose.size())
                    else:
                        loop_pose.append_residue_by_bond(ref_pose.residue(j))
                for j in range(1, loop1_pose_v[i].size()+1):
                    loop_pose.append_residue_by_bond(loop1_pose_v[i].residue(j))
                for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                    if 'Nterm' in ref_pose.residue(j).name():
                        loop_pose.append_residue_by_jump(ref_pose.residue(j),loop_pose.size())
                    else:
                        loop_pose.append_residue_by_bond(ref_pose.residue(j))        

        '''                 
        ## now we build a new pose by piecing together the reference pose and new superimposed loops
        ## be careful with index, additions, and subtrations
        my_new_pose = pyrosetta.rosetta.core.pose.Pose()
        my_new_pose.append_residue_by_jump(ref_pose.residue(1), 1)
        for i in range(0, num_repeats):
            if i == 0: ### handle the first repeat becasue we had to start the pose by jump ...
                for j in range(2, loop1_s_e[0][0]+repeat_len*i):
                    #start of repeat
                    if 'Nterm' in ref_pose.residue(j).name():
                        my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                    else:
                        my_new_pose.append_residue_by_bond(ref_pose.residue(j)) 
                for j in range(1, loop1_pose_v[i].size()+1):
                    my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j)) # loop1
                for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                    # second part of repeat
                    if 'Nterm' in ref_pose.residue(j).name():
                        my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                    else:
                        my_new_pose.append_residue_by_bond(ref_pose.residue(j))
                ##for j in range(1, loop2_pose_v[i].size()+1):
                ##    my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j)) # loop2
            else: ### handle internal repeats and last repeat
                for j in range(repeat_len*i+1, loop1_s_e[0][0]+repeat_len*i):
                    if 'Nterm' in ref_pose.residue(j).name():
                        my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                    else:
                        my_new_pose.append_residue_by_bond(ref_pose.residue(j))
                for j in range(1, loop1_pose_v[i].size()+1):
                    my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j))
                for j in range(loop1_s_e[1][0]+repeat_len*i+1, repeat_len*(i+1)+1):
                    if 'Nterm' in ref_pose.residue(j).name():
                        my_new_pose.append_residue_by_jump(ref_pose.residue(j),my_new_pose.size())
                    else:
                        my_new_pose.append_residue_by_bond(ref_pose.residue(j))
                #for j in range(1, loop2_pose_v[i].size()+1):
                #    my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j))
        '''


        
        
        #return my_new_pose 

    #<<<<<<=============================  TJ's repeat propagation ======================================



    def get_name(self):
        return self.__class__.__name__

    def apply(self, pose):
        #pose = pose.get()
        self.propogate_loops_idealize_and_symmetrize(pose, self.ref_pose, self.num_repeats)

#<<<<<==============================  TJ's repeat propagation as a mover ==================================




#===================================  symmetric relax/min ================================>>>>>>


# fills in a SymmetryInfo object with the necessary info
def setup_repeat_symminfo(repeatlen, symminfo, nrepeat, base_repeat):
    #print('doing setup_repeat_symminfo')
    base_offset = (base_repeat-1)*repeatlen
    i=1
    while i <= nrepeat: 
        if i == base_repeat: 
            i=i+1
            continue
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.add_bb_clone(base_offset+j, offset+j )
            symminfo.add_chi_clone( base_offset+j, offset+j )
            j=j+1
        i=i+1


    symminfo.num_virtuals( 1 ) # the one at the end...
    symminfo.set_use_symmetry( True )
    symminfo.set_flat_score_multiply( nrepeat*repeatlen+1, 1 )
    symminfo.torsion_changes_move_other_monomers( True ) # note -- this signals that we are folding between repeats

    ### what is the reason to do this???
    ### If there is a good reason, why not do for repeats after base_repeat???
    ### 
    # repeats prior to base_repeat have score_multiply ==> 0
    """
    i=1
    while i < base_repeat:
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.set_score_multiply( offset+j, 0 )
            j=j+1
        i=i+1
    """
    symminfo.update_score_multiply_factor()
    #print('finished setup_repeat_symminfo')


# sets up a repeat pose, starting from a non-symmetric pdb with nres=repeatlen*nrepeat
def setup_repeat_pose(pose, numb_repeats_=4, base_repeat=2):
    #print('doing setup_repeat_pose')
    if pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False # not to begin with...
    repeatlen = int(pose.size()/numb_repeats_)
    nrepeat = numb_repeats_
    
    if not nrepeat * repeatlen == pose.size():
        return False

    if not base_repeat > 1:
        return False
    # why? well, the base repeat should get the right context info from nbring repeats
    # but note that with base_repeat>1 we probably can't use linmem_ig and there are other places in the code that
    # assume that monomer 1 is the independent monomer. These should gradually be fixed. Let me (PB) know if you run into
    # trouble.

    nres_protein = nrepeat * repeatlen
    pyrosetta.rosetta.core.pose.remove_upper_terminus_type_from_pose_residue( pose, pose.size() )
    vrtrsd = pyrosetta.rosetta.core.conformation.Residue(pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue
           (pyrosetta.rosetta.core.pose.get_restype_for_pose(pose, "VRTBB" )))
    pose.append_residue_by_bond( vrtrsd, True ) # since is polymer...
    #pose.append_residue_by_jump( vrtrsd, True )
    pose.conformation().insert_chain_ending( nres_protein )
    f = pyrosetta.rosetta.core.kinematics.FoldTree( pose.size() )
    f.reorder( pose.size() )
    pose.fold_tree( f )
    symminfo = pyrosetta.rosetta.core.conformation.symmetry.SymmetryInfo()
    setup_repeat_symminfo( repeatlen, symminfo, nrepeat, base_repeat )

    # now make symmetric
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_pose(pose, symminfo)
    
    if not pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False

    ### what is the purpose of this???
    ###
    ##TJ adding these to Phil's function
    base_offset = (base_repeat-1)*repeatlen
    """
    for i in range(0,nrepeat):
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            pose.set_phi  ( j+i*repeatlen, pose.phi  (pos) )
            pose.set_psi  ( j+i*repeatlen, pose.psi  (pos) )
            pose.set_omega( j+i*repeatlen, pose.omega(pos) )
            j=j+1
       
    for i in range(0,nrepeat): 
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            oldrsd = pyrosetta.rosetta.core.conformation.Residue( pose.residue(pos).clone() )
            pose.replace_residue( j+i*repeatlen, oldrsd, False )
            j=j+1
    """
    #print('finished setup_repeat_pose')


def setup_movemap(pose, bblist=[], chilist=[]):
    #print('doing setup_movemap')
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    if len(bblist) == 0 and len(chilist) == 0:
        mm.set_chi( True )
        mm.set_bb( True )        
    else:
        mm.set_chi( False )
        mm.set_bb( False )
        for resid in range(1, pose.size()+1):
            if resid in bblist:
                mm.set_bb( resid, True )
                mm.set_chi( resid, True )
            elif resid in chilist:
                mm.set_chi( resid, True )
    mm.set_jump( True )
    mm.set_bb ( pose.size(), False ) # # for the virtual residue?
    mm.set_chi( pose.size(), False ) # for the virtual residue?
    #print('finished setup_movemap')
    return mm

def seal_jumps(pose):   
    #print('doing seal_jumps')
    i=1
    while i <= pose.size():
        if pose.residue_type(i).name() == "VRTBB":
            pose.conformation().delete_residue_slow(i)
        i=i+1
    ii=1
    while ii <= pose.size()-1:
        if ( pose.residue( ii ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER ) 
            and pose.residue( ii+1 ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER ) ):
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER, ii )
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER, ii+1 )
        ii=ii+1
    ft = pose.fold_tree()
    ft.clear()
    ft.add_edge(1,pose.size(),-1)
    pose.fold_tree(ft)
    #print('finished seal_jumps')


def relax_pose(pose, cartesian_=False, bblist=[], chilist=[], rmsd_check_resids=[]):
    #print('doing relax_pose')
    relax_iterations_ = 1
    pdb_pose = pose.clone()
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)        
    
    sf.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 0.1)
    fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax( sf , relax_iterations_ )
    #fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.ramp_down_constraints(False)
    fastrelax.min_type('lbfgs_armijo_nonmonotone')
    if cartesian_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0) # has to be zero
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 0.5) # what are good values
        fastrelax.cartesian(True)
        fastrelax.minimize_bond_angles(True)
        fastrelax.minimize_bond_lengths(True)
    movemap = setup_movemap(pose, bblist, chilist)
    
   
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    #movemap.clear() #for testing purposes only, prevents relax from doing anything
    fastrelax.set_movemap( movemap )
    pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    fastrelax.apply( pose )
    #rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(pose, pdb_pose)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(pose, rmsd_check_resids[0], rmsd_check_resids[-1], pdb_pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from relax: ",rmsd)
    #print('finished relax_pose')


def minimize_pose(pose, cartesian_=False, coordcst_=False):
    #print('doing minimize_pose')
    movemap = setup_movemap(pose)
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)    
    if coordcst_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
        pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    
    use_nblist = True
    deriv_check = True
    deriv_check_verbose = False
    #min_mover = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
    #                                                                      deriv_check,deriv_check_verbose)
    min_mover = pyrosetta.rosetta.protocols.simple_moves.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
                                                                          deriv_check,deriv_check_verbose)
    min_mover.max_iter(1)
    #min_mover.min_type('lbfgs_armijo_nonmonotone')
      
    if cartesian_:
        min_mover.cartesian(True)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)

    min_mover.apply( pose )
    #print('finished minimize_pose')

def RepeatProteinRelax_apply(pose, modify_symmetry_and_exit_=False, remove_symm_=False, minimize_=False, cartesian_=False, bblist=[], chilist=[], rmsd_check_resids=[], num_repeats=4, base_repeat=2):
    #print('doing RepeatProteinRelax_apply')
    if modify_symmetry_and_exit_ and remove_symm_:
        pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
        seal_jumps(pose)
        return True
    setup_repeat_pose(pose, numb_repeats_=num_repeats, base_repeat=base_repeat)
    setup_movemap(pose, bblist, chilist);
    if modify_symmetry_and_exit_ and not remove_symm_:
        return True
    if minimize_:
        minimize_pose(pose, cartesian_)
    else:
        relax_pose(pose, cartesian_, bblist, chilist, rmsd_check_resids)
    pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
    seal_jumps(pose)

#<<<<<<=============================  symmetric relax/min ======================================



#===================================  loop direction check ================================>>>>>>


def get_com(pose, reslist=[], atomtype=['CA']):
    if len(reslist) == 0:
        reslist = list(range(1,pose.size()+1))  
    coords = np.zeros(len(atomtype)*len(reslist)*3).reshape(len(atomtype)*len(reslist), 3)
    for i, resid in enumerate(reslist):
        for j, atm in enumerate(atomtype):
            coords[i*len(atomtype)+j] = pose.residue(resid).xyz(atm)
    return coords.sum(axis=0)/coords.shape[0]


# OBSOLETE!!!
def compute_loop_direction(loop_pose, loop1_s, loop1_e, repeatlen, num_repeats=4, max_dist_type = 'Ccap'):
    #repeatlen = int(loop_pose.size()/num_repeats) # doesn't work when there's VRTATM

    # center of mass of repeat unit 1 (loop excluded)
    r1_reslist = [x for x in range(1, repeatlen+1) if x not in range(loop1_s, loop1_e+1)]
    com_r1 = np.array(get_com(loop_pose, r1_reslist))
    #print(com_r1)

    # center of mass of repeat unit 2, or the central (loop excluded)
    if num_repeats%2 == 0:
        repeat_ahead = repeatlen * int(num_repeats/2-1)
    else:
        repeat_ahead = repeatlen * int(num_repeats/2) 
    r2_reslist = [x for x in range(repeat_ahead + 1, repeat_ahead+repeatlen+1) if x not in range(loop1_s+repeat_ahead, loop1_e+repeat_ahead+1)]
    com_r2 = np.array(get_com(loop_pose, r2_reslist))
    #print(com_r2)
    if num_repeats%2 == 0:
        r3_reslist = [x+repeatlen for x in r2_reslist]
    else:
        r3_reslist = []
    r23_reslist = r2_reslist+r3_reslist
    com_r23 = np.array(get_com(loop_pose, r23_reslist))


    # center of mass of last repeat unit (loop excluded)
    r4_reslist = [x for x in range(repeatlen * (num_repeats-1) + 1, (repeatlen*num_repeats)+1) if x not in range(loop1_s+(repeatlen*(num_repeats-1)), loop1_e+(repeatlen*(num_repeats-1))+1)]
    com_r4 = np.array(get_com(loop_pose, r4_reslist))
    #print(com_r4)


    # A circle passing com_r1, com_r23 and com_r4
    a = np.linalg.norm(com_r4 - com_r23)
    b = np.linalg.norm(com_r4 - com_r1)
    c = np.linalg.norm(com_r23 - com_r1)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))  # radius
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((com_r1, com_r23, com_r4)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3 # center of circle
    #print(P, R)
    #print(np.linalg.norm(com_r1 - P), np.linalg.norm(com_r23 - P), np.linalg.norm(com_r4 - P))

    # plane of com_r1, com_r23 and com_r4
    norm_vec = np.cross(com_r1 - com_r23, com_r4 - com_r23)
    circle_a, circle_b, circle_c = norm_vec
    circle_d = -1 * np.dot(norm_vec, com_r23)

    '''
    ###################### Obsolete plane def below! ######################
    if num_repeats%2 == 0:
        # center of mass of the region spanning +/- 8 residues from the middle residue of the pose
        rmid = int(loop_pose.size()/2)
        com_middle = np.array(get_com(loop_pose, list(range(rmid-8,rmid+8 + 1))))
    else:
        # center of mass of the central unit excluding loop +/- 6 residues
        loop_mid_s, loop_mid_e = loop1_s + repeatlen * int(num_repeats/2), loop1_e + repeatlen * int(num_repeats/2)
        loop_mid_range = range(loop_mid_s - 6, loop_mid_e + 6 + 1)
        mid_reslist = []
        for resi in range(1, repeatlen+1):
            resi += repeatlen * int(num_repeats/2)
            if resi not in loop_mid_range:
                mid_reslist.append(resi)
        com_middle = np.array(get_com(loop_pose, mid_reslist))
    #print(com_middle)

    # plane equation ax+by+cz+d = 0, containing points: com_middle, com_r1 and com_r4
    norm_vec = np.cross(com_r1 - com_middle, com_r4 - com_middle)
    a, b, c = norm_vec
    d = np.dot(norm_vec, com_middle)
    ###################### Obsolete plane def above! ######################
    '''


    ###################### Current plane def below! ######################
    # com_middle: center of mass of the capping residues (hardcoded 3) of the middle helices
    # say n = number of repeats
    # if num_repeats%2 == 2 the helices involved are helices before/after loops n/2 and n/2+1 (loop2&3 for 4 repeats)
    # if num_repeats%2 == 1 the helices involved are helices before/after loop n/2+1

    com_cap_size = 3
    cap_reslist_baselist = [x-com_cap_size+loop1_s for x in range(com_cap_size)] + [x+1+loop1_e for x in range(com_cap_size)]
    cap_reslist = [x+repeat_ahead for x in cap_reslist_baselist]

    if num_repeats%2 == 0:
        cap_reslist += [x+repeatlen for x in cap_reslist]

    if _DEBUG:
        print('DEBUG:       num_repeats: ', num_repeats)
        print('DEBUG:       cap_reslist: ', cap_reslist)
    com_middle = np.array(get_com(loop_pose, cap_reslist))


    # plane equation ax+by+cz+d = 0, containing points: com_middle, com_r1 and com_r4
    norm_vec = np.cross(com_r1 - com_middle, com_r4 - com_middle)
    a, b, c = norm_vec
    d = -1 * np.dot(norm_vec, com_middle)

    ###################### Current plane def above! ######################

    #print('a,b,c,d:  ', a, b, c, d)
    #print('pseudoatom com_r1, pos=[{}]'.format(','.join([str(x) for x in com_r1])))
    #print('pseudoatom com_r2, pos=[{}]'.format(','.join([str(x) for x in com_r2])))
    #print('pseudoatom com_r4, pos=[{}]'.format(','.join([str(x) for x in com_r4])))
    #print('pseudoatom com_middle, pos=[{}]'.format(','.join([str(x) for x in com_middle])))



    #print(a,b,c,d)

    # find loop residues farthest from com_r2
    #
    #  todo: instead of farthest point from com_r2, try farthest point from the plane
    #
    '''
    # max dist to com_r2
    max_dist_res = [-1, 0, [0,0,0]]
    for resid in range(loop1_s+repeat_ahead, loop1_e+repeat_ahead+1):
            x,y,z = loop_pose.residue(resid).xyz('CA')[0], loop_pose.residue(resid).xyz('CA')[1], loop_pose.residue(resid).xyz('CA')[2]
            dist = np.linalg.norm( np.array([x,y,z]) - com_r2)
            if dist > max_dist_res[1]:
                    max_dist_res = [resid, dist, np.array([x,y,z])]
    '''

    if max_dist_type == 'plane':
        # max dist to the plane
        max_dist_res = [-1, 0, [0,0,0]]
        for resid in range(loop1_s+repeat_ahead, loop1_e+repeat_ahead+1):
                x,y,z = loop_pose.residue(resid).xyz('CA')[0], loop_pose.residue(resid).xyz('CA')[1], loop_pose.residue(resid).xyz('CA')[2]
                k_p = -1 * float(a*x+b*y+c*z+d) / (a*a+b*b+c*c)
                proj_p = np.array([x+k_p*a, y+k_p*b, z+k_p*c])
                dist = np.linalg.norm( np.array([x,y,z]) - proj_p )
                if dist > max_dist_res[1]:
                        max_dist_res = [resid, dist, np.array([x,y,z])]
    elif max_dist_type == 'Ccap':
        # max dist to loop Cterm
        max_dist_res = [-1, 0, [0,0,0]]
        base_resid = loop1_s+repeat_ahead-1
        base_coord = np.array([loop_pose.residue(base_resid).xyz('CA')[0], loop_pose.residue(base_resid).xyz('CA')[1], loop_pose.residue(base_resid).xyz('CA')[2]])
        for resid in range(loop1_s+repeat_ahead, loop1_e+repeat_ahead+1):
                x,y,z = loop_pose.residue(resid).xyz('CA')[0], loop_pose.residue(resid).xyz('CA')[1], loop_pose.residue(resid).xyz('CA')[2]
                dist = np.linalg.norm( np.array([x,y,z]) - base_coord )
                if dist > max_dist_res[1]:
                        max_dist_res = [resid, dist, np.array([x,y,z])]
    else:
        # max dist type not defined, return max cos value
        return 1


    #print(loop1_s+repeat_ahead, loop1_e+repeat_ahead+1, max_dist_res)

    # projection of farthest residue CA onto the plane com_r1, com_r23 and com_r4 (ref: https://stackoverflow.com/questions/9971884/computational-geometry-projecting-a-2d-point-onto-a-plane-to-determine-its-3d)
    e,f,g = max_dist_res[-1]
    k = -1 * float(circle_a*e+circle_b*f+circle_c*g+circle_d) / (circle_a*circle_a+circle_b*circle_b+circle_c*circle_c)
    proj_circle = np.array([e+k*circle_a, f+k*circle_b, g+k*circle_c])
    #print(proj_circle)
    if np.linalg.norm( proj_circle - P ) > R:
        # loop is pointing away from center of circle
        return 1

    # projection of farthest residue CA onto the plane com_r1, com_middle and com_r4 
    k = -1 * float(a*e+b*f+c*g+d) / (a*a+b*b+c*c)
    proj = np.array([e+k*a, f+k*b, g+k*c])
    #print('pseudoatom com_point, pos=[{}]'.format(','.join([str(x) for x in max_dist_res[-1]])))
    #print('pseudoatom com_proj, pos=[{}]'.format(','.join([str(x) for x in proj])))

    '''
    ###################### Obsolete angle def below! ######################
    # compute the angle between com_r2->proj and com_r2->max_dist_res[-1]
    unit_vec = (max_dist_res[-1]-com_r2) / np.linalg.norm(max_dist_res[-1]-com_r2)
    unit_vec_proj = (proj-com_r2) / np.linalg.norm(proj-com_r2)
    cos_angle = np.dot(unit_vec, unit_vec_proj)
    ###################### Obsolete angle def above! ######################
    '''

    ###################### new angle def below! ######################
    # compute the angle between com_r2_proj->proj and com_r2_proj->max_dist_res[-1]
    com_r2_e,com_r2_f,com_r2_g = com_r2
    k = -1 * float(a*com_r2_e+b*com_r2_f+c*com_r2_g+d) / (a*a+b*b+c*c)
    com_r2_proj = np.array([com_r2_e+k*a, com_r2_f+k*b, com_r2_g+k*c])
    #print('pseudoatom com_r2_proj, pos=[{}]'.format(','.join([str(x) for x in com_r2_proj])))
    unit_vec = (max_dist_res[-1]-com_r2_proj) / np.linalg.norm(max_dist_res[-1]-com_r2_proj)
    unit_vec_proj = (proj-com_r2_proj) / np.linalg.norm(proj-com_r2_proj)
    cos_angle = np.dot(unit_vec, unit_vec_proj)
    ###################### new angle def above! ######################


    #print(max_dist_res)
    #print('{} {}'.format(line.strip(), cos_angle))
    return cos_angle



def compute_loop_direction_nonperfect_repeats(pose, loop_idx=1, bturn_side='ncap'):
    # loop_idx = index of the long loop to be considered

    helices = find_helices_by_dssp(pose)
    loops = find_loop_by_dssp(pose)

    rn_reslist = []
    for i in range(len(helices)-1):
        if helices[i][-1] < loops[0][0] and loops[0][-1] < helices[i+1][0]:
            rn_reslist += [x for x in range(helices[i][0], helices[i][1]+1)]
            rn_reslist += [x for x in range(helices[i+1][0], helices[i+1][1]+1)]
    rn_com = get_com(pose, rn_reslist)

    rloopidx_reslist = []
    for i in range(len(helices)-1):
        if helices[i][-1] < loops[loop_idx][0] and loops[loop_idx][-1] < helices[i+1][0]:
            rloopidx_reslist += [x for x in range(helices[i][0], helices[i][1]+1)]
            rloopidx_reslist += [x for x in range(helices[i+1][0], helices[i+1][1]+1)]
    rloopidx_com = get_com(pose, rloopidx_reslist)    

    rc_reslist = []
    for i in range(len(helices)-1):
        if helices[i][-1] < loops[-1][0] and loops[-1][-1] < helices[i+1][0]:
            rc_reslist += [x for x in range(helices[i][0], helices[i][1]+1)]
            rc_reslist += [x for x in range(helices[i+1][0], helices[i+1][1]+1)]
    rc_com = get_com(pose, rc_reslist)            

    if len(loops) < 3:
        print(f'Error: loop projection calculation requires at least 3 long loops. {len(loops)} found ...')
        return 1
    if len(loops)%2 == 0:
        rmid_idx_list = [int(len(loops)/2)-1, int(len(loops)/2)]
    else:
        rmid_idx_list = [int(len(loops)/2)]

    rmid_reslist = []
    for rmid_idx in rmid_idx_list:
        for i in range(len(helices)-1):
            if helices[i][-1] < loops[rmid_idx][0] and loops[rmid_idx][-1] < helices[i+1][0]:
                rmid_reslist += [x for x in range(helices[i][0], helices[i][1]+1)]
                rmid_reslist += [x for x in range(helices[i+1][0], helices[i+1][1]+1)]
    rmid_com = get_com(pose, rmid_reslist)

    #print('pseudoatom rn_com, pos=', list(rn_com))
    #print('pseudoatom rc_com, pos=', list(rc_com))
    #print('pseudoatom rmid_com, pos=', list(rmid_com))

    # A circle passing rn_com, rmid_com and rc_com
    a = np.linalg.norm(rc_com - rmid_com)
    b = np.linalg.norm(rc_com - rn_com)
    c = np.linalg.norm(rmid_com - rn_com)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))  # radius
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((rn_com, rmid_com, rc_com)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3 # center of circle
    #print('pseudoatom circle_com, pos=', list(P))


    # plane of rn_com, rmid_com and rc_com
    norm_vec = np.cross(rn_com - rmid_com, rc_com - rmid_com)
    circle_a, circle_b, circle_c = norm_vec
    circle_d = -1 * np.dot(norm_vec, rmid_com)


    ###################### Current plane def below! ######################
    # com_middle: center of mass of the capping residues (hardcoded 3) of the middle helices

    com_cap_size = 3
    cap_reslist = [x-com_cap_size+loops[loop_idx][0] for x in range(com_cap_size)] + [x+1+loops[loop_idx][-1] for x in range(com_cap_size)]
    com_middle = np.array(get_com(pose, cap_reslist))


    # plane equation ax+by+cz+d = 0, containing points: com_middle, rn_com and rc_com
    norm_vec = np.cross(rn_com - com_middle, rc_com - com_middle)
    a, b, c = norm_vec
    d = -1 * np.dot(norm_vec, com_middle)

    ###################### Current plane def above! ######################


    if bturn_side == 'ccap':
        # max dist to loop Nterm
        dist_target_resid = loops[loop_idx][-1]+1
    elif bturn_side == 'ncap':
        # max dist to loop Cterm
        dist_target_resid = loops[loop_idx][0]-1

    else:
        # max dist type not defined, return max cos value
        if _DEBUG:
            print(f'Error: unsupported bturn_side value {bturn_side}')
        sys.exit(1)

    max_dist_res = [-1, 0, [0,0,0]]
    for resid in range(loops[loop_idx][0], loops[loop_idx][-1]+1):
        dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose.residue(resid).xyz('CA'), pose.residue(dist_target_resid).xyz('CA'))
        if dist > max_dist_res[1]:
            max_dist_res = [resid, dist, np.array(list(pose.residue(resid).xyz('CA')))]

    #print(loops[loop_idx][0]+repeat_ahead, loops[loop_idx][-1]+repeat_ahead+1, max_dist_res)

    # projection of farthest residue CA onto the plane rn_com, rmid_com and rc_com (ref: https://stackoverflow.com/questions/9971884/computational-geometry-projecting-a-2d-point-onto-a-plane-to-determine-its-3d)
    e,f,g = max_dist_res[-1]
    k = -1 * float(circle_a*e+circle_b*f+circle_c*g+circle_d) / (circle_a*circle_a+circle_b*circle_b+circle_c*circle_c)
    proj_circle = np.array([e+k*circle_a, f+k*circle_b, g+k*circle_c])
    #print(proj_circle)
    if np.linalg.norm( proj_circle - P ) > R:
        # loop is pointing away from center of circle
        if _DEBUG:
            print('loop is pointing away from center of circle, return max cos value (1) ...')
        return 1

    # projection of farthest residue CA onto the plane rn_com, com_middle and rc_com 
    k = -1 * float(a*e+b*f+c*g+d) / (a*a+b*b+c*c)
    proj = np.array([e+k*a, f+k*b, g+k*c])
    #print('pseudoatom com_point, pos=[{}]'.format(','.join([str(x) for x in max_dist_res[-1]])))
    #print('pseudoatom com_proj, pos=[{}]'.format(','.join([str(x) for x in proj])))


    ###################### new angle def below! ######################
    # compute the angle between rloopidx_com_proj->proj and rloopidx_com_proj->max_dist_res[-1]
    rloopidx_com_e,rloopidx_com_f,rloopidx_com_g = rloopidx_com
    k = -1 * float(a*rloopidx_com_e+b*rloopidx_com_f+c*rloopidx_com_g+d) / (a*a+b*b+c*c)
    rloopidx_com_proj = np.array([rloopidx_com_e+k*a, rloopidx_com_f+k*b, rloopidx_com_g+k*c])
    #print('pseudoatom rloopidx_com_proj, pos=[{}]'.format(','.join([str(x) for x in rloopidx_com_proj])))
    unit_vec = (max_dist_res[-1]-rloopidx_com_proj) / np.linalg.norm(max_dist_res[-1]-rloopidx_com_proj)
    unit_vec_proj = (proj-rloopidx_com_proj) / np.linalg.norm(proj-rloopidx_com_proj)
    cos_angle = np.dot(unit_vec, unit_vec_proj)
    ###################### new angle def above! ######################


    #print(max_dist_res)
    #print('{} {}'.format(line.strip(), cos_angle))
    return cos_angle


#<<<<<<=============================  loop direction check ======================================


#===================================  loop hairpin shape check ================================>>>>>>
def get_ca_dist(pose, resi1, resi2):
    coord1 = [pose.residue(resi1).xyz('CA')[i] for i in range(3)]
    coord2 = [pose.residue(resi2).xyz('CA')[i] for i in range(3)]
    return np.linalg.norm( np.array(coord1) - np.array(coord2) )

def find_beta_turn(pose, loop1, sf=''):
    if sf == '':
        sf = pyrosetta.get_score_function()
    sf(pose)
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    hb = False
    # find beta turn (i-hb-i+3)
    beta_turn_check = False
    interloop_hb_don, interloop_hb_acc = -1, -1
    for resid in range(loop1[0],loop1[1]+1):
        if beta_turn_check:
            break
        for hb_i in hbond_set.residue_hbonds(resid):
            if (hb_i.acc_res() == resid+3 or hb_i.don_res() == resid+3) \
                and hb_i.acc_atm_is_backbone() and hb_i.don_hatm_is_backbone():
                beta_turn_check = True
                interloop_hb_don = hb_i.don_res()
                interloop_hb_acc = hb_i.acc_res()
                break
    #print(interloop_hb_don, interloop_hb_acc)
    if interloop_hb_acc == -1 or interloop_hb_don == -1:
        #print('No inter beta turn bb hb found!')
        return [-1,-1]

    return sorted([interloop_hb_don, interloop_hb_acc])


def check_loop_hairpin_shape(pose, sf, cutoffs=[-1], bturn=[]):
    if cutoffs[0] == -1:
        return True
    num_d = len(cutoffs) # number of dists to compute
    loops = find_loop_by_dssp(pose)
    if len(loops) <= 0:
        # cannot find loops
        return False
    loop = loops[0]
    if len(bturn) == 0:
        bturn = find_beta_turn(pose, loop, sf)  # this fails when more than one beta turns in the loop
    bturn = [bturn[0]+1, bturn[1]-1]
    for i in range(len(cutoffs)):
        d = get_ca_dist(pose, bturn[0]-i, bturn[1]+i)
        if d > cutoffs[i]:
            return False
    return True


#<<<<<<=============================  loop hairpin shape check ======================================





def align_cap_pose_to_anchor_coords(cap, coords, ncap=True):
    # ncap: True (ncap), False (ccap)
    if ncap:
        moveable_coords = get_anchor_coordinates_from_pose(cap, [cap.size()-1])
    else:
        moveable_coords = get_anchor_coordinates_from_pose(cap, [2])
    R, t = np_utils.rigid_transform_3D(moveable_coords, coords)
    #np_utils.rotate_pose(p, np_utils.numpy_to_rosetta(R)) # JHL, sth wrong w/ dig's pyrosetta: xx() not callable, but xx directly accessible
    my_own_rotate_pose(cap, my_own_2D_numpy_to_rosetta(R))  # JHL, so I had to rewrite np->rosetta and rotation function to change xx() to xx
    #np_utils.translate_pose(cap, np_utils.numpy_to_rosetta(t.T))
    my_own_translate_pose(cap, numpy_to_rosetta(t.T))
    return cap

def generate_cap(phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=True, aacomp_cap_pro_file="-1"):
    cap_seq = aa*(len(phipsi)+2)
    cap = pyrosetta.pose_from_sequence(cap_seq) # two more residues serve as dummy termini for phipsi definition
    for i in range(1,len(phipsi)+1):
        cap.set_phi(i+1,phipsi[i-1][0])
        cap.set_psi(i+1,phipsi[i-1][1])
        # omega is 180 by default

    #
    # TODO
    # get the binselector to work!!
    #
    if insert_GLy: # put pro and gly in if its their bin
        pack_reslist = []
        for i in range(1,len(phipsi)+1):
            # gly
            if phipsi[i-1][0] > 0: # CAUTION!! hard coded way of identify gly bin 
                mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(i+1, 'GLY')
                mt.apply(cap)
            else:
                pack_reslist.append(i+1)
    else:
        pack_reslist = [x+1 for x in range(1, len(phipsi)+1)]

    if insert_Pro:        

        if aacomp_cap_pro_file != "-1":
            p_bin = pyrosetta.rosetta.core.select.residue_selector.BinSelector()
            p_bin.set_bin_params_file_name('PRO_DPRO')
            p_bin.set_bin_name('LPRO')
            p_bin.initialize_and_check()
            aacomp_pro = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
            aacomp_pro.create_constraint_from_file(aacomp_cap_pro_file)
            aacomp_pro.add_residue_selector(p_bin)
            aacomp_pro.apply(cap) 

        # pro (need the bin selector here)
        #sf.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2)

        ## old packer
        #pack_rotamers = pyrosetta.rosetta.protocols.simple_moves.PackRotamersMover(sf_cap)
        #pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa='AP'))

        pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sf_cap)
        pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa='AP'))
        pack_rotamers.apply(cap)

    return cap

def insert_n_cap(p, anchor_pos, cap):
    # inserting ncap to the tail side of the gap in pdb (closer to pdb's cterm)
    # assuming ncap is inserted before ccap!
    p_anchor_coord = get_anchor_coordinates_from_pose(p, [anchor_pos+1])
    align_cap_pose_to_anchor_coords(cap, p_anchor_coord, ncap=True)
    new_p = pyrosetta.rosetta.core.pose.Pose()
    new_p.append_residue_by_jump(p.residue(1), 1)
    for i in range(2,anchor_pos+1):
        new_p.append_residue_by_bond(p.residue(i))
    new_p.append_residue_by_jump(cap.residue(2), anchor_pos)
    for i in range(3,cap.size()):
        new_p.append_residue_by_bond(cap.residue(i))
    for i in range(anchor_pos+2, p.size()+1): # skipping the overlapping resiude on the scaffold
        if 'Nterm' in p.residue(i).name():
            new_p.append_residue_by_jump(p.residue(i),new_p.size())
        else:
            new_p.append_residue_by_bond(p.residue(i))
    return new_p

def insert_c_cap(p, anchor_pos, cap):
    # inserting ccap to the head side of the gap in pdb (closer to pdb's nterm)
    # assuming ncap has been inserted before ccap!
    p_anchor_coord = get_anchor_coordinates_from_pose(p, [anchor_pos])
    align_cap_pose_to_anchor_coords(cap, p_anchor_coord, ncap=False)
    new_p = pyrosetta.rosetta.core.pose.Pose()
    new_p.append_residue_by_jump(p.residue(1), 1)
    for i in range(2,anchor_pos): # skipping the overlapping resiude on the scaffold
        new_p.append_residue_by_bond(p.residue(i))
    for i in range(2,cap.size()):
        new_p.append_residue_by_bond(cap.residue(i))
    new_p.append_residue_by_jump(p.residue(anchor_pos+1), anchor_pos+cap.size()-3)
    for i in range(anchor_pos+2, p.size()+1): 
        if 'Nterm' in p.residue(i).name():
            new_p.append_residue_by_jump(p.residue(i),new_p.size())
        else:        
            new_p.append_residue_by_bond(p.residue(i))
    return new_p


def fragment_gen_kic_mover(p, before_frag_length, after_frag_length, phipsi, n_cap_phipsi, c_cap_phipsi, anchor_pos, n_cys, c_cys,
                  scorefxn, selection_scorefxn, perturb=0, preselection_flank=1, preselection_mover=None):

    '''
        n_cys, c_cys: resids of residues before/after the chain break
    '''

    num_res_added = before_frag_length + len(phipsi) + after_frag_length

    if len(phipsi) == 0:
        if before_frag_length == 0 and after_frag_length == 0:
            print('Error: before_frag_length: {} after_frag_length: {}, cannot be both 0s'.format(before_frag_length, after_frag_length))
            return False                 
        if before_frag_length+after_frag_length == 1:
            mid_pivot = anchor_pos + 1
        else:
            mid_pivot = anchor_pos + int((before_frag_length+after_frag_length)/2)
    elif before_frag_length != 0:
        mid_pivot = anchor_pos + before_frag_length
    elif after_frag_length != 0:
        mid_pivot = anchor_pos + len(phipsi) + 1
    else:
        print('Error: before_frag_length: {} after_frag_length: {}, cannot be both 0s'.format(before_frag_length, after_frag_length))
        return False       

    # pivot_res: [achor_pos,'CA',mid_pivot,'CA',anchor_pos+num_res_added+1,'CA']
    pivot_res = list(chain.from_iterable([(anchor_pos, 'CA'),
                                          (mid_pivot, 'CA'),  # make sure pivot is not in the fragment!!
                                          (anchor_pos + num_res_added + 1, 'CA')]))

    gk = pyrosetta.rosetta.protocols.generalized_kinematic_closure.GeneralizedKIC()
    gk.set_selector_type('lowest_energy_selector')

    gk.set_selector_scorefunction(selection_scorefxn)
    #gk.set_selector_scorefunction(scorefxn)

    #gk.set_closure_attempts(100)
    gk.set_closure_attempts(1000)
    gk.set_min_solution_count(1) # get whatever works

    def _add_dihedral_to_perturber(atm1, atm2):
        atom_set = pyrosetta.rosetta.utility.vector1_core_id_NamedAtomID()
        atom_set.append(pyrosetta.rosetta.core.id.NamedAtomID(*atm1))
        atom_set.append(pyrosetta.rosetta.core.id.NamedAtomID(*atm2))
        gk.add_atomset_to_perturber_atomset_list(atom_set)



    for res_num in range(anchor_pos,anchor_pos + num_res_added + 2):
        gk.add_loop_residue(res_num)


    for res_num in pivot_res:
        if type(res_num) != int: 
            continue
        # The pivot residues are not necessarily in good regions of
        # Ramachandran space, so we should filter by the rama_prepro energy
        # of pivot positions.
        gk.add_filter('rama_prepro_check')
        gk.set_filter_resnum(res_num)
        gk.set_filter_rama_cutoff_energy(0.5)
        #gk.set_filter_rama_cutoff_energy(10000)

    gk.add_filter('loop_bump_check') 

    # the close bond call must occur BEFORE the randomize dihedral perturber
    gk.close_bond(n_cys, _peptide_bond_params['atom1'],
                  c_cys, _peptide_bond_params['atom2'],
                  0, '', 0, '',  # optional params -- use default values
                  _peptide_bond_params['bond_length'],
                  _peptide_bond_params['angle1'],
                  _peptide_bond_params['angle2'],
                  _peptide_bond_params['torsion'],
                  True, False)
                  #0., True, False)


    # perturbers 
    # randomization
    gk.add_perturber('randomize_backbone_by_rama_prepro')
    for res_num in range(anchor_pos + 1,anchor_pos + num_res_added + 1):
        if res_num <= anchor_pos + before_frag_length or res_num > anchor_pos + before_frag_length + len(phipsi):
            #if _DEBUG:
            #    print('DEBUG: Randomizing residue ', res_num)
            gk.add_residue_to_perturber_residue_list(res_num)

    # fragment
    if len(phipsi) > 0:
        #if _DEBUG:
        #    print('DEBUG: Inserting fragment ...')
        for i in range(len(phipsi)):
            # phi
            gk.add_perturber('set_dihedral')
            gk.add_value_to_perturber_value_list(phipsi[i][0])
            _add_dihedral_to_perturber(('N', anchor_pos+before_frag_length+i+1), ('CA', anchor_pos+before_frag_length+i+1)) # phi
            # psi
            gk.add_perturber('set_dihedral')
            gk.add_value_to_perturber_value_list(phipsi[i][1])
            _add_dihedral_to_perturber(('CA', anchor_pos+before_frag_length+i+1), ('C', anchor_pos+before_frag_length+i+1)) # psi         
    else:
        #if _DEBUG:
        #    print('DEBUG: No fragment inserted ...')
        pass

    # sample around the fragment
    if perturb > 0 and len(phipsi) > 0:
        gk.add_perturber('perturb_dihedral')
        gk.add_value_to_perturber_value_list(perturb)
        for i in range(len(phipsi)):
            if _DEBUG:
                print('DEBUG: Perturbing fragment residue {} by {} degrees ..'.format(anchor_pos+before_frag_length+i+1, perturb))
            _add_dihedral_to_perturber(('N', anchor_pos+before_frag_length+i+1), ('CA', anchor_pos+before_frag_length+i+1)) # phi
            _add_dihedral_to_perturber(('CA', anchor_pos+before_frag_length+i+1), ('C', anchor_pos+before_frag_length+i+1)) # psi         

    # omega
    gk.add_perturber('set_dihedral')
    gk.add_value_to_perturber_value_list(180.)
    for pos in range(anchor_pos, anchor_pos + num_res_added + 1):
        #if _DEBUG:
        #    print('DEBUG: Setting omega for: ', pos, pos+1)
        _add_dihedral_to_perturber(('C', pos), ('N', pos + 1))

    '''
    # perturb dihedral (instead of randomizing)
    rama_dev = 30.
    gk.add_perturber('perturb_dihedral')
    gk.add_value_to_perturber_value_list(rama_dev)
    for pos in range(anchor_pos, anchor_pos + num_res_added + 2):
        #if pos in [anchor_pos+mid_res, anchor_pos+mid_res+1]: # prevent perturbing the frag residues
        #    continue
        _add_dihedral_to_perturber(('N', pos), ('CA', pos))
        _add_dihedral_to_perturber(('CA', pos), ('C', pos))
    '''

    #if _DEBUG:
    #    print('GeneralizedKIC: {}'.format(pivot_res))

    gk.set_pivot_atoms(*pivot_res)

    
    if preselection_mover != None:
        gk.set_preselection_mover(preselection_mover)
    

    '''
    gk.set_preselection_mover(_setup_fast_design(p, scorefxn, terminus_ranges,
                              preselection_flank))
    '''
    gk.apply(p)
    return gk.get_last_move_status()


class SymPose:
    '''
        sympose: the pose itself
        labels: list of labels to be added into pdbinfo later
        used_sc_resids: resids of residues whose side chains have been used for existing features
    '''

    def __init__(self, sympose, labels=[], used_sc_resids={}):
        self._sympose = sympose
        self._labels = labels
        self._used_sc_resids = used_sc_resids

    def pose(self):
        return self._sympose

    def labels(self):
        return self._labels

    def used_sc_resids(self):
        return self._used_sc_resids

    def set_pose(self, pose):
        self._sympose = pose

    def set_labels(self, labels):
        self._labels = labels

    def set_used_sc_resids(self, used_sc_resids):
        self._used_sc_resids = used_sc_resids



def find_hbonds(pose, search_reslist, target_reslist, skip=[]):
    '''
        skip = [] for each element k:
            hbonds between residue i and residue i+k are skipped (e.g. when k = 2, gamma turns are skipped)
        my_hbond_set = {'bb-bb':[(don_resid, acc_resid, hb_i object)] ...}
    '''
    my_hbond_set = {'bb-bb':[],'bb-sc':[],'sc-sc':[]}
    my_hbond_list = []
    #sc_pack_list = [] # residues in this list have sc hbonding to bb, and therfore should be kept
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pose.update_residue_neighbors()  # no need to update as I just scored the pose 
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    for i in search_reslist:
        for hb_i in hbond_set.residue_hbonds(i):
            if hb_i.acc_res() in target_reslist or hb_i.don_res() in target_reslist:

                # e.g. skip gamma turn hbond i-i+2
                if len(skip) > 0:
                    skip_check = False
                    for k in skip:
                        if hb_i.acc_res() == hb_i.don_res() + k or hb_i.acc_res() == hb_i.don_res() - k:
                            skip_check = True
                            break
                    if skip_check:
                        continue

                # skip repeat of hbonds (happens when there are internal hb pairs)
                if (hb_i.don_res(), hb_i.acc_res(), hb_i) in my_hbond_list:
                    continue
                my_hbond_list.append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )

                if hb_i.acc_atm_is_backbone() and hb_i.don_hatm_is_backbone():
                    my_hbond_set['bb-bb'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                elif hb_i.acc_atm_is_backbone() or hb_i.don_hatm_is_backbone():
                    #print(hb_i.acc_res(), hb_i.don_res())
                    my_hbond_set['bb-sc'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                else:
                    my_hbond_set['sc-sc'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )

    return my_hbond_set

def find_bidentate_hbond(pose, hbond_list):
    '''
        Definition of bidentate hbond: 
            at least two different HEAVY atoms from the same residue participate in hbonds within two different atoms

        hbond_list = [(don_resid, acc_resid, hb_i object), ...]
    '''

    hbonds_by_sc_res = {} # {sidechain_res:[(don_res,don_atom,acc_res,acc_atom)]}
    for hb in hbond_list:

        hb_i = hb[-1] # hb object

        if hb_i.don_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.don_res()] = []
        if hb_i.acc_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.acc_res()] = []

        hb_info = ( hb_i.don_res(), pose.residue(hb_i.don_res()).atom_name(hb_i.don_hatm()).strip(), \
                    hb_i.acc_res(), pose.residue(hb_i.acc_res()).atom_name(hb_i.acc_atm()).strip() ) # don_res, don_atm_name, acc_res, acc_atm_name
        if not hb_i.don_hatm_is_backbone():
            hbonds_by_sc_res[hb_i.don_res()].append( hb_info )
        if not hb_i.acc_atm_is_backbone():
            hbonds_by_sc_res[hb_i.acc_res()].append( hb_info )

    bidentate_hbonds_by_sc_res = {}
    for res in hbonds_by_sc_res:
        if len(hbonds_by_sc_res[res]) > 1:
            bidentate_hbonds_by_sc_res[res] = hbonds_by_sc_res[res]
    return bidentate_hbonds_by_sc_res


def simple_hbond_finder(pose, reslist1, reslist2, delta_HA=3., delta_theta=30., 
                        reslist1_atom_type=['bb','sc'], reslist2_atom_type=['bb','sc'],verbose=False):
    '''
    input:
        pose
        reslist1
        reslist2 (can overlap with reslist1)
        delta_HA: max distance cutoff
        delta_theta: max angle deviation cutoff
        reslist1/reslist2_type: type of atom to be considered
    output:
        list of hbond atom pairs: [(don_res1,don_atm1,accpt_res2,accpt_atm2,delta_HA,theta),(),()...]
    ''' 
    
    def _get_don_accpt_residue_atom_indices(pose, reslist, atom_type=['bb','sc']):
        '''
        don_list = [(resid,Hpol_atomid,base_atomid),()..]
        accpt_list = [(resid,accpt_pos,base_atomid),()..]
        '''
        don_list, accpt_list = [], []
        for resid in reslist:
            bb_list = [x for x in range(1,pose.residue(resid).natoms()+1) if pose.residue(resid).atom_is_backbone(x)]
            for atomid in pose.residue(resid).Hpol_index():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                don_list.append( (resid, atomid, pose.residue(resid).atom_base(atomid)) )
            for atomid in pose.residue(resid).accpt_pos():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                accpt_list.append( (resid, atomid, pose.residue(resid).atom_base(atomid)) )
        return don_list, accpt_list
    
    reslist1_don, reslist1_accpt = _get_don_accpt_residue_atom_indices(pose, reslist1, reslist1_atom_type)
    reslist2_don, reslist2_accpt = _get_don_accpt_residue_atom_indices(pose, reslist2, reslist2_atom_type)
    #print(reslist1_don)
    #print(reslist1_accpt)
    #print(reslist2_don)
    #print(reslist2_accpt)
    
    hbonds = []
    
    
    for don in reslist1_don:
        for accpt in reslist2_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by 3 atom xyzs
            angle = pyrosetta.rosetta.numeric.angle_degrees_double( pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by vectors (not working as no reference angle for this)
            #don_vec = np.array( pose.residue(don[0]).xyz(don[1]) - pose.residue(don[0]).xyz(don[2]) )
            #accpt_vec = np.array( pose.residue(accpt[0]).xyz(accpt[1]) - pose.residue(accpt[0]).xyz(accpt[2]) )
            #angle = np.arccos(np.dot(don_vec, accpt_vec) / (np.linalg.norm(don_vec) * np.linalg.norm(accpt_vec)))
            #angle = angle*180/np.pi
            if verbose:
                print(don, pose.residue(don[0]).atom_name(don[1]).strip(), accpt, pose.residue(accpt[0]).atom_name(accpt[1]).strip(), dist,angle)
            if dist <= delta_HA and angle >= 180-delta_theta:
                # always write out reslist1 first
                hbonds.append( (don[0],don[1],accpt[0],accpt[1],dist,angle,'don-accpt') )
    for don in reslist2_don:
        for accpt in reslist1_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by 3 atom xyzs
            angle = pyrosetta.rosetta.numeric.angle_degrees_double( pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by vectors (not working as no reference angle for this)
            #don_vec = np.array( pose.residue(don[0]).xyz(don[1]) - pose.residue(don[0]).xyz(don[2]) )
            #accpt_vec = np.array( pose.residue(accpt[0]).xyz(accpt[1]) - pose.residue(accpt[0]).xyz(accpt[2]) )
            #angle = np.arccos(np.dot(don_vec, accpt_vec) / (np.linalg.norm(don_vec) * np.linalg.norm(accpt_vec)))
            #angle = angle*180/np.pi            
            if verbose:
                print(don, pose.residue(don[0]).atom_name(don[1]).strip(), accpt, pose.residue(accpt[0]).atom_name(accpt[1]).strip(), dist,angle)
            if dist <= delta_HA and angle >= 180-delta_theta:
                # always write out reslist1 first
                hbonds.append( (accpt[0],accpt[1],don[0],don[1],dist,angle,'accpt-don') )               

                
    return hbonds

def find_potential_bidentate_hbond(pose, reslist1, reslist2, delta_HA=3., delta_theta=30.,
                                   reslist1_atom_type=['sc'], reslist2_atom_type=['bb']):
    hbonds = simple_hbond_finder(pose, reslist1, reslist2, delta_HA=delta_HA, delta_theta=delta_theta,
                                 reslist1_atom_type=reslist1_atom_type, reslist2_atom_type=reslist2_atom_type)
    hbonds_by_sc_res = {} # {sidechain_res:[(don_res,don_atom,acc_res,acc_atom,distance,angle)]}
    for hb in hbonds:
        if hb[0] not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb[0]] = []
        if hb[6] == 'don-accpt':
            hbonds_by_sc_res[hb[0]].append( (hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[4], hb[5]) ) 
        elif hb[6] == 'accpt-don':
            hbonds_by_sc_res[hb[0]].append( (hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[4], hb[5]) )
        else:
            print('Error: incorrect donor acceptor info: {}'.fomrat(hb[6]))
        
    bidentate_hbonds_by_sc_res = {}
    for res in hbonds_by_sc_res:
        if len(hbonds_by_sc_res[res]) > 1:
            bidentate_hbonds_by_sc_res[res] = hbonds_by_sc_res[res]
    return bidentate_hbonds_by_sc_res

def check_fully_satisfied_bidenates(pose, bidentates, bidentate_resid):

    this_residue = pose.residue(bidentate_resid)
    sc_heavy_atms = {}

    hpol_map = {}
    for hpol in this_residue.Hpol_index():
        if not this_residue.atom_is_backbone(hpol):
            #print(hpol, this_residue.atom_name(hpol))
            hpol_map[hpol] = False

    for atom_id in range(1, len(this_residue.atoms())+1):
        atm_name = this_residue.atom_name(atom_id).strip()
        if 'H' not in atm_name:
            if 'N' in atm_name or 'O' in atm_name: # heavy atoms
                if not this_residue.atom_is_backbone(atom_id): # avoid backbone 
                    hb_group = [atm_name]
                    for hpol in hpol_map:
                        if not hpol_map[hpol] and hpol in this_residue.bonded_neighbor(atom_id):
                            hb_group.append(this_residue.atom_name(hpol).strip())
                            hpol_map[hpol] = True
                    sc_heavy_atms[tuple(hb_group)] = False
    #print(sc_heavy_atms)

    for hb in bidentates[bidentate_resid]:
        for hb_i in [0,2]: # don, acc        
            if hb[hb_i] == bidentate_resid:
                for hb_group in sc_heavy_atms:
                    if hb[hb_i+1] in hb_group:
                        sc_heavy_atms[hb_group] = True

    #print(bidentate_resid, sc_heavy_atms)

    for hb_group in sc_heavy_atms:
        if sc_heavy_atms[hb_group] == False:
            return False

    return True


def find_all_beta_turns(pose, loop1):
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pose.update_residue_neighbors()  # no need to update as I just scored the pose 
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    beta_turn_check = False
    # find beta turn (i-hb-i+3)
    beta_turn_list = []
    for resid in range(loop1[0],loop1[-1]+1):
        if beta_turn_check:
            break
        for hb_i in hbond_set.residue_hbonds(resid):
            if (hb_i.acc_res() == resid+3 or hb_i.don_res() == resid+3) \
                and hb_i.acc_atm_is_backbone() and hb_i.don_hatm_is_backbone():
                beta_turn_check = True
                interloop_hb_don = hb_i.don_res()
                interloop_hb_acc = hb_i.acc_res()
                if [interloop_hb_don, interloop_hb_acc] not in beta_turn_list:
                    beta_turn_list.append([interloop_hb_don, interloop_hb_acc])
    return beta_turn_list


def find_all_pseudo_beta_turns(pose, loop1, delta_HA=3., delta_theta=30.):
    loop_reslist = list(range(loop1[0],loop1[-1]+1))
    pseudo_hbonds = simple_hbond_finder(pose, loop_reslist, loop_reslist,
                                         delta_HA=delta_HA, delta_theta=delta_theta, 
                                        reslist1_atom_type=['bb'], reslist2_atom_type=['bb'],verbose=False)
    #if _DEBUG:
    #    print('DEBUG:             find_all_pseudo_beta_turns:  loop_reslist: ', loop_reslist)
    #    print('DEBUG:             find_all_pseudo_beta_turns:  pseudo_hbonds: ', pseudo_hbonds)

    beta_turn_list = []
    for hb in pseudo_hbonds:
        if (hb[0] == hb[2] + 3 or hb[0] == hb[2] - 3) and sorted([hb[0],hb[2]]) not in beta_turn_list:
            beta_turn_list.append(sorted([hb[0],hb[2]]))
    return beta_turn_list


def add_atom_pair_cst_to_pose(pose, res1_id_vec, atm1_name_vec, res2_id_vec, atm2_name_vec, func_vec):
    '''
    res1_id_vec, res2_id_vec: pyrosetta.rosetta.utility.vector1_unsigned_long()
    atm1_name_vec, atm2_name_vec, func_vec: pyrosetta.rosetta.utility.vector1_std_string()
    func_vec elements: e.g. 'HARMONIC 2.0 0.5'
    '''
    dist_cst = pyrosetta.rosetta.protocols.cyclic_peptide.CreateDistanceConstraint()
    dist_cst.set(res1_id_vec, atm1_name_vec, res2_id_vec, atm2_name_vec, func_vec)
    dist_cst.apply(pose)


def add_bidentate_atom_pair_cst_to_pose(pose, bidentate_hbonds, cst_func='HARMONIC 2.0 0.5'):
    '''
    bidentate_hbonds: {resid_of_sidechain-hbonded_residue:[hb1_tuple,hb2_tuple],[],..}
    hb_tuple: (don_resid,don_atom_name,acc_resid,acc_atom_name,distance,angle)
    '''
    res1_id_vec = pyrosetta.rosetta.utility.vector1_unsigned_long()
    atm1_name_vec = pyrosetta.rosetta.utility.vector1_std_string()
    res2_id_vec = pyrosetta.rosetta.utility.vector1_unsigned_long()
    atm2_name_vec = pyrosetta.rosetta.utility.vector1_std_string()
    func_vec = pyrosetta.rosetta.utility.vector1_std_string()
    
    for b_hbond_resid in bidentate_hbonds:
        for cst_id in range(len(bidentate_hbonds[b_hbond_resid])):
            res1_id_vec.append(bidentate_hbonds[b_hbond_resid][cst_id][0])
            atm1_name_vec.append(bidentate_hbonds[b_hbond_resid][cst_id][1])
            res2_id_vec.append(bidentate_hbonds[b_hbond_resid][cst_id][2])
            atm2_name_vec.append(bidentate_hbonds[b_hbond_resid][cst_id][3])
            func_vec.append(cst_func)
    add_atom_pair_cst_to_pose(pose, res1_id_vec, atm1_name_vec, res2_id_vec, atm2_name_vec, func_vec)

def add_hbond_atom_pair_cst_to_pose(pose, hbonds, cst_func='HARMONIC 2.0 0.5'):
    '''
    hbonds: [(accpt_resi,accpt_atmi,don_resi,don_atmi,dist,angle,'accpt-don'),...]
    '''
    res1_id_vec = pyrosetta.rosetta.utility.vector1_unsigned_long()
    atm1_name_vec = pyrosetta.rosetta.utility.vector1_std_string()
    res2_id_vec = pyrosetta.rosetta.utility.vector1_unsigned_long()
    atm2_name_vec = pyrosetta.rosetta.utility.vector1_std_string()
    func_vec = pyrosetta.rosetta.utility.vector1_std_string()
    
    for hb in hbonds:
        res1_id_vec.append(hb[0])
        atm1_name_vec.append(pose.residue(hb[0]).atom_name(hb[1]).strip())
        res2_id_vec.append(hb[2])
        atm2_name_vec.append(pose.residue(hb[2]).atom_name(hb[3]).strip())
        func_vec.append(cst_func)
    add_atom_pair_cst_to_pose(pose, res1_id_vec, atm1_name_vec, res2_id_vec, atm2_name_vec, func_vec)


def propagate_bidentate_hbonds(bidentate_hbonds, num_repeats, repeatlen):
    pose_size = num_repeats * repeatlen
    repeatlen = int(pose_size/num_repeats)
    new_bidentate_hbonds = {}
    for b_hbond_resid in bidentate_hbonds:
        # only propagate bidentates from the first repeat
        if b_hbond_resid > repeatlen:
            new_b_hbond_resid = b_hbond_resid%repeatlen
            if new_b_hbond_resid == 0:
                new_b_hbond_resid = repeatlen
        else:
            new_b_hbond_resid = b_hbond_resid
        res_diff = new_b_hbond_resid - b_hbond_resid

        for repeat_id in range(num_repeats):
            range_check = True
            new_entry = []
            for cst_id in range(len(bidentate_hbonds[b_hbond_resid])):
                hb_info = bidentate_hbonds[b_hbond_resid][cst_id]
                res1_id = hb_info[0] + repeat_id*repeatlen + res_diff
                res2_id = hb_info[2] + repeat_id*repeatlen + res_diff
                if res1_id < 1 or res1_id > pose_size or res2_id < 1 or res2_id > pose_size:
                    range_check = False
                    break
                new_entry.append((res1_id,hb_info[1],res2_id,hb_info[3],hb_info[4],hb_info[5]))
            if range_check:                
                new_bidentate_hbonds[repeat_id*repeatlen+new_b_hbond_resid] = new_entry
    return new_bidentate_hbonds


def compute_motifscore_for_residue_pair(pose, res1, res2, dist_cutoff=12, 
                                        dssp_obj=None, motif_hash_man=None, max_motif_per_res=3.0):
    '''
        modified from core/scoring/methods/CenPairMotifEnergy.cc
                    protocol/simple_filters/SSElementMotifContactFilter::get_SSelements_in_contact
    '''
    
    tree = pose.fold_tree()
    
    # do not score terminal residues (lack bb torsion) or self pairig or jump point in foldtree
    if res1 in [1, pose.size()] or res2 in [1, pose.size()] or res1 == res2 or \
        tree.is_jump_point(res1) or tree.is_jump_point(res2):
        return 0
    
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
        #dssp = dssp_obj.get_dssp_secstruct()
    
    if motif_hash_man == None:
        motif_hash_man = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
           
    dist = pose.residue(res1).xyz('CA').distance(pose.residue(res2).xyz('CA'))
    # skip motif searching if residues too far away from each other
    if dist > dist_cutoff:
        return 0
    
    ss1 = dssp_obj.get_dssp_secstruct(res1)
    aa1 = pose.residue(res1).name1()
    bb_stub1 = pyrosetta.rosetta.core.pose.motif.get_backbone_reference_frame(pose, res1)

    ss2 = dssp_obj.get_dssp_secstruct(res2)
    aa2 = pose.residue(res2).name1()    
    bb_stub2 = pyrosetta.rosetta.core.pose.motif.get_backbone_reference_frame(pose, res2)

    #print(ss1,aa1,ss2,aa2,dist)
    
    # Xbb = bb_stub1.inverse() * bb_stub2
    # numeric/xyzTransform.hh: line 127:
    # friend Transform operator *( Transform const & a, Transform const & b ){ return Transform( a.R*b.R, a.R*b.t + a.t ); }
    bb_stub1_inv = bb_stub1.inverse()
    Xbb = pyrosetta.rosetta.numeric.xyzTransform_double_t( bb_stub1_inv.R*bb_stub2.R, bb_stub1_inv.R*bb_stub2.t + bb_stub1_inv.t )
    
    xs_bb_fxn1 = motif_hash_man.get_xform_score_BB_BB(ss1,ss2,aa1,aa2)
    xs_bb_fxn2 = motif_hash_man.get_xform_score_BB_BB(ss2,ss1,aa2,aa1)
    
    try:        
        score1 = xs_bb_fxn1.score_of_bin(Xbb)
    except AttributeError:
        score1 = 0    
    try:        
        score2 = xs_bb_fxn2.score_of_bin(Xbb.inverse())
    except AttributeError:
        score2 = 0
        
    tmpScore = min(score1+score2, max_motif_per_res)
        
    return tmpScore



def _setup_fast_relax(p, scorefxn, resids, cartesian=False, include_neighbor=True, itr=1):
    mm = pyrosetta.MoveMap()
    #ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector('{}'.format(resids))
    ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in resids]))
    ind = ind_selector.apply(p)
    mm.set_bb(ind)
    if include_neighbor:
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(ind_selector, 6.0)
        nbr = nbr_selector.apply(p)
        #print(nbr)
        mm.set_chi(nbr)
    else:
        mm.set_chi(ind)
    #mm.show()               

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(itr)
    fr.set_scorefxn(scorefxn)
    fr.set_movemap(mm)
    fr.cartesian(cartesian)
    return fr

def _setup_min_mover(p, scorefxn, resids, cartesian=False, include_neighbor=True):
    mm = pyrosetta.MoveMap()
    #ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector('{}'.format(resids))
    ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in resids]))
    ind = ind_selector.apply(p)
    mm.set_bb(ind)
    if include_neighbor:
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(ind_selector, 6.0)
        nbr = nbr_selector.apply(p)
        #print(nbr)
        mm.set_chi(nbr)
    else:
        mm.set_chi(ind)
    m = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    m.score_function(scorefxn)
    m.movemap(mm)
    m.cartesian(cartesian)
    return m


def get_xml_string(anchor_pos, loop_len, repeatlen, loop_flank=3, min_intraloop_hbond=2, min_interloop_hbond=0, MAX_UNSATS=1, MAX_GAMMA=0):

    '''
        disabled unsats check for now, tricky to distinguish real unsats from the ones salvageable by seq design

        repeatlen is the length of repeat before genkic!

    '''



    #script_string  = '<ROSETTASCRIPTS>\n'
    script_string = ''
    #script_string += '  <SCOREFXNS>\n'
    #script_string += '        <ScoreFunction name="ref_cst" weights="beta">\n'
    #script_string += '            <Reweight scoretype="coordinate_constraint" weight="1" />\n'
    #script_string += '            <Reweight scoretype="atom_pair_constraint" weight="1" />\n'
    #script_string += '            <Reweight scoretype="dihedral_constraint" weight="1" />\n'
    #script_string += '            <Reweight scoretype="angle_constraint" weight="1" />\n'
    #script_string += '        </ScoreFunction>\n'
    #script_string += '  </SCOREFXNS>\n'
    script_string += '  <RESIDUE_SELECTORS>\n'
    script_string += f'       <Index name="loop1" resnums="{anchor_pos-loop_flank}-{anchor_pos+loop_len+loop_flank}"/>\n'
    script_string += f'       <Index name="loop12" resnums="{anchor_pos-loop_flank}-{anchor_pos+loop_len+loop_flank},{anchor_pos+repeatlen+loop_len-loop_flank}-{anchor_pos+repeatlen+loop_len+loop_len+loop_flank}"/>\n'
    #script_string += '        <Unsat name="unsats" check_acceptors="false" hbond_energy_cutoff="-0.2" scorefxn="ref_cst"/>\n'
    #script_string += '        <And name="unsats_loop" selectors="unsats,loop1"/>\n'
    script_string += '  </RESIDUE_SELECTORS>\n'
    #script_string += '  <SIMPLE_METRICS>\n'
    #script_string += '      <SelectedResidueCountMetric name="num_unsats_on_added" residue_selector="unsats_loop" />\n'
    #script_string += '  </SIMPLE_METRICS>\n'
    #script_string += '  <FILTERS>\n'
    #script_string += f'      <SimpleMetricFilter name="UNSATS_ON_APPEDED_RESIDUES" metric="num_unsats_on_added" cutoff="{MAX_UNSATS}" comparison_type="lt_or_eq"/>\n'
    #script_string += '  </FILTERS>\n'
    #script_string += '  <TASKOPERATIONS>\n'
    #script_string += '  </TASKOPERATIONS>\n'
    script_string += '  <FILTERS>\n'
    script_string += '       <PeptideInternalHbondsFilter\n'
    script_string += '           name="intraloop_hbond"\n'
    script_string += '           backbone_backbone="true"\n'
    script_string += '           backbone_sidechain="false"\n'
    script_string += '           sidechain_sidechain="false"\n'
    script_string += '           exclusion_distance="2"\n'
    script_string += '           hbond_energy_cutoff="-0.25"\n'
    script_string += '           residue_selector="loop1"\n'    
    script_string += f'          hbond_cutoff="{min_intraloop_hbond}"\n'
    script_string += '           confidence="1.0" />\n'
    script_string += '       <PeptideInternalHbondsFilter\n'
    script_string += '           name="interloop_hbond"\n'
    script_string += '           backbone_backbone="true"\n'
    script_string += '           backbone_sidechain="false"\n'
    script_string += '           sidechain_sidechain="false"\n'
    script_string += f'           exclusion_distance="{repeatlen-loop_len}"\n'
    script_string += '           hbond_energy_cutoff="-0.25"\n'
    script_string += '           residue_selector="loop12"\n'    
    script_string += f'          hbond_cutoff="{min_interloop_hbond}"\n'
    script_string += '           confidence="1.0" />\n'
    script_string += '       <PeptideInternalHbondsFilter\n'
    script_string += '           name="BETA_TURNS_AND_LONGER"\n'
    script_string += '           backbone_backbone="true"\n'
    script_string += '           backbone_sidechain="false"\n'
    script_string += '           sidechain_sidechain="false"\n'
    script_string += '           exclusion_distance="2"\n'
    script_string += '           hbond_energy_cutoff="-0.25"\n'
    script_string += '           residue_selector="loop1"\n'  
    script_string += '           hbond_cutoff="1"\n'
    script_string += '           confidence="0.0" />\n'
    script_string += '       <PeptideInternalHbondsFilter\n'
    script_string += '           name="GAMMA_TURNS_AND_LONGER"\n'
    script_string += '           backbone_backbone="true"\n'
    script_string += '           backbone_sidechain="false"\n'
    script_string += '           sidechain_sidechain="false"\n'
    script_string += '           exclusion_distance="1"\n'
    script_string += '           hbond_energy_cutoff="-0.25"\n'
    script_string += '           residue_selector="loop1"\n'   
    script_string += '           hbond_cutoff="1"\n'
    script_string += '           confidence="0.0" />\n'
    script_string += '       <OversaturatedHbondAcceptorFilter\n'
    script_string += '           name="KIC_OVERSAT_FILTER"\n'
    script_string += '           max_allowed_oversaturated="0"\n'
    script_string += '           hbond_energy_cutoff="-0.2"\n'
    script_string += '           consider_mainchain_only="true"\n'
    script_string += '           donor_selector="loop1"\n'
    script_string += '           acceptor_selector="loop1"\n'  
    script_string += '           confidence="1.0" />\n'
    script_string += f'      <CombinedValue name="num_gamma_turns" threshold="{MAX_GAMMA}">\n'
    script_string += '          <Add filter_name="GAMMA_TURNS_AND_LONGER" factor="1.0"/>\n'
    script_string += '          <Add filter_name="BETA_TURNS_AND_LONGER" factor="-1.0"/>\n'
    script_string += '      </CombinedValue>\n'
    script_string += '  </FILTERS>\n'
    script_string += '  <MOVERS>\n'
    # format parsed protocol for genKIC
    script_string += '      <ParsedProtocol name="preselection_pp">\n'
    script_string += '          <Add filter="intraloop_hbond"/>\n'
    script_string += '          <Add filter="interloop_hbond"/>\n'
    script_string += '          <Add filter="KIC_OVERSAT_FILTER"/>\n'
    script_string += '          <Add filter="num_gamma_turns"/>\n'
    #script_string += '          <Add filter="UNSATS_ON_APPEDED_RESIDUES"/>\n'
    script_string += '      </ParsedProtocol>\n'
    # close movers, protocols, and rosetta scripts
    script_string += ' </MOVERS>'
    #script_string += f' </MOVERS>\n<PROTOCOLS>\n</PROTOCOLS>\n</ROSETTASCRIPTS>'

    return script_string 


class do_nothing_mover(pyrosetta.rosetta.protocols.moves.Mover):
    '''
        a dummy mover
    '''
    def __init__(self):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)

    def __str__(self):
        return 'I do nothing!'

    def get_name(self):
        return self.__class__.__name__

    def apply(self, pose):
        pass


def str_to_int_list(s):
    out = []
    for i in s.split(','):
        if '-' in i:
            start, end = i.split('-')
            for j in range(int(start), int(end)+1):
                out.append(j)
        else:
            out.append(int(i))
    return out

def read_torsion_file(torsion_file):
    phipsi = []
    if torsion_file == '':
        return phipsi
    elif 'angle' in torsion_file:
        readlist = [torsion_file]
    else:
        readlist = open(torsion_file,'r').readlines()
    for m in readlist:
        pp = []
        c = 0
        lines = open(m.strip(),'r').readlines()
        for line in lines:
            angles = [float(x) for x in line.split()[1:3]]
            if (c == 0 or c == len(lines)-1) and 0 in angles:  # hacky check to skip the terminal residues
                continue
            pp.append(angles)
            c += 1
        phipsi.append([pp, m.strip()])
    return phipsi




def build_loop(param_file, cut_combination, input_anchor_pos, before_frag_length_list, after_frag_length_list, phipsi_list, n_cap_phipsi_list, c_cap_phipsi_list, in_fname, base_fname, \
                  out_dir, out_put_fname, num_repeats=-1, neighbor_flank=3, perturb=0, genkic_trials=100, aacomp_cap_pro_file="-1", \
                  ban_helix=-1, loop_direction=-1, bturn_side='ncap', loop_helix_contact_cutoff=8, loop_heix_contact_skip_resnum=5, loop_heix_contact_max_zero_contacts=-1, \
                  loop_pca_subspan_resnum=4, loop_pca_min_subspan_proj=-1, min_intraloop_hbond_num=0, min_intraloop_bbbb_hbond_num=0, min_interloop_hbond_num=0, min_interloop_bbbb_hbond_num=0, min_potential_interloop_bbbb_hbond_num=0,  \
                  bidentate_resn='N', min_num_bidentate=0, min_num_pseudo_bidentate=0, pseudo_bidentate_anchor_sstype='all', delta_HA=4, delta_theta=90, disable_combine_pseudo_bidentate=False, disable_relax=False, \
                  bbrmsd_cutoff=0.5, max_relax_rmsd_allowed=3.0, motif_dist_cutoff=10, worst_score_allowed=-1, design=0, layer_design=False, aacomp_design_file="-1", \
                  aacomp_design_pro_file="-1", aacomp_design_Ebin_file="-1", aacomp_design_Gbin_file="-1", checkpoint_frequency=300, sheetlike_turns=False, min_loop_motif=0, disable_motif_packing=False, total_score_cutoff=0, info_only=False \
                  ):

    '''
    neighbor_flank      number of residues flanking the loop to be allowed with chi movement during relax

    '''

    do_nothing = do_nothing_mover()


    # centroid sf w/ torsion, hbonds terms for genkic
    sf_cen = pyrosetta.get_score_function()
    sf_cen = sf_cen.clone_as_base_class()  # this is all what asymmetrize_scorefunction() does...
    for term in sf_cen.get_nonzero_weighted_scoretypes():
        sf_cen.set_weight(term, 0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.0)  # use fa_rep instead of vdw (vdw yields error)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.pair, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.env, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.cbeta, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.rama_prepro, 0.45)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.omega, 0.4)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.chainbreak, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 1.0)
    sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 1.0)

    sf_cen_highhb = sf_cen.clone()
    sf_cen_highhb.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 3.0)
    sf_cen_highhb.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 5.0)
    sf_cen_highhb.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 3.0)
    sf_cen_highhb.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 3.0)    

    sf_sym = pyrosetta.get_score_function()
    sf_sym_cst = sf_sym.clone()
    sf_sym_cst.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    sf_sym_cst.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    sf_sym_cst.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    sf_sym_cst.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)    

    sf_sym_cst_design = sf_sym_cst.clone()
    sf_sym_cst_design.set_weight(pyrosetta.rosetta.core.scoring.aa_composition, 1.0)
    #sf_sym_cst_design.set_weight(pyrosetta.rosetta.core.scoring.buried_unsatisfied_penalty, 1.0) # this leads to seg faults w/ repeat_symmetry...
    sf_sym_cst_design.set_weight(pyrosetta.rosetta.core.scoring.approximate_buried_unsat_penalty, 5.0)


    sf_sym_pack = sf_sym.clone()
    sf_sym_pack.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.4) #0.55
    sf_sym_pack.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 0.5) #0.7

    sf_high_hbond_sym = sf_sym.clone()
    sf_high_hbond_sym.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.4) #0.55
    sf_high_hbond_sym.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 0.5) #0.7
    sf_high_hbond_sym.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 3.0) #1.0  # Sidechain-backbone hydrogen bond energy

    sf = sf_sym.clone()
    #sf = pyrosetta.rosetta.core.scoring.symmetry.asymmetrize_scorefunction(sf)  # only works in older pyrosetta
    sf = sf.clone_as_base_class()  # this is all what asymmetrize_scorefunction() does...

    sf_high_hbond = sf.clone()
    sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.4) #0.55
    sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 0.5) #0.7
    #sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1.5)  # Backbone-backbone hbonds close in primary sequence
    #sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 2)   # Backbone-backbone hbonds distant in primary sequence
    sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 3.0) #1.0  # Sidechain-backbone hydrogen bond energy
    #sf_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 3.0) #1.0   #  Sidechain-sidechain hydrogen bond energy

    sf_cap = sf.clone()
    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2) # soft rep to encourage PRO packing for capping residues
    if aacomp_cap_pro_file != "-1":
        sf_cap.set_weight(pyrosetta.rosetta.core.scoring.aa_composition, 1) # turn on aacomp

    sf_cst = sf.clone()
    sf_cst.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    sf_cst.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    sf_cst.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    sf_cst.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)


    sf_2015_soft = pyrosetta.create_score_function('ref2015_soft')

    #sf_2015_soft_sym = sf_2015_soft.clone()   # only works in older pyrosetta
    #sf_2015_soft_sym = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(sf_2015_soft_sym)  # only works in older pyrosetta
    sf_2015_soft_sym = pyrosetta.rosetta.core.scoring.symmetry.SymmetricScoreFunction()  # this is all what symmetrize_scorefunction() does...
    sf_2015_soft_sym.assign(sf_2015_soft)  # this is all what symmetrize_scorefunction() does...

    sf_2015_soft_sym_high_hbond = sf_2015_soft_sym.clone()
    sf_2015_soft_sym_high_hbond.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 3.0) #1.0  # Sidechain-backbone hydrogen bond energy    


    start_pose = pyrosetta.pose_from_file(in_fname)
    scaffold = start_pose.clone()
    scaffold_score = sf(scaffold)


    # parser param file if it's specified
    comb_list, prob_list = [], []
    param_file_comb_term_list = []
    if param_file != '-1':
        param_in = open(param_file, 'r')
        lines = param_in.readlines()
        param_file_comb_term_list = lines[0].split(',')
        count_list = []
        for line in lines[1:]:
            items = line.strip().split(',')
            comb_list.append(items[:-1])
            count_list.append(int(items[-1]))
        total_count = sum(count_list)
        prob_list = [x*1.0/total_count for x in count_list]
        param_in.close()
        if _DEBUG:
            print('DEBUG:  Reading from the following paramter combinations ...')
            print('DEBUG:  ', param_file_comb_term_list[:-1]+['probability'])
            for comb_ind in range(len(comb_list)):
                print('DEBUG:    {}  {}'.format(comb_list[comb_ind], prob_list[comb_ind]))


    posepool = []
    info_pool = []
    comb_term_list = ['cut_start','cut_end','ccap','ncap','frag','before_frag_numres','after_frag_numres']

    #for mdl in range(genkic_trials):
    mdl = -1
    while mdl < genkic_trials:

        mdl += 1

        if mdl == 1:
            # read in all the pdb (if any) previously written into current directory (this can happen when using backfill)
            infofile = glob.glob(out_dir+'/'+base_fname+'*info*')
            if len(infofile) > 0:
                #print('Reading infofile: ', infofile)
                info_fin = open(infofile[0].strip(),'r')
                lines = info_fin.readlines()
                for line in lines:
                    info_line = line.strip()
                    items = info_line.split()
                    if info_only:
                        info_pool.append(info_line+'\n')
                    else:
                        pdb = glob.glob(out_dir+'/'+items[0]+'*.pdb')
                        if len(pdb) > 0:
                            pdb = pdb[0].strip()
                            #print('Loading pdb: ', pdb)
                            pose = pyrosetta.pose_from_file(pdb)
                            pose_score = sf(pose)
                            current_out_put_fname = pdb
                            new_pose_entry = [pose, pose_score, current_out_put_fname, info_line+'\n']
                            posepool.append(new_pose_entry)

                info_fin.close()
                # update mdl based on the number in infofile
                try:
                    mdl = int(infofile[0].split('.tmp.')[-1])
                    #if _DEBUG:
                    #    print('DEBUG:  Found existing info file, updated mdl to {}'.format(mdl))
                    print('Found existing info file, updated mdl to {}'.format(mdl))
                except:
                    #print('Warning: failed to extract and update mdl from info file: {}, restart sampling from mdl = 1...'.format(infofile[0]))
                    print('Warning: failed to extract and update mdl from info file: {}, assuming sampling is done, stop now ...'.format(infofile[0]))
                    return None
                    

        elif mdl > 1 and mdl%checkpoint_frequency == 0: # just in case the program gets killed before finishing, writing check points for every 300 trials
            #os.system('rm -f '+out_dir+'/'+base_fname+'_info.dat.tmp.{}'.format(mdl-checkpoint_frequency))
            os.system('rm -f '+out_dir+'/'+base_fname+'_info.dat.tmp.*')
            os.system('rm -f '+out_dir+'/'+base_fname+'_combination_stats.dat.tmp.*')
            os.system('rm -f '+out_dir+'/'+'*tmp{}.pdb'.format(mdl-checkpoint_frequency))
            fout = open(out_dir+'/'+base_fname+'_info.dat.tmp.{}'.format(mdl),'w')
            fout.write('name\tcut_start\tcut_end\tccap\tncap\tfrag\tbefore_frag_numres\tafter_frag_numres\ttrial\tloop_start\tloop_end\ttotal_score\tloopene_perres\tloop_direction\n')

            comb_pool = {}
            header_line = 'name\tcut_start\tcut_end\tccap\tncap\tfrag\tbefore_frag_numres\tafter_frag_numres\ttrial\tloop_start\tloop_end\ttotal_score\tloopene_perres\tloop_direction\n'
            term_ind_dict = {term:header_line.split().index(term) for term in comb_term_list}
            if info_only:
                for info_line in info_pool:
                    fout.write(info_line)
            else:
                for pose_ind in range(len(posepool)):
                    # if pose's score is higher than % of the best (assuming every score is negative), skip output, as the score indicate the pose is bad
                    ref_score = max([posepool[0][1],scaffold_score])
                    if worst_score_allowed != -1:
                        current_score_cutoff = worst_score_allowed * ref_score if ref_score < 0 else ref_score / worst_score_allowed
                    else:
                        current_score_cutoff = total_score_cutoff
                    
                    # comment out the following to output all
                    if posepool[pose_ind][1] > total_score_cutoff or posepool[pose_ind][1] > current_score_cutoff:
                        continue
                    
                    #if _DEBUG:
                    #    print('DEBUG: label info: ', posepool[pose_ind][4])
                    #    print('DEBUG: pdb_info: ', posepool[pose_ind][0].pdb_info())
                    #    print('DEBUG: pose sequence: ', posepool[pose_ind][0].sequence())

                    posepool[pose_ind][0].pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(posepool[pose_ind][0].total_residue()))
                    # for poses that have labels, add the label (poses read from previous run don't have label_list)
                    if len(posepool[pose_ind]) > 4:
                        for l in posepool[pose_ind][4]:
                            posepool[pose_ind][0].pdb_info().add_reslabel(l[0], l[1])
                    posepool[pose_ind][0].dump_scored_pdb(posepool[pose_ind][2].replace('.pdb', '_tmp{}.pdb'.format(mdl)), sf)
                    fout.write(posepool[pose_ind][3])

                    items = posepool[pose_ind][3].split()
                    comb = tuple([items[term_ind_dict[x]] for x in comb_term_list])
                    if comb not in comb_pool:
                        comb_pool[comb] = 0
                    comb_pool[comb] += 1


                    score = [posepool[pose_ind][1]]
                    score += [sf.score_by_scoretype(posepool[pose_ind][0],st) for st in sf.get_nonzero_weighted_scoretypes()]
                    #print('Pose {}: '.format(pose_ind+1)+' '.join(['%.3f' % x for x in score]))
            fout.close()

            # write combination stats
            comb_pool_keys = sorted(list(comb_pool.keys()), key=lambda x:comb_pool[x], reverse=True)
            fout = open(out_dir+'/'+base_fname+'_combination_stats.dat.tmp.{}'.format(mdl),'w')
            fout.write('{},count\n'.format(','.join(comb_term_list)))
            for comb in comb_pool_keys:
                fout.write('{},{}\n'.format(','.join(comb), comb_pool[comb]))
            fout.close()

            if _DEBUG:
                if info_only:
                    print('DEBUG: Genkic trial: {}, Current info poolsize: {}'.format(mdl+1, len(info_pool)))
                else:
                    #print('DEBUG: Genkic trial: {}, Current poolsize: {}, Best score: {}'.format(mdl+1, len(posepool), posepool[0][1] if len(posepool) > 0 else 'NA'))
                    print('DEBUG: Genkic trial: {}, Current poolsize: {}, Scores: {}'.format(mdl+1, len(posepool), ' '.join(['%.2f' % posepool[x][1] for x in range(len(posepool))])))

        #print('Genkic trial: {}, Current poolsize: {}, Best score: {}'.format(mdl+1, len(posepool), posepool[0][1] if len(posepool) > 0 else 'NA'))
        #print('\n\n\n\nGenkic trial: {}, Current poolsize: {}, Best score: {}'.format(mdl+1, len(posepool), posepool[0][1] if len(posepool) > 0 else 'NA'))


        #if _DEBUG:
        #    print(f'DEBUG:  Reading param files for trial {mdl+1} ...')

        # TODO
        # read the param files once at the beginner instead of repeating this in the loop!!!

        # sample from param_file instead, if the file is specified
        cut_start, cut_end = -1, -1
        phipsi, phipsi_name = [], 'None'
        n_cap_phipsi, n_cap_phipsi_name = [], 'None'
        c_cap_phipsi, c_cap_phipsi_name = [], 'None'

        if param_file != '-1':
            this_comb_ind = np.random.choice(list(range(len(comb_list))), p=prob_list)
            this_comb = comb_list[this_comb_ind]
            # allow overwriting cut_start/end from command line
            if len(cut_combination) > 0:
                cut_pairs = random.sample(cut_combination, 1)[0]
                cut_start = cut_pairs[0]
                cut_end = cut_pairs[1]
            else:
                cut_start = int(this_comb[param_file_comb_term_list.index('cut_start')])
                cut_end = int(this_comb[param_file_comb_term_list.index('cut_end')])
            c_cap_phipsi_name = this_comb[param_file_comb_term_list.index('ccap')]
            c_cap_phipsi = read_torsion_file(c_cap_phipsi_name)[0][0]
            n_cap_phipsi_name = this_comb[param_file_comb_term_list.index('ncap')]
            n_cap_phipsi = read_torsion_file(n_cap_phipsi_name)[0][0]            
            phipsi_name = this_comb[param_file_comb_term_list.index('frag')]
            phipsi = read_torsion_file(phipsi_name)[0][0]          
            before_frag_length = int(this_comb[param_file_comb_term_list.index('before_frag_numres')])
            after_frag_length = int(this_comb[param_file_comb_term_list.index('after_frag_numres')])

        else:
            # randomly select motifs and number of residues from their lists
            before_frag_length = random.sample(before_frag_length_list, 1)[0] if len(before_frag_length_list) > 0 else 0
            after_frag_length = random.sample(after_frag_length_list, 1)[0] if len(after_frag_length_list) > 0 else 0
            if len(phipsi_list) > 0:
                phipsi, phipsi_name = random.sample(phipsi_list, 1)[0]

            # CAUTION: ncap at cterm-side of loop, ccap at nterm-side of loop!
            if len(n_cap_phipsi_list) > 0:
                n_cap_phipsi, n_cap_phipsi_name = random.sample(n_cap_phipsi_list, 1)[0]

            if len(c_cap_phipsi_list) > 0:
                c_cap_phipsi, c_cap_phipsi_name = random.sample(c_cap_phipsi_list, 1)[0]

            #phipsi_name = phipsi_name.split('/')[-1]
            #n_cap_phipsi_name = n_cap_phipsi_name.split('/')[-1]
            #c_cap_phipsi_name = c_cap_phipsi_name.split('/')[-1]

            #print(before_frag_length, after_frag_length, phipsi_name, n_cap_phipsi_name, c_cap_phipsi_name)
            #print(n_cap_phipsi)
            #print(c_cap_phipsi)
            #print(phipsi)


            #print('Genkic trial: {}, Current poolsize: {}, Best score: {}'.format(mdl+1, len(posepool), posepool[0][1] if len(posepool) > 0 else 'NA'), \
            #    cut_start, cut_end, input_anchor_pos, before_frag_length, after_frag_length, phipsi_name, n_cap_phipsi_name, c_cap_phipsi_name)
            #print(cut_start, cut_end, input_anchor_pos, before_frag_length, after_frag_length, phipsi_name, n_cap_phipsi_name, c_cap_phipsi_name, '\n')

            #if _DEBUG:
            #    input_score = sf(p)
            #    print('DEBUG: input score: ', input_score)

            if len(cut_combination) > 0:
                cut_pairs = random.sample(cut_combination, 1)[0]
                cut_start = cut_pairs[0]
                cut_end = cut_pairs[1]


        neighbor_flank_n = len(c_cap_phipsi) + neighbor_flank
        neighbor_flank_c = len(n_cap_phipsi) + neighbor_flank


        #if _DEBUG:
        #    print(f'DEBUG:  Setting up pose for trial {mdl+1} ...')


        p = start_pose.clone()

        #if _DEBUG:
        #    print('num_repeats: ', num_repeats)

        if cut_start != -1 and cut_end != -1:
            if cut_start + 1 >= cut_end:
                print('invalid cut site creation (start: {} end: {})! skip this trial...'.format(cut_start, cut_end))
                continue

            if num_repeats == -1:
                num_repeats = find_num_repeats_by_rmsd(scaffold)
            scaffold_repeatlen = int(p.size()/num_repeats)

            # TODO: double check if this mutation code is working
            # mutate cut site and flanking residues to ala
            base_cut_start = cut_start%scaffold_repeatlen if cut_start%scaffold_repeatlen != 0 else scaffold_repeatlen
            base_cut_end = cut_end%scaffold_repeatlen if cut_end%scaffold_repeatlen != 0 else scaffold_repeatlen
            for r in range(num_repeats):
                for mt_resid in range(base_cut_start - neighbor_flank, base_cut_end + neighbor_flank + 1): 
                    this_mt_resid = scaffold_repeatlen*r+mt_resid
                    #if _DEBUG:
                    #    print('DEBUG: mutation: {} {}'.format(this_mt_resid, p.residue(this_mt_resid).name3().strip()))
                    # mutate only non-gly and non-pro
                    if p.residue(this_mt_resid).name3().strip() not in ['PRO','GLY']: 
                        mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(this_mt_resid, 'ALA')
                        mt.apply(p)


            # record cut site residue phi/psi, as one dihedral will be lost after cutting
            cut_start_phi, cut_start_psi = p.phi(cut_start-1), p.psi(cut_start-1)
            cut_end_phi, cut_end_psi = p.phi(cut_end+1), p.psi(cut_end+1)
            #if _DEBUG:
            #    print('cut_start, cut_start_phi, cut_start_psi:', cut_start, cut_start_phi, cut_start_psi)
            #    print('cut_end, cut_end_phi, cut_end_psi:', cut_end, cut_end_phi, cut_end_psi)
            # creat cut site
            p.delete_residue_range_slow(cut_start, cut_end)
            p.update_residue_neighbors()

            # input_anchor_pos is overwritten when cut sites specified!!!
            input_anchor_pos = cut_start - 1

            #if _DEBUG:
            #    print('DEBUG: cut_start, cut_end: ', cut_start, cut_end)
            #    print('DEBUG: before_length, after_length: ', before_frag_length, after_frag_length)


        #if _DEBUG:
        #    print(f'DEBUG:  Generating helix caps for trial {mdl+1} ...')


        # add caps (do replacing in future?)
        # n cap (closer to the tail)
        if len(n_cap_phipsi) > 0:
            #ncap = generate_cap(n_cap_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=True)
            #ncap = generate_cap(n_cap_phipsi, sf_cap, aa='A', insert_Pro=False, insert_GLy=False)

            # new scheme: add the phi/psi of anchor residue for alignment
            # anchor_phipsi = [[p.phi(input_anchor_pos+1),p.psi(input_anchor_pos+1)]]
            anchor_phipsi = [[cut_start_phi, cut_start_psi]]
            ncap = generate_cap(n_cap_phipsi+anchor_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=False, aacomp_cap_pro_file=aacomp_cap_pro_file) 
            p = insert_n_cap(p, input_anchor_pos, ncap)
            #if _DEBUG:
            #    p.dump_pdb('post_ncap.pdb')

        # c cap
        if len(c_cap_phipsi) > 0:
            #ccap = generate_cap(c_cap_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=True)
            #ccap = generate_cap(c_cap_phipsi, sf_cap, aa='A', insert_Pro=False, insert_GLy=False)

            # new scheme: add the phi/psi of anchor residue for alignment
            #anchor_phipsi = [[p.phi(input_anchor_pos),p.psi(input_anchor_pos)]]
            anchor_phipsi = [[cut_end_phi, cut_end_psi]]
            ccap = generate_cap(anchor_phipsi+c_cap_phipsi, sf_cap, aa='A', insert_Pro=False, insert_GLy=True, aacomp_cap_pro_file="-1")
            p = insert_c_cap(p, input_anchor_pos, ccap)
            #if _DEBUG:
            #    p.dump_pdb('post_ccap.pdb')

        #p.dump_pdb('test_post_cap.pdb')
      
        #if _DEBUG:
        #    print(f'DEBUG:  Setting up GenKIC for trial {mdl+1} ...')

        ### To be tested! use fixed anchor points (not included in genkic, e.g. bridge res and capping res)
        ### in this case, the last residue of ccap and first residue of ncap are set as pivots (therefore a dummy residue required?)
        anchor_pos = input_anchor_pos + len(c_cap_phipsi)

        num_res_added = before_frag_length + len(phipsi) + after_frag_length
        loop_s, loop_e = anchor_pos, anchor_pos+num_res_added+1
        #if _DEBUG:
        #    print('DEBUG:   loop_s, loop_e: ', loop_s, loop_e)
        #    print('DEBUG:   scaffold_repeatlen: ', scaffold_repeatlen)

        # make a cut in the fold tree
        # Caution: I hard coded the cut site 
        
        #if _DEBUG:
        #    print(f'DEBUG:  Setting up GenKIC:fold_tree for trial {mdl+1} ...')

        ft = pyrosetta.FoldTree()
        ft.add_edge(1, anchor_pos - 3, -1)
        ft.add_edge(anchor_pos - 3, anchor_pos, -1)
        ft.add_edge(anchor_pos - 3, anchor_pos + 3, 1)
        ft.add_edge(anchor_pos + 3, anchor_pos + 1, -1)
        ft.add_edge(anchor_pos + 3, p.size(), -1)
        if _DEBUG:        
            #print(ft)
            ft.check_fold_tree()
        p.fold_tree(ft)
        

        #print('Starting length of pose: {}'.format(p.size()))
        #print('Inserting {} residues at {}'.format(num_res_added, anchor_pos))


        #if _DEBUG:
        #    print(f'DEBUG:  Setting up GenKIC:add_loop and close_bond for trial {mdl+1} ...')

        extend_residue(p, num_res_added, anchor_pos)

        # res_indices is a tuple of positions for the connection site
        res_indices = (anchor_pos + num_res_added, anchor_pos + num_res_added + 1)

        declare_bond(p, res_indices)

        #p.dump_pdb('test_pre_kic.pdb')


        # make sure the pose neighbors are up-to-date
        p.update_residue_neighbors()

        # sf_cen
        #st = fragment_gen_kic_mover(p, before_frag_length, after_frag_length, phipsi, n_cap_phipsi, c_cap_phipsi, anchor_pos,
        #                   res_indices[0], res_indices[1], sf_cen, sf_cen_highhb, perturb=perturb)

        # sf_high_hbond
        #st = fragment_gen_kic_mover(p, before_frag_length, after_frag_length, phipsi, n_cap_phipsi, c_cap_phipsi, anchor_pos,
        #                   res_indices[0], res_indices[1], sf, sf_high_hbond, perturb=perturb)


        '''
        # this way of setting up preselection mover costs ~6.5s (whereas setting up from python costs only ~0.0003s!!!)
        if _DEBUG:
            print(f'DEBUG:  Setting up GenKIC:preselection_mover:filters for trial {mdl+1} ...')
            time_s = time.time()
    
        # filter intra hbond thru preselection filter in genkic
        script_string = get_xml_string(anchor_pos, num_res_added, 
                                        scaffold_repeatlen, 
                                        loop_flank=1, 
                                        min_intraloop_hbond=min_intraloop_hbond_num, 
                                        min_interloop_hbond=min_interloop_hbond_num, 
                                        MAX_UNSATS=10000, MAX_GAMMA=10) #unsats tricky to judge at this stage
        xml_objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string( script_string )
        preselection_pp = xml_objs.get_mover('preselection_pp')
        if _DEBUG:
            time_e = time.time()
            print(f'DEBUG:  Time: {time_e-time_s}')
        '''


        #if _DEBUG:
        #    print(f'DEBUG:  Setting up GenKIC:preselection_mover:repeat_propagate_mover for trial {mdl+1} ...')
        rp = repeat_propagate_mover(start_pose, _num_repeats=num_repeats)
        #preselection_mover = pyrosetta.rosetta.protocols.moves.SequenceMover(rp, preselection_pp)  # for setting up preselection by xml_obj



        #if _DEBUG:
        #    print(f'DEBUG:  Setting up GenKIC:ParsedProtocol for trial {mdl+1} ...')
        #    time_s = time.time()

        # making movers and filters in python
        genkic_loop_hbfilter_flank = 1
        genkic_loop1_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(f'{anchor_pos-genkic_loop_hbfilter_flank}-{anchor_pos+num_res_added+genkic_loop_hbfilter_flank}')
        genkic_loop12_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(f'{anchor_pos-genkic_loop_hbfilter_flank}-{anchor_pos+num_res_added+genkic_loop_hbfilter_flank},'+\
                                                                                                f'{anchor_pos+scaffold_repeatlen+num_res_added-genkic_loop_hbfilter_flank}'+\
                                                                                                f'-{anchor_pos+scaffold_repeatlen+num_res_added+num_res_added+genkic_loop_hbfilter_flank}')

        # NOTE: default bb-bb: true, bb-sc: false, sc-sc: false
        intraloop_hbfilter = pyrosetta.rosetta.protocols.cyclic_peptide.PeptideInternalHbondsFilter() 
        intraloop_hbfilter.set_exclusion_distance(2)
        intraloop_hbfilter.set_hbond_energy_cutoff(-0.25)
        intraloop_hbfilter.set_residue_selector(genkic_loop1_selector)
        intraloop_hbfilter.set_hbond_cutoff(min_intraloop_hbond_num)

        interloop_hbfilter = pyrosetta.rosetta.protocols.cyclic_peptide.PeptideInternalHbondsFilter() 
        interloop_hbfilter.set_exclusion_distance(scaffold_repeatlen-num_res_added)
        interloop_hbfilter.set_hbond_energy_cutoff(-0.25)
        interloop_hbfilter.set_residue_selector(genkic_loop12_selector)
        interloop_hbfilter.set_hbond_cutoff(min_interloop_hbond_num)

        bturn_n_longer_hbfilter = pyrosetta.rosetta.protocols.cyclic_peptide.PeptideInternalHbondsFilter() # beta turn
        bturn_n_longer_hbfilter.set_exclusion_distance(2)
        bturn_n_longer_hbfilter.set_hbond_energy_cutoff(-0.25)
        bturn_n_longer_hbfilter.set_residue_selector(genkic_loop1_selector)
        bturn_n_longer_hbfilter.set_hbond_cutoff(1)
        gturn_n_longer_hbfilter = pyrosetta.rosetta.protocols.cyclic_peptide.PeptideInternalHbondsFilter()  # gamma turn
        gturn_n_longer_hbfilter.set_exclusion_distance(1)
        gturn_n_longer_hbfilter.set_hbond_energy_cutoff(-0.25)
        gturn_n_longer_hbfilter.set_residue_selector(genkic_loop1_selector)
        gturn_n_longer_hbfilter.set_hbond_cutoff(1)
        combined_hbfilter = pyrosetta.rosetta.protocols.filters.CombinedFilter()
        combined_hbfilter.add_filter(gturn_n_longer_hbfilter, 1.0)
        combined_hbfilter.add_filter(bturn_n_longer_hbfilter, -1.0)

        os_hbfilter = pyrosetta.rosetta.protocols.cyclic_peptide.OversaturatedHbondAcceptorFilter()
        os_hbfilter.set_max_allowed_oversaturated(0)
        os_hbfilter.set_hbond_energy_cutoff(-0.2)
        os_hbfilter.set_consider_mainchain_only(True)
        os_hbfilter.set_donor_selector(genkic_loop1_selector)
        os_hbfilter.set_acceptor_selector(genkic_loop1_selector)

        preselection_mover = pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol()
        preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(do_nothing, 'do_nothing', intraloop_hbfilter) )
        preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(do_nothing, 'do_nothing', combined_hbfilter) )
        preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(do_nothing, 'do_nothing', os_hbfilter) )
        preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(rp, 'repeat_propagate', interloop_hbfilter) )
        #true_filter = pyrosetta.rosetta.protocols.filters.TrueFilter()
        #preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(rp, 'repeat_propagate', true_filter) )
        #preselection_mover.add_step( pyrosetta.rosetta.protocols.rosetta_scripts.ParsedProtocol.ParsedProtocolStep(do_nothing, 'do_nothing', interloop_hbfilter) )


        #if _DEBUG:
        #    time_e = time.time()
        #    print(f'DEBUG:  Time: {time_e-time_s}')
        


        #if _DEBUG:
        #    print(f'DEBUG:  Now starting genKIC for trial {mdl+1} ...')


        st = fragment_gen_kic_mover(p, before_frag_length, after_frag_length, phipsi, n_cap_phipsi, c_cap_phipsi, anchor_pos,
                           res_indices[0], res_indices[1], sf, sf_high_hbond, perturb=perturb, preselection_mover=preselection_mover)        




        # posepool update
        if st == pyrosetta.rosetta.protocols.moves.MoverStatus.MS_SUCCESS:


            if _DEBUG:
                #print('DEBUG:   Successful closure for trial ', mdl+1)
                print('DEBUG:   Successful closure for trial {}: cut_start({}) cut_end({}), before_length({}), after_length({}), ccap({}), ncap({}), turn({})'.format(mdl+1, \
                        cut_start, cut_end, before_frag_length, after_frag_length, c_cap_phipsi_name.split('/')[-1], n_cap_phipsi_name.split('/')[-1], phipsi_name.split('/')[-1]))

            pose = p.clone()
            pose_score = sf(pose)
            #if _DEBUG:
            #    print('DEBUG: post-genKIC score: ', pose_score)

            #if pose_score < total_score_cutoff:
            #    pose.dump_pdb('tmp_{}.pdb'.format(mdl+1))


            # helix element filtering
            if ban_helix > 0:
                dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(p)
                dssp_obj.dssp_reduced()
                dssp = dssp_obj.get_dssp_secstruct()
                if 'H'*ban_helix in dssp[anchor_pos:anchor_pos+num_res_added]:
                    if _DEBUG:
                        print('DEBUG:     HELIX_BAN: helix in loop {} longer than the cutoff {}, skipping ... '.format(dssp[anchor_pos:anchor_pos+num_res_added], ban_helix))
                    continue


            '''
            # now I already propagate the loop during genkic, this is nolonger required!
            # determine repeatlen
            if len(cut_combination) == 0 and param_file == '-1':
                #  propagate repeats by repeatlen
                repeatlen = find_repeatlen_by_rmsd(pose, max_rmsd=0.5, sliding_window=10, min_repeatlen=20) # hard coded param here
                pose_prop = poorman_repeat_propagate(pose, repeatlen, num_repeat=num_repeats, overhang=1)
            else:
                # propagate repeats by num_repeats
                pose_prop = propogate_loops_idealize_and_symmetrize(p, start_pose, num_repeats=num_repeats)
                repeatlen = int(pose_prop.size()/num_repeats)
            '''
            pose_prop = pose.clone()
            repeatlen = int(pose_prop.size()/num_repeats)

            if _DEBUG:
                print('DEBUG:     num_repeats: ', num_repeats)
                print('DEBUG:     repeatlen: ', repeatlen)



            # determine feature search/target/minimize list
            loops, loops_all = [], []
            loop_s_base = loop_s%scaffold_repeatlen if loop_s%scaffold_repeatlen != 0 else scaffold_repeatlen
            #loop_e_base = loop_e%scaffold_repeatlen if loop_e%scaffold_repeatlen != 0 else scaffold_repeatlen  # error when loop_e on 1st loop but > scaffold_repeatlen!
            loop_e_base = loop_s_base + (loop_e-loop_s)

            for k in range(num_repeats):
                tmp = []
                for i in range(loop_s_base, loop_e_base+1):
                    tmp.append(k*repeatlen+i)
                    loops_all.append(k*repeatlen+i)
                loops.append(tmp)

            loops_n_cap, loops_n_cap_all = [], []
            for k in range(num_repeats):
                tmp = []
                for i in range(loop_s_base - neighbor_flank_n, loop_e_base + neighbor_flank_c +1):
                    tmp.append(k*repeatlen+i)            
                    loops_n_cap_all.append(k*repeatlen+i)  
                loops_n_cap.append(tmp)          

            min_reslist = loops_n_cap_all

            #search_reslist = list(range(loop_s, loop_e))
            #search_reslist = list(range(loop_s - neighbor_flank_n, loop_e + neighbor_flank_c +1))
            #search_reslist = list(range(loop_s + repeatlen - neighbor_flank_n, loop_e + repeatlen + neighbor_flank_c +1))  #loop2
            #target_reslist = [x+repeatlen for x in search_reslist]   # loop3


            if _DEBUG:
                print('DEBUG:     loops: ', loops)
                print('DEBUG:     loops_n_cap: ', loops_n_cap)
            #    print('DEBUG:     search_reslist: ', search_reslist)
            #    print('DEBUG:     min_reslist: ', min_reslist)



            RepeatProteinRelax_apply(pose_prop, modify_symmetry_and_exit_=True, remove_symm_=False, bblist=loops_n_cap_all, chilist=loops_n_cap_all, num_repeats=num_repeats)
            #fr = _setup_fast_relax(pose_prop, sf_sym_cst, min_reslist, cartesian=False, include_neighbor=True)
            #fr.apply(pose_prop)

            # score the symmetric pose once
            sym_score = sf_sym(pose_prop)
            dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose_prop)
            dssp_obj.dssp_reduced()
            dssp = dssp_obj.get_dssp_secstruct()


            pose_prop_candidates = [SymPose(pose_prop, labels=[], used_sc_resids={})]


            ############################### filters before relax ###############################>>>>>


            # loop shape check: contacts with helix caps
            if loop_heix_contact_max_zero_contacts != -1:
                if _DEBUG:
                    print('DEBUG:     FILTER: loop-helix contacts')
                loop_contacts_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    pose_prop = sympose.pose()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()

                    search_reslist = loops[1]
                    target_reslist = [x for x in loops_n_cap[0]+loops_n_cap[1]+loops_n_cap[2] if x not in loops_all]      
                    if bturn_side == 'ncap':
                        subset_search_reslist = search_reslist[:(len(search_reslist)-loop_heix_contact_skip_resnum)]
                    elif bturn_side == 'ccap':
                        subset_search_reslist = search_reslist[loop_heix_contact_skip_resnum:]
                    else:
                        print(f'Error: unsupported bturn_side value {bturn_side}!')
                        sys.exit(1)
                    if _DEBUG:
                        print('DEBUG:         search_reslist', search_reslist)                    
                        print('DEBUG:         subset_search_reslist', subset_search_reslist)                    
                        print('DEBUG:         target_reslist', target_reslist) 
                    subset_contact_count_list = [] # [num_of_contacts_for_res1, num_of_contacts_for_res2 ...]
                    for resi in subset_search_reslist:
                        contact_count = 0
                        for resj in target_reslist:
                            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose_prop.residue(resi).xyz('CA'), pose_prop.residue(resj).xyz('CA'))
                            if dist <= loop_helix_contact_cutoff:
                                contact_count += 1
                        subset_contact_count_list.append(contact_count)        
                    num_zero_contacts = sum([1 if x == 0 else 0 for x in subset_contact_count_list])

                    if num_zero_contacts > loop_heix_contact_max_zero_contacts:
                        if _DEBUG:
                            print('DEBUG:       failed to pass loop-helix contact filter ({}, cutoff at {}), skipping ...'.format(num_zero_contacts, loop_heix_contact_max_zero_contacts))
                        continue

                    loop_contacts_pose_prop_candidates.append(SymPose(pose_prop, labels=label_list, used_sc_resids=used_sc_resids))

                pose_prop_candidates = loop_contacts_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after loop-helix contact check, skipping ...')
                    continue   


            # loop shape check: shape of loop tip (bturn) by PCA (loop bturn region should have certain span on top component from PCA)
            if loop_pca_min_subspan_proj != -1:
                if _DEBUG:
                    print('DEBUG:     FILTER: loop PCA span check')
                loop_pca_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    pose_prop = sympose.pose()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()

                    # compute PCA 
                    loop_ca_coords = np.array([list(pose_prop.residue(i).xyz('CA')) for i in loops[1]])
                    pca = PCA(n_components=3) # 3 components as its in 3D space??
                    pc = pca.fit(loop_ca_coords)
                    if bturn_side == 'ncap':
                        subspan_ca_coords_proj = pca.transform(loop_ca_coords[len(loop_ca_coords)-loop_pca_subspan_resnum:])[:,0] # project at top component
                    elif bturn_side == 'ccap':
                        subspan_ca_coords_proj = pca.transform(loop_ca_coords[:loop_pca_subspan_resnum])[:,0] # project at top component
                    else:
                        print(f'Error: unsupported bturn_side value {bturn_side}!')
                        sys.exit(1)
                    subspan_proj_span = float(max(subspan_ca_coords_proj) - min(subspan_ca_coords_proj))

                    #print('cos_angle ', cos_angle)
                    if subspan_proj_span < loop_pca_min_subspan_proj:
                        if _DEBUG:
                            print('DEBUG:       failed to pass loop PCA filter ({}, cutoff at {}), skipping ...'.format(subspan_proj_span, loop_pca_min_subspan_proj))
                        continue
                    loop_pca_pose_prop_candidates.append(SymPose(pose_prop, labels=label_list, used_sc_resids=used_sc_resids))

                    
                pose_prop_candidates = loop_pca_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after loop PCA check, skipping ...')
                    continue   



            # loop shape check: loop direction 
            if loop_direction != -1:
                if _DEBUG:
                    print('DEBUG:     FILTER: loop direction')
                loop_direction_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    pose_prop = sympose.pose()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()

                    cos_angle = compute_loop_direction_nonperfect_repeats(pose_prop, loop_idx=1, bturn_side=bturn_side)
                    #print('cos_angle ', cos_angle)
                    if cos_angle > loop_direction:
                        if _DEBUG:
                            print('DEBUG:       failed to pass loop direction filter ({}, cutoff at {}), skipping ...'.format(cos_angle, loop_direction))
                        continue
                    loop_direction_pose_prop_candidates.append(SymPose(pose_prop, labels=label_list, used_sc_resids=used_sc_resids))

                pose_prop_candidates = loop_direction_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after loop direction check, skipping ...')
                    continue   


            ''' 
            ## no need for this here since I'm now filtering this in genKIC
            # intra loop hbond filters
            if min_intraloop_hbond_num > 0 or min_intraloop_bbbb_hbond_num > 0:
                if _DEBUG:
                    print('DEBUG:     FILTER: intraloop_hbond')
                intraloop_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    pose = sympose.pose()
                    intraloop_hb_set = find_hbonds(pose, loops[1], loops[1], skip=[])  # 2nd loop
                    num_intraloop_bbbb_hb = len(intraloop_hb_set['bb-bb'])
                    num_intraloop_hb = len(intraloop_hb_set['bb-bb']) + len(intraloop_hb_set['bb-sc']) + len(intraloop_hb_set['sc-sc'])
                    if num_intraloop_bbbb_hb < min_intraloop_bbbb_hbond_num:
                        if _DEBUG:
                            print('DEBUG:       {} intraloop bb-bb hbonds, {} required, skipping ...'.format(num_intraloop_bbbb_hb, min_intraloop_bbbb_hbond_num))
                        continue
                    if num_intraloop_hb < min_intraloop_hbond_num:
                        if _DEBUG:
                            print('DEBUG:       {} intraloop total hbonds, {} required, skipping ...'.format(num_intraloop_hb, min_intraloop_hbond_num))
                        continue    
                    intraloop_pose_prop_candidates.append(SymPose(pose,[],{}))
                pose_prop_candidates = intraloop_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after intraloop hbond search, skipping ...')
                    continue 
            '''     


            '''
            ## no need since now I filter interloop bb-bb hb in genkic
            # search for potential interloop bb-bb hb
            if min_potential_interloop_bbbb_hbond_num > 0:
                if _DEBUG:
                    print('DEBUG:     FILTER: potential_interloop_bbbb_hbond_num')
                potential_interloop_bbbb_hb_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    pose_prop = sympose.pose()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()
                    # hard coded hbond metric
                    #potential_interloop_bbbb_hb = simple_hbond_finder(pose_prop, loop_1, loop_2, delta_HA=3.5, delta_theta=60., reslist1_atom_type=['bb'], reslist2_atom_type=['bb'], verbose=False)
                    potential_interloop_bbbb_hb = simple_hbond_finder(pose_prop, loops[1], loops[2], delta_HA=3.5, delta_theta=60., reslist1_atom_type=['bb'], reslist2_atom_type=['bb'], verbose=False) # 2nd-3rd loops
                    if len(potential_interloop_bbbb_hb) < min_potential_interloop_bbbb_hbond_num:
                        if _DEBUG:
                            print('DEBUG:       only {} potential bb-bb hbond found, {} required ...'.format(len(potential_interloop_bbbb_hb), min_potential_interloop_bbbb_hbond_num))
                        continue
                    if _DEBUG:
                        print('DEBUG:       found potential bb-bb hbonds: {}'.format(potential_interloop_bbbb_hb))

                    #### add hb dist constraints
                    add_hbond_atom_pair_cst_to_pose(pose_prop, potential_interloop_bbbb_hb, cst_func='HARMONIC 2.0 0.5')
                    #### add bb-bb hb residue labels
                    label_potential_interloop_bbhb_list = []
                    for hb in potential_interloop_bbbb_hb:
                        reduced_i, reduced_j = hb[0]%repeatlen, hb[2]%repeatlen
                        if reduced_i not in label_potential_interloop_bbhb_list:
                            label_potential_interloop_bbhb_list.append(reduced_i)
                        if reduced_j not in label_potential_interloop_bbhb_list:
                            label_potential_interloop_bbhb_list.append(reduced_j)
                    for i in label_potential_interloop_bbhb_list:
                        for r in range(num_repeats):
                            label_list.append( (r*repeatlen+i,'POTENTIAL_INTERLOOP_BB-BB_HBONDS') )

                    potential_interloop_bbbb_hb_pose_prop_candidates.append(SymPose(pose_prop, labels=label_list, used_sc_resids=used_sc_resids))
                pose_prop_candidates = potential_interloop_bbbb_hb_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    continue                
            '''


            # search for potential interloop sc-bb bidentate hbonds
            if min_num_pseudo_bidentate > 0:
                if _DEBUG:
                    print('DEBUG:     FILTER: pseudo_bidentate')
                potential_bidentate_pose_candidates = []  # list of SymPoses
                pseudo_bidentate_anchor_sstype = pseudo_bidentate_anchor_sstype.strip()
                if pseudo_bidentate_anchor_sstype == 'all':
                    bidentate_search_reslist = loops_n_cap[1]
                elif pseudo_bidentate_anchor_sstype == 'helix':

                    #pose_prop.dump_pdb('test.pdb')

                    bidentate_search_reslist = []
                    # select by short helices 
                    tmp_helices = find_helices_by_dssp(pose_prop, min_helix_length=1, search_range=[loops_n_cap[1][0], loops_n_cap[1][-1]])
                    for tmp_helix in tmp_helices:
                        # include the flanking residue to include capping residues (as sometimes capping residues can contribute bidentates)
                        bidentate_search_reslist += list(range(tmp_helix[0]-1, tmp_helix[-1]+2))
                    # select by dssp
                    #for tmp_resi in loops_n_cap[1]:
                    #    if dssp[tmp_resi-1] == 'H':
                    #        bidentate_search_reslist.append(tmp_resi)
                    bidentate_search_reslist = sorted(list(set(bidentate_search_reslist)))
                else:
                    print(f'Error: unsupported pseudo_bidentate_anchor_sstype: {pseudo_bidentate_anchor_sstype}')
                    sys.exit(1)
                if _DEBUG:
                    print('DEBUG:       pseudo_bidentate search residues: ', bidentate_search_reslist)
                    

                for sympose in pose_prop_candidates:
                    pose_prop = sympose.pose()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()

                    this_potential_bidentate_pose_candidates = []  #  list of [pose,[resid],[resn],label_list,hbs]
                    #for resid in loops_n_cap[1]: # second loop and its capping residues plus the neighbors on primary sequence
                    for resid in bidentate_search_reslist:
                        this_pose_prop = pose_prop.clone()

                        pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_high_hbond_sym)   
                        #pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_2015_soft_sym_high_hbond)   
                        pack_rotamers.task_factory(_config_my_task_factory([x*repeatlen+(resid%repeatlen) for x in range(num_repeats)], allowed_aa='A'+bidentate_resn))   
                        #pack_rotamers.task_factory(_config_my_task_factory([x*repeatlen+(resid%repeatlen) for x in range(num_repeats)], allowed_aa=bidentate_resn))   
                        pack_rotamers.apply(this_pose_prop)

                        # for labels in pdb_info
                        this_label_list = [x for x in label_list] # deep copy from parent label list

                        #### pseudo bidentate search (with loose hbond definition) ####
                        # target list is the third loop and its capping residues plus the neighbors on primary sequence
                        raw_potential_bidentate_hbonds = find_potential_bidentate_hbond(this_pose_prop, [resid], loops_n_cap[2], delta_HA=delta_HA, delta_theta=delta_theta, reslist1_atom_type=['sc'], reslist2_atom_type=['bb'])

                        # only allow fully satisfied bidentates
                        potential_bidentate_hbonds = {x:raw_potential_bidentate_hbonds[x] for x in raw_potential_bidentate_hbonds 
                                                        if check_fully_satisfied_bidenates(this_pose_prop, raw_potential_bidentate_hbonds, x)}

                        if len(potential_bidentate_hbonds) < min_num_pseudo_bidentate:
                            #if _DEBUG:
                            #    print('DEBUG:     at residue {}, only {} potential bidentate bb-sc hbond found, {} required ...'.format(resid, len(potential_bidentate_hbonds), min_num_pseudo_bidentate))
                            continue

                        ### CAUTION!! here I only allow HIS to pack on helix or capping residues
                        if this_pose_prop.sequence(resid, resid) == 'H' and resid in loops[1][1:-1]: 
                            continue

                        if _DEBUG:
                            print('DEBUG:       at residue {}, found potential bidentate hbonds: {}'.format(resid, potential_bidentate_hbonds))

                        potential_bidentate_hbonds = propagate_bidentate_hbonds(potential_bidentate_hbonds, num_repeats, repeatlen)

                        #### relax with distance constraints on bidentate atoms ####
                        add_bidentate_atom_pair_cst_to_pose(this_pose_prop, potential_bidentate_hbonds, cst_func='HARMONIC 2.0 0.5')
                        #this_pose_prop.dump_pdb(out_put_fname.format(dir=out_dir, base=base_fname, mdl_no=mdl+1, LS=loop_s, LE=loop_e).replace('.pdb','.debug_pre_fr.pdb'))

                        # add labels to residue in pdb
                        label_sc_list, label_bb_list = [], []
                        for i in sorted(potential_bidentate_hbonds.keys()):
                            reduced_i = i%repeatlen
                            if reduced_i not in label_sc_list:
                                label_sc_list.append(reduced_i)
                            for hb in potential_bidentate_hbonds[i]:
                                for j in [0,2]:
                                    if hb[j] != i:
                                        reduced_j = hb[j]%repeatlen
                                        if reduced_j not in label_bb_list:
                                            label_bb_list.append(reduced_j)
                        for sc in label_sc_list:
                            for r in range(num_repeats):
                                this_label_list.append( (r*repeatlen+sc,'POTENTIAL_BIDENTATE_HBONDS_SC') )
                                #this_pose_prop.pdb_info().add_reslabel(sc,'POTENTIAL BIDENTATE HBONDS SC')
                        for bb in label_bb_list:
                            for r in range(num_repeats):
                                this_label_list.append( (r*repeatlen+bb,'POTENTIAL_BIDENTATE_HBONDS_BB') )
                                #this_pose_prop.pdb_info().add_reslabel(bb,'POTENTIAL BIDENTATE HBONDS BB')   
                        used_sc_resids[resid] = 'POTENTIAL_BIDENTATE_HBONDS_SC'
                        this_potential_bidentate_pose_candidates.append([this_pose_prop, [resid], [this_pose_prop.residue(resid).name1()], this_label_list, potential_bidentate_hbonds])                         


                    if not disable_combine_pseudo_bidentate and len(this_potential_bidentate_pose_candidates) > 0:
                        # make combinations if more than one pseudo bidentate found (for now only pairs)
                        bidentate_combination_list = [x for x in combinations(this_potential_bidentate_pose_candidates, 2)]

                        if _DEBUG:
                            print('DEBUG:       bidentate_combination_list_size ', len(bidentate_combination_list))
                            #print('DEBUG:     bidentate_combination_list: ', bidentate_combination_list)

                        for comb in bidentate_combination_list:
                            this_pose_prop = pose_prop.clone()
                            resids, resns, pot_bhb = [], [], []
                            comb_label_list = []
                            for comb_i in range(len(comb)):

                                '''
                                # assemble by packing
                                pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_high_hbond_sym)   
                                #pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_2015_soft_sym_high_hbond)   
                                pack_rotamers.task_factory(_config_my_task_factory([x*repeatlen+comb[comb_i][1][0] for x in range(num_repeats)], allowed_aa=comb[comb_i][2][0]))   
                                pack_rotamers.apply(this_pose_prop)
                                '''

                                # assemble by grafting
                                this_pose_prop = pyrosetta.rosetta.protocols.grafting.replace_region(comb[comb_i][0], this_pose_prop, comb[comb_i][1][0], comb[comb_i][1][0], 1)

                                add_bidentate_atom_pair_cst_to_pose(this_pose_prop, comb[comb_i][4], cst_func='HARMONIC 2.0 0.5')
                                resids += comb[comb_i][1]
                                resns += comb[comb_i][2]
                                comb_label_list += comb[comb_i][3]
                                pot_bhb += comb[comb_i][4]

                            comb_label_list = list(set(comb_label_list))
                            this_potential_bidentate_pose_candidates.append( [ this_pose_prop, resids, resns, comb_label_list, pot_bhb ] )

                    potential_bidentate_pose_candidates += [SymPose(x[0], labels=x[3], used_sc_resids={y:'POTENTIAL_BIDENTATE_HBONDS_SC' for y in x[1]}) for x in this_potential_bidentate_pose_candidates] # append new entry in the form of SymPose

                if _DEBUG:
                    print('DEBUG:       length of potential_bidentate_pose_candidates: ', len(potential_bidentate_pose_candidates))

                pose_prop_candidates = potential_bidentate_pose_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after potential bidentate search, skipping ...')
                    continue      




            ############################### filters before relax ###############################<<<<<

            ##
            #
            #  TODO: add hb cst to existing bbhb (e.g. beta turn)
            #
            ##



            # relax
            if not disable_relax:  
                if _DEBUG:
                    print('DEBUG:     BB_RELAX')
                relaxed_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    this_pose_prop_ref = this_pose_prop.clone()
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()                

                    # TODO
                    # add hbond cst to intra/interloop hbonds!!!!


                    coord_cst = pyrosetta.rosetta.protocols.constraint_movers.AddConstraintsToCurrentConformationMover()
                    non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_all])) # loops only
                    #non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_n_cap_all]))  # loops, caps and flanks
                    coord_cst.residue_selector(non_loop_residue_selector)
                    coord_cst.apply(this_pose_prop)

                    # add virtual root to eliminate levered arm effect
                    vr = pyrosetta.rosetta.protocols.simple_moves.VirtualRootMover()
                    vr.apply(this_pose_prop)

                    #fr = _setup_fast_relax(pose, sf_cst, min_reslist, cartesian=False, include_neighbor=True)
                    fr = _setup_fast_relax(this_pose_prop, sf_sym_cst, loops_n_cap_all, cartesian=False, include_neighbor=True)
                    fr.apply(this_pose_prop)
                    fr_rmsd = rmsd_by_ndxs_atoms(this_pose_prop, 1, num_repeats*repeatlen, this_pose_prop_ref, 1, num_repeats*repeatlen)
                    if fr_rmsd > max_relax_rmsd_allowed:
                        if _DEBUG:
                            print('DEBUG:       the pose had a rmsd drift of {}, more than the cutoff {}, skipping this pose ...'.format(fr_rmsd, max_relax_rmsd_allowed))
                        continue
                    relaxed_pose_prop_candidates.append(SymPose(this_pose_prop, label_list, used_sc_resids))
                pose_prop_candidates = relaxed_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after relax, skipping ...')
                    continue  



            ############################### post-relax filters #################################>>>>>




            # bidentate
            if min_num_bidentate > 0:

                if _DEBUG:
                    print('DEBUG:     FILTER: bidentate')

                bidentate_pose_prop_candidates = []

                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()     
                    new_used_sc_resids = {}                

                    ####
                    #     TODO: scan loop sequence, keep pose directly if no non-ala/gly residue found (the base bbhb pose will then be kept till the end)
                    ####


                    bidentate_total_count = 0
                    potential_bidentate_sc_list = [x for x in used_sc_resids if used_sc_resids[x] == 'POTENTIAL_BIDENTATE_HBONDS_SC']
                    if len(potential_bidentate_sc_list) > 0:
                        search_reslist = potential_bidentate_sc_list
                    else:
                        # this happens when pseudo bidentate search not performed
                        # use the entire 2nd loop
                        search_reslist = loops_n_cap[1] 
                    target_reslist = loops_n_cap[2]  # 3rd loop, cap, nei
                    for resid in search_reslist:
                        # CAUTION: a TEMPORARY hard-coded less strigent criterion for his
                        if this_pose_prop.residue(resid).name1().strip() == 'H':
                            raw_current_bidentate_hbonds_by_sc_res = find_potential_bidentate_hbond(this_pose_prop, [resid], target_reslist, delta_HA=3., delta_theta=60., reslist1_atom_type=['sc'], reslist2_atom_type=['bb'])
                        else:
                            # assess hb use rosetta's definition
                            current_hbond_set = find_hbonds(this_pose_prop, [resid], target_reslist, skip=[])
                            raw_current_bidentate_hbonds_by_sc_res = find_bidentate_hbond(this_pose_prop, current_hbond_set['bb-sc'])

                        #this_pose_prop.dump_pdb(out_put_fname.format(dir=out_dir, base=base_fname, mdl_no=mdl+1, LS=loop_s, LE=loop_e).replace('.pdb','.debug_post_fr.pdb'))

                        # only allow fully satisfied bidentates
                        current_bidentate_hbonds_by_sc_res = {x:raw_current_bidentate_hbonds_by_sc_res[x] for x in raw_current_bidentate_hbonds_by_sc_res 
                                                        if check_fully_satisfied_bidenates(this_pose_prop, raw_current_bidentate_hbonds_by_sc_res, x)}

                        if len(current_bidentate_hbonds_by_sc_res) == 0:
                            if _DEBUG:
                                print('DEBUG:       at residue {}, bidentate bb-sc hbond not found after relax ...'.format(resid))
                            #break
                        else:
                            if _DEBUG:
                                print('DEBUG:       at residue {}, found bidentate hbonds: {}'.format(resid, current_bidentate_hbonds_by_sc_res))
                            bidentate_total_count += len(current_bidentate_hbonds_by_sc_res)

                        # add labels to residue in pdb
                        label_sc_list, label_bb_list = [], []
                        for i in sorted(current_bidentate_hbonds_by_sc_res.keys()):
                            reduced_i = i%repeatlen
                            if reduced_i not in label_sc_list:
                                label_sc_list.append(reduced_i)
                            for hb in current_bidentate_hbonds_by_sc_res[i]:
                                for j in [0,2]:
                                    if hb[j] != i:
                                        reduced_j = hb[j]%repeatlen
                                        if reduced_j not in label_bb_list:
                                            label_bb_list.append(reduced_j)
                        for sc in label_sc_list:
                            for r in range(num_repeats):
                                label_list.append( (r*repeatlen+sc,'BIDENTATE_HBONDS_SC') )
                                #this_pose_prop.pdb_info().add_reslabel(sc,'BIDENTATE HBONDS SC')
                        for bb in label_bb_list:
                            for r in range(num_repeats):                          
                                label_list.append( (r*repeatlen+bb,'BIDENTATE_HBONDS_BB') )
                                #this_pose_prop.pdb_info().add_reslabel(bb,'BIDENTATE HBONDS BB')  

                        new_used_sc_resids[resid] = 'BIDENTATE_HBONDS_SC'


                    if bidentate_total_count < min_num_bidentate:
                        if _DEBUG:
                            print('DEBUG:       only {} bidentate hbonds found, but {} required ...'.format(bidentate_total_count, min_num_bidentate))
                        continue

                    if len(potential_bidentate_sc_list) > 0 and bidentate_total_count < len(potential_bidentate_sc_list):
                        if _DEBUG:
                            print('DEBUG:       {} bidentate hbonds lost after relax, original {} post-relax {}, skipping ...'.format(len(potential_bidentate_sc_list)-bidentate_total_count, len(potential_bidentate_sc_list), bidentate_total_count))
                        continue

                    bidentate_pose_prop_candidates.append(SymPose(this_pose_prop, label_list, new_used_sc_resids))  # potential bidentate resid don't matter no more if they didn't turn into a real bidentate

                pose_prop_candidates = bidentate_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after interloop bidentate hbonds search, skipping ...')
                    continue                 



            # interloop bbhb
            # still keep this just in case the hbond got lost after relax
            if min_interloop_hbond_num > 0 or min_interloop_bbbb_hbond_num > 0:

                if _DEBUG:
                    print('DEBUG:     FILTER: interloop_hbond')

                interloophb_pose_prop_candidates = []

                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()     

                    interloop_hb_set = find_hbonds(this_pose_prop, loops[1], loops[2], skip=[])  # 2nd-3rd loops
                    num_interloop_bbbb_hb = len(interloop_hb_set['bb-bb'])
                    num_interloop_hb = len(interloop_hb_set['bb-bb']) + len(interloop_hb_set['bb-sc']) + len(interloop_hb_set['sc-sc'])
                    if num_interloop_bbbb_hb < min_interloop_bbbb_hbond_num:
                        if _DEBUG:
                            print('DEBUG:       {} interloop bb-bb hbonds, {} required, skipping ...'.format(num_interloop_bbbb_hb, min_interloop_bbbb_hbond_num))
                        continue
                    if num_interloop_hb < min_interloop_hbond_num:
                        if _DEBUG:
                            print('DEBUG:       {} interloop total hbonds, {} required, skipping ...'.format(num_interloop_hb, min_interloop_hbond_num))
                        continue  
                    label_bbhb_list = []
                    for hb in interloop_hb_set['bb-bb']:
                        reduced_i, reduced_j = hb[0]%repeatlen, hb[1]%repeatlen
                        if reduced_i not in label_bbhb_list:
                            label_bbhb_list.append(reduced_i)
                        if reduced_j not in label_bbhb_list:
                            label_bbhb_list.append(reduced_j)
                    for i in label_bbhb_list:
                        for r in range(num_repeats):
                            label_list.append( (r*repeatlen+i,'INTERLOOP_BB-BB_HBONDS') )

                    interloophb_pose_prop_candidates.append(SymPose(this_pose_prop, label_list, used_sc_resids))

                pose_prop_candidates = interloophb_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after interloop hbond search, skipping ...')
                    continue  


            # filter for sheet-like beta hairpins
            if sheetlike_turns:

                #if _DEBUG:
                #    print('DEBUG:     FILTER: sheet-like beta turns')

                sheetlike_pose_prop_candidates = []

                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids() 

                    second_loop_bturns = find_all_pseudo_beta_turns(this_pose_prop, loops[1], delta_HA=3., delta_theta=60.)

                    #if _DEBUG:
                    #    print('DEBUG:             second_loop_bturns: ', second_loop_bturns)

                    bb_hb_check = False
                    for bturn in second_loop_bturns:

                        bturn = sorted(bturn) # needed for hbond searching code below

                        if bb_hb_check:
                            break

                        neighbor_loops_reslist = list(range(loops[0][0], loops[0][-1]+1))+list(range(loops[2][0], loops[2][-1]+1)) 

                        # bturn residues: i-i+3
                        #    i+1's N and i+2'O has to bb-bb hbond to their image in neighbor loops

                        '''
                        hbond_set_1 = find_hbonds(this_pose_prop, [bturn[0]+1], neighbor_loops_reslist)
                        if len(hbond_set_1['bb-bb']) > 0 and hbond_set_1['bb-bb'][0][2].don_hatm.name() == 'N':
                            bb_hb_check = True
                        hbond_set_2 = find_hbonds(this_pose_prop, [bturn[0]+2], neighbor_loops_reslist)
                        if len(hbond_set_2['bb-bb']) > 0 and hbond_set_2['bb-bb'][0][2].acc_atm.name() == 'O':
                            bb_hb_check = True
                        '''

                        hbond_set_1 = simple_hbond_finder(this_pose_prop, [bturn[0]+1], neighbor_loops_reslist, delta_HA=3., delta_theta=60., 
                                        reslist1_atom_type=['bb'], reslist2_atom_type=['bb'],verbose=False)
                        #if _DEBUG:
                        #    print('DEBUG:             hbond_set_1: ', hbond_set_1)
                        if len(hbond_set_1) > 0 and this_pose_prop.residue(hbond_set_1[0][0]).atom_name(hbond_set_1[0][1]).strip() == 'N':
                            bb_hb_check = True

                        hbond_set_2 = simple_hbond_finder(this_pose_prop, [bturn[0]+2], neighbor_loops_reslist, delta_HA=3., delta_theta=60., 
                                        reslist1_atom_type=['bb'], reslist2_atom_type=['bb'],verbose=False)
                        #if _DEBUG:
                        #    print('DEBUG:             hbond_set_2: ', hbond_set_2)
                        if len(hbond_set_2) > 0 and this_pose_prop.residue(hbond_set_2[0][0]).atom_name(hbond_set_2[0][1]).strip() == 'O':
                            bb_hb_check = True
                    if bb_hb_check:
                        sheetlike_pose_prop_candidates.append(sympose)

                pose_prop_candidates = sheetlike_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after sheet-like beta hairpin check, skipping ...')
                    continue  


            # motifscore
            if min_loop_motif > 0:

                if _DEBUG:
                    print('DEBUG:     FILTER: motifscore')

                motif_pose_prop_candidates = []

                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()     
                    label_reslist = [x for x in used_sc_resids if used_sc_resids[x] == 'BIDENTATE_HBONDS_SC']  # only real bidentates matter at this stage
                    label_reslist_reduced = [x%repeatlen for x in label_reslist]


                    # search 2nd loop to avoid misssing contacts when using 1st loop
                    ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in loops[1]]))  # 2nd loop
                    ind = ind_selector.apply(this_pose_prop)
                    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(ind_selector, motif_dist_cutoff)
                    nbr = nbr_selector.apply(this_pose_prop)
                    #nbr_reslist = [i+1 for i,j in enumerate(nbr) if j and i+1 not in loops[1]] # other than 2nd loop
                    nbr_reslist = [i+1 for i,j in enumerate(nbr) if j and i+1 not in loops_all] # noneloop residues
                    #if _DEBUG:
                    #    print('DEBUG: motifscore: nbr_reslist, ', nbr_reslist)
                    #print('+'.join([str(x) for x in nbr_reslist]))
                    dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(this_pose_prop)
                    dssp_obj.dssp_reduced()
                    mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
                    total_motifscore = 0
                    motif_list = []
                    for res1 in loops[1]: # 2nd loop
                        if res1%repeatlen in label_reslist_reduced: # skip residues already having labels (from bidentate)
                            continue
                        for res2 in nbr_reslist:
                            if res2%repeatlen in label_reslist_reduced: # skip residues already having labels (from bidentate)
                                continue
                            #if dssp_obj.get_dssp_secstruct(res2) != 'H': # skip non helix residues # TODO: check if this is necessary
                            #    continue
                            motifscore = compute_motifscore_for_residue_pair(this_pose_prop, res1, res2, dist_cutoff=motif_dist_cutoff, dssp_obj=dssp_obj, motif_hash_man=mman_)
                            if motifscore != 0:
                                #print(this_pose_prop.residue(res1).name1(), res1, this_pose_prop.residue(res2).name1(), res2, motifscore)
                                motif_list.append([this_pose_prop.residue(res1).name1(), res1, this_pose_prop.residue(res2).name1(), res2, motifscore])
                                if not disable_motif_packing:
                                    used_sc_resids[res1] = 'MOTIF'
                                    used_sc_resids[res2] = 'MOTIF'
                                total_motifscore += motifscore
                    total_motifscore = -1*total_motifscore / this_pose_prop.size()
                    if len(motif_list) < min_loop_motif:
                        if _DEBUG:
                            print('DEBUG:       no motif found, skipping ...')
                        continue
                    if _DEBUG:
                        print('DEBUG:       found motif: {} {}'.format(total_motifscore, motif_list))


                    # update label and sympack motifs
                    motif_id_dict = {}
                    uniq_motif_list = []
                    motif_reslist_sym = []
                    motif_id = 0
                    for m in motif_list:
                        base_motif_pair = tuple([ x%repeatlen if x%repeatlen != 0 else repeatlen for x in [m[1], m[3]] ])
                        if base_motif_pair not in uniq_motif_list:
                            motif_id_dict[base_motif_pair] = 'MOTIF_{}'.format(motif_id+1)
                            motif_id += 1
                            uniq_motif_list.append(base_motif_pair)
                    for m in uniq_motif_list:
                        for r in range(num_repeats):
                            label_list.append( ( r*repeatlen+m[0], motif_id_dict[m] ) )
                            label_list.append( ( r*repeatlen+m[1], motif_id_dict[m] ) )
                            motif_reslist_sym.append( r*repeatlen+m[0] )
                            motif_reslist_sym.append( r*repeatlen+m[1] )
                    if _DEBUG:
                        print('DEBUG:         uniq_motif_list: ', uniq_motif_list)
                        print('DEBUG:         motif_reslist_sym: ', motif_reslist_sym)

                    if not disable_motif_packing:
                        pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_sym_pack)   
                        #pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf_2015_soft_sym)   
                        pack_rotamers.task_factory(_config_my_task_factory(motif_reslist_sym, allowed_aa='AFILMPV'))   
                        #pack_rotamers.task_factory(_config_my_task_factory(motif_reslist_sym, allowed_aa='FILMV'))   
                        pack_rotamers.apply(this_pose_prop)
                    else:
                        if _DEBUG:
                            print('DEBUG:           motif packing disabled ...')
                        pass


                    motif_pose_prop_candidates.append(SymPose(this_pose_prop, label_list, used_sc_resids))

                pose_prop_candidates = motif_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after motif search, skipping ...')
                    continue  



            ############################### post-relax filters #################################<<<<<



            ####################################### design #####################################>>>>>

            if design > 0:

                if _DEBUG:
                    print('DEBUG:     DESIGN')

                design_pose_prop_candidates = []
                for sympose in pose_prop_candidates:
                    this_pose_prop = sympose.pose() 
                    label_list = sympose.labels()
                    used_sc_resids = sympose.used_sc_resids()  
                    # exapnd used_sc_resids to all repeats
                    used_sc_resids_sym = {}
                    for resid in used_sc_resids:
                        if resid > repeatlen:
                            new_resid = resid%repeatlen
                            if new_resid == 0:
                                new_resid = repeatlen
                        else:
                            new_resid = resid
                        for r in range(num_repeats):
                            used_sc_resids_sym[r*repeatlen+new_resid] = used_sc_resids[resid]


                    design_reslist = [x for x in loops_n_cap_all if x not in used_sc_resids_sym]
                    design_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in design_reslist]))
                    design_nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(design_selector, 6.0)
                    design_nbr = design_nbr_selector.apply(this_pose_prop)
                    design_nbr_reslist = [i+1 for i,j in enumerate(design_nbr) if j and i+1 not in design_reslist] 

                    if _DEBUG:
                        print('DEBUG:       used_sc_resids_sym: ', sorted(list(used_sc_resids_sym.keys())))
                        print('DEBUG:       design_reslist: ', design_reslist)


                    loops_all_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in loops_all]))
                    loops_n_cap_all_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in loops_n_cap_all]))
                    special_selector_list = []

                    #
                    # bin selectors applied to loops_n_cap_all (or loops_all)
                    #
                    p_bin = None
                    p_bin_loop = None
                    if aacomp_design_pro_file != "-1":
                        p_bin = pyrosetta.rosetta.core.select.residue_selector.BinSelector()
                        p_bin.set_bin_params_file_name('PRO_DPRO')
                        p_bin.set_bin_name('LPRO')
                        p_bin.initialize_and_check()
                        p_bin_vec = p_bin.apply(this_pose_prop)
                        p_bin_reslist = [i+1 for i,j in enumerate(p_bin_vec) if j] 
                        p_bin_loop =  pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(p_bin, loops_n_cap_all_selector)
                        p_bin_loop_vec = p_bin_loop.apply(this_pose_prop)
                        p_bin_loop_reslist = [i+1 for i,j in enumerate(p_bin_loop_vec) if j] 
                        if _DEBUG:
                            print('DEBUG:         p_bin reslist: ', p_bin_reslist)
                            print('DEBUG:         p_bin_loop reslist: ', p_bin_loop_reslist)
                        aacomp_pro = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
                        aacomp_pro.create_constraint_from_file(aacomp_design_pro_file)
                        aacomp_pro.add_residue_selector(p_bin_loop)
                        aacomp_pro.apply(this_pose_prop)    
                    special_selector_list.append(p_bin_loop)   


                    Ebin = None
                    Ebin_loop = None
                    if aacomp_design_Ebin_file != "-1":
                        Ebin = pyrosetta.rosetta.core.select.residue_selector.BinSelector()
                        Ebin.set_bin_params_file_name('ABEGO')
                        Ebin.set_bin_name('E')
                        Ebin.initialize_and_check()
                        Ebin_vec = Ebin.apply(this_pose_prop)
                        Ebin_reslist = [i+1 for i,j in enumerate(Ebin_vec) if j] 
                        Ebin_loop = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(Ebin, loops_n_cap_all_selector)
                        Ebin_loop_vec = Ebin_loop.apply(this_pose_prop)
                        Ebin_loop_reslist = [i+1 for i,j in enumerate(Ebin_loop_vec) if j]                         
                        if _DEBUG:
                            print('DEBUG:         Ebin reslist: ', Ebin_reslist)
                            print('DEBUG:         Ebin_loop reslist: ', Ebin_loop_reslist)
                        aacomp_Ebin = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
                        aacomp_Ebin.create_constraint_from_file(aacomp_design_Ebin_file)
                        aacomp_Ebin.add_residue_selector(Ebin_loop)
                        aacomp_Ebin.apply(this_pose_prop)             
                    special_selector_list.append(Ebin_loop)   


                    Gbin = None
                    Gbin_loop = None
                    if aacomp_design_Gbin_file != "-1":
                        Gbin = pyrosetta.rosetta.core.select.residue_selector.BinSelector()
                        Gbin.set_bin_params_file_name('ABEGO')
                        Gbin.set_bin_name('G')
                        Gbin.initialize_and_check()
                        Gbin_vec = Gbin.apply(this_pose_prop)
                        Gbin_reslist = [i+1 for i,j in enumerate(Gbin_vec) if j] 
                        Gbin_loop = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(Gbin, loops_n_cap_all_selector)
                        Gbin_loop_vec = Gbin_loop.apply(this_pose_prop)
                        Gbin_loop_reslist = [i+1 for i,j in enumerate(Gbin_loop_vec) if j]                           
                        if _DEBUG:
                            print('DEBUG:         Gbin reslist: ', Gbin_reslist)
                            print('DEBUG:         Gbin_loop reslist: ', Gbin_loop_reslist)
                        aacomp_Gbin = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
                        aacomp_Gbin.create_constraint_from_file(aacomp_design_Gbin_file)
                        aacomp_Gbin.add_residue_selector(Gbin_loop)
                        aacomp_Gbin.apply(this_pose_prop)    
                    special_selector_list.append(Gbin_loop)   


                    if aacomp_design_file != "-1":
                        design_normal_selector = design_selector
                        for special_selector in special_selector_list:
                            if special_selector != None:
                                not_special_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(special_selector)
                                not_special_selector.apply(this_pose_prop)
                                design_normal_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(design_normal_selector, not_special_selector)
                        design_normal_vec = design_normal_selector.apply(this_pose_prop)
                        design_normal_reslist = [i+1 for i,j in enumerate(design_normal_vec) if j] 
                        if _DEBUG:
                            print('DEBUG:         design_normal reslist size: ', len(design_normal_reslist))
                            print('DEBUG:         design_normal reslist: ', design_normal_reslist)
                        aacomp_normal = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
                        aacomp_normal.create_constraint_from_file(aacomp_design_file)
                        aacomp_normal.add_residue_selector(design_normal_selector)
                        aacomp_normal.apply(this_pose_prop)


                    # helix nterm capping residues: 1st residue in loop after a helix
                    if _DEBUG:
                        print('DEBUG:           searching for helix ncap residues ...')
                    dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(this_pose_prop)
                    dssp_obj.dssp_reduced()
                    dssp = 'X' + str(dssp_obj.get_dssp_secstruct()) # pad a letter at Nterm to make index consistent with residue numbering
                    min_helix_length = 6 # hardcoded minimum helix length for each helix
                    helix_ncap_reslist = []
                    helix_ncap_selector = None
                    for resid in loops_n_cap[0]:
                        if dssp[resid] != 'H' and dssp[resid+1:resid+min_helix_length+1] == 'H'*min_helix_length:
                            helix_ncap_reslist.append(resid)
                    if len(helix_ncap_reslist) == 0:
                        if _DEBUG:
                            print('DEBUG:           could not find helix ncap residue ...')
                    elif aacomp_design_pro_file != "-1":
                        for r in range(1, num_repeats):
                            helix_ncap_reslist.append(helix_ncap_reslist[0]+(r*repeatlen))
                        if _DEBUG:
                            print('DEBUG:           found helix ncap residues: ', helix_ncap_reslist)    
                        helix_ncap_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in helix_ncap_reslist]))
                        # force pro capping
                        helix_ncap_pro_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x+1) for x in helix_ncap_reslist])) # pro position is resi_cap + 1
                        aacomp_cap_pro = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
                        aacomp_cap_pro.create_constraint_from_file(aacomp_design_pro_file)
                        aacomp_cap_pro.add_residue_selector(helix_ncap_pro_selector)
                        aacomp_cap_pro.apply(this_pose_prop)    




                    if layer_design:

                        if _DEBUG:
                            print('DEBUG:         using layer design...')

                        # somehow the layer selector works only in asymmetric setup here ...
                        RepeatProteinRelax_apply(this_pose_prop, modify_symmetry_and_exit_=True, remove_symm_=True, bblist=loops_n_cap_all, chilist=loops_n_cap_all, num_repeats=num_repeats)

                        # use 2nd repeat unit, select layers, then propagate to all repeats
                        design_reslist_2nd_repeat = [x for x in design_reslist if x > repeatlen and x <= repeatlen*2]
                        second_loop_n_cap_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in design_reslist_2nd_repeat]))

                        layer_list = ['core','boundary','surface']
                        layer_selectors = {}
                        layer_reslists = {}
                        for layer in layer_list:
                            layer_all_selector = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
                            layer_bool_list = [True if l == layer else False for l in layer_list ]
                            layer_all_selector.set_layers(*layer_bool_list) # core, boundary, surface
                            layer_2nd_unit_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(layer_all_selector, second_loop_n_cap_selector)
                            layer_2nd_unit_vec = layer_2nd_unit_selector.apply(this_pose_prop)
                            #layer_2nd_unit_reslist = [i+1 for i,j in enumerate(layer_2nd_unit_vec) if j]
                            layer_2nd_unit_reslist = [i+1 for i,j in enumerate(layer_2nd_unit_vec) if j and i+1 not in helix_ncap_reslist]
                            layer_reslist = []
                            for x in layer_2nd_unit_reslist:
                                new_x = x%repeatlen
                                if new_x == 0:
                                    new_x = repeatlen
                                for r in range(num_repeats):                            
                                    layer_reslist.append(r*repeatlen+new_x)
                            layer_reslist = sorted(layer_reslist)
                            layer_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in layer_reslist]))
                            layer_selectors[layer] = layer_selector
                            layer_reslists[layer] = layer_reslist

                        if _DEBUG:
                            for layer in layer_list:
                                print('DEBUG:           {} reslist: {}'.format(layer, layer_reslists[layer]))


                        RepeatProteinRelax_apply(this_pose_prop, modify_symmetry_and_exit_=True, remove_symm_=False, bblist=loops_n_cap_all, chilist=loops_n_cap_all, num_repeats=num_repeats)

                        coord_cst = pyrosetta.rosetta.protocols.constraint_movers.AddConstraintsToCurrentConformationMover()
                        #non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_all])) # loops only
                        non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_n_cap_all]))  # loops, caps and flanks
                        coord_cst.residue_selector(non_loop_residue_selector)
                        coord_cst.apply(this_pose_prop)

                        # add virtual root to eliminate levered arm effect
                        vr = pyrosetta.rosetta.protocols.simple_moves.VirtualRootMover()
                        vr.apply(this_pose_prop)

                        # re-apply the aacomp as they (may) got removed during sym_info setting
                        if aacomp_design_pro_file != "-1":
                            aacomp_pro.apply(this_pose_prop)    
                        if aacomp_design_Ebin_file != "-1":
                            aacomp_Ebin.apply(this_pose_prop)             
                        if aacomp_design_Gbin_file != "-1":
                            aacomp_Gbin.apply(this_pose_prop)    
                        if aacomp_design_file != "-1":
                            aacomp_normal.apply(this_pose_prop)
                        if len(helix_ncap_reslist) != 0 and aacomp_design_pro_file != "-1":
                            aacomp_cap_pro.apply(this_pose_prop) 


                        if _DEBUG:
                            print('DEBUG:           now designing ...')

                        #fast_design_rosettacon2018_layer(this_pose_prop, sf_sym_cst_design, layer_selectors['core'], layer_selectors['boundary'], layer_selectors['surface'], helix_ncap_selector,
                        #                                     core_aa='AFILMPV', boundary_aa='ADEGHIKLMNPQRSTVWY', surface_aa='ADEGHKNPQRST', helix_ncap_aa='DNST', rmsd_check_resids=loops_n_cap[1], include_neighbor=True)

                        fast_design_layer_relaxscript(this_pose_prop, sf_sym_cst_design, layer_selectors['core'], layer_selectors['boundary'], layer_selectors['surface'], helix_ncap_selector,
                                                             core_aa='AFILMPV', boundary_aa='ADEGHIKLMNPQRSTVWY', surface_aa='ADEGHKNPQRST', helix_ncap_aa='DNST', rmsd_check_resids=loops_n_cap[1], include_neighbor=True)

                    else:

                        if _DEBUG:
                            print('DEBUG:         layer design disabled ...')


                        coord_cst = pyrosetta.rosetta.protocols.constraint_movers.AddConstraintsToCurrentConformationMover()
                        #non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_all])) # loops only
                        non_loop_residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join([str(x) for x in range(1,this_pose_prop.size()) if x not in loops_n_cap_all]))  # loops, caps and flanks
                        coord_cst.residue_selector(non_loop_residue_selector)
                        coord_cst.apply(this_pose_prop)

                        # add virtual root to eliminate levered arm effect
                        vr = pyrosetta.rosetta.protocols.simple_moves.VirtualRootMover()
                        vr.apply(this_pose_prop)

                        fast_design_rosettacon2018(this_pose_prop, sf_sym_cst_design, design_reslist=design_reslist, allowed_aa='ADEGHIKLMNPQRSTV', rmsd_check_resids=loops_n_cap[1], include_neighbor=True)




                    ### post design min/relax using sf w/o aacomp and buns_penalty
                    #if _DEBUG:
                    #    print('DEBUG:     POST-DESIGN: relax')
                    #fr = _setup_fast_relax(this_pose_prop, sf_sym_cst, loops_n_cap_all, cartesian=False, include_neighbor=True)
                    #fr.apply(this_pose_prop)
                    if _DEBUG:
                        print('DEBUG:     POST-DESIGN: min')
                    min_mover = _setup_min_mover(this_pose_prop, sf_sym_cst, loops_n_cap_all, cartesian=False, include_neighbor=True)
                    min_mover.apply(this_pose_prop)


                    designed_score = sf_sym(this_pose_prop)                         


                    if design > 1:
                        #
                        #  TODO: genkic perturb and redesign
                        #
                        pass

                    #######################################  post design filters #######################################>>>>>

                    design_pose_prop_candidates.append(SymPose(this_pose_prop, label_list, used_sc_resids_sym))

                    #######################################  post design filters #######################################<<<<<


                pose_prop_candidates = design_pose_prop_candidates
                if len(pose_prop_candidates) == 0:
                    if _DEBUG:
                        print('DEBUG:       no pose remains after design, skipping ...')
                    continue  


            ####################################### design #####################################<<<<<




            for pose_prop_id in range(len(pose_prop_candidates)):
                this_pose_prop = pose_prop_candidates[pose_prop_id].pose() 
                label_list = pose_prop_candidates[pose_prop_id].labels()
                used_sc_resids = pose_prop_candidates[pose_prop_id].used_sc_resids()  


                # remove symmetry before scoring?
                RepeatProteinRelax_apply(this_pose_prop, modify_symmetry_and_exit_=True, remove_symm_=True, bblist=min_reslist, chilist=min_reslist, num_repeats=num_repeats)
                this_pose_prop_score = sf(this_pose_prop)

                #
                # TODO: expand info lines 
                #

                loop_scores = [this_pose_prop.energies().residue_total_energy(x) for x in range(loop_s_base-neighbor_flank_n, loop_e_base+neighbor_flank_c+1)]
                loop_energy = sum(loop_scores) / float(len(loop_scores))
                if len(cut_combination) == 0 and param_file == '-1':
                    cut_start = -1
                    cut_end = -1
                #loop_direction_cos_angle = compute_loop_direction(this_pose_prop, loops[1][0], loops[1][-1], repeatlen, num_repeats=num_repeats)
                loop_direction_cos_angle = compute_loop_direction_nonperfect_repeats(this_pose_prop, loop_idx=1, bturn_side=bturn_side)
                info_line = '{base}_{mdl_no:04}\t{cut_start}\t{cut_end}\t{ccap}\t{ncap}\t{frag}\t{before}\t{after}\t{trial}\t{ls}\t{le}\t{pose_ene}\t{loopene}\t{loop_direction}\n'.format(base=base_fname, mdl_no=mdl+1, \
                    cut_start=cut_start, cut_end=cut_end, ccap=c_cap_phipsi_name, ncap=n_cap_phipsi_name, frag=phipsi_name, before=before_frag_length, after=after_frag_length, \
                    trial='{}_{}'.format(mdl,pose_prop_id), ls=loops[0][0], le=loops[0][-1], pose_ene=this_pose_prop_score, loopene=loop_energy, loop_direction=loop_direction_cos_angle)

                if info_only:
                    info_pool.append(info_line)
                    continue

                rmsd_check = bbrmsd_check(this_pose_prop, posepool, loops[0][0], loops[0][-1], cutoff=bbrmsd_cutoff)
                #new_pose_entry = [this_pose_prop, this_pose_prop_score, out_put_fname.format(dir=out_dir, base=base_fname, mdl_no=mdl+1, LS=loop_s, LE=loop_e), info_line]
                new_pose_entry = [this_pose_prop, this_pose_prop_score, out_put_fname.format(dir=out_dir, base=base_fname, mdl_no=mdl+1, LS=loops[0][0], LE=loops[0][-1]).replace('.pdb','_{}.pdb'.format(pose_prop_id)), info_line, label_list]
                if rmsd_check[0]:
                    if len(posepool) < pyrosetta.rosetta.basic.options.get_integer_option('out:nstruct'):
                        posepool.append(new_pose_entry)
                    else:
                        #print('replacing the worst existing pose in the pool!!')
                        posepool[-1] = new_pose_entry
                else: # replace the one in posepool with small rmsd but high score
                    if _DEBUG:
                        print('DEBUG:     bbrmsd less than cutoff ', bbrmsd_cutoff)
                    if this_pose_prop_score < posepool[rmsd_check[1]][1]:
                        posepool[rmsd_check[1]] = new_pose_entry
                posepool = sorted(posepool, key=lambda x:x[1])
        else:
            if _DEBUG:
                #print('DEBUG:   Failed closure for trial ', mdl+1)
                print('DEBUG:   Failed closure for trial {}: cut_start({}) cut_end({}), before_length({}), after_length({}), ccap({}), ncap({}), turn({})'.format(mdl+1, \
                        cut_start, cut_end, before_frag_length, after_frag_length, c_cap_phipsi_name.split('/')[-1], n_cap_phipsi_name.split('/')[-1], phipsi_name.split('/')[-1]))
            pass


    os.system('rm -f '+out_dir+'/'+base_fname+'_info.dat.tmp.*')
    os.system('rm -f '+out_dir+'/'+base_fname+'_combination_stats.dat.tmp.*')
    os.system('rm -f '+out_dir+'/'+'*tmp*.pdb')
    fout = open(out_dir+'/'+base_fname+'_info.dat','w')
    fout.write('name\tcut_start\tcut_end\tccap\tncap\tfrag\tbefore_frag_numres\tafter_frag_numres\ttrial\tloop_start\tloop_end\ttotal_score\tloopene_perres\tloop_direction\n')

    comb_pool = {}
    header_line = 'name\tcut_start\tcut_end\tccap\tncap\tfrag\tbefore_frag_numres\tafter_frag_numres\ttrial\tloop_start\tloop_end\ttotal_score\tloopene_perres\tloop_direction\n'
    term_ind_dict = {term:header_line.split().index(term) for term in comb_term_list}
    if info_only:
        for info_line in info_pool:
            fout.write(info_line)
    else:
        if len(posepool) == 0:
            print('No qualified poses remain after filtering!')

        else:
            if posepool[0][1] > total_score_cutoff:
                print('Skip output: all of the poses are above total_score_cutoff!')
            else:

                #for l in posepool[0][4]:
                #    posepool[0][0].pdb_info().add_reslabel(l[0],l[1])
                #posepool[0][0].dump_scored_pdb(posepool[0][2], sf)
                #fout.write(posepool[0][3])
                #for pose_entry in posepool[1:]:
                for pose_entry in posepool:
                     
                    # comment out the following to output all
                    if worst_score_allowed != -1:
                        current_score_cutoff = worst_score_allowed * posepool[0][1] if posepool[0][1] < 0 else posepool[0][1] / worst_score_allowed
                    else:
                        current_score_cutoff = total_score_cutoff
                    
                    # comment out the following to output all
                    if pose_entry[1] > total_score_cutoff or pose_entry[1] > current_score_cutoff:
                        continue

                    if len(pose_entry) > 4:
                        for l in pose_entry[4]:
                            try:
                                pose_entry[0].pdb_info().add_reslabel(l[0], l[1])
                            except AttributeError:
                                if _DEBUG:
                                    print('DEBUG: missing pdb_info for  {} of the posepool, reinitialize pdb_info ...'.format(pose_entry))
                                pose_entry[0].pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(pose_entry[0],True))
                                pose_entry[0].pdb_info().add_reslabel(l[0], l[1])

                    pose_entry[0].dump_scored_pdb(pose_entry[2], sf)
                    fout.write(pose_entry[3])

                    items = pose_entry[3].split()
                    comb = tuple([items[term_ind_dict[x]] for x in comb_term_list])
                    if comb not in comb_pool:
                        comb_pool[comb] = 0
                    comb_pool[comb] += 1

    fout.close()


    # write combination stats
    comb_pool_keys = sorted(list(comb_pool.keys()), key=lambda x:comb_pool[x], reverse=True)
    fout = open(out_dir+'/'+base_fname+'_combination_stats.dat','w')
    fout.write('{},count\n'.format(','.join(comb_term_list)))
    for comb in comb_pool_keys:
        fout.write('{},{}\n'.format(','.join(comb), comb_pool[comb]))
    fout.close()


def parse_cut_pairs(cut_pairs):
    cut_combination = []
    # example input:
    # 1:7,8;2:8,9;3,4:9,10
    for pair in cut_pairs.split(';'):
        cstart = str_to_int_list(pair.split(':')[0])
        cend = str_to_int_list(pair.split(':')[1])
        for c in list(itertools.product(cstart,cend)):
            cut_combination.append(c)
    return cut_combination


def find_helices_by_dssp(p, min_helix_length=6, search_range=[-1,-1]):
    dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(p)
    dssp_obj.dssp_reduced()
    dssp = dssp_obj.get_dssp_secstruct()
    if search_range != [-1,-1]:
       resrange = list(range(search_range[0]-1, search_range[1])) # -1 because here idx starts at 0
    else:
       resrange = range(len(dssp))
    #print('search_range:', resrange)
    #print('search_dssp: ', dssp[resrange[0]-1:resrange[-1]-1])
    helices = []
    prev = ''
    start = -1
    for i in resrange:
        if dssp[i] == 'H' and dssp[i] != prev:
            start = i+1
        elif dssp[i] != 'H' and prev == 'H' and start != -1 and i - start >= min_helix_length:
            helices.append([start,i])
            start = -1
        elif i == resrange[-1] and prev == 'H' and start != -1 and i - start >= min_helix_length: # if last helix is at the end
            helices.append([start,i+1]) # +1 to correct index mismatch 
            start = -1
        prev = dssp[i]
    #print(helices)
    return helices


def find_loop_by_dssp(p, min_helix_length=6, min_loop_length=6):
    '''
        look for the two loops connecting Helix1-Helix2 and Helix2-Helix3
    '''
    helices = find_helices_by_dssp(p, min_helix_length=min_helix_length)
    loops = []
    #print(helices)
    for i in range(len(helices)-1):
      #print(i, len(helices))
      ls, le = helices[i][1]+1, helices[i+1][0]-1
      if le >= ls + min_loop_length-1:
        loops.append([ls, le])
    return loops


def get_CA_sccom_vector(pose, resid):
    '''
    get the vector of CA -> centerofmass of side chain heavy atoms
    for the given residue id
    '''
    res = pose.residue(resid)
    coords = []
    for atomid in range(1, len(res.atoms())+1):
        if (not res.atom_is_hydrogen(atomid)) and (not res.atom_is_backbone(atomid)):
            #print(res.atom_name(atomid))
            coords.append(list(res.xyz(atomid)))
    coords_sum = [0]*3
    for c in coords:
        for j in range(3):
            coords_sum[j] += c[j]
    sc_com = [x/len(coords) for x in coords_sum] # center of mass of side chain heavy atom
    ca_coord = res.xyz(res.atom_index('CA'))
    #print(sc_com)
    #print(ca_coord)
    return [x-y for x,y in zip(sc_com,ca_coord)]

def get_cos_angle_of_vecs(vec1, vec2):
    return np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2))


def compute_handedness(pose):
    '''
        1: left handed -> cut 1st repeat
        -1: right handed -> cut last repeat
    '''

    helices = find_helices_by_dssp(pose)

    r2_reslist = list(range(helices[2][0], helices[2][1]+1)) + list(range(helices[3][0], helices[3][1]+1))
    com_r2 = np.array(get_com(pose, r2_reslist))
    
    # get vectors (com 2nd repeat to coms of CA of 1st/last 3 residues in each helix of 2nd repeat)
    # com_r2 to end of the 1st helix
    #h1_vec_s_list = [helices[2][0]+x for x in range(3)]
    #h1_vec_s = np.array(get_com(pose, h1_vec_s_list))
    h1_vec_e_list = [helices[2][1]-x for x in range(3)]
    h1_vec_e = np.array(get_com(pose, h1_vec_e_list))
    #print('pseudoatom h1_vec_s, pos=[{}]'.format(','.join([str(x) for x in h1_vec_s])))
    #print('pseudoatom h1_vec_e, pos=[{}]'.format(','.join([str(x) for x in h1_vec_e])))
    h1_vec = h1_vec_e - com_r2

    # com_r2 to start of the 2nd helix
    #h2_vec_s_list = [helices[3][1]-x for x in range(3)]
    #h2_vec_s = np.array(get_com(pose, h2_vec_s_list))
    h2_vec_e_list = [helices[3][0]+x for x in range(3)]
    h2_vec_e = np.array(get_com(pose, h2_vec_e_list))
    #print('pseudoatom h2_vec_s, pos=[{}]'.format(','.join([str(x) for x in h2_vec_s])))
    #print('pseudoatom h2_vec_e, pos=[{}]'.format(','.join([str(x) for x in h2_vec_e])))
    #h2_vec = h2_vec_e - h2_vec_s
    h2_vec = h2_vec_e - com_r2

    # get 3rd vector (com 2nd repeat to com of CA of 3rd repeat)
    r3_reslist = list(range(helices[4][0], helices[4][1]+1)) + list(range(helices[5][0], helices[5][1]+1))
    com_r3 = np.array(get_com(pose, r3_reslist))
    #print('pseudoatom com_r2, pos=[{}]'.format(','.join([str(x) for x in com_r2])))
    #print('pseudoatom com_r3, pos=[{}]'.format(','.join([str(x) for x in com_r3])))
    vec_r2_r3 = com_r3 - com_r2

    # vector direction comparison
    h1_h2_cross = np.cross(h1_vec, h2_vec)
    handedness = np.dot(vec_r2_r3, h1_h2_cross)
    #print(handedness)
    
    return handedness / abs(handedness) 


def find_cap_anchors(pose, ncap_res_num=9, min_cos=0, check_handedness=True):
    '''
        ncap_res_num = 9 # number of residues from start of 2nd helix to be considered
        min_cos = 0 # minimum cos value of angle between the CA-com_of_sc_heavy_atom of potential ncap anchor residues (0 indicate 90 degree)
        check_handedness = True # by default select the repeat to minimize the potential loop-helix clashes (and therefore genkic loop bump check is turned on)
    '''

    #print(pose.sequence())


    loops = find_loop_by_dssp(pose, min_helix_length=6, min_loop_length=1)


    '''
    ################### single ccap anchor ###################

    # ccap anchor
    # -2 register of the 1st L residue in 1st loop
    ccap_resid = loops[0][0] - 2
    print(ccap_resid)
    ccap_vec = get_CA_sccom_vector(pose, ccap_resid)

    # ncap anchors
    # Top  (must be less than 9 residues far away and closest to the start of 2nd helix) residues whose sidechain (CA-<com of sc heavy atoms>) vectors
    # are at equal or less than 90 w/ the (CA-<com of sc heavy atoms>) vector of ccap residue, i.e. cos(angle) > 0
    ncap_vec_list = [[loops[0][-1]+1+x,get_CA_sccom_vector(pose,loops[0][-1]+1+x)] for x in range(ncap_res_num)]
    for i in range(len(ncap_vec_list)):
        ncap_vec_list[i].append(get_cos_angle_of_vecs(ccap_vec,ncap_vec_list[i][1]))
    ncap_vec_list = sorted(ncap_vec_list, key=lambda x:x[2], reverse=True)
    #print(ncap_vec_list)

    ncap_res_list = [x[0] for x in ncap_vec_list if x[2] >= min_cos]
    print(sorted(ncap_res_list))
    '''

        
    ################### mutiple ccap anchors (single +/- 1 register) ###################

    loop_index = -1
    if check_handedness:
        handedness = compute_handedness(pose)
        if handedness > 0:
            loop_index = 0
        else:
            loop_index = -1
    else:
        # cut first loop if no handedness checking
        loop_index = 0

    # ccap anchor
    # -3,-2,-1 registers of the 1st L residue in 1st loop
    ccap_resids = [x + loops[loop_index][0] - 3 for x in range(3)]
    #print(ccap_resids)
    ccap_vecs = [get_CA_sccom_vector(pose,x) for x in ccap_resids]

    # ncap anchors
    # For each ccap, select top (must be less than 9 residues far away and closest to the start of 2nd helix) residues whose sidechain (CA-<com of sc heavy atoms>) vectors
    # are at equal or less than 90 w/ the (CA-<com of sc heavy atoms>) vector of ccap residue, i.e. cos(angle) > 0
    ncap_vec_list = [[loops[loop_index][-1]+1+x,get_CA_sccom_vector(pose,loops[loop_index][-1]+1+x)] for x in range(ncap_res_num)]
    for i in range(len(ncap_vec_list)):
        ncap_vec_list[i].append([get_cos_angle_of_vecs(ccap_vecs[x],ncap_vec_list[i][1]) for x in range(len(ccap_vecs))])
    #for i in range(len(ccap_resids)):
    #    print('{}: {}'.format(ccap_resids[i],','.join([str(ncap_vec_list[x][0]) for x in range(len(ncap_vec_list)) if ncap_vec_list[x][2][i] >= min_cos ])))
    cut_pair_list = []
    for i in range(len(ccap_resids)):
        ncap_reslist = [str(ncap_vec_list[x][0]) for x in range(len(ncap_vec_list)) if ncap_vec_list[x][2][i] >= min_cos ]
        if len(ncap_reslist) > 0:
            cut_pair_list.append('{}:{}'.format(ccap_resids[i],','.join(ncap_reslist)))
    cut_pairs = ';'.join(cut_pair_list)

    # the following generate unreadable string when no ncap_res found for given ccap res
    #cut_pairs = ';'.join(['{}:{}'.format(ccap_resids[i],','.join([str(ncap_vec_list[x][0]) for x in range(len(ncap_vec_list)) if ncap_vec_list[x][2][i] >= min_cos ])) for i in range(len(ccap_resids))])

    return cut_pairs




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_fname', type=str,
                        help='input filename')
    parser.add_argument('-p', '--input_anchor_pos', type=int, default=-1,
                        help='anchor residue index for loop building') 
    parser.add_argument('-n', '--nstruct', type=int, default=1,
                        help='number of structures for output') 
    parser.add_argument('-b', '--before_frag_length', type=str, default='1',
                        help='numbers of residues before fragment separated by comma') 
    parser.add_argument('-a', '--after_frag_length', type=str, default='1',
                        help='numbers of residues after fragment separated by comma') 
    parser.add_argument('-d', '--output_dir', type=str, default='',
                        help='name of output directory') 

    parser.add_argument('--cut_start', type=str, default='-1', 
                        help='first residue(s) to be cut off for loop insertion')
    parser.add_argument('--cut_end', type=str, default='-1', 
                        help='last residue(s) to be cut off for loop insertion')
    parser.add_argument('--cut_pairs', type=str, default='-1', 
                        help='pairs of first and last residue(s) to be cut off for loop insertion')
    parser.add_argument('--find_cut_sites', action='store_true', 
                        help='if enabled, ignore cut_start, cut_end and cut_pairs, automatically search for cut sites')


    parser.add_argument('--num_repeats', type=int, default=-1, 
                        help='number of repeats, default=-1 (auto detection)')    

    parser.add_argument('-t', '--frag_angle_file', type=str, default='',
                        help='file/filelist containing phi/psi of fragment residues') 
    parser.add_argument('--helix_n_cap_angle_file', type=str, default='',
                        help='file/filelist containing phi/psi of helix N-terminal capping residues, i.e. C-terminus of loop') 
    parser.add_argument('--helix_c_cap_angle_file', type=str, default='',
                        help='file/filelist containing phi/psi of helix C-terminal capping residues, i.e. N-terminus of loop') 

    parser.add_argument('--param_file', type=str, default='-1', 
                        help='if set, read and sample combinations of parameters according to the distribution file')


    parser.add_argument('--ban_helix', type=int, default=-1,
                        help='prohibit helical fragments equal or longer than certain length (i.e. >= H*X in dssp))') 
    parser.add_argument('--loop_direction', type=float, default=-1, 
                        help='cos angle cutoff for loop direction, only loops with smaller value (i.e. larger angle) pass, default=-1 (not calculated), recommended value 0.95')
    parser.add_argument('--loop_hairpin_shape', type=str, default="-1", 
                        help='float cutoffs separated by commas for distances of residue pairs on at the beta turn. e.g. 4,6,7,8 (which is the best combination I found so far) \
                        indicates max dists between beta turn residues is 4, the next pair 6 and the farther pair at 7 and the farthest at 8...')
    parser.add_argument('--bturn_side', type=str, default='ncap', 
                        help='which cap is closer to the bturn fragment, the approximate bturn region will be skipped for contact measurement\
                        allowed inputs: ncap, ccap')
    parser.add_argument('--loop_helix_contact_cutoff', type=int, default=8, 
                        help='distance cutoff for computing loop-helix_cap contacts. recommended value: 8')    
    parser.add_argument('--loop_heix_contact_skip_resnum', type=int, default=5, 
                        help='skip this num of residues from bturn_side for contact measurement, recommended value 5, \
                        because 4 from bturn + 1 for extra residue sometimes added in genkic')    
    parser.add_argument('--loop_heix_contact_max_zero_contacts', type=int, default=-1, 
                        help='max num of subset residue alowed in loop to have 0 contacts w/ helix cap. activated when != -1, recommended value 4')    
    parser.add_argument('--loop_pca_subspan_resnum', type=int, default=4, 
                        help='compute span of only this number of residues starting from the bturn-side cap, recommended value 4')    
    parser.add_argument('--loop_pca_min_subspan_proj', type=int, default=-1, 
                        help='the span of projection of the subset residues on the top component from PCA should be larger than this cutoff. \
                        activated when != -1. recommended value 4')  


    parser.add_argument('--iter', type=int, default=100,
                        help='number of genkic iterations') 
    parser.add_argument('-e', '--perturb', type=float, default=0,
                        help='rama deviation allowed for perturbing the frag/capping residues') 
    parser.add_argument('--neighbor_flank', type=int, default=3,
                        help='flanking regions of loops+capping to be considered for relax/design') 
    parser.add_argument('--relax_repeats', type=int, default=1, 
                        help='number of repeats for fastrelax and fastdesign')
    parser.add_argument('--bbrmsd_cutoff', type=float, default=0.5, 
                        help='minimal bbrmsd allowed for any two poses in the posepool')
    parser.add_argument('--max_relax_rmsd_allowed', type=float, default=3.0, 
                        help='max whole-pose rmsd drift allowed during symmetric relaxation (for both backbone stage and design)')


    parser.add_argument('--min_intraloop_hbond_num', type=int, default=0, 
                        help='minimal number of hbond within each loop, checked before propagation, so might end up having less after relax')
    parser.add_argument('--min_intraloop_bbbb_hbond_num', type=int, default=0, 
                        help='minimal number of backbone-backbone hbond within each loop, checked before propagation, so might end up having less after relax')
    parser.add_argument('--min_interloop_hbond_num', type=int, default=0, 
                        help='minimal number of hbond between loops, checked after propgation and design')
    parser.add_argument('--min_interloop_bbbb_hbond_num', type=int, default=0, 
                        help='minimal number of backbone-backbone hbond between loops, checked after propgation and design')
    parser.add_argument('--min_potential_interloop_bbbb_hbond_num', type=int, default=0, 
                        help='minimal number of pseudo backbone-backbone hbond between loops, once satified, distance cst will be applied to pull the bond length')


    parser.add_argument('--min_num_bidentate', type=int, default=0, 
                        help='minimal number of sc-bb bidentate hbonds between loop and neighbor repeat (helix/loop) required')
    parser.add_argument('--bidentate_resn', type=str, default='N',
                        help='one letter names for amino acids used for bidentate searching')
    parser.add_argument('--min_num_pseudo_bidentate', type=int, default=0, 
                        help='minimal number of  pseudo sc-bb bidentate hbonds between loop and neighbor repeat (helix/loop) required')
    parser.add_argument('--pseudo_bidentate_anchor_sstype', type=str, default='all',
                        help='define the allowed second structural type of the bidentate anchor residue. supported types: all (default), helix') 
    parser.add_argument('--delta_HA', type=float, default=4.0, 
                        help='HA distance cutoff for pseudo hbond')
    parser.add_argument('--delta_theta', type=float, default=90.0, 
                        help='D-H-A angle cutoff for pseudo hbond')
    parser.add_argument('--disable_combine_pseudo_bidentate', action='store_true', 
                        help='if specified, disallows combine the discovered pseudo bidentate hbonds')
    parser.add_argument('--disable_relax', action='store_true', 
                        help='if specified, skip relaxation after propagation')

    parser.add_argument('--sheetlike_turns', action='store_true', 
                        help='if specified, check and filter for sheet-like beta hairpins, whose beta turns at hairpin tip form bb-bb hbonds in sheet-like manner')


    parser.add_argument('--min_loop_motif', type=int, default=0, 
                        help='minimum motif hits required for a new loop pose to be accepted')
    parser.add_argument('--motif_database', type=str, 
                        help='database for motifscore') 
    parser.add_argument('--max_motif_per_res', type=float, default=3.0, 
                        help='max motifscore allowed for each residue')
    parser.add_argument('--motif_dist_cutoff', type=float, default=10.0, 
                        help='max distance cutoff for residue pair searching during motifscore calculation')
    parser.add_argument('--disable_motif_packing', action='store_true', 
                        help='if set, skip motif packing (useful if design protocol runs after)')


    parser.add_argument('--design', type=int, default=0, 
                        help='number of design iterations to do, default=0 (no design)')
    parser.add_argument('--layer_design', action='store_true', 
                        help='if set, use layer selector and fastdesign')

    parser.add_argument('--aacomp_cap_pro_file', type=str, default='-1', 
                        help='aacomposition .comp file for proline bins in ncap of helices')
    parser.add_argument('--aacomp_design_file', type=str, default='-1', 
                        help='aacomposition .comp file for sequence design')
    parser.add_argument('--aacomp_design_pro_file', type=str, default='-1', 
                        help='aacomposition .comp file for controlling proline composition during sequence design')
    parser.add_argument('--aacomp_design_Ebin_file', type=str, default='-1', 
                        help='aacomposition .comp file for controlling Ebin composition during sequence design')
    parser.add_argument('--aacomp_design_Gbin_file', type=str, default='-1', 
                        help='aacomposition .comp file for controlling Gbin composition during sequence design')


    parser.add_argument('--worst_score_allowed', type=float, default=-1, 
                        help='worst scoring poses allowed to proceed for loop propagation/ouput with the score as a fraction of the best score, \
                        e.g. when set to0.6, the worst scoring poses has score <= 0.6*best_score; default=-1: not used when set to -1')
    parser.add_argument('--total_score_cutoff', type=float, default=0.0, 
                        help='highest total score allowed, \
                        default=0.0, i.e. all outputs have to have total score greater than 0')
    parser.add_argument('--checkpoint_frequency', type=int, default=300, 
                        help='number of iterations between actions of writing checkpoint info/pdb')
    parser.add_argument('--verbose', action='store_true', 
                        help='if enabled, turn off -mute all options')
    parser.add_argument('--debug', action='store_true', 
                        help='if enabled, print out all the debug messages')
    parser.add_argument('--info_only', action='store_true', 
                        help='if enabled, only print info file and do not store or output pdbs (might be useful for parameter scanning)')


    args = parser.parse_args()    

    mut_arg = '-mute all ' if not args.verbose else ''
    pyrosetta.init(options='{}-beta -optimization::default_max_cycles 200 -symmetry_definition stoopid'.format(mut_arg) +
       ' -old_sym_min true -rebuild_disulf false -detect_disulf false -nstruct {} -relax:default_repeats {}'.format(args.nstruct, args.relax_repeats) +
       ' -mh:path:scores_BB_BB {} -score:max_motif_per_res {}'.format(args.motif_database, args.max_motif_per_res), extra_options='')
    #pyrosetta.init(options='{}-beta'.format(mut_arg) +
    #' -rebuild_disulf false -detect_disulf false -nstruct {} -relax:default_repeats {}'.format(args.nstruct, args.relax_repeats), extra_options='')

    global _DEBUG 
    _DEBUG= args.debug

    inputpose = pyrosetta.pose_from_file(args.in_fname)

    #base_fname = args.in_fname.split('.')[0]
    base_fname = args.in_fname.split('/')[-1].replace('.pdb','')
    out_put_fname = '{dir}/{base}_{mdl_no:04}_L{LS}-{LE}.pdb'

    out_dir = '{}_genKIC'.format(base_fname) if args.output_dir == '' else args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #print('\n\n\n Running ab initio loop sampling \n\n\n')

    if args.param_file == '-1':
        cut_combination = []
        if args.find_cut_sites:
            cut_pairs = find_cap_anchors(inputpose, ncap_res_num = 9, min_cos = 0)
            print(cut_pairs)
            cut_combination = parse_cut_pairs(cut_pairs)
        elif args.cut_pairs != '-1':
            # example input:
            # 1:7,8;2:8,9;3,4:9,10
            cut_combination = parse_cut_pairs(args.cut_pairs)
        else:
            cut_start_list = str_to_int_list(args.cut_start)
            cut_end_list = str_to_int_list(args.cut_end)
            cut_combination = list(itertools.product(cut_start_list, cut_end_list))

        if len(cut_combination) > 0:
            if _DEBUG:
                print('cut_combination is set. input_anchor_pos will be overwritten!!')
        else:
            if args.input_anchor_pos != -1:
                if args.num_repeats == -1:
                    print('Error: must set num_repeats when using input_anchor_pos!!')
                    sys.exit(1)         
            else:   
                print('Error: must set one of read_param_from_file, input_anchor_pos and cut start/end/combination!!')
                sys.exit(1)

        before_frag_length_list = str_to_int_list(args.before_frag_length)
        after_frag_length_list = str_to_int_list(args.after_frag_length)
        phipsi_list = read_torsion_file(args.frag_angle_file)
        n_cap_phipsi_list = read_torsion_file(args.helix_n_cap_angle_file)
        c_cap_phipsi_list = read_torsion_file(args.helix_c_cap_angle_file)

    else:
        # overwrite cut_start/end if specified explicitly
        if args.cut_start != '-1' and args.cut_end != '-1':
            cut_start_list = str_to_int_list(args.cut_start)
            cut_end_list = str_to_int_list(args.cut_end)
            cut_combination = list(itertools.product(cut_start_list, cut_end_list))
        else:
            cut_combination = []
        before_frag_length_list = []
        after_frag_length_list = []
        phipsi_list = []
        n_cap_phipsi_list = []
        c_cap_phipsi_list = []



    build_loop(args.param_file, cut_combination, args.input_anchor_pos, before_frag_length_list, after_frag_length_list, phipsi_list, n_cap_phipsi_list, c_cap_phipsi_list, args.in_fname, base_fname, \
      out_dir, out_put_fname, num_repeats=args.num_repeats, neighbor_flank=args.neighbor_flank, perturb=args.perturb, genkic_trials=args.iter, aacomp_cap_pro_file=args.aacomp_cap_pro_file, \
        ban_helix=args.ban_helix, loop_direction=args.loop_direction, bturn_side=args.bturn_side, loop_helix_contact_cutoff=args.loop_helix_contact_cutoff, loop_heix_contact_skip_resnum=args.loop_heix_contact_skip_resnum, \
        loop_heix_contact_max_zero_contacts=args.loop_heix_contact_max_zero_contacts, loop_pca_subspan_resnum=args.loop_pca_subspan_resnum, loop_pca_min_subspan_proj=args.loop_pca_min_subspan_proj, \
        min_intraloop_hbond_num=args.min_intraloop_hbond_num, min_intraloop_bbbb_hbond_num=args.min_intraloop_bbbb_hbond_num, min_interloop_hbond_num=args.min_interloop_hbond_num, min_interloop_bbbb_hbond_num=args.min_interloop_bbbb_hbond_num, min_potential_interloop_bbbb_hbond_num=args.min_potential_interloop_bbbb_hbond_num, \
      bidentate_resn=args.bidentate_resn, min_num_bidentate=args.min_num_bidentate, min_num_pseudo_bidentate=args.min_num_pseudo_bidentate, pseudo_bidentate_anchor_sstype=args.pseudo_bidentate_anchor_sstype, \
      delta_HA=args.delta_HA, delta_theta=args.delta_theta, disable_combine_pseudo_bidentate=args.disable_combine_pseudo_bidentate, disable_relax=args.disable_relax, \
      bbrmsd_cutoff=args.bbrmsd_cutoff, max_relax_rmsd_allowed=args.max_relax_rmsd_allowed, motif_dist_cutoff=args.motif_dist_cutoff, worst_score_allowed=args.worst_score_allowed, design=args.design, layer_design=args.layer_design, aacomp_design_file=args.aacomp_design_file, \
      aacomp_design_pro_file=args.aacomp_design_pro_file, aacomp_design_Ebin_file=args.aacomp_design_Ebin_file, aacomp_design_Gbin_file=args.aacomp_design_Gbin_file, checkpoint_frequency=args.checkpoint_frequency, \
       sheetlike_turns=args.sheetlike_turns, min_loop_motif=args.min_loop_motif, disable_motif_packing=args.disable_motif_packing, total_score_cutoff=args.total_score_cutoff, info_only=args.info_only)




if __name__ == '__main__':
    main()
