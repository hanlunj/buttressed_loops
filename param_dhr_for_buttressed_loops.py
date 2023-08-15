import pyrosetta
import pyrosetta.toolbox.numpy_utils as np_utils
import numpy as np
import os
import sys
import argparse
import copy
from sklearn.cluster import KMeans, MiniBatchKMeans

import time
from param_dhr_for_buttressed_loops_helpers import *
    


def scan_new_geometric_param_list_for_param_poses(param_poses, param_list, param_name='',
                                   translation_vector=[0,0,0], rotation_axis=[0,0,0], radian=False):
    '''
        param_poses: list of ParamPose objects
    '''

    assert(param_name!='')
    assert(len(param_list)>0)
    param_list = sorted(list(set(param_list)))
    new_param_poses = []
    for this_input_param_pose in param_poses:
        
        for this_param in param_list:
        
            this_param_pose = ParamPose(this_input_param_pose.pose, copy.deepcopy(this_input_param_pose.params))

            this_param_pose.params[param_name] = this_param
            
            if list(rotation_axis) != [0,0,0]:
                this_param_pose.set_axis_vec_and_theta(np.array(rotation_axis), this_param, radian=radian)
                this_param_pose.rotate(rotate_by_R=False)
            
            if list(translation_vector) != [0,0,0]:
                translation_vector = np.array(translation_vector)
                trans_vec_norm = translation_vector / np.linalg.norm(translation_vector)
                this_param_pose.t = trans_vec_norm*this_param   
                this_param_pose.translate()      

            new_param_poses.append(this_param_pose)
        
    return new_param_poses    


def filter_param_poses_by_fa_rep(param_poses, sf_farep=None, cutoff=100):
    if sf_farep == None:
        sf_farep = pyrosetta.get_score_function()
        for st in sf_farep.get_nonzero_weighted_scoretypes():
            sf_farep.set_weight(st, 0)
        sf_farep.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 1)   
    new_param_poses = []
    for this_param_pose in param_poses:
        if sf_farep(this_param_pose.pose) <= cutoff:
            new_param_poses.append(this_param_pose)
    return new_param_poses


def filter_param_poses_by_ss_degree(param_poses, helices=[], worst_ss_degree_cutoff=1, best_ss_degree_cutoff=1,
                                    avg_ss_degree_cutoff=1, motifscore_cutoff=-0.01, mman_=None,
                                   ignore_terminal_helix=False, only_second_repeat=False, debug=False):
    if mman_ == None:
        mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance() 
    new_param_poses = []
    for this_param_pose in param_poses:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(this_param_pose.pose)
        dssp_obj.dssp_reduced()
        # pass helices to function may speed up
        if len(helices) == 0:
            helices = find_loopless_dhr_chains(this_param_pose.pose)
        if debug:
            print(this_param_pose.params)
            helices_ss_degree_debug = []
            for i, helix in enumerate(helices):
                ss_degree_list, ss_degree = compute_ss_degree_quick_n_dirty(this_param_pose.pose, helices, i, 
                                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, 
                                            mman_=mman_, debug=True) 
                #ss_degree_list, ss_degree = compute_ss_degree(this_param_pose.pose, helices, i, 
                #                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, 
                #                            mman_=mman_, debug=True) 
                print(i, ss_degree, ss_degree_list)
                helices_ss_degree_debug.append(ss_degree)    
            print(helices_ss_degree_debug)
        helices_ss_degree = []
        if only_second_repeat:
            for i in [2,3]:
                helices_ss_degree.append( compute_ss_degree_quick_n_dirty(this_param_pose.pose, helices, i, 
                                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) )
                #helices_ss_degree.append( compute_ss_degree(this_param_pose.pose, helices, i, 
                #                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) )
        elif ignore_terminal_helix:
            for i in range(1, len(helices)-1):
                helices_ss_degree.append( compute_ss_degree_quick_n_dirty(this_param_pose.pose, helices, i, 
                                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) )
                #helices_ss_degree.append( compute_ss_degree(this_param_pose.pose, helices, i, 
                #                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) )
        else:
            for i, helix in enumerate(helices):
                helices_ss_degree.append( compute_ss_degree_quick_n_dirty(this_param_pose.pose, helices, i, 
                                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) )  
                #helices_ss_degree.append( compute_ss_degree(this_param_pose.pose, helices, i, 
                #                        motifscore_cutoff=motifscore_cutoff, dssp_obj=dssp_obj, mman_=mman_) ) 
        worst_ss_degree = sorted(helices_ss_degree)[0]
        if worst_ss_degree < worst_ss_degree_cutoff:
            continue
        best_ss_degree = sorted(helices_ss_degree, reverse=True)[0]
        if best_ss_degree < best_ss_degree_cutoff:
            continue
        avg_ss_degree = sum(helices_ss_degree) * 1.0 / len(helices_ss_degree)
        if avg_ss_degree < avg_ss_degree_cutoff:
            continue
        new_param_poses.append(this_param_pose)
    return new_param_poses


def filter_param_poses_by_core_residue_percentage(param_poses, num_repeats=4, percentage_cutoff=0.28, 
                                                    core_cutoff=5.2, helix_only=False, debug=False):
    new_param_poses = []
    for this_param_pose in param_poses:
        if helix_only:
            dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(this_param_pose.pose)
            dssp_obj.dssp_reduced()
            dssp = dssp_obj.get_dssp_secstruct()
            reslist = [x+1 for x in range(len(dssp)) if dssp[x]=='H']        
        else:
            reslist = list(range(1, this_param_pose.pose.size()))
        # use 2nd repeat unit
        this_repeat_len = int(this_param_pose.pose.size()/num_repeats)
        r2_reslist = [x for x in reslist if x > this_repeat_len and x <= this_repeat_len*2]
        r2_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                                            ','.join([str(x) for x in r2_reslist]))
        # select layers
        layer_core_selector = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_core_selector.set_cutoffs(core_cutoff, 2.0) # core, surface, default: 5.2, 2.0
        layer_core_selector.set_layers(True, False, False) # core, boundary, surface
        layer_r2_core_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
                                                                    layer_core_selector, r2_selector)
        layer_r2_core_vec = layer_r2_core_selector.apply(this_param_pose.pose)
        layer_r2_core_reslist = [i+1 for i,j in enumerate(layer_r2_core_vec) if j]
        

        this_percent_core = len(layer_r2_core_reslist)*1.0/this_repeat_len

        if this_percent_core < percentage_cutoff:
            if debug:
                print('DEBUG:      failed percent_core filter: {}   minimum required: {}'.format(
                                                        this_percent_core, percentage_cutoff))
            continue
        new_param_poses.append(this_param_pose)
    return new_param_poses
        

def filter_param_poses_by_helical_cap_com_dist(param_poses, helices=[], dist_cutoff=15, num_cap_res=4):
    new_param_poses = []
    for this_param_pose in param_poses:
        # pass helices to function may speed up
        if len(helices) == 0:
            helices = find_loopless_dhr_chains(this_param_pose.pose)
        com_h1_ccap = np.array(get_com(this_param_pose.pose, 
                                       list(range(helices[0][1]-num_cap_res+1, helices[0][1]+1))))
        com_h2_ncap = np.array(get_com(this_param_pose.pose, 
                                       list(range(helices[1][0], helices[1][0]+num_cap_res))))
        d1 = np.linalg.norm(com_h2_ncap - com_h1_ccap)
        if d1 > dist_cutoff:
            continue
        com_h2_ccap = np.array(get_com(this_param_pose.pose, 
                                       list(range(helices[1][1]-num_cap_res+1, helices[1][1]+1))))
        com_h3_ncap = np.array(get_com(this_param_pose.pose, 
                                       list(range(helices[2][0], helices[2][0]+num_cap_res))))
        d2 = np.linalg.norm(com_h3_ncap - com_h2_ccap)
        if d2 > dist_cutoff:
            continue
        new_param_poses.append(this_param_pose)
    return new_param_poses




            
#
#  TODOS
#
#  post cap farep (maybe not for now --- will over filter due to cap motif clashes)
#
###################
#
#  Future TODOS
#
#  double check centroid residue generation (compare w/ remodel)
#  silent file output
#  update motifscore to Will's
#  speed up
#  multiprocessor for hyak
#
def build_param_dhrs(
    sf,
    sf_sym,
    num_repeats=4,
    residue_name3='ALA',
    
    h1_len=20, #    length of h1 in each repeat
    r_z1_list=list(np.arange(0,359.999,60)), #      'self-rotation degree' of h1 about the Z axis
    r_r1_list=[0], #       rotation degree of h1 about the radius line
    r_t1_list=[0], #       rotation degree of h1 about the tangent line
    t_x1_list=[30], #      (radius) translation of h1 along X axis

    h2_len=20, #    length of h2 in each repeat

    sample_h2=False, #      sample h2 instead of generate h2 parametrically
    h1h2_term_com_dist_range=[9.5, 10.5], # generate h2 by sampling conformations of h2 around h1
    h1h2_term_phi_range_degree = [70,110], # range of angle between h1_cterm_com->h2_nterm_com and h1
    h1h2_term_theta_range_degree = [-30, 30],     # rotation angle of h2 around h1, starts (when=0) on X axis
    h2_cterm_perturb_radius = 1.5,    # max distance of deviation of h2_cterm_com from initial position
    num_trial_h2_nterm = 10,     #   number of trials for placing nterm and cterm of h2
    num_trial_h2_cterm = 10,    #   number of trials for placing nterm and cterm of h2
    num_top_h2 = 5, # keep this number of h2 sampled for each h1 (through clustering)

    r_z2_list=list(np.arange(0,359.999,60)), #      'self-rotation degree' of h2 about the Z axis
    t_x2_list=[10.0], #      (radius of h2 from h1) translation along X axis of h2 from h1
    t_z2_list=[0], #    translation along Z axis of h2 from h1
    r_r2_list=[0], #      rotation degree of h2 about h1

    sample_r2=False, #      sample r2 instead of generate r2 parametrically
    r1h2_r2h1_term_com_dist_range=[9.5, 10.5], #[9.5, 10.5]
    r1h1_r2h1_term_com_dist_range=[9.5, 12.0],
    r1h2_r2h1_term_theta_range_degree = [0,359.99999], # rotation angle of r2h1 around r1h2, starts (when=0) on X axis
    num_trial_r2h1_nterm=10, # number of trials for placing nterm of r2h1, #10*num_trial_h2_nterm
    num_trial_r2h1_cterm=10, # number of trials for placing cterm of r2h1,  #10*num_trial_h2_cterm
    num_top_r2h1=3, # keep this number of h2 sampled for each h1 (through clustering)

    dist_r1h1_r2h1_nterm_list=[-1],
    handedness=-1,
    r_zr_list=[20], # (remodel's twist) rotation degree of r2 from r1 about Z axis
    r_hr_list=[0], #  (Kobe's repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis (r1's Y)
    t_zr_list=[0], #      (rise) translation of repeat2 (r2) from r1 along Z axis    
    

    farep_cutoff=100,
    
    motifscore_cutoff=-0.01,
    filter_h1h2_ss_degree=True,  # only keep repeat unit (h1h2, r1) if worst_ss_degree > 0
    filter_r1r2_ss_degree=True,  # only keep r1r2 repeat pair if worst_ss_degree > 0, best_ss_degree > 1
    worst_ss_degree_cutoff=1,
    best_ss_degree_cutoff=1,
    avg_ss_degree_cutoff=1,
    
    min_core_residue_percentage=0,
    core_residue_SCN_cutoff=5.2,
    
    max_helix_cap_com_dist=15,  # max dist allowed for helix caps to be connected by loops
    
    add_helix_capping_motif=False,
    inner_helix_trim_size=4,   # trim this number of residues from inner helix ncap site for long loops
    min_helix_height_diff=3.0,  
    max_helix_height_diff=7.0,
    min_cos_angle=0.85,
    min_cos_angle_scaffold=0.5,    

    output_params_file = 'params.dat',
    
    workdir='./',
    c_cap_phipsi_file='cap_files/angle_ccap_ank1_4aa_SKGA',
    n_cap_phipsi_file='cap_files/angle_ncap_ank1_4aa_RTPL',
    output_dir='DHR_scan',
    
    suppress_dhr_output = False,
    
    debug=False,
    ):
    '''
    parametric terms:
    
        h1_len:    length of helix1 (h1) in each repeat
        r_z1:      'self-rotation degree' of h1 about the Z axis
        r_r1:       rotation degree of h1 about the radius line (X axis)
        r_t1:       rotation degree of h1 about the tangent line (Y axis)
        t_x1:      (radius) translation of h1 along X axis
        
        h2_len:    length of h2 in each repeat
        r_z2:      'self-rotation degree' of h2 about the Z axis
                       (no r_r2, r_t2 as assuming h2 parallel to h1 at the beginning)
        t_x2:      (radius of h2 from h1) translation along X axis of h2 from h1
        r_r2:      rotation degree of h2 about h1
        
        r_zr:       (remodel's twist) rotation degree of r2 from r1 about Z axis
        r_hr:       (Kobe's repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis 
        t_zr:      (rise) translation of repeat2 (r2) from r1 along Z axis        
        
    Full parametric terms (not all implemented here):
        h1_len               length
        t_x1, t_y1, t_z1     translation
        r_x1, r_y1, r_z1     rotation
        h2_len
        t_x2, t_y2, t_z2
        r_x2, r_y2, r_z2     
        t_xr, t_yr, t_zr     between repeats
        r_xr, r_yr, r_zr

    '''
    
    time_start = time.time()

    sf_motif = sf.clone()
    for st in sf_motif.get_nonzero_weighted_scoretypes():
        sf_motif.set_weight(st, 0)
    sf_motif.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cen_pair_motifs, 1)

    sf_farep = sf.clone()
    for st in sf_farep.get_nonzero_weighted_scoretypes():
        sf_farep.set_weight(st, 0)
    sf_farep.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 1)    

    if workdir[-1] != '/':
        workdir += '/'
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    if output_dir[:5] == '/home':
        output_dir.replace(workdir, '')
    os.system('rm -rf {}{}'.format(workdir, output_dir))
    os.system('mkdir {}{}'.format(workdir, output_dir))
    #os.system('mkdir -p {}{}'.format(workdir, output_dir))

    mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
    
    chain_ids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    if debug:
        total_param_combs = 1
    
    #
    #  generate h1
    #    
    
    time_h1_start = time.time()
    
    this_h1_total_param_combs = {}
    
    # list of h1 parametric poses
    h1_list = []
    

    h1 = ParamHelix(helix_len_=h1_len, residue_name3_=residue_name3)
    h1.generate_pose()

    # flip to make ncap along Z axis
    h1.set_axis_vec_and_theta(np.array([1,0,0]), np.pi, radian=True)
    h1.update_R_by_axis_vec_and_theta()
    h1.rotate(rotate_by_R=True)

    #h1.pose.dump_pdb('{}{}/h1_parent.pdb'.format(workdir, output_dir))

    h1_list.append(ParamPose(template_pose_=h1.pose, params_={}))

    # r_z1:      'self-rotation degree' of h1 about the Z axis
    h1_list = scan_new_geometric_param_list_for_param_poses(h1_list, r_z1_list, param_name='r_z1',
                               rotation_axis=[0,0,1], radian=False)
    if debug:
        this_num_of_params = len(r_z1_list)
        if this_num_of_params >= 1:
            #print('r_z1: ', this_num_of_params)
            this_h1_total_param_combs['r_z1'] = this_num_of_params

    
    # r_r1:       rotation degree of h1 about the radius line (X axis) (-: ccw, +: cw)
    h1_list = scan_new_geometric_param_list_for_param_poses(h1_list, r_r1_list, param_name='r_r1',
                               rotation_axis=[1,0,0], radian=False)
    if debug:
        this_num_of_params = len(r_r1_list)
        if this_num_of_params >= 1:
            #print('r_r1: ', this_num_of_params)
            this_h1_total_param_combs['r_r1'] = this_num_of_params


    # r_t1:       rotation degree of h1 about the tangent line (Y axis) (-: ccw, +: cw)
    h1_list = scan_new_geometric_param_list_for_param_poses(h1_list, r_t1_list, param_name='r_t1',
                               rotation_axis=[0,1,0], radian=False)
    if debug:
        this_num_of_params = len(r_t1_list)
        if this_num_of_params >= 1:
            #print('r_t1: ', this_num_of_params)
            this_h1_total_param_combs['r_t1'] = this_num_of_params


    # t_x1:      (radius) translation of h1 along X axis 
    h1_list = scan_new_geometric_param_list_for_param_poses(h1_list, t_x1_list, param_name='t_x1',
                               translation_vector=[1,0,0])
    if debug:
        this_num_of_params = len(t_x1_list)
        if this_num_of_params >= 1:
            #print('t_x1: ', this_num_of_params)
            this_h1_total_param_combs['t_x1'] = this_num_of_params


    if debug:
        print(this_h1_total_param_combs)
        for this_param in this_h1_total_param_combs:
            total_param_combs *= this_h1_total_param_combs[this_param]
    if debug:
        print('num of h1: ', len(h1_list))
        for h1_id, this_h1 in enumerate(h1_list):
            print(this_h1.params)
        #    this_h1.pose.dump_pdb('{}{}/h1_{}.pdb'.format(workdir, output_dir, h1_id))

    
    time_h1_end = time.time()
    if debug:    
        print('time h1 gen: ', time_h1_end - time_h1_start)
    

    #
    #  generate h2 to form the repeat unit r1
    #
    
    time_h2_start = time.time()
    
    h2_dummy = ParamHelix(helix_len_=h2_len, residue_name3_=residue_name3)
    h2_dummy.generate_pose()

    this_h2_total_param_combs = {}


    r1_list = []
    for h1_id, this_h1 in enumerate(h1_list):

        this_h1_r1_list = []
        
        # record the transform from flipped h1 to current h1, use this to transform h2 to h1's position
        h1_dummy = ParamPose(h1.pose)
        h1_dummy.set_axis_vec_and_theta(np.array([0,0,1]), this_h1.params['r_z1'], radian=False)
        h1_dummy.rotate(rotate_by_R=False)
        h1_R, h1_t = compute_rigid_3D_transform_for_poses(h1_dummy.pose, list(range(1,this_h1.pose.size()+1)), 
                                         this_h1.pose, list(range(1, this_h1.pose.size()+1)),
                                         atomtype=['CA'])

        h1_cterm_com = get_com(this_h1.pose, list(range(this_h1.pose.size()-3,this_h1.pose.size()+1)))
        h1_nterm_com = get_com(this_h1.pose, list(range(1,4)))


        if sample_h2:
            # generate h2 by sampling conformations of h2 around h1
                
            # range of angle between h1_cterm_com->h2_nterm_com and h1
            # so that h2_nterm_com is a resonable position and not clashing into h1
            # more reference and plot here: http://corysimon.github.io/articles/uniformdistn-on-sphere/
            h1h2_term_phi_range = [np.radians(x) for x in h1h2_term_phi_range_degree]
            h1h2_term_phi_proj_range = []
            for this_phi in h1h2_term_phi_range:
                assert(this_phi<=np.pi)
                h1h2_term_phi_proj_range.append(np.cos(this_phi))
            h1h2_term_phi_proj_range = sorted(h1h2_term_phi_proj_range)

            # rotation angle of h2 around h1, starts (when=0) on X axis
            # more reference and plot here: http://corysimon.github.io/articles/uniformdistn-on-sphere/
            h1h2_term_theta_range = [np.radians(x) for x in h1h2_term_theta_range_degree]


            h2_start = ParamPose(h2_dummy.pose)
            h2_start.R = h1_R
            h2_start.rotate(rotate_by_R=True)
            h2_start.t = h1_t
            h2_start.translate()

            h2_start_nterm_com = get_com(h2_start.pose, list(range(1,4)))


            for trial_h2_nterm in range(num_trial_h2_nterm):

                h1h2_term_com_dist = np.random.sample() * (h1h2_term_com_dist_range[1]-h1h2_term_com_dist_range[0]) \
                                        + h1h2_term_com_dist_range[0]

                this_theta = h1h2_term_theta_range[0] + np.random.sample()*(h1h2_term_theta_range[1]-h1h2_term_theta_range[0])
                # sample on h1's Z axis instead on theta to get uniform spherical surface distribution
                this_phi = np.arccos( h1h2_term_phi_proj_range[0] + \
                                     np.random.sample()*(h1h2_term_phi_proj_range[1]-h1h2_term_phi_proj_range[0]) )
                x = np.cos(this_theta)*np.sin(this_phi)
                y = np.sin(this_theta)*np.sin(this_phi)
                z = np.cos(this_phi)
                
                # rotate the Z axis to h1 direction (so the sphere has h1 passing one of its diameters)
                h2_nterm_com = np.dot(h1_R, np.array([x,y,z]))
                    
                # 'normalized sphere surface point'*radius + sphere_origin_position
                h2_nterm_com_xyz = np.array([x,y,z])*h1h2_term_com_dist + h1_cterm_com
                h2_nterm_com = h2_nterm_com*h1h2_term_com_dist + h1_cterm_com


                '''
                if DEBUG:
                    print('h1h2_term_com_dist: ', h1h2_term_com_dist)
                    print('this_theta: ', np.degrees(this_theta))
                    print('this_phi: ', np.degrees(this_phi))
                    print('x, y, z: ', x, y, z)
                    print(_pml_cmd_for_com(h2_nterm_com, 'h2_nterm_com'))
                    print(_pml_cmd_for_com(h2_nterm_com_xyz, 'h2_nterm_com_xyz'))
                '''
                
                # build h2, sample h2 orientation w/ const until output accepted
                for trial_h2_cterm in range(num_trial_h2_cterm):      
                    
                    h2 = ParamPose(h2_start.pose, copy.deepcopy(h2_start.params))

                    h2.params['h1h2_term_com_dist'] = h1h2_term_com_dist
                    h2.params['h2_nterm_theta'] = np.degrees(this_theta)
                    h2.params['h2_nterm_phi'] = np.degrees(this_phi)
                    
                    # superimposing h1clone's Nterm COM on h2_nterm_com
                    #    in this way, h2 initial orientation is parallel to h1
                    #    we can perturb the orientation further on from here
                    h2.t = h2_nterm_com - h2_start_nterm_com
                    h2.translate()
                    #
                    # perturb h2 orientation
                    #        use h2_nterm_com as origin, sample a cone by h2_cterm_com
                    #        param: deviation dist of h2_cterm_com from initial place (e.g. max 1A)
                    #
                    #  w/ fixed h2_nterm_com, sampling h2_cterm_com becomes another sphere surface sampling task
                    #       the new phi is [0, arcsin(h2_cterm_perturb_radius/2/dist_h2_nc_com)*2]
                    #
                    h2_cterm_com = get_com(h2.pose, list(range(h2.pose.size()-3,h2.pose.size()+1)))
                    h2_nc_dist = np.linalg.norm(h2_cterm_com - h2_nterm_com)
                    h2_cterm_sphere_point, this_cterm_theta, this_cterm_phi = sample_point_on_sphere_surface([0,2*np.pi], 
                                                                  [0, 2*np.arcsin(h2_cterm_perturb_radius/2/h2_nc_dist)])

                    h2.params['h2_cterm_theta'] = np.degrees(this_cterm_theta)
                    h2.params['h2_cterm_phi'] = np.degrees(this_cterm_phi)

                    # rotate h2
                    h2.R = rotation_matrix_from_vectors(np.array([0,0,1]), h2_cterm_sphere_point)
                    h2.t = -1*h2_nterm_com
                    h2.translate()
                    h2.rotate(rotate_by_R=True)
                    h2.t = h2_nterm_com
                    h2.translate()
                    
                    #  self rotate h2 along it's own Z axis (via r_z2)
                    h2_z_vec = h2_cterm_sphere_point
                    this_h2_list = [h2]
                    this_h2_list = scan_new_geometric_param_list_for_param_poses(this_h2_list, r_z2_list, param_name='r_z2',
                                               rotation_axis=h2_z_vec, radian=False)
                    if debug:
                        this_num_of_params = len(r_z2_list)
                        if this_num_of_params >= 1:
                            #print('r_z2: ', this_num_of_params)
                            this_h2_total_param_combs['r_z2'] = this_num_of_params

                    this_h2_r1_list = []
                    for this_h2 in this_h2_list:
                        r1 = ParamPose(this_h1.pose)
                        r1.pose.append_pose_by_jump(this_h2.pose, r1.pose.size())
                        r1.params = {**this_h1.params, **this_h2.params}
                        this_h2_r1_list.append(r1)

                    # filter r1 by farep
                    this_h2_r1_list = filter_param_poses_by_fa_rep(this_h2_r1_list, sf_farep, farep_cutoff)
                    # filter r1 by ss_degree
                    if filter_h1h2_ss_degree:
                        this_h2_r1_list = filter_param_poses_by_ss_degree(this_h2_r1_list, helices=[[1, h1_len],[h1_len+1, h1_len+h2_len]],
                                                worst_ss_degree_cutoff=1, motifscore_cutoff=motifscore_cutoff, mman_=mman_)

                    this_h1_r1_list += this_h2_r1_list


            # cluster r1 (actually h2) and save only num_top_h2 objs
            if len(this_h1_r1_list) > num_top_h2:
                if debug:
                    print('clustering {} h2 into {} groups for output ...'.format(len(this_h1_r1_list), num_top_h2))
                coords = []
                for r1 in this_h1_r1_list:
                    # just nc_term_com of h2 should be enough
                    this_nterm_com = get_com(r1.pose, list(range(h1_len+1,h1_len+4)))
                    this_cterm_com = get_com(r1.pose, list(range(r1.pose.size()-2,r1.pose.size()+1)))
                    coords.append(np.concatenate((this_nterm_com,this_cterm_com)))
                coords = np.array(coords)
                # use minibatch if large sample size (numbers to be optimized!)
                if len(coords) > 100:
                    kmeans = MiniBatchKMeans(n_clusters=num_top_h2, batch_size=50).fit(coords)
                else:
                    kmeans = KMeans(n_clusters=num_top_h2).fit(coords)
                # select 1st member from each cluster (TODO: maybe ranked by motifscore?)
                if debug:
                    print('kmeans.labels_:  ', kmeans.labels_)
                clusters = {x:[] for x in kmeans.labels_}
                for i, label_ in enumerate(kmeans.labels_):
                    clusters[label_].append(this_h1_r1_list[i])
                    #if DEBUG:
                    #    h2_list[i].pose.dump_pdb(workdir+'h2_{}_{}.pdb'.format(label_, i))
                clustered_this_h1_r1_list = [clusters[x][0] for x in list(clusters.keys())]
                this_h1_r1_list = clustered_this_h1_r1_list
            r1_list += this_h1_r1_list
                    


        else:
            # generate h2 parametrically
            h2 = ParamHelix(helix_len_=h2_len, residue_name3_=residue_name3)
            h2.generate_pose()

            # list of h2 parametric poses
            h2_list = [ParamPose(template_pose_=h2.pose, params_={})]

            # r_z2:      'self-rotation degree' of h2 about the Z axis
            h2_list = scan_new_geometric_param_list_for_param_poses(h2_list, r_z2_list, param_name='r_z2',
                                       rotation_axis=[0,0,1], radian=False)
            if debug:
                this_num_of_params = len(r_z2_list)
                if this_num_of_params >= 1:
                    #print('r_z2: ', this_num_of_params)
                    this_h2_total_param_combs['r_z2'] = this_num_of_params


            # t_x2:      (radius of h2 from h1) translation along new 'X axis' of h2 from h1
            #            so the distance between h1 and h2 remains the same
            h2_list = scan_new_geometric_param_list_for_param_poses(h2_list, t_x2_list, param_name='t_x2',
                                       translation_vector=[1,0,0])
            if debug:
                this_num_of_params = len(t_x2_list)
                if this_num_of_params >= 1:
                    #print('t_x2: ', this_num_of_params)
                    this_h2_total_param_combs['t_x2'] = this_num_of_params


            # t_z2:     translation along Z axis of h2 from h1
            h2_list = scan_new_geometric_param_list_for_param_poses(h2_list, t_z2_list, param_name='t_z2',
                                       translation_vector=[0,0,1])
            if debug:
                this_num_of_params = len(t_z2_list)
                if this_num_of_params >= 1:
                    #print('t_z2: ', this_num_of_params)
                    this_h2_total_param_combs['t_z2'] = this_num_of_params

            
            # r_r2:      rotation degree of h2 about h1
            h2_list = scan_new_geometric_param_list_for_param_poses(h2_list, r_r2_list, param_name='r_r2',
                                       rotation_axis=[0,0,1], radian=False)
            if debug:
                this_num_of_params = len(r_r2_list)
                if this_num_of_params >= 1:
                    #print('r_r2: ', this_num_of_params)
                    this_h2_total_param_combs['r_r2'] = this_num_of_params


            # now place h2 around h1 by transforming h2 the same way of initial h1 to current h1 (but w/o Z-axis flipping)
            # get t annd R for flipped initial h1 to current h1, use them to position h2
            for h2 in h2_list:
                h2.R = h1_R
                h2.rotate(rotate_by_R=True)
                h2.t = h1_t
                h2.translate()


            #
            #  generate r1
            #

            for h2 in h2_list:
                r1 = ParamPose(this_h1.pose)
                r1.pose.append_pose_by_jump(h2.pose, r1.pose.size())
                r1.params = {**this_h1.params, **h2.params}
                this_h1_r1_list.append(r1)

            # filter r1 by farep
            this_h1_r1_list = filter_param_poses_by_fa_rep(this_h1_r1_list, sf_farep, farep_cutoff)
            # filter r1 by ss_degree
            if filter_h1h2_ss_degree:
                this_h1_r1_list = filter_param_poses_by_ss_degree(this_h1_r1_list, helices=[[1, h1_len],[h1_len+1, h1_len+h2_len]],
                                        worst_ss_degree_cutoff=1, motifscore_cutoff=motifscore_cutoff, mman_=mman_)
            r1_list += this_h1_r1_list


    if debug:
        print(this_h2_total_param_combs)
        for this_param in this_h2_total_param_combs:
            total_param_combs *= this_h2_total_param_combs[this_param]

    
    if debug:
        print('num of r1s: ', len(r1_list))
        #for r1_id, r1 in enumerate(r1_list):
        #    r1.pose.dump_pdb('{}{}/r1_{}.pdb'.format(workdir, output_dir, r1_id))

    
    time_h2_end = time.time()
    if debug:
        print('time h2 gen: ', time_h2_end - time_h2_start)    




    #
    #  generate 2nd repeat unit
    #
    
    time_r2_start = time.time()
    
    this_r1r2_total_param_combs = {}
    
    r1r2_list = []


    for r1 in r1_list:

        this_r1_r1r2_list = []

        if debug:
            print('r1.params: ')
            print(r1.params)


        if sample_r2:
            # generate r2 by sampling conformations of r2h1 around r1

            # range of angle between r1h2_cterm_com->r2h1_nterm_com and r1h2
            #   this should in general be the same as h1h2_term_phi_range_degree
            r1h2_r2h1_term_phi_range_degree = h1h2_term_phi_range_degree
            r1h2_r2h1_term_phi_range = [np.radians(x) for x in r1h2_r2h1_term_phi_range_degree]
            r1h2_r2h1_term_phi_proj_range = []
            for this_phi in r1h2_r2h1_term_phi_range:
                assert(this_phi<=np.pi)
                r1h2_r2h1_term_phi_proj_range.append(np.cos(this_phi))
            r1h2_r2h1_term_phi_proj_range = sorted(r1h2_r2h1_term_phi_proj_range)

            # rotation angle of r2h1 around r1h2, starts (when=0) on X axis
            r1h2_r2h1_term_theta_range = [np.radians(x) for x in r1h2_r2h1_term_theta_range_degree]

            # max distance of deviation of r2h1_cterm_com from initial position
            #    this should in general be the same as h2_cterm_perturb_radius
            r2h1_cterm_perturb_radius = h2_cterm_perturb_radius


            r1h1_nterm_com = get_com(r1.pose, list(range(1,4)))
            r1h2_cterm_com = get_com(r1.pose, list(range(r1.pose.size()-3,r1.pose.size()+1)))
            '''
            if debug:
                print('r1h1_nterm_com: ', r1h1_nterm_com)
                print('r1h2_cterm_com: ', r1h2_cterm_com)
                print('h1h2_com_dist: ', np.linalg.norm(r1h2_cterm_com-r1h1_nterm_com))
            '''
            # get r1h2_R to align the sphere to r1h2
            _reslist_for_xform = list(range(1, h2_dummy.pose.size()+1))
            r1h2_R, r1h2_t = compute_rigid_3D_transform_for_poses(h2_dummy.pose, _reslist_for_xform, 
                                                     r1.pose, [x+h1_len for x in _reslist_for_xform],
                                                     atomtype=['CA'])

            # create r2h1 obj, move it to r1h2 position 
            r2h1_start = ParamPose(h1.pose)

            # move to r1h2 position
            r2h1_start.R = r1h2_R
            r2h1_start.rotate(rotate_by_R=True)
            r2h1_start.t = r1h2_t
            r2h1_start.translate()

            r2h1_start_nterm_com = get_com(r2h1_start.pose, list(range(1,4)))

            for trial_r2h1_nterm in range(num_trial_r2h1_nterm):

                r1h2_r2h1_term_com_dist = np.random.sample() * (r1h2_r2h1_term_com_dist_range[1]-r1h2_r2h1_term_com_dist_range[0]) \
                                        + r1h2_r2h1_term_com_dist_range[0]

                this_theta = r1h2_r2h1_term_theta_range[0] + np.random.sample()*(r1h2_r2h1_term_theta_range[1]-r1h2_r2h1_term_theta_range[0])
                # sample on r1h2's Z axis instead on theta to get uniform spherical surface distribution
                this_phi = np.arccos( r1h2_r2h1_term_phi_proj_range[0] + \
                                     np.random.sample()*(r1h2_r2h1_term_phi_proj_range[1]-r1h2_r2h1_term_phi_proj_range[0]) )
                x = np.cos(this_theta)*np.sin(this_phi)
                y = np.sin(this_theta)*np.sin(this_phi)
                z = np.cos(this_phi)

                # rotate the Z axis to r1h2 direction (so the sphere has r1h2 passing one of its diameters)
                r2h1_nterm_com = np.dot(r1h2_R, np.array([x,y,z]))

                # 'normalized sphere surface point'*radius + sphere_origin_position
                r2h1_nterm_com_xyz = np.array([x,y,z])*r1h2_r2h1_term_com_dist + r1h2_cterm_com
                r2h1_nterm_com = r2h1_nterm_com*r1h2_r2h1_term_com_dist + r1h2_cterm_com

                # check r2h1_nterm_com to r1h1_nterm_com distance
                # TODO: enumerate the point on the overlapping circle instead of sampling
                dist_r2h1_r1h1 = np.linalg.norm(r2h1_nterm_com-r1h1_nterm_com)
                if dist_r2h1_r1h1 < r1h1_r2h1_term_com_dist_range[0] or \
                dist_r2h1_r1h1 > r1h1_r2h1_term_com_dist_range[1]:
                    continue


                if debug:
                    print('dist_r2h1_r1h1: ', dist_r2h1_r1h1)

                    '''
                    print('r1h2_r2h1_term_com_dist: ', r1h2_r2h1_term_com_dist)
                    print('this_theta: ', np.degrees(this_theta))
                    print('this_phi: ', np.degrees(this_phi))
                    print('x, y, z: ', x, y, z)
                    print(_pml_cmd_for_com(r2h1_nterm_com, 'r2h1_nterm_com'))
                    print(_pml_cmd_for_com(r2h1_nterm_com_xyz, 'r2h1_nterm_com_xyz'))
                    '''

                # build r2h1, sample r2h1 orientation w/ const until output accepted
                for trial_r2h1_cterm in range(num_trial_r2h1_cterm):

                    # superimposing r2h1_nterm_com (similar to h2)    
                    r2h1 = ParamPose(r2h1_start.pose)
                    r2h1.params['r1h2_r2h1_term_com_dist'] = r1h2_r2h1_term_com_dist
                    r2h1.params['r2h1_nterm_theta'] = np.degrees(this_theta)
                    r2h1.params['r2h1_nterm_phi'] = np.degrees(this_phi)

                    r2h1.t = r2h1_nterm_com -r2h1_start_nterm_com
                    r2h1.translate()  

                    # sample r2h1_cterm_com based on r2h1_cterm_perturb_radius
                    # perturb r2h1 orientation (similar to h2)               
                    r2h1_cterm_com = get_com(r2h1.pose, list(range(r2h1.pose.size()-3,r2h1.pose.size()+1)))
                    r2h1_nc_dist = np.linalg.norm(r2h1_cterm_com - r2h1_nterm_com)
                    r2h1_cterm_sphere_point, this_cterm_theta, this_cterm_phi = sample_point_on_sphere_surface([0,2*np.pi], 
                                                                  [0, 2*np.arcsin(r2h1_cterm_perturb_radius/2/r2h1_nc_dist)])

                    r2h1.params['r2h1_cterm_theta'] = np.degrees(this_cterm_theta)
                    r2h1.params['r2h1_cterm_phi'] = np.degrees(this_cterm_phi)

                    # rotate r2h1
                    r2h1.R = rotation_matrix_from_vectors(np.array([0,0,1]), r2h1_cterm_sphere_point)
                    r2h1.t = -1*r2h1_nterm_com
                    r2h1.translate()
                    r2h1.rotate(rotate_by_R=True)
                    r2h1.t = r2h1_nterm_com
                    r2h1.translate()

                    r1r2 = ParamPose(r1.pose)
                    r1r2.pose.append_pose_by_jump(r2h1.pose, r1r2.pose.size())
                    r1r2.params = {**r1.params, **r2h1.params}
                    # propagate to generate r2h2
                    r1r2.pose = poorman_repeat_propagate(r1r2.pose, h1_len+h2_len, num_repeat=2)
                    this_r2_r1r2_list = [r1r2]

                    # filter by farep
                    this_r2_r1r2_list = filter_param_poses_by_fa_rep(this_r2_r1r2_list, sf_farep, farep_cutoff)
                    # filter by ss_degree
                    if filter_r1r2_ss_degree:
                        r1_len = h1_len + h2_len
                        r1r2_helices = [[1, h1_len], [h1_len+1, h1_len+h2_len],
                                        [r1_len+1, r1_len+h1_len], [r1_len+h1_len+1, r1_len*2]]
                        this_r2_r1r2_list = filter_param_poses_by_ss_degree(this_r2_r1r2_list, helices=r1r2_helices,
                                                                worst_ss_degree_cutoff=1, best_ss_degree_cutoff=2,
                                                motifscore_cutoff=motifscore_cutoff, mman_=mman_) #, debug=True)
                    #  filter out the poses whose helices' ends are too far 
                    this_r2_r1r2_list = filter_param_poses_by_helical_cap_com_dist(this_r2_r1r2_list, dist_cutoff=max_helix_cap_com_dist)

                    this_r1_r1r2_list += this_r2_r1r2_list # one element list


            # TODO: cluster based on r2 instead of just r2h1
            # cluster r2h1 (equivalent to r2 and r1r2): 
            #    if len(r1r2_list) < num_top_r2h1: output all
            #    else: do k=num_top_r2h1 kmeans
            if len(this_r1_r1r2_list) > num_top_r2h1:
                if debug:
                    print('clustering {} r2h1 into {} groups for output ...'.format(len(this_r1_r1r2_list), num_top_r2h1))
                coords = []      

                for r1r2 in this_r1_r1r2_list:
                    # just nc_term_com should be enough
                    this_nterm_com = get_com(r1r2.pose, [x+h1_len+h2_len for x in range(1,4)])
                    this_cterm_com = get_com(r1r2.pose, [x+h1_len+h2_len+h1_len for x in range(-2,1)])
                    coords.append(np.concatenate((this_nterm_com,this_cterm_com)))
                coords = np.array(coords)
                # use minibatch if large sample size (numbers to be optimized!)
                if len(coords) > 100:
                    kmeans = MiniBatchKMeans(n_clusters=num_top_r2h1, batch_size=50).fit(coords)
                else:
                    kmeans = KMeans(n_clusters=num_top_r2h1).fit(coords)
                # select 1st member from each cluster (TODO: maybe ranked by motifscore?)
                if debug:
                    print('kmeans.labels_:  ', kmeans.labels_)
                clusters = {x:[] for x in kmeans.labels_}
                for i, label_ in enumerate(kmeans.labels_):
                    clusters[label_].append(this_r1_r1r2_list[i])
                    #if DEBUG:
                    #    this_r1_r1r2_list[i].pose.dump_pdb(workdir+'r2h1_{}_{}_{}.pdb'.format(r1_id, label_, i))
                clustered_this_r1_r1r2_list = [clusters[x][0] for x in list(clusters.keys())]
                this_r1_r1r2_list = clustered_this_r1_r1r2_list

            #if debug:
            #    for r1r2_i, r1r2 in enumerate(this_r1_r1r2_list):
            #        r1r2.pose.dump_pdb(workdir+'r1r2_{}_{}.pdb'.format(r1_id, r1r2_i))

            r1r2_list += this_r1_r1r2_list


        else:            
            # generate r2 parametrically

            # list of 2nd parametric repeat unit
            r2_list = [ParamPose(r1.pose, copy.deepcopy(r1.params))]


            # r_zr:   (remodel's twist) rotation degree of r2 from r1 about Z axis
            # BUT overwrite r_zr if dist_r1h1_r2h1_nterm_list specified
            if dist_r1h1_r2h1_nterm_list != [-1]:
                r_zr_list = []
                for this_dist in dist_r1h1_r2h1_nterm_list:
                    # assuming dist_r1h1_r2h1_nterm measure COM dist between r1h1_nterm and r2h1_nterm
                    r1h1_nterm_com = get_com(r1.pose, list(range(1,4)))
                    this_t_x1 = np.linalg.norm(r1h1_nterm_com[:2]) # dist of r1h1_nterm_com to Z axis
                    # assuming dist_r1h1_r2h1_nterm measures COM dist between r1h1 and r2h1
                    #this_t_x1 = r1.params['t_x1']
                    r_zr_list.append(handedness * np.degrees( 2 * np.arcsin( 0.5 * this_dist / this_t_x1 ) ))
                r_zr_list = sorted(r_zr_list)

            r2_list = scan_new_geometric_param_list_for_param_poses(r2_list, r_zr_list, param_name='r_zr',
                                       rotation_axis=[0,0,1], radian=False)
            if dist_r1h1_r2h1_nterm_list != [-1]:
                for r2_i, r2 in enumerate(r2_list):
                    r2.params['d12'] = dist_r1h1_r2h1_nterm_list[r2_i%len(dist_r1h1_r2h1_nterm_list)]
            if debug:
                this_num_of_params = len(r_zr_list)
                if this_num_of_params >= 1:
                    #print('r_zr: ', this_num_of_params)
                    this_r1r2_total_param_combs['r_zr'] = this_num_of_params


            # r_hr:   (Kobe's repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis (r1's Y axis)
            # move r2  to origin, rotate, then move it back
            r2_com_list = []
            for r2 in r2_list:
                r2_com = get_com(r2.pose)
                r2_com_list.append(r2_com)
                r2.t = -1*r2_com
                r2.translate()

            r2_list = scan_new_geometric_param_list_for_param_poses(r2_list, r_hr_list, param_name='r_hr',
                                       rotation_axis=[0,1,0], radian=False)
            if debug:
                this_num_of_params = len(r_hr_list)
                if this_num_of_params >= 1:
                    #print('r_hr: ', this_num_of_params)
                    this_r1r2_total_param_combs['r_hr'] = this_num_of_params

            for r2_id, r2 in enumerate(r2_list):
                r2.t = r2_com_list[r2_id]
                r2.translate()


            # t_zr:   (rise) translation of repeat2 (r2) from r1 along Z axis 
            r2_list = scan_new_geometric_param_list_for_param_poses(r2_list, t_zr_list, param_name='t_zr',
                                       translation_vector=[0,0,1])
            if debug:
                this_num_of_params = len(t_zr_list)
                if this_num_of_params >= 1:
                    #print('t_zr: ', this_num_of_params)
                    this_r1r2_total_param_combs['t_zr'] = this_num_of_params


            # merge r1 and r2 together
            for r2 in r2_list:
                r1r2 = ParamPose(r1.pose)
                r1r2.pose.append_pose_by_jump(r2.pose, r1r2.pose.size())
                r1r2.params = {**r1.params, **r2.params}
                this_r1_r1r2_list.append(r1r2)

            # filter by farep
            this_r1_r1r2_list = filter_param_poses_by_fa_rep(this_r1_r1r2_list, sf_farep, farep_cutoff)
            # filter by ss_degree
            if filter_r1r2_ss_degree:
                r1_len = h1_len + h2_len
                r1r2_helices = [[1, h1_len], [h1_len+1, h1_len+h2_len],
                                [r1_len+1, r1_len+h1_len], [r1_len+h1_len+1, r1_len*2]]
                this_r1_r1r2_list = filter_param_poses_by_ss_degree(this_r1_r1r2_list, helices=r1r2_helices,
                                                        worst_ss_degree_cutoff=1, best_ss_degree_cutoff=2,
                                        motifscore_cutoff=motifscore_cutoff, mman_=mman_) #, debug=True)
            #  filter out the poses whose helices' ends are too far 
            this_r1_r1r2_list = filter_param_poses_by_helical_cap_com_dist(this_r1_r1r2_list, dist_cutoff=max_helix_cap_com_dist)

            r1r2_list += this_r1_r1r2_list



    if debug:
        print(this_r1r2_total_param_combs)
        for this_param in this_r1r2_total_param_combs:
            total_param_combs *= this_r1r2_total_param_combs[this_param]

    
    if debug:
        print('num of r1r2s: ', len(r1r2_list))
        #for i, this_r1r2 in enumerate(r1r2_list):
        #    print(i, this_r1r2.params)
        #    this_r1r2.pose.dump_pdb('{}{}/HJDHR_{}.pdb'.format(workdir, output_dir, i))
    
    time_r2_end = time.time()
    if debug:
        print('time r2 gen: ', time_r2_end - time_r2_start)


    #
    #  propagate
    #
    
    time_prop_start = time.time()
    
    param_dhrs = []
    for this_r1r2 in r1r2_list:
        this_pose = poorman_repeat_propagate(this_r1r2.pose, h1_len+h2_len, num_repeat=num_repeats)
        this_param = this_r1r2.params
        param_dhrs.append(ParamPose(this_pose, copy.deepcopy(this_param)))
        
    # filter by ss_degree
    #    only filter on the sandwiched 2nd repeat so ank-like dhr can also pass the filter
    helices = []
    r1_len = h1_len + h2_len
    for i in range(num_repeats):
        helices.append([r1_len*i+1, r1_len*i+h1_len])
        helices.append([r1_len*i+h1_len+1, r1_len*(i+1)])
    param_dhrs = filter_param_poses_by_ss_degree(param_dhrs, helices=helices,
                    worst_ss_degree_cutoff=worst_ss_degree_cutoff, best_ss_degree_cutoff=best_ss_degree_cutoff,
                     avg_ss_degree_cutoff=avg_ss_degree_cutoff, motifscore_cutoff=motifscore_cutoff, 
                    mman_=mman_, only_second_repeat=True) # debug=True
    
    # filter by core residue percentage
    param_dhrs = filter_param_poses_by_core_residue_percentage(param_dhrs, num_repeats=num_repeats, 
                                                        percentage_cutoff=min_core_residue_percentage, 
                                                        core_cutoff=core_residue_SCN_cutoff, helix_only=False)
                                                              #debug=True)


    if debug:
        print('num of param dhrs: ', len(param_dhrs))
    
    time_prop_end = time.time()
    if debug:
        print('time propagate and filter: ', time_prop_end - time_prop_start)


    if len(param_dhrs) == 0:
        print('WARNING: no more param_dhr poses left after filtering ... done running this job ...') 
        return None

  

    #
    #  add helix capping motifs
    #
    
    if add_helix_capping_motif:
        
        time_cap_start = time.time()

        # trim inner row ncap side to make way for loops
        repeat_len = h1_len + h2_len
        new_repeat_len = repeat_len - inner_helix_trim_size
        capped_param_dhrs = []
        capped_param_dhrs_base = []
        
        fout_param_cap_base = open('{}{}/{}_capped_base.dat'.format(workdir, output_dir, 
                                               output_params_file.replace('.dat','')),'w')
        fout_param_cap_base.write('description,{}\n'.format(','.join(
                                            sorted(param_dhrs[0].params.keys()))))

        base_count = 0

        fout_param_cap = open('{}{}/{}_capped.dat'.format(workdir, output_dir, 
                                               output_params_file.replace('.dat','')),'w')
        fout_param_cap.write('description,{}\n'.format(','.join(
                                            sorted(list(param_dhrs[0].params.keys())+['repeat_len']) )))

    
        for this_dhr in param_dhrs:
            this_dhr_base_pose = this_dhr.pose.clone()
            this_dhr_base_pose.delete_residue_range_slow(repeat_len+1, repeat_len+inner_helix_trim_size)
            this_dhr_base_pose.delete_residue_range_slow(1, inner_helix_trim_size)
            this_dhr_base_pose = poorman_repeat_propagate(this_dhr_base_pose, new_repeat_len, 
                                                                  num_repeat=num_repeats)
            
            # helix_capped_list
            #   return: [pose_ncap, pose_ccap_list[c_id][1], pose_ccap_list[c_id][2], cos_angle, i]
            helix_capped_list = add_caps_n_filter_direction( this_dhr_base_pose,
                                    sf,
                                    sf_farep,
                                    farep_cutoff,
                                      c_cap_phipsi_file,
                                      n_cap_phipsi_file,
                                    h1_len=h1_len-inner_helix_trim_size, 
                                    h2_len=h2_len,
                                    cap_search_range=4,
                                    num_res_for_helix_center_com=4,
                                    min_cos_angle=min_cos_angle,
                                    min_helix_height_diff=min_helix_height_diff,
                                    max_helix_height_diff=max_helix_height_diff)
            # helix_capped_list returned list
            #.   [ [pose, ccap_cos, ccap_ind, ncap_cos, ncap_ind, new_repeat_len] ]
            #print(len(helix_capped_list))
            
            '''
            # filter capped by farep
            filtered_helix_capped_list = []
            for capped in helix_capped_list:
                if sf_farep(capped.pose) < farep_cutoff:
                    filtered_helix_capped_list.append(capped)
            helix_capped_list = filtered_helix_capped_list
            '''
            
            if len(helix_capped_list) > 0:
               
                if not suppress_dhr_output:
                    # TODO: (improve this)add param info to reslabel (as no se what else way 
                    #                                        to write info into pdb...)
                    this_dhr.pose.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(this_dhr.pose.size()))
                    for j, this_p in enumerate(sorted(this_dhr.params.keys())):
                        this_dhr.pose.pdb_info().add_reslabel(j+1, '{}:{}'.format(this_p, 
                                                                            this_dhr.params[this_p]))
                    this_dhr.pose.dump_pdb('{}{}/HJDHR_{}_base.pdb'.format(workdir, output_dir, base_count))
                    fout_param_cap_base.write( 'HJDHR_{}_base,{}\n'.format(base_count, ','.join(
                            [str(this_dhr.params[x]) for x in sorted(param_dhrs[0].params.keys())])) )
                
            
                for capped_id, capped in enumerate(helix_capped_list):
                    capped_pose_prop = poorman_repeat_propagate(capped[0], capped[5], num_repeat=num_repeats)
                    capped_params = copy.deepcopy(this_dhr.params)
                    capped_params['repeat_len'] = capped[5]
                    capped_param_dhrs.append(ParamPose(capped_pose_prop, copy.deepcopy(capped_params)))
                    #capped_param_dhrs.append(ParamPose(capped[0], capped_params))

                    if debug:
                        print(base_count, capped_id, sf_farep(capped_pose_prop), capped_params)
                    # TODO: (improve this)add param info to reslabel (as no se what else way to write info into pdb...)
                    capped_pose_prop.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(capped_pose_prop.size()))
                    for j, this_p in enumerate(sorted(capped_params.keys())):
                        capped_pose_prop.pdb_info().add_reslabel(j+1, '{}:{}'.format(this_p, capped_params[this_p]))
                    capped_pose_prop.dump_pdb('{}{}/HJDHR_{}_{}.pdb'.format(workdir, output_dir, 
                                                                                   base_count, capped_id))
                    fout_param_cap.write( 'HJDHR_{},{}\n'.format(base_count, ','.join([str(capped_params[x]) for x in 
                                                                     sorted(capped_param_dhrs[0].params.keys())])) )                
                
                base_count += 1

        fout_param_cap_base.close()
        fout_param_cap.close()
                
        if debug:
            print('num of capped param dhrs in output: ', len(capped_param_dhrs))
                   
        time_cap_end = time.time()
        if debug:
            print('time capping: ', time_cap_end - time_cap_start)
            print('scanned parameter combinations: ', total_param_combs)

    
    else:
        # output dhr w/o adding helix caps
        if debug:
            print('num of param dhrs in output: ', len(param_dhrs))
        
        if len(param_dhrs) > 0:
            fout_param = open('{}{}/{}'.format(workdir, output_dir, output_params_file),'w')
            fout_param.write('description,{}\n'.format(','.join(sorted(param_dhrs[0].params.keys()))))
            for i, this_dhr in enumerate(param_dhrs):
                if debug:
                    print(i, sf_farep(this_dhr.pose), this_dhr.params)
                # TODO: (improve this)add param info to reslabel (as no se what else way to write info into pdb...)
                this_dhr.pose.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(this_dhr.pose.size()))
                for j, this_p in enumerate(sorted(this_dhr.params.keys())):
                    this_dhr.pose.pdb_info().add_reslabel(j+1, '{}:{}'.format(this_p, this_dhr.params[this_p]))
                this_dhr.pose.dump_pdb('{}{}/HJDHR_{}.pdb'.format(workdir, output_dir, i))
                fout_param.write( 'HJDHR_{},{}\n'.format(i, ','.join([str(this_dhr.params[x]) for x in 
                                                                 sorted(param_dhrs[0].params.keys())])) )
            fout_param.close()

    time_end = time.time()
    if debug:
        print('total run time: ', time_end - time_start)
    
    return None        
        


def parse_scan_range(scan_range):
    '''
        scan_range(str): a comma seperated strlist or slash seperated min;max;delta
    '''
    if '/' in scan_range:
        items = scan_range.split('/')
        assert(len(items)==3)
        min_val, max_val, delta = [float(x) for x in items]
        out = []
        x = min_val
        while x <= max_val:
            out.append(x)
            x += delta
        return out
    else:
        return [float(x) for x in scan_range.split(',')]



def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--num_repeats', type=int, default=5,
                        help='number of repeats in output dhr')  

    parser.add_argument('--residue_name3', type=str, default='ALA',
                        help='residue type to use for building dhr') 


    # h1 parameters
    parser.add_argument('--h1_len', type=int, default=20, 
                        help='length of parental helix h1') 

    parser.add_argument('--r_z1', type=str, default='0;359.99;60.0',
                        help='self-rotation degree of h1 about the Z axis') 

    parser.add_argument('--r_r1', type=str, default='0.0',
                        help='rotation degree(s) of h1 about the radius line; comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--r_t1', type=str, default='0.0',
                        help='rotation degree(s) of h1 about the tangent line; comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--t_x1', type=str, default='30.0',
                        help='(radius) translation of h1 along X axis; comma-separated list or slash-separated min;max;stride') 



    # h2 parameters
    parser.add_argument('--h2_len', type=int, default=20, 
                        help='length of parental helix h2') 


    parser.add_argument('--sample_h2', action='store_true', 
                        help='if enabled, sample h2 instead of generate h2 parametrically')

    parser.add_argument('--h1h2_term_com_dist_range', type=str, default='9.5,10.5',
                        help='(for h2 sampling) comma-separated list of min,max for h1h2_term_com_dist') 

    parser.add_argument('--h1h2_term_phi_range_degree', type=str, default='70,110',
                        help='(for h2 sampling) comma-separated list of min,max for range of angle between h1_cterm_com->h2_nterm_com and h1') 

    parser.add_argument('--h1h2_term_theta_range_degree', type=str, default='-30,30',
                        help='(for h2 sampling) comma-separated list of min,max for rotation angle of h2_nterm_com around h1_cterm_com, starts (when=0) on X axis') 

    parser.add_argument('--h2_cterm_perturb_radius', type=float, default=1.5,
                        help='(for h2 sampling) max distance of deviation of h2_cterm_com from initial position')  

    parser.add_argument('--num_trial_h2_nterm', type=int, default=10, 
                        help='(for h2 sampling) number of trials for placing nterm of h2') 

    parser.add_argument('--num_trial_h2_cterm', type=int, default=10, 
                        help='(for h2 sampling) number of trials for placing cterm of h2') 

    parser.add_argument('--num_top_h2', type=int, default=5, 
                        help='(for h2 sampling) keep this number of h2 sampled for each h1 (through clustering)') 



    parser.add_argument('--r_z2', type=str, default='0;359.99;60.0',
                        help='self-rotation degree of h2 about its own Z axis; comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--t_x2', type=str, default='2.0',
                        help='translation along h2_s own X axis of h2 from h1; comma-separated list or slash-separated min;max;stride')  

    parser.add_argument('--t_z2', type=str, default='1.0',
                        help='translation along Z axis of h2 from h1; comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--r_r2', type=str, default='10.0',
                        help='rotation degree(s) of h2 about h1; comma-separated list or slash-separated min;max;stride') 



    # inter-repeat parameters
    parser.add_argument('--dist_r1h1_r2h1_nterm', type=str, default='-1',
                        help='distance(s) between inner row helix (h1) Nterminus COM, this term with t_x1 overwrite r_zr; \
                        comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--handedness', type=int, default=-1, 
                        help='handedness of dhr') 



    parser.add_argument('--sample_r2', action='store_true', 
                        help='if enabled, sample r2 instead of generate r2 parametrically')

    parser.add_argument('--r1h2_r2h1_term_com_dist_range', type=str, default='9.5,10.5',
                        help='(for r2 sampling) comma-separated list of min,max for r1h2 cterm r2h1 nterm com dist') 

    parser.add_argument('--r1h1_r2h1_term_com_dist_range', type=str, default='9.5,12.0',
                        help='(for r2 sampling) comma-separated list of min,max for r1h1_r2h1 nterm com dist') 

    parser.add_argument('--r1h2_r2h1_term_theta_range_degree', type=str, default='0,359.999',
                        help='(for r2 sampling) comma-separated list of min,max for rotation angle of r2h1 around r1h2, starts (when=0) on X axis') 

    # r2h1's phi range degree and cterm perturb radius use the same value as h2's

    parser.add_argument('--num_trial_r2h1_nterm', type=int, default=10, 
                        help='(for r2 sampling) number of trials for placing nterm of r2h1') 

    parser.add_argument('--num_trial_r2h1_cterm', type=int, default=10, 
                        help='(for r2 sampling) number of trials for placing cterm of r2h1') 

    parser.add_argument('--num_top_r2h1', type=int, default=3, 
                        help='(for r2 sampling) keep this number of r2h1 sampled for each r1 (through clustering)') 



    parser.add_argument('--r_zr', type=str, default='10.0',
                        help='(remodel_s twist) rotation degree of r2 from r1 about Z axis; comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--r_hr', type=str, default='10.0',
                        help='(Kobe_s repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis; \
                        comma-separated list or slash-separated min;max;stride') 

    parser.add_argument('--t_zr', type=str, default='0.0',
                        help='stride for (rise) translation of repeat2 (r2) from r1 along Z axis; comma-separated list or slash-separated min;max;stride') 



    # filters
    parser.add_argument('--farep_cutoff', type=float, default=100,
                        help='max farep allowed, for clashes')   

    parser.add_argument('--motifscore_cutoff', type=float, default=-0.01,
                        help='max motifscore allowed') 
    parser.add_argument('--filter_h1h2_ss_degree', action='store_true', 
                        help='if enabled, only keep repeat unit (h1h2, r1) if worst_ss_degree > 0')
    parser.add_argument('--filter_r1r2_ss_degree', action='store_true', 
                        help='if enabled, only keep r1r2 repeat pair if worst_ss_degree > 0, best_ss_degree > 1')
    parser.add_argument('--worst_ss_degree_cutoff', type=float, default=2,
                        help='worst ss_degree using my own motifscore') 
    parser.add_argument('--best_ss_degree_cutoff', type=float, default=3,
                        help='best ss_degree using my own motifscore') 
    parser.add_argument('--avg_ss_degree_cutoff', type=float, default=2.5,
                        help='average ss_degree using my own motifscore') 

    parser.add_argument('--min_core_residue_percentage', type=float, default=0.28,
                        help='minimum percentage of core residues required') 
    parser.add_argument('--core_residue_SCN_cutoff', type=float, default=5.2,
                        help='cutoff for core layer in the layer selector with use_sidechain_neighbors option') 

    parser.add_argument('--max_helix_cap_com_dist', type=float, default=16.0,
                        help='max dist allowed for helix caps to be connected by loops') 



    # add capping options
    parser.add_argument('--add_helix_capping_motif', action='store_true', 
                        help='if enabled, add canonical helical capping motif for long loop modeling later')
    parser.add_argument('--suppress_dhr_output', action='store_true', 
                        help='if enabled, do not output the base, pre-capped dhr, output only the capped scaffold')
    parser.add_argument('--inner_helix_trim_size', type=int, default=4, 
                        help='trim this number of residues from inner helix ncap site for long loops') 
    parser.add_argument('--min_helix_height_diff', type=float, default=3.0,
                        help='minimum height difference between inner and outter row helix after capping along Z-axis. \
                            this is used to avoid clashing between capping motif (and later the loop) with inner row helix') 
    parser.add_argument('--max_helix_height_diff', type=float, default=6.0,
                        help='maximum height difference between inner and outter row helix after capping along Z-axis. \
                            this is used to avoid capping motifs being too far away, resulting in big holes near loops') 
    parser.add_argument('--min_cos_angle', type=float, default=0.8,
                        help='min_cos_angle for controlling capping motif anchor positions') 



    # output 
    parser.add_argument('--workdir', type=str, default=os.getcwd(),
                        help='workdir')   
    parser.add_argument('--output_dir', type=str, default=os.getcwd(),
                        help='output_dir, based on workdir')   
    parser.add_argument('--c_cap_phipsi_file', type=str, 
                        help='c_cap_phipsi_file, use the 4aa verison')      
    parser.add_argument('--n_cap_phipsi_file', type=str, 
                        help='n_cap_phipsi_file, use the 4aa version')  
    parser.add_argument('--output_params_file', type=str, default='params.dat',
                        help='name of output params files')  


    # debug
    parser.add_argument('--verbose', action='store_true', 
                        help='if enabled, turn off -mute all options')
    parser.add_argument('--debug', action='store_true', 
                        help='if enabled, print out all the debug messages')


    args = parser.parse_args()



    cmd_option = []
    cmd_option.append('-indexed_structure_store:fragment_store ss_grouped_vall_helix_shortLoop.h5')
    cmd_option.append('-symmetry_definition stoopid -old_sym_min true')
    cmd_option.append('-relax:default_repeats 1')
    cmd_option.append('-score:max_motif_per_res 3.0')
    mut_arg = '-mute all ' #if not args.verbose else '' 
    cmd_option.append(mut_arg)

    pyrosetta.init(' '.join(cmd_option))

    
    
    sf_sym = pyrosetta.get_score_function()
    sf = sf_sym.clone()
    sf = sf.clone_as_base_class()  # this is all what asymmetrize_scorefunction() does...

    # 
    #  TODO: double check centroid residue generation (compare w/ remodel)
    # 
    # remodel's centroid weights (w/o terms of helical params, cst)
    sf_cen_sym = sf_sym.clone()
    for term in sf_cen_sym.get_nonzero_weighted_scoretypes():
        sf_cen_sym.set_weight(term, 0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.0)  # use fa_rep instead of vdw (vdw yields error)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.rama_prepro, 0.5)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.omega, 1.0)
    #sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.pair, 1.0)
    #sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.env, 1.0)
    #sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cbeta, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cen_pair_smooth, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cen_env_smooth, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cbeta_smooth, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cenpack_smooth, 1.0)
    #sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.sidechain_neighbors, 0.1) # nonexist
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.rg, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.cen_pair_motifs, 1.0)
    sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1.0)
    #sf_cen_sym.set_weight(pyrosetta.rosetta.core.scoring.chainbreak, 1.0)
    #sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 1.0)
    #sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 1.0)
    #sf_cen.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 1.0)

    sf_cen = sf_cen_sym.clone_as_base_class()  # this is all what asymmetrize_scorefunction() does...


    # parameter sanity check
    assert(args.handedness**2 == 1)  # handedness can only be 1 or -1





    build_param_dhrs(
    #sf,
    #sf_sym,
    sf_cen,
    sf_cen_sym,

    num_repeats=args.num_repeats,
    residue_name3=args.residue_name3,

    h1_len=args.h1_len, #    length of h1 in each repeat
    r_z1_list = parse_scan_range(args.r_z1), #      'self-rotation degree' of h1 about the Z axis
    r_r1_list = parse_scan_range(args.r_r1), #       rotation degree of h1 about the radius line
    r_t1_list = parse_scan_range(args.r_t1), #       rotation degree of h1 about the tangent line
    t_x1_list = parse_scan_range(args.t_x1), #      (radius) translation of h1 along X axis

    h2_len=args.h2_len, #    length of h2 in each repeat
    
    sample_h2=args.sample_h2, #      sample h2 instead of generate h2 parametrically
    h1h2_term_com_dist_range=parse_scan_range(args.h1h2_term_com_dist_range), # generate h2 by sampling conformations of h2 around h1
    h1h2_term_phi_range_degree=parse_scan_range(args.h1h2_term_phi_range_degree), # range of angle between h1_cterm_com->h2_nterm_com and h1
    h1h2_term_theta_range_degree=parse_scan_range(args.h1h2_term_theta_range_degree),     # rotation angle of h2 around h1, starts (when=0) on X axis
    h2_cterm_perturb_radius=args.h2_cterm_perturb_radius,    # max distance of deviation of h2_cterm_com from initial position
    num_trial_h2_nterm=args.num_trial_h2_nterm,     #   number of trials for placing nterm and cterm of h2
    num_trial_h2_cterm=args.num_trial_h2_cterm,    #   number of trials for placing nterm and cterm of h2
    num_top_h2=args.num_top_h2, # keep this number of h2 sampled for each h1 (through clustering)

    r_z2_list = parse_scan_range(args.r_z2), #      'self-rotation degree' of h2 about the Z axis
    t_x2_list = parse_scan_range(args.t_x2), #      (radius of h2 from h1) translation along X axis of h2 from h1
    t_z2_list = parse_scan_range(args.t_z2), #    translation along Z axis of h2 from h1
    r_r2_list = parse_scan_range(args.r_r2), #      rotation degree of h2 about h1

    sample_r2=args.sample_r2, #      sample r2 instead of generate r2 parametrically
    r1h2_r2h1_term_com_dist_range=parse_scan_range(args.r1h2_r2h1_term_com_dist_range), #[9.5, 10.5]
    r1h1_r2h1_term_com_dist_range=parse_scan_range(args.r1h1_r2h1_term_com_dist_range),
    r1h2_r2h1_term_theta_range_degree=parse_scan_range(args.r1h2_r2h1_term_theta_range_degree), # rotation angle of r2h1 around r1h2, starts (when=0) on X axis
    num_trial_r2h1_nterm=args.num_trial_r2h1_nterm, # number of trials for placing nterm of r2h1, #10*num_trial_h2_nterm
    num_trial_r2h1_cterm=args.num_trial_r2h1_cterm, # number of trials for placing cterm of r2h1,  #10*num_trial_h2_cterm
    num_top_r2h1=args.num_top_r2h1, # keep this number of h2 sampled for each h1 (through clustering)

    dist_r1h1_r2h1_nterm_list = parse_scan_range(args.dist_r1h1_r2h1_nterm),
    handedness = args.handedness,
    r_zr_list = parse_scan_range(args.r_zr), # (remodel's twist) rotation degree of r2 from r1 about Z axis
    r_hr_list = parse_scan_range(args.r_hr), #  (Kobe's repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis (r1's Y)
    t_zr_list = parse_scan_range(args.t_zr), #      (rise) translation of repeat2 (r2) from r1 along Z axis    


    farep_cutoff=args.farep_cutoff,

    motifscore_cutoff=args.motifscore_cutoff,
    filter_h1h2_ss_degree=args.filter_h1h2_ss_degree,  # only keep repeat unit (h1h2, r1) if worst_ss_degree > 0
    filter_r1r2_ss_degree=args.filter_r1r2_ss_degree,  # only keep r1r2 repeat pair if worst_ss_degree > 0, best_ss_degree > 1
    worst_ss_degree_cutoff=args.worst_ss_degree_cutoff,
    best_ss_degree_cutoff=args.best_ss_degree_cutoff,
    avg_ss_degree_cutoff=args.avg_ss_degree_cutoff,

    min_core_residue_percentage=args.min_core_residue_percentage,
    core_residue_SCN_cutoff=args.core_residue_SCN_cutoff,

    max_helix_cap_com_dist=args.max_helix_cap_com_dist,  # max dist allowed for helix caps to be connected by loops

    add_helix_capping_motif=args.add_helix_capping_motif,
    inner_helix_trim_size=args.inner_helix_trim_size,   # trim this number of residues from inner helix ncap site for long loops
    min_helix_height_diff=args.min_helix_height_diff,  
    max_helix_height_diff=args.max_helix_height_diff,
    min_cos_angle=args.min_cos_angle,

    output_params_file=args.output_params_file,

    workdir=args.workdir,
    c_cap_phipsi_file=args.c_cap_phipsi_file,
    n_cap_phipsi_file=args.n_cap_phipsi_file,
    output_dir=args.output_dir,

    suppress_dhr_output=args.suppress_dhr_output,
    debug=args.debug,
    )



if __name__ == '__main__':
    main()

