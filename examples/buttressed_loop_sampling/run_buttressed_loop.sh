#!/bin/bash

input_pdb=RBL_input.pdb
cutstart=27
cutend=34
nstruct=300
B=3,4,5
A=0
iter=30
turn=turn-list
ncap=N-cap-list
ccap=C-cap-list
param_file_dir=../../params


python ../../buttressed_loop_builder_for_param_dhr.py \
-i $input_pdb \
--cut_start $cutstart \
--cut_end $cutend \
--helix_n_cap_angle_file $ncap \
--helix_c_cap_angle_file $ccap \
-t $turn \
-n $nstruct -b $B -a $A \
-d buttressed_loops_iter${iter} \
--num_repeats 4 \
--perturb 0 \
--worst_score_allowed -1 \
--bturn_side=ncap \
--relax_repeats 1 \
--min_potential_interloop_bbbb_hbond_num 0 \
--min_intraloop_hbond_num 0 \
--min_intraloop_bbbb_hbond_num 0 \
--min_interloop_hbond_num 0 \
--min_interloop_bbbb_hbond_num 0 \
--bidentate_resn DNHQ \
--min_num_pseudo_bidentate 0 \
--pseudo_bidentate_anchor_sstype=all \
--min_num_bidentate 0 \
--delta_HA 4 \
--delta_theta 90 \
--min_loop_motif 0 \
--motif_database ../../motifs/xs_bb_ss_FILV_FILV_resl0.5_smooth1.3_msc0.3_mbv1.0/xs_bb_ss_FILV_FILV_resl0.5_smooth1.3_msc0.3_mbv1.0 \
--max_motif_per_res 3 \
--disable_motif_packing \
--design 0 \
--layer_design \
--aacomp_design_file $param_file_dir/design_composition.comp \
--aacomp_design_pro_file $param_file_dir/design_pro_composition.comp \
--aacomp_cap_pro_file $param_file_dir/cap_pro_composition.comp \
--aacomp_design_Ebin_file $param_file_dir/design_Ebin_composition.comp \
--aacomp_design_Gbin_file $param_file_dir/design_Gbin_composition.comp \
--bbrmsd_cutoff 0.3 \
--checkpoint_frequency 100 \
--total_score_cutoff 0 \
--iter $iter \
--debug
    






