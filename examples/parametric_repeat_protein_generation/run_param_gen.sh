#!/bin/bash

output_dir=outputs
script_dir=../../
motif_dir=../../motifs
H1_LENGTH=16
H2_LENGTH=18

for h1_len in ${H1_LENGTH}; do # length of helix1 (h1)
  for r_z1_d in 120; do # stride of 'self-rotation degree' of h1 about the Z axis
    for r_x1 in 10; do  # rotation degree of h1 about the radius line
      for r_y1 in 10; do  # rotation degree of h1 about the tangent line
        for t_x1 in 30; do # (radius) translation of h1 along X axis
          for d12 in 9.5; do  # distance between inner row helix (h1), this term with t_x1 overwrite r_zr
            for handedness in -1; do  # handedness of dhr
              for h2_len in ${H2_LENGTH}; do # length of helix2 (h2)
                for r_z2_d in 120; do # stride of 'self-rotation degree' of h2 about the Z axis
                  for t_x2 in 0; do  # translation along X axis of h2 from h1
                    for t_z2 in 0; do  # translation along Y axis of h2 from h1
                      for r_r2 in 0; do  # rotation degree of h2 about h1
                        for r_zr in -14.1; do  # (remodel's twist) rotation degree of r2 from r1 about Z axis
                          for r_xr in 0; do # rotation degree of r2 from r1 about X axis
                            for r_yr in 0; do  # (Kobe's repeat twist) rotation degree of r2 from r1 about intra-dhr helical axis
                              for t_xr in 0; do  # translation of repeat2 (r2) from r1 along X axis
                                for t_yr in 0; do  # translation of repeat2 (r2) from r1 along Y axis
                                  for t_zr in 0; do  # (rise) translation of repeat2 (r2) from r1 along Z axis

     


python $script_dir/param_dhr_for_buttressed_loops.py \
--residue_name3=ALA \
--num_repeats=5 \
--h1_len=$h1_len \
--r_z1=0/359.999/${r_z1_d} \
--r_x1=$r_x1 \
--r_y1=$r_y1 \
--t_x1=$t_x1 \
--dist_r1h1_r2h1_nterm=$d12 \
--handedness=$handedness \
--h2_len=$h2_len \
--sample_h2 \
--h1h2_term_com_dist_range 8.5,10.5 \
--h1h2_term_phi_range_degree=80,100 \
--h1h2_term_theta_range_degree=-45,45 \
--h2_cterm_perturb_radius=6 \
--num_trial_h2_nterm 20 \
--num_trial_h2_cterm 20 \
--num_top_h2 100 \
--r_z2=0/359.999/${r_z2_d} \
--t_x2=$t_x2 \
--t_z2=$t_z2 \
--r_r2=$r_r2 \
--r_zr=$r_zr \
--r_xr=$r_xr \
--r_yr=$r_yr \
--t_xr=$t_xr \
--t_yr=$t_yr \
--t_zr=$t_zr \
--farep_cutoff 1000 \
--motifscore_cutoff -0.001 \
--path_to_motifs=../../motifs \
--filter_h1h2_ss_degree \
--filter_r1r2_ss_degree \
--worst_ss_degree_cutoff 2 \
--best_ss_degree_cutoff 3 \
--avg_ss_degree_cutoff 2.5 \
--min_core_residue_percentage 0.28 \
--core_residue_SCN_cutoff 5.2 \
--min_helix_height_diff 3.0 \
--max_helix_height_diff 8.0 \
--ccap_min_cos_angle -0.3 \
--ccap_direction 1 \
--ncap_min_cos_angle 0.8 \
--max_helix_cap_com_dist 18 \
--add_helix_capping_motif \
--inner_helix_trim_size 4 \
--output_dir $output_dir \
--c_cap_phipsi_file $motif_dir/angle_ccap.dat \
--n_cap_phipsi_file $motif_dir/angle_ncap.dat \
--output_params_file params.dat \
--suppress_dhr_output \
--debug 

#--output_silent \
#--output_silent_prefix=out \
#--max_num_pose_per_silent_file 10 \


echo "Done with $output_dir ..."

                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
