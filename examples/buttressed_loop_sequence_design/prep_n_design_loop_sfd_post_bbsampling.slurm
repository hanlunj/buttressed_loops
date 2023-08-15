#!/bin/bash
##SBATCH -p short
#SBATCH -p backfill
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -J dhrd
#SBATCH -o prep_n_design_loop_sfd.log
##SBATCH --array=1-4

module purge
source /software/conda/etc/profile.d/conda.sh
conda activate pyrosetta

script_dir=/home/jianghl/Projects/loop_design/scripts/param_dhr_loop/scripts/param_dhr_loop/parametric_dhr
pssm_file=pssm.dat
num_repeats=4
suffix=fd

WORKDIR=$PWD
index=0
for i in `cat pdblist-sfd-loop` ; do
  let index=index+1
  if [ $SLURM_ARRAY_TASK_ID -eq $index ] ; then
    cd $WORKDIR

d=`dirname $i`
cd $d
pdbname=`basename $i .pdb`
reslists=${pdbname}.layer_design_reslists
#rm -rf $pdbname
mkdir -p $pdbname
cd $pdbname

if [ ! -f ${pssm_file} ]; then
# generate sequence profile (counts)
#/software/rosetta/latest/bin/rosetta_scripts.hdf5.linuxgccrelease \
/home/jianghl/software/Rosetta/main/source/cmake/build_release_hdf5_logs_odds_profile/rosetta_scripts \
-mute all \
-beta \
-parser:protocol $script_dir/xmls/pssm_from_frags.xml \
-indexed_structure_store:fragment_store /home/jianghl/database/tj/hdf5/unpublished/ss_grouped_vall_all.h5 \
-s ../$pdbname.pdb \
-out:nooutput &> profile_${pdbname}.log

# convert profile counts to pssm
python $script_dir/pyrosetta/cnorn_profile_make_pssm_from_count_files.py \
--eff_count_method=2 \
--pseudoCountWeight=50 \
--countAA=profile \
--pssmOut=${pssm_file} &>> profile_${pdbname}.log
fi


hostname &> design_${suffix}.log
env &>> design_${suffix}.log

if [ ! -f ${pssm_file} ]; then
  echo "WARNING:  design pssm file not found, skipping design for $i ..." >> design_${suffix}.log
  #echo "WARNING:  design pssm file not found, skipping design for $i ..." > design_shp_${pdbname}.log
  continue
fi

if [ ! -f ${pdbname}_${suffix}_0001.pdb ]; then

rm -f score_${suffix}.sc

if [ ! -f $reslists ]; then
  python $script_dir/pyrosetta/run_layer_selector.py ../$pdbname.pdb $num_repeats &>> design_${suffix}.log
fi

/software/rosetta/latest/bin/rosetta_scripts.hdf5.linuxgccrelease \
-beta_nov16 \
-overwrite \
-indexed_structure_store:fragment_store /home/jianghl/database/tj/hdf5/ss_grouped_vall_helix_shortLoop.h5 \
-parser:protocol $script_dir/xmls/param_dhr_design_20210503_fastd_loopsfd.xml \
-s ../$pdbname.pdb \
-nstruct 1 \
-holes:dalphaball /home/jianghl/software/DAlphaBall.gcc \
-symmetry_definition stoopid -old_sym_min true \
-parser:script_vars num_repeats=${num_repeats} pssm_file=${pssm_file} \
start_dummy=`grep start_dummy $reslists | awk -F':' '{print $2}'` \
surface_AND_helix_start=`grep 1_surface_AND_helix_start $reslists | awk -F':' '{print $2}'` \
surface_AND_helix=`grep 2_surface_AND_helix $reslists | awk -F':' '{print $2}'` \
surface_AND_sheet=`grep 3_surface_AND_sheet $reslists | awk -F':' '{print $2}'` \
surface_AND_loop=`grep 4_surface_AND_loop $reslists | awk -F':' '{print $2}'` \
boundary_AND_helix_start=`grep 5_boundary_AND_helix_start $reslists | awk -F':' '{print $2}'` \
boundary_AND_helix=`grep 6_boundary_AND_helix $reslists | awk -F':' '{print $2}'` \
boundary_AND_sheet=`grep 7_boundary_AND_sheet $reslists | awk -F':' '{print $2}'` \
boundary_AND_loop=`grep 8_boundary_AND_loop $reslists | awk -F':' '{print $2}'` \
core_AND_helix_start=`grep 9_core_AND_helix_start $reslists | awk -F':' '{print $2}'` \
core_AND_helix=`grep 10_core_AND_helix $reslists | awk -F':' '{print $2}'` \
core_AND_sheet=`grep 11_core_AND_sheet $reslists | awk -F':' '{print $2}'` \
core_AND_loop=`grep 12_core_AND_loop $reslists | awk -F':' '{print $2}'` \
helix_cap=`grep 13_helix_cap $reslists | awk -F':' '{print $2}'` \
end_dummy=`grep end_dummy $reslists | awk -F':' '{print $2}'` \
length=`grep length $reslists | awk -F':' '{print $2}'` \
BIDENTATE_HBONDS_BB=`grep BIDENTATE_HBONDS_BB $reslists | awk -F':' '{print $2}'` \
BIDENTATE_HBONDS_BB_NH=`grep BIDENTATE_HBONDS_BB_NH $reslists | awk -F':' '{print $2}'` \
BIDENTATE_HBONDS_BB_CO=`grep BIDENTATE_HBONDS_BB_CO $reslists | awk -F':' '{print $2}'` \
BIDENTATE_HBONDS_SC=`grep BIDENTATE_HBONDS_SC $reslists | awk -F':' '{print $2}'` \
-out:suffix _${suffix} &>> design_${suffix}.log

fi


  fi
done



