#!/bin/bash

pdb=RBL_input_0013_L30-40_0.pdb
#rosetta=$PATH_TO_rosetta_scripts
num_repeats=4
suffix=design

pdbname=`basename $pdb .pdb`
reslists=${pdbname}.layer_design_reslists

if [ ! -f $reslists ]; then
  python ../../run_layer_selector.py $pdb $num_repeats 
fi

$rosetta \
-beta_nov16 \
-overwrite \
-parser:protocol ../../buttressed_loops_design.xml \
-s $pdb \
-nstruct 1 \
-symmetry_definition stoopid -old_sym_min true \
-parser:script_vars num_repeats=${num_repeats} \
params=../../params \
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
-out:suffix _${suffix} 





