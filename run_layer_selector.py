import sys
import pyrosetta

pyrosetta.init('-mute all')

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


in_fname = sys.argv[1]
try:
    num_repeats = int(sys.argv[2])
except:
    num_repeats = 4

if in_fname[-4:] == '.pdb':
    inputlist = [in_fname.strip()]
else:
    fin = open(in_fname, 'r')
    inputlist = fin.read().split('\n')[:-1]
    fin.close()

for input_pdb in inputlist:
    pose = pyrosetta.pose_from_file(input_pdb)

    helices = find_helices_by_dssp(pose, min_helix_length=5)
    helix_reslist = []
    for helix in helices:
        for resi in range(helix[0], helix[1]+1):
            helix_reslist.append(resi)

    assert(pose.size()%num_repeats==0)
    repeatlen = int(pose.size()/num_repeats)

    second_repeat_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(f'{repeatlen+1}-{repeatlen*2}')

    # SS selectors
    entire_helix_selector = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    entire_helix_selector.set_overlap(0)
    entire_helix_selector.set_minH(3)
    entire_helix_selector.set_minE(2)
    entire_helix_selector.set_include_terminal_loops(False)
    entire_helix_selector.set_use_dssp(True)
    entire_helix_selector.set_selected_ss('H')

    sheet_selector = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    sheet_selector.set_overlap(0)
    sheet_selector.set_minH(3)
    sheet_selector.set_minE(2)
    sheet_selector.set_include_terminal_loops(False)
    sheet_selector.set_use_dssp(True)
    sheet_selector.set_selected_ss('E')

    entire_loop_selector = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    entire_loop_selector.set_overlap(0)
    entire_loop_selector.set_minH(3)
    entire_loop_selector.set_minE(2)
    entire_loop_selector.set_include_terminal_loops(True)
    entire_loop_selector.set_use_dssp(True)
    entire_loop_selector.set_selected_ss('L')

    # helix cap
    entire_helix_start_selector = pyrosetta.rosetta.core.select.residue_selector.PrimarySequenceNeighborhoodSelector()
    entire_helix_start_selector.set_lower_residues(1)
    entire_helix_start_selector.set_upper_residues(0)
    entire_helix_start_selector.set_selector(entire_helix_selector)
    helix_cap_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(entire_loop_selector, entire_helix_start_selector)

    # helix start
    entire_helix_cap_start_selector = pyrosetta.rosetta.core.select.residue_selector.PrimarySequenceNeighborhoodSelector()
    entire_helix_cap_start_selector.set_lower_residues(0)
    entire_helix_cap_start_selector.set_upper_residues(1)
    entire_helix_cap_start_selector.set_selector(helix_cap_selector)
    helix_start_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(entire_helix_selector, entire_helix_cap_start_selector)

    # rest of the helices
    not_helix_start_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(helix_start_selector)
    helix_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(entire_helix_selector, not_helix_start_selector)

    # rest of the loops
    loop_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(entire_loop_selector, not_helix_start_selector)


    # layer selectors
    layer_list = ['core','boundary','surface']
    layer_selectors = {}
    for layer in layer_list:
        layer_all_selector = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_bool_list = [True if l == layer else False for l in layer_list ]
        layer_all_selector.set_layers(*layer_bool_list) # core, boundary, surface
        layer_selectors[layer] = layer_all_selector


    # now focus all selectors on 2nd repeat unit, then propagate to the rest of the repeats
    base_selectors = {
        'core': layer_selectors['core'],
        'boundary': layer_selectors['boundary'],
        'surface': layer_selectors['surface'],
        'helix_start': helix_start_selector,
        'helix_cap': helix_cap_selector,
        'helix': helix_selector,
        'sheet': sheet_selector,
        'loop': loop_selector,
    }
    global_selectors_list = [
        "surface AND helix_start",
        "surface AND helix",
        "surface AND sheet",
        "surface AND loop",
        "boundary AND helix_start",
        "boundary AND helix",
        "boundary AND sheet",
        "boundary AND loop",
        "core AND helix_start",
        "core AND helix",
        "core AND sheet",
        "core AND loop",
        "helix_cap",    
    ]
    global_selectors = {
        "surface AND helix_start": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['surface'], base_selectors['helix_start']),  
        "surface AND helix": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['surface'], base_selectors['helix']),        
        "surface AND sheet": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['surface'], base_selectors['sheet']),        
        "surface AND loop": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['surface'], base_selectors['loop']),         
        "boundary AND helix_start": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['boundary'], base_selectors['helix_start']), 
        "boundary AND helix": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['boundary'], base_selectors['helix']),       
        "boundary AND sheet": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['boundary'], base_selectors['sheet']),       
        "boundary AND loop": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['boundary'], base_selectors['loop']),        
        "core AND helix_start": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['core'], base_selectors['helix_start']),     
        "core AND helix": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['core'], base_selectors['helix']),           
        "core AND sheet": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['core'], base_selectors['sheet']),          
        "core AND loop": pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(base_selectors['core'], base_selectors['loop']),            
        "helix_cap": base_selectors['helix_cap'],                
    }
    empty_selector_exists = False
    output_reslists = {x:[] for x in global_selectors_list}
    for selector in global_selectors_list:
        this_selector = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(global_selectors[selector], second_repeat_selector)
        this_vec = this_selector.apply(pose)
        this_reslist = [i+1 for i,j in enumerate(this_vec) if j]
        for x in this_reslist:
            new_x = x%repeatlen
            if new_x == 0:
                new_x = repeatlen
            for r in range(num_repeats):
                output_reslists[selector].append(r*repeatlen+new_x)
        output_reslists[selector] = sorted(output_reslists[selector])
        # check for empty select list
        if len(output_reslists[selector]) == 0:
            empty_selector_exists = True
        #print(selector)
        #print(f'show sticks, resi {"+".join([str(x) for x in output_reslists[selector]])}')

    # correct false helix starts
    #     'helices' shorter than 5 residues do not need a capping motif (?)
    new_helix_reslist = []
    for resi in output_reslists['helix_cap']:
        if resi+1 in helix_reslist:
            new_helix_reslist.append(resi)
    output_reslists['helix_cap'] = new_helix_reslist

    # helix_start and helix_cap may overlap with other selectors
    #     needs to remove their residues from other selectors' lists
    check_selectors_list = [
        "surface AND helix_start",
        "boundary AND helix_start",
        "core AND helix_start",
        "helix_cap",    
    ]
    for check_selector in check_selectors_list:
        print('check_selectors_list: ', check_selector, output_reslists[check_selector])
        for selector in global_selectors_list:
            if selector not in check_selectors_list:
                print('    global_selectors_list: ', selector, output_reslists[selector])
                filtered_reslist = output_reslists[selector].copy()
                for resi in output_reslists[selector]:
                    if resi in output_reslists[check_selector]:
                        print('        removing: ', resi)
                        filtered_reslist.remove(resi)
                output_reslists[selector] = filtered_reslist

    dummy_reslist = []
    dummy_check = False
    for resi in range(len(output_reslists['surface AND helix'])):
        if output_reslists['surface AND helix'][resi] not in output_reslists['surface AND helix_start']:
            for r in range(num_repeats):
                dummy_reslist.append(r*repeatlen+output_reslists['surface AND helix'][resi])
            dummy_check = True
            break
    if not dummy_check and empty_selector_exists:
        print('Error: '+input_pdb+' Could not find a dummy residue and there exists empty selector! This will cause crash in ROSETTA!')
        continue 


    # add label copying step here (need to read the bidentate labels, as the labels get lost after idealizer and propagation...)
    bidentate_selector_list = [
        'BIDENTATE_HBONDS_BB',
        'BIDENTATE_HBONDS_BB_NH',
        'BIDENTATE_HBONDS_BB_CO',
        'BIDENTATE_HBONDS_SC',
    ]
    bidentate_selector_dict = {}
    for bidentate in bidentate_selector_list:
        bidentate_selector = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector( bidentate )
        bidentate_vec = bidentate_selector.apply( pose )
        bidentate_reslist = [i+1 for i, j in enumerate(list(bidentate_vec)) if j]
        bidentate_selector_dict[bidentate] = ','.join([str(x) for x in bidentate_reslist])
   

    #with open(input_pdb.replace('.pdb', '.layer_design_reslists'), 'w') as fout:
    with open(input_pdb.split('/')[-1].replace('.pdb', '.layer_design_reslists'), 'w') as fout:
        out = ",".join([str(x) for x in dummy_reslist])
        fout.write(f'start_dummy:{out}\n')
        for selector_id, selector in enumerate(global_selectors_list):
            if len(output_reslists[selector]) > 0:
                out = ",".join([str(x) for x in output_reslists[selector]])
            else:
                #out = "0"
                # HACKY FIX
                out = ",".join([str(x) for x in dummy_reslist])
            #fout.write(f'{selector}:{out}\n')
            fout.write(f'{selector_id+1}_{"_".join([x for x in selector.split()])}:{out}\n')
        out = ",".join([str(x) for x in dummy_reslist])
        fout.write(f'end_dummy:{out}\n')
        fout.write(f'length:{pose.size()}\n') # for repeat propagation after idealizer
      
        # add bidentate stuff
        for bidentate in bidentate_selector_list:
            fout.write(f'{bidentate}:{bidentate_selector_dict[bidentate]}\n')

