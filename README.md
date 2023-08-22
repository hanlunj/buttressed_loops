### Installation requirement
PyRosetta4 (release 2019.22 or later), available at https://www.pyrosetta.org/downloads  
sklearn (0.22.1 or later)  
numpy (1.18.1 or later)  

### Examples
Example commands for running parametric repeat protein generation can be found at:  
examples/parametric_repeat_protein_generation/i  
run_param_gen.sh provides simple and quick results in outputs/  
run_param_gen.full_parameters.sh samples full parameter combination (Caution: long running time)  
  
Example commands for sampling buttressed loops can be found at:  
examples/buttressed_loop_sampling/  
run_buttressed_loop.sh samples buttressed loops from an input parametric repeat protein  
  
Example commands for buttressed loop sequence design can be found at:  
examples/buttressed_loop_sequence_design/  
run_sequence_design.sh uses a customized Rosetta sequence design protocol  
