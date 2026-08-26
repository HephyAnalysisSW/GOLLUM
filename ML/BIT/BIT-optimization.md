# BIT Optimization Project

## Goal
Optimize the Boosted Information Tree (BIT) algorithm for:
1. **Training time** (primary)
2. **Evaluation/inference time** (primary)
3. **Memory consumption** (important, spikes are damaging)
4. **I/O / data loading** (secondary, optimize if it shows up as bottleneck)

## Files to Optimize
All in `/users/robert.schoefbeck/claude/GOLLUM/ML/BIT/`:
- `NumbaMultiNode.py` — core tree node logic, split finding, Numba kernels
- `NumbaBIT.py` — boosted tree training loop
- `pdf_bit_training.py` — training script / entry point
- `../../data/RDataLoader.py` — data loader (optimize only if I/O is a bottleneck)

## Working Directory
Always run benchmarks from:
```
/users/robert.schoefbeck/claude/GOLLUM/ML/BIT
```

## Benchmark Command
```bash
memray run --output benchmark.bin pdf_bit_training.py \
  ../../configs/benchmark/unbinned_delphes_6_RunII.yaml \
  --every 1 \
  --job bit_NG_PDF4LHC21_6_tt2l_delphes \
  --overwrite \
  --max_n_files 1 \
  --profile \
  --postfix <current_iteration_label>
```

Followed immediately by:
```bash
memray stats benchmark.bin
```

## Mandatory Flags (always include these)
| Flag | Reason |
|------|--------|
| `--every 1` | Required for meaningful results |
| `--profile` | Enables CPU profiling output to stdout |
| `--job bit_NG_PDF4LHC21_6_tt2l_delphes` | Selects the benchmark job |
| `--overwrite` | Overwrites earlier results so comparisons are clean |

## Variable Flags
- `--max_n_files 1` — default; use a small number for quick iterations. Increase (e.g. to 3 or 5) if a heavier test is needed to confirm a result is not noise.
- `--postfix <label>` — change this each iteration (e.g. `baseline`, `iter-01-numba-prange`, `iter-02-vectorized-predict`) so plot directories are traceable.

## Profiling Output
- **CPU profile** prints directly to stdout during the run — read it carefully
- **Memory profile** is captured by memray; always run `memray stats benchmark.bin` after each run
- Pay special attention to **memory spikes** (peak >> current at end of run), not just peak usage

## Optimization Log
Keep a running log here after each iteration.

### Baseline
- postfix: `first-try`
- What was changed: nothing, baseline measurement
- CPU hotspots: 
================= cProfile (sorted by tottime) =================                                                                                                                                               
         1070 function calls in 10.007 seconds                                                                                                                                                                 
                                                                                                                                                                                                               
   Ordered by: internal time                                                                                                                                                                                   
   List reduced from 123 to 60 due to restriction <60>                                                                                                                                                         
                                                                                                                                                                                                               
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)                                                                                                                                        
        1    9.707    9.707    9.707    9.707 NumbaMultiNode.py:47(_mse_neg_loss_gains)                                                                                                                        
        1    0.123    0.123    9.919    9.919 NumbaMultiNode.py:312(get_split_vectorized)                                                                                                                      
        2    0.043    0.021    0.043    0.021 {built-in method numpy.array}                                                                                                                                    
        8    0.040    0.005    0.040    0.005 {method 'astype' of 'numpy.ndarray' objects}                                                                                                                     
        1    0.039    0.039    0.039    0.039 {method 'reduceat' of 'numpy.ufunc' objects}                                                                                                                     
        1    0.017    0.017    0.017    0.017 {method 'argsort' of 'numpy.ndarray' objects}                                                                                                                    
        1    0.014    0.014    0.014    0.014 {method 'searchsorted' of 'numpy.ndarray' objects}                                                                                                               
        1    0.013    0.013    0.013    0.013 {method 'partition' of 'numpy.ndarray' objects}                                                                                                                  
        1    0.005    0.005   10.008   10.008 NumbaMultiNode.py:119(__init__)                                                                                                                                  
       10    0.004    0.000    0.004    0.000 {method 'reduce' of 'numpy.ufunc' objects}                                                                                                                       
        3    0.001    0.000    0.001    0.000 {method 'flatten' of 'numpy.ndarray' objects}                                                                                                                    
        1    0.000    0.000    0.000    0.000 _linalg.py:1639(svd)                                                                                                                                             
      756    0.000    0.000    0.000    0.000 {built-in method _functools.reduce}                                                                                                                              
        1    0.000    0.000    0.014    0.014 _function_base_impl.py:4771(_quantile)                                                                                                                           
        1    0.000    0.000    0.000    0.000 {method 'nonzero' of 'numpy.ndarray' objects}                                                                                                                    
        1    0.000    0.000    0.000    0.000 _linalg.py:2010(matrix_rank)                                                                                                                                     
        2    0.000    0.000    0.000    0.000 _arraysetops_impl.py:339(_unique1d)                                                                                                                              
        2    0.000    0.000    0.000    0.000 {method 'cumsum' of 'numpy.ndarray' objects}                                                                                                                     
        1    0.000    0.000    0.000    0.000 function_base.py:25(linspace)                                                                                                                                    
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:4736(_get_indexes)                                                                                                                        
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:4639(_lerp)                                                                                                                               
        5    0.000    0.000    0.031    0.006 fromnumeric.py:51(_wrapfunc)                                                                                                                                     
        2    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}                                                                                                                                    
        1    0.000    0.000    0.000    0.000 {built-in method builtins.sum}                                                                                                                                   
        2    0.000    0.000    0.000    0.000 {method 'sort' of 'numpy.ndarray' objects}                                                                                                                       
        2    0.000    0.000    0.000    0.000 _arraysetops_impl.py:145(unique)                                                                                                                                 
        1    0.000    0.000    0.000    0.000 numeric.py:137(ones)                                                                                                                                             
        1    0.000    0.000    0.000    0.000 fromnumeric.py:89(_wrapreduction_any_all)                                                                                                                        
        2    0.000    0.000    0.004    0.002 fromnumeric.py:69(_wrapreduction)                                                                                                                                
        1    0.000    0.000    0.015    0.015 _function_base_impl.py:3834(_ureduce)                                                                                                                            
        1    0.000    0.000    0.000    0.000 NumbaMultiNode.py:142(<dictcomp>)                                                                                                                                
        1    0.000    0.000    0.001    0.001 NumbaMultiNode.py:149(<listcomp>)                                                                                                                                
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:107(<lambda>)                                                                                                                             
        4    0.000    0.000    0.000    0.000 numerictypes.py:471(issubdtype)                                                                                                                                  
        1    0.000    0.000    0.015    0.015 _function_base_impl.py:4697(_quantile_ureduce_func)                                                                                                              
        1    0.000    0.000    0.000    0.000 fromnumeric.py:1904(ravel)                                                                                                                                       
        1    0.000    0.000    0.000    0.000 numeric.py:450(count_nonzero)                                                                                                                                    
        1    0.000    0.000    0.000    0.000 numeric.py:646(flatnonzero)                                                                                                                                      
        1    0.000    0.000    0.015    0.015 _function_base_impl.py:4283(quantile)                                                                                                                            
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:4615(_get_gamma)                                                                                                                          
        1    0.000    0.000    0.000    0.000 _ufunc_config.py:440(__enter__)                                                                                                                                  
       31    0.000    0.000    0.000    0.000 {built-in method builtins.sorted}                                                                                                                                
        8    0.000    0.000    0.000    0.000 numerictypes.py:289(issubclass_)                                                                                                                                 
        1    0.000    0.000    0.000    0.000 _linalg.py:148(_commonType)                                                                                                                                      
        1    0.000    0.000    0.000    0.000 getlimits.py:493(__new__)                                                                                                                                        
       15    0.000    0.000    0.000    0.000 {built-in method builtins.issubclass}                                                                                                                            
        3    0.000    0.000    0.000    0.000 multiarray.py:180(concatenate)                                                                                                                                   
        1    0.000    0.000    9.919    9.919 NumbaMultiNode.py:583(split)                                                                                                                                     
        2    0.000    0.000    0.000    0.000 fromnumeric.py:2879(cumsum)                                                                                                                                      
        1    0.000    0.000    0.000    0.000 NumbaMultiNode.py:168(<listcomp>)                                                                                                                                
        1    0.000    0.000    0.002    0.002 fromnumeric.py:3190(min)                                                                                                                                         
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}                                                                                                                                    
        1    0.000    0.000    0.015    0.015 _function_base_impl.py:4541(_quantile_unchecked)                                                                                                                 
        5    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}                                                                                                                                 
        1    0.000    0.000    0.000    0.000 _ufunc_config.py:430(__init__)                                                                                                                                   
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:4561(_quantile_is_valid)                                                                                                                  
       10    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}                                                                                                                               
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}                                                                                                                                   
       15    0.000    0.000    0.000    0.000 {built-in method builtins.setattr}                                                                                                                               
       39    0.000    0.000    0.000    0.000 {built-in method builtins.len}           

- Memory current / peak: (fill in after first run)
{ clip-login-0:~/claude/GOLLUM/ML/BIT }$ memray stats benchmark.bin
📏 Total allocations:
    17321848

📦 Total memory allocated:
    217.067GB

📊 Histogram of allocation size:
    min: 1.000B
    ----------------------------------------------
    < 6.000B   :  101813 ▇
    < 43.000B  :  817168 ▇▇▇
    < 282.000B : 8000678 ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
    < 1.808KB  : 7803531 ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
    < 11.858KB :  242286 ▇
    < 77.790KB :   58730 ▇
    < 510.266KB:  108477 ▇
    < 3.269MB  :  186866 ▇
    < 21.441MB :    1229 ▇
    <=140.640MB:    1070 ▇
    ----------------------------------------------
    max: 140.640MB

📂 Allocator type distribution:
     MALLOC: 17136975
     POSIX_MEMALIGN: 95337
     REALLOC: 70508
     CALLOC: 17274
     MMAP: 1754

🥇 Top 5 largest allocating locations (by size):
    - <stack trace unavailable> -> 165.051GB
    - get_split_vectorized:/users/robert.schoefbeck/claude/GOLLUM/ML/BIT/../../ML/BIT/NumbaMultiNode.py:378 -> 30.291GB
    - <listcomp>:/users/robert.schoefbeck/claude/GOLLUM/pdf/PODBasis.py:296 -> 4.912GB
    - _wrapfunc:/groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-claude/lib/python3.10/site-packages/numpy/_core/fromnumeric.py:57 -> 4.607GB
    - get_data:<frozen importlib._bootstrap_external>:1073 -> 1.403GB

🥇 Top 5 largest allocating locations (by number of allocations):
    - <listcomp>:/users/robert.schoefbeck/claude/GOLLUM/pdf/PODBasis.py:296 -> 12291012
    - evaluate:/users/robert.schoefbeck/claude/GOLLUM/pdf/PODBasis.py:294 -> 2048968
    - <stack trace unavailable> -> 1877441
    - <listcomp>:/users/robert.schoefbeck/claude/GOLLUM/pdf/PODBasis.py:243 -> 129469
    - __init__:/groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-claude/lib/python3.10/ctypes/__init__.py:374 -> 121684

- Notes:

### Iteration 01
- postfix: `iter-01-...`
- What was changed:
- CPU hotspots before/after:
- Memory before/after:
- Verdict: improvement / no change / regression

*(add iterations here)*

## Ground Rules
- Always explain what you are about to do and why before making changes
- Never change algorithmic correctness — only optimize numerics and data throughput
- After each change, run the full benchmark and compare to the previous iteration
- If a change is a regression, revert it and note why
- Commit working improvements to git with a descriptive message before moving to the next iteration
- If unsure whether a heavier test is needed, increase `--max_n_files` to 3 before concluding

## Known Optimization Targets (starting hypotheses)
- `vectorized_predict` uses `eval()` on dynamically constructed strings — replace with proper numpy tree traversal
- Feature loop in `get_split_vectorized` is sequential — `numba.prange` could parallelize across features
- `np.cumsum` in exact mode allocates full `(N, D)` copy of `training_weights` — potential spike source
- Memory layout of `training_weights`: accessed row-wise in cumsum but column-wise in Numba kernels — transposing upfront may help cache performance
