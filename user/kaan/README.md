# Toy Generation and Confidence Level (CL) Evaluation

**Important Notes**

- Please pay attention not to use someone else's scratch-cbe directory.
  
Step 1: generate toys  
Step 2: fit the toys  
Step 3: plot
  
---

### Scripts

- `user/kaan/generate_toys_v2.py` — Generates toy datasets and computes the test statistic for each toy.
- `user/kaan/fit_toys.py` — Minimises the likelihood with the POIs and the nuissances floating.
- `user/kaan/submit_toy_fits.py` — Batch submission for the toy fits.

### Notebooks (for plotting)

- `user/kaan/notebooks/plot_fit_results.ipynb` — Plot fit results

### Toy generation

Generate toys with a mock-up shape variation:

`python3 user/kaan/generate_toys.py configs/unbinned/unbinned_2016APV.yaml --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_2016APV.json --n-toys 1000 --shape_2 1.0`

Generate toys based on a target PDF:

`python3 user/kaan/generate_toys_v2.py configs/unbinned_v5/2016/unbinned_2016_6.yaml   PDF4LHC21_mc   --n-toys 10000   --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_unbinned_2016_6_unbinned_2016_v5.json`

### Single toy fit (on the login node)
`export OMP_NUM_THREADS=1`  

`python3 -u user/kaan/fit_toys.py configs/unbinned_v5/2016/unbinned_2016_6.yaml /scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_PDF4LHC21_mc_m0_rotate_N10000.npz --toy-number 0 --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_unbinned_2016_6_unbinned_2016_v5.json --print-every 1 --minuit-print-level 2 --out /scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_nominal_N10000/toy_0`

### Batch submission of the toy fits.

**Warning: Please modify the paths in this script to submit the fits on the correct toys.**  
`python3 user/kaan/submit_toy_fits.py`