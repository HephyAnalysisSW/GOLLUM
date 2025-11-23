# Toy Generation and Confidence Level (CL) Evaluation

**Important Notes**

- The scripts currently work for hardcoded c1=1e-3 hypothesis only.
- I am not quite sure of the validity of the codes, a lot of checks must be performed.
- There might be hidden issues, as much of the repository is coded with the aid of ChatGPT.

---

### Scripts

- `user/kaan/generate_toys_v2.py` — Generates toy datasets and computes the test statistic for each toy.
- `user/kaan/check_CL_v2.py` — Evaluates the Asimov test statistic and computes the expected CL using the generated toys.

### Toy generation
`python -u user/kaan/generate_toys_v2.py configs/unbinned_merged.yaml --mode fixed --fixed-N 1000 --n-toys 100 >user/kaan/out.log`

### CL check
`python user/kaan/check_CL_v2.py configs/unbinned_merged.yaml --c1 1e-3`


