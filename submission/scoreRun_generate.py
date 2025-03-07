from Model import Model

import h5py
import pandas as pd
import numpy as np
import os
import pickle
from toy_generator import run_pseudo_experiments
from tqdm import tqdm
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import common.user as user


import argparse
parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument("--mu", type=float, default=None)
parser.add_argument("--tes", type=float, default=None)
parser.add_argument("--jes", type=float, default=None)
parser.add_argument("--met", type=float, default=None)
parser.add_argument("--ttbar", type=float, default=None)
parser.add_argument("--diboson", type=float, default=None)
parser.add_argument("--bkg", type=float, default=None)
parser.add_argument("--postfix", type=str, default="")
parser.add_argument("--Ntoys", type=int, default=10)
parser.add_argument("--freeze", type=str, default=None)
parser.add_argument('--nJobs', action='store',type=int, default=1)
parser.add_argument('--job', action='store',type=int, default=0)
parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
parser.add_argument('--config_path', action='store', help='path to yaml config')
parser.add_argument('--tmp_path', action='store', help='path to tmp_data')
args = parser.parse_args()

initialSeed = 0
Ntoys_this_job = args.Ntoys
if args.nJobs > 1:
    args.postfix = args.postfix+"_"+str(args.job)
    initialSeed = (args.job)*(Ntoys_this_job)
    Ntoys_this_job = args.Ntoys // args.nJobs
    remainder = args.Ntoys % args.nJobs

    if args.job == args.nJobs-1:
        Ntoys_this_job += remainder


print(f"Run {Ntoys_this_job} toys in this job.")

def coveragePenalty(coverage, Ntoys):
    sigma_68 = np.sqrt( (1-0.6827)*0.6827/Ntoys)
    if coverage < 0.6827 - 2*sigma_68:
        f_penalty = 1 + ( (coverage - (0.6827-2*sigma_68))/sigma_68 )**4
    elif coverage > 0.6827 + 2*sigma_68:
        f_penalty = 1 + ( (coverage - (0.6827+2*sigma_68))/sigma_68 )**3
    else:
        f_penalty = 1
    return f_penalty

width_sum = 0
Ntoys = 0
c_sum = 0

mu_true = []
mu_measured = []
mu_measured_up = []
mu_measured_down = []
mu_measured_up_scan = []
mu_measured_down_scan = []
toypaths = []

nu_tes = []
nu_jes = []
nu_met = []
nu_bkg = []
nu_tt = []
nu_diboson = []

limits = None
if args.freeze is not None:
    limits = {}
    if "jes" in args.freeze:
        limits["nu_jes"] = (-0.0001, 0.0001)
    if "tes" in args.freeze:
        limits["nu_tes"] = (-0.0001, 0.0001)
    if "met" in args.freeze:
        limits["nu_met"] = (0.0, 0.0001)
    if "bkg" in args.freeze:
        limits["nu_bkg"] = (-0.0001, 0.0001)
    if "ttbar" in args.freeze:
        limits["nu_tt"] = (-0.0001, 0.0001)
    if "diboson" in args.freeze:
        limits["nu_diboson"] = (-0.0001, 0.0001)


m = Model(get_train_set=None, systematics=None, config_path = args.config_path)
m.cfg["tmp_path"] = args.tmp_path

for i in tqdm(range(Ntoys_this_job)):
    seed = initialSeed+i
    if args.mu is not None:
        mu_input = args.mu
    else:
        mu_input = np.random.uniform(0.1, 3.0)

    toy_sample, trueValues = run_pseudo_experiments(
        tes=args.tes,
        jes=args.jes,
        soft_met=args.met,
        ttbar_scale=args.ttbar,
        diboson_scale=args.diboson,
        bkg_scale=args.bkg,
        num_pseudo_experiments=1,
        num_of_sets=1,
        ground_truth_mus=[mu_input],
        seed_input = seed
    )

    print(trueValues, f"seed = {seed}")
    # Run fit
    # results = m.predict(toy_sample, limits)
    results = m.predict(toy_sample)

    Ntoys += 1
    # Check fit results and compare with true values
    width_sum += abs(results["p84"] - results["p16"])
    if trueValues["mu"] > results["p16"] and trueValues["mu"] < results["p84"]:
        c_sum += 1

    mu_true.append(trueValues["mu"] )
    mu_measured.append(results["mu_hat"])
    mu_measured_up.append(results["p84"])
    mu_measured_down.append(results["p16"])
    # mu_measured_up_scan.append(results["p84_scan"])
    # mu_measured_down_scan.append(results["p16_scan"])
    # toypaths.append("generated")
    # nu_tes.append(results["nu_tes"])
    # nu_jes.append(results["nu_jes"])
    # nu_met.append(results["nu_met"])
    # nu_bkg.append(results["nu_bkg"])
    # nu_tt.append(results["nu_tt"])
    # nu_diboson.append(results["nu_diboson"])

# Calculate the score
average_width = width_sum/Ntoys
coverage = c_sum/Ntoys
f_penalty = coveragePenalty(coverage, Ntoys)
epsilon = 0.01
score = -np.log( (average_width+epsilon)*f_penalty )

# print results
print("=======================================================================")
print("NTOYS =", Ntoys)
print("COVERAGE =", coverage)
print("AVG. WIDTH =", average_width)
print("PENALTY =", f_penalty)
print("SCORE =", score)

import uuid
from datetime import datetime
unique_id = uuid.uuid4().hex  # Shorter hex representation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(user.output_directory, "toyFits", f"output_{args.postfix}_{timestamp}_{unique_id}.npz")

np.savez(filename,
    mu_true=np.array(mu_true),
    mu_measured=np.array(mu_measured),
    mu_measured_up=np.array(mu_measured_up),
    mu_measured_down=np.array(mu_measured_down),
    # mu_measured_up_scan=np.array(mu_measured_up_scan),
    # mu_measured_down_scan=np.array(mu_measured_down_scan),
    # toypaths=np.array(toypaths, dtype=object),
    # nu_tes=np.array(nu_tes),
    # nu_jes=np.array(nu_jes),
    # nu_met=np.array(nu_met),
    # nu_bkg=np.array(nu_bkg),
    # nu_tt=np.array(nu_tt),
    # nu_diboson=np.array(nu_diboson),
    )
print("Saved to file:", filename)
