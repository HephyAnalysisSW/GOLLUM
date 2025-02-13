import   numpy as np
import   ROOT
import   array

def make_TH1F( h, ignore_binning = False):
    # remove infs from thresholds
    vals, thrs = h
    if ignore_binning:
        histo = ROOT.TH1F("h","h",len(vals),0,len(vals))
    else:
        histo = ROOT.TH1F("h","h",len(thrs)-1,array.array('d', thrs))
    for i_v, v in enumerate(vals):
        if v<float('inf'): # NAN protection
            histo.SetBinContent(i_v+1, v)
    return histo

def make_TGraph( coords ):
    tgraph = ROOT.TGraph(len(coords), array.array('d', [c[0] for c in coords]), array.array('d', [c[1] for c in coords]))
    return tgraph

def make_TH2F( h, ignore_binning = False):
    # remove infs from thresholds
    vals, thrs_x, thrs_y = h
    if ignore_binning:
        histo = ROOT.TH2F("h","h",len(vals[0]),0,len(vals[0]),len(vals),0,len(vals))
    else:
        histo = ROOT.TH2F("h","h",len(thrs_x)-1,array.array('d', thrs_x),len(thrs_y)-1,array.array('d', thrs_y))
    for ix, _ in enumerate(vals):
        for iy, v in enumerate(vals[ix]):
            if v<float('inf'): # NAN protection
                histo.SetBinContent(histo.FindBin(thrs_x[ix], thrs_y[iy]), v)
    return histo

import os, shutil
def copyIndexPHP( directory ):
    ''' Copy index.php to directory
    '''
    index_php = os.path.join( directory, 'index.php' )
    if not os.path.exists( directory ): os.makedirs( directory )
    shutil.copyfile( os.path.join(os.path.dirname(__file__), 'scripts/php/index.php'), index_php )

def coveragePenalty(coverage, Ntoys):
    sigma_68 = np.sqrt( (1-0.6827)*0.6827/Ntoys)
    if coverage < 0.6827 - 2*sigma_68:
        f_penalty = 1 + ( (coverage - (0.6827-2*sigma_68))/sigma_68 )**4
    elif coverage > 0.6827 + 2*sigma_68:
        f_penalty = 1 + ( (coverage - (0.6827+2*sigma_68))/sigma_68 )**3
    else:
        f_penalty = 1
    return f_penalty

def calculateScore(mu_true, mu_down, mu_up):
    width_sum = 0
    c_sum = 0
    Ntoys = 0
    Ntoys = len(mu_true)
    if len(mu_up) != Ntoys:
        print("[ERROR] upper limits have different length than true mu" )
        return None
    if len(mu_down) != Ntoys:
        print("[ERROR] lower limits have different length than true mu" )
        return None
    for i in range(len(mu_true)):
        width_sum += mu_up[i] - mu_down[i]
        if mu_true[i] > mu_down[i] and mu_true[i] < mu_up[i]:
            c_sum += 1

    # Calculate the score
    average_width = width_sum/Ntoys
    coverage = c_sum/Ntoys
    f_penalty = coveragePenalty(coverage, Ntoys)
    epsilon = 0.01
    score = -np.log( (average_width+epsilon)*f_penalty )
    return score
