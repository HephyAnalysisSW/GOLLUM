import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
'''Implement the model from https://arxiv.org/pdf/1706.01816.pdf 
'''

import ROOT
import csv
import os
import array

dir_path = os.path.dirname(os.path.realpath(__file__))

h_pdf = {} 
for c_pdf in [ "u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar", "b", "bbar", "gluon"]:

    with open(os.path.join(dir_path, "./pdf_data/pdf_%s.txt"%c_pdf)) as f:
        reader = csv.reader(f)
        data = list(reader)
        thresholds = []
        values     = []
        for s_thr, s_val in data:
            thresholds.append(float(s_thr))
            values.append(float(s_val))
        h_pdf[c_pdf] = ROOT.TH1D( c_pdf, c_pdf, len(thresholds), array.array('d', thresholds+[1.001] ) )
        for thr, val in zip(thresholds, values):
            h_pdf[c_pdf].SetBinContent( h_pdf[c_pdf].FindBin( thr ), val ) 

pdg = {1:"d", 2:"u", 3:"s", 4:"c", 5:"b", -1:"dbar", -2:"ubar", -3:"sbar", -4:"cbar", -5:"bbar", 21:"gluon"}

def pdf( x, f ):
    histo = h_pdf[f] if type(f)==str else h_pdf[pdg[f]]
    if x<histo.GetXaxis().GetXmin() or x>1:
        raise RuntimeError("Minimum is %5.5f, maximum is 1, you asked for %5.5f" %( histo.GetXaxis().GetXmin(), x))
    return max(0, histo.Interpolate(x))

# EWSB constants
e   = 0.3028 #sqrt(4*pi*alpha)
s2w = 0.23122
sw  = sqrt(s2w)
c2w = 1-s2w
cw  = sqrt(c2w)
g   = e/sw
gZ  = g/cw
v   = 246.22
# boson masses and widths
m = {
    'W':80.379,
    'Z':91.1876,
    'H':125.1,
    }
Gamma = {
    'W':2.085,
    'Z':2.4952,
    }

sqrt_2 = sqrt(2)

E_LHC       = 13000
s_hat_max   = E_LHC**2
#s_hat_clip  = 0.03
#s_hat_clip  = 0.5 #FIXME
s_hat_clip  = 0.03 

# Qq
Qq  = {1:-1./3., 2: 2./3., 3:-1/3., 4:2./3., 5:-1./3.}
T3q = {1:-.5,    2:.5, 3: -.5, 4:.5, 5:-.5}

# EFT settings, parameters, defaults
wilson_coefficients    = ['cHW', 'cHWtil', 'cHQ3']
tex                    = {"cHW":"C_{HW}", "cHWtil":"C_{H#tilde{W}}", "cHQ3":"C_{HQ^{(3)}}"}

default_eft_parameters = { 'Lambda':1000. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

first_derivatives = [('cHQ3',), ('cHW',), ('cHWtil',)]
second_derivatives= [('cHQ3','cHQ3'), ('cHW','cHW'), ('cHWtil', 'cHWtil'), ('cHQ3','cHW'), ('cHQ3','cHWtil'), ('cHW', 'cHWtil'), ]
derivatives       = [tuple()] + first_derivatives + second_derivatives

# minimum transverse Z mass in this model
pT_min = 0

def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in wilson_coefficients+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )
sm         = make_eft()

feature_names =  ['sqrt_s_hat', 'pT', 'y', 'cos_theta', 'phi_hat', 'cos_theta_hat', 
                  'fLL', 'f1TT', 'f2TT' , 'f1LT', 'f2LT', 'f1tildeLT', 'f2tildeLT', 'fTTprime', 'ftildeTTprime']

def make_s_cos_theta( N_events, pT_min = pT_min):

    # theta of boson in the qq restframe
    cos_theta = np.random.uniform(-1,1,N_events)

    # kinematics
    s_hat_min   = (m['H'] + m['Z'])**2
    #s_hat_min   = (2*pT_min)**2 #minimum s_hat for sin(theta)=1
    s_hat       = s_hat_min+(s_hat_max-s_hat_min)*np.random.uniform(0, s_hat_clip, N_events)
    sqrt_s_hat  = np.sqrt(s_hat)

    #pT          = 0.5*sqrt_s_hat*np.sqrt(1-cos_theta**2)
    pT          = np.sqrt(0.25*(1-cos_theta**2)/s_hat*(s_hat-(m['H']+m['Z'])**2)*(s_hat-(m['H']-m['Z'])**2))

    return s_hat, sqrt_s_hat, pT, cos_theta

# qq -> ZH
def _getEvents(N_events_requested):

    # correct efficiency of pt cut
    _, _, pT, _ = make_s_cos_theta( N_events_requested, pT_min = pT_min)
    N_events_corrected = int(round(N_events_requested/(np.count_nonzero(pT>pT_min)/float(N_events_requested))))

    # simulate events and compensate pT efficiency
    s_hat, sqrt_s_hat, pT, cos_theta = make_s_cos_theta( N_events_corrected, pT_min = pT_min)

    # apply selection 
    sel = pT>pT_min
    s_hat, sqrt_s_hat, pT, cos_theta = s_hat[sel], sqrt_s_hat[sel], pT[sel], cos_theta[sel]
    N_events = len(s_hat)
    # remind myself
    print(("Requested %i events. Simulated %i events and %i survive pT_min cut of %i." %( N_events_requested, N_events_corrected, N_events, pT_min ) ))

    phi_hat = pi*np.random.uniform(-1,1,N_events)
    cos_theta_hat = np.random.uniform(-1,1,N_events)

    x_min       = np.sqrt( s_hat/s_hat_max )
    abs_y_max   = - np.log(x_min)
    y           = np.random.uniform(-1,1, N_events)*abs_y_max

    # Eq. 3.6 of Spannowsky (https://arxiv.org/pdf/1912.07628.pdf)
    C2_theta     =  cos_theta**2  
    C2_theta_hat =  cos_theta_hat**2  

    S2_theta     =  1. - C2_theta 
    S2_theta_hat =  1. - C2_theta_hat
    S_theta     = np.sqrt(S2_theta) 
    S_theta_hat = np.sqrt(S2_theta_hat) 
    C_phi_hat   = np.cos(phi_hat)
    S_phi_hat   = np.sin(phi_hat)
    C_2phi_hat   = np.cos(2*phi_hat)
    S_2phi_hat   = np.sin(2*phi_hat)

    fLL         = S2_theta*S2_theta_hat
    f1TT        = cos_theta*cos_theta_hat
    f2TT        = (1+C2_theta)*(1+C2_theta_hat)
    f1LT        = C_phi_hat*S_theta*S_theta_hat
    f2LT        = f1LT*cos_theta*cos_theta_hat 
    f1tildeLT   = S_phi_hat*S_theta*S_theta_hat
    f2tildeLT   = f1tildeLT*cos_theta*cos_theta_hat 
    fTTprime    = C_2phi_hat*fLL
    ftildeTTprime=S_2phi_hat*fLL

    return np.transpose(np.array( [sqrt_s_hat, pT, y, cos_theta, phi_hat, cos_theta_hat, fLL, f1TT, f2TT, f1LT, f2LT, f1tildeLT, f2tildeLT, fTTprime, ftildeTTprime]))

def _getWeights(features, eft):

    sqrt_s_hat    = features[:,feature_names.index('sqrt_s_hat')]
    y             = features[:,feature_names.index('y')]
    cos_theta     = features[:,feature_names.index('cos_theta')]
    phi_hat       = features[:,feature_names.index('phi_hat')]
    cos_theta_hat = features[:,feature_names.index('cos_theta_hat')]

    s_hat         = sqrt_s_hat**2
    sin_theta     = np.sqrt( 1. - cos_theta**2 ) 
    sin_theta_hat = np.sqrt( 1. - cos_theta_hat**2 ) 

    w           = (s_hat + m['Z']**2-m['H']**2)/(2*np.sqrt(s_hat))
    k           = np.sqrt( w**2-m['Z']**2 )
    #k_Z         = 0.5*sqrt_s_hat #Limit for large s_hat

    x1          = sqrt_s_hat/E_LHC*np.exp(y)
    x2          = sqrt_s_hat/E_LHC*np.exp(-y)

    Cf  = 1.

    #gZsigma[sigma_quark][pdg_quark]
    gZsigma = {1:{1: gZ*(-Qq[1]*s2w), 2: gZ*(-Qq[2]*s2w), 3:gZ*(-Qq[3]*s2w), 4:gZ*(-Qq[4]*s2w), 5:gZ*(-Qq[5]*s2w)}, 
              -1:{1: gZ*(T3q[1]-Qq[1]*s2w), 2:gZ*(T3q[2]-Qq[2]*s2w), 3:gZ*(T3q[3]-Qq[3]*s2w), 4:gZ*(T3q[4]-Qq[4]*s2w), 5:gZ*(T3q[5]-Qq[5]*s2w)}}
    gZtaull = {1:gZ*s2w, -1:gZ*(-0.5+s2w) }

    constZH   = 1 
    #constZH = 12288*pi**3*Gamma['Z']*E_LHC**2

    N_events  = len(features)
    dsigmaZH  = {der:np.zeros(N_events).astype('complex128') for der in derivatives}

    for pdg_quark in [1, 2, 3, 4, 5]:
        qx1     = np.array( [ pdf( x,  pdg_quark ) for x in x1 ] ) 
        qbarx2  = np.array( [ pdf( x, -pdg_quark ) for x in x2 ] ) 

        qbarx1  = np.array( [ pdf( x, -pdg_quark ) for x in x1 ] ) 
        qx2     = np.array( [ pdf( x,  pdg_quark ) for x in x2 ] ) 
        for sigma_quark in [+1, -1]:
            M_lambda_sigma_qbarq = {}
            M_lambda_sigma_qqbar = {}
            t3q      = T3q[pdg_quark] if sigma_quark == -1 else 0
            dtau         = {}
            dtau_flipped = {}

            kappa_ZZ           = 2*v**2/eft['Lambda']**2*cw**2*eft['cHW']
            kappa_Zgamma       = 2*v**2/eft['Lambda']**2*cw*sw*eft['cHW']

            kappa_tilde_ZZ     = 2*v**2/eft['Lambda']**2*cw**2*eft['cHWtil']
            kappa_Zgamma_tilde = 2*v**2/eft['Lambda']**2*cw*sw*eft['cHWtil']

            g_h_Z_sigma        = -2*g/cw*v**2/eft['Lambda']**2*(-t3q*eft['cHQ3'])
            propagator         = gZsigma[sigma_quark][pdg_quark]/(s_hat-m['Z']**2)

            for lambda_boson in [+1, -1, 0]:
                if abs(lambda_boson)==1:
                    prefac   = gZ*m['Z']*sqrt_s_hat 
                    #print c2w, pdg_quark, sigma_quark, gZsigma[sigma_quark][pdg_quark], "c2w_factor", c2w_factor, "2nd term", Qq[pdg_quark]*g*s2w*cw/gZsigma[sigma_quark][pdg_quark]

                    Mhat = {tuple()   : prefac*(propagator + g_h_Z_sigma/(2*m['Z']**2) + 0.5*(1+(s_hat-m['H']**2)/m['Z']**2)*(propagator*kappa_ZZ + Qq[pdg_quark]*e/s_hat*kappa_Zgamma ) -1j*lambda_boson*k*sqrt_s_hat/(m['Z']**2)*(propagator*kappa_tilde_ZZ + Qq[pdg_quark]*e/s_hat*kappa_Zgamma_tilde)),
                           ('cHQ3',)  : prefac*(-2*g/cw*v**2/eft['Lambda']**2*(-t3q))/(2*m['Z']**2),
                           ('cHW',)   : prefac*(0.5*(1+(s_hat-m['H']**2)/m['Z']**2)*(propagator*2*v**2/eft['Lambda']**2*cw**2 + Qq[pdg_quark]*e/s_hat*2*v**2/eft['Lambda']**2*cw*sw )),
                           ('cHWtil',): prefac*(-1j*lambda_boson*k*sqrt_s_hat/(m['Z']**2)*(propagator*2*v**2/eft['Lambda']**2*cw**2 + Qq[pdg_quark]*e/s_hat*2*v**2/eft['Lambda']**2*cw*sw)),
                            }

## FIXME drop Qe term from gamma
#                    Mhat = {tuple()   : prefac*(propagator + g_h_Z_sigma/(2*m['Z']**2) + 0.5*(1+(s_hat-m['H']**2)/m['Z']**2)*(propagator*kappa_ZZ ) -1j*lambda_boson*k*sqrt_s_hat/(m['Z']**2)*(propagator*kappa_tilde_ZZ + Qq[pdg_quark]*e/s_hat*kappa_Zgamma_tilde)),
#                           ('cHQ3',)  : prefac*(-2*g/cw*v**2/eft['Lambda']**2*(-t3q))/(2*m['Z']**2),
#                           ('cHW',)   : prefac*(0.5*(1+(s_hat-m['H']**2)/m['Z']**2)*(propagator*2*v**2/eft['Lambda']**2*cw**2 )),
#                           ('cHWtil',): prefac*(-1j*lambda_boson*k*sqrt_s_hat/(m['Z']**2)*(propagator*2*v**2/eft['Lambda']**2*cw**2)),
#                            }
                    #print Mhat
                    M_lambda_sigma_qqbar[lambda_boson] = {k: sigma_quark*(1+sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat[k] for k in list(Mhat.keys())} 
                    M_lambda_sigma_qbarq[lambda_boson] = {k:-sigma_quark*(1-sigma_quark*lambda_boson*cos_theta)/sqrt(2.)*Mhat[k] for k in list(Mhat.keys())}
                else:
                    prefac = -gZ*w*sin_theta*sqrt_s_hat
                    M_lambda_sigma_qqbar[lambda_boson] = \
                           {tuple()   : prefac*(propagator + g_h_Z_sigma/(2*m['Z']**2)+0.5*(1+(s_hat-m['H']**2)/m['Z']**2-2*k**2*sqrt_s_hat/(m['Z']**2*w))*(propagator*kappa_ZZ+ Qq[pdg_quark]*e/s_hat*kappa_Zgamma)),
                           ('cHW',)   : prefac*(+0.5*(1+(s_hat-m['H']**2)/m['Z']**2-2*k**2*sqrt_s_hat/(m['Z']**2*w))*(propagator*2*v**2/eft['Lambda']**2*cw**2+ Qq[pdg_quark]*e/s_hat*2*v**2/eft['Lambda']**2*cw*sw)), 
                           ('cHQ3',)  : prefac*(-2*g/cw*v**2/eft['Lambda']**2*(-t3q))/(2*m['Z']**2),
                           ('cHWtil',): np.zeros(N_events),
                            }

## FIXME drop Qe term from gamma
#                    M_lambda_sigma_qqbar[lambda_boson] = \
#                           {tuple()   : prefac*(propagator + g_h_Z_sigma/(2*m['Z']**2)+0.5*(1+(s_hat-m['H']**2)/m['Z']**2-2*k**2*sqrt_s_hat/(m['Z']**2*w))*(propagator*kappa_ZZ)),
#                           ('cHW',)   : prefac*(+0.5*(1+(s_hat-m['H']**2)/m['Z']**2-2*k**2*sqrt_s_hat/(m['Z']**2*w))*(propagator*2*v**2/eft['Lambda']**2*cw**2)), 
#                           ('cHQ3',)  : prefac*(-2*g/cw*v**2/eft['Lambda']**2*(-t3q))/(2*m['Z']**2),
#                           ('cHWtil',): np.zeros(N_events),
#                            }
                    M_lambda_sigma_qbarq[lambda_boson] = M_lambda_sigma_qqbar[lambda_boson]

                dtau[lambda_boson] = {}
                dtau_flipped[lambda_boson] = {}
                ## flipped dtau:
                ## theta_hat -> pi - theta_hat
                ## phi       -> pi + phi_hat
                ## sin(pi-theta) = sin(theta), cos(pi-theta) = -cos(theta), sin(pi+phi) = -sin(phi), cos(pi+phi) = -cos(phi)
                for tau in [+1, -1]:
                    if abs(tau)==1:
                        dtau[lambda_boson][tau]         = tau*(1+lambda_boson*tau*cos_theta_hat)/sqrt(2.)*np.exp(1j*lambda_boson*phi_hat) 
                        dtau_flipped[lambda_boson][tau] = tau*(1-lambda_boson*tau*cos_theta_hat)/sqrt(2.)*( -np.exp(1j*lambda_boson*phi_hat))
                    else:
                        dtau[lambda_boson][tau]         = sin_theta_hat 
                        dtau_flipped[lambda_boson][tau] = sin_theta_hat 

            for lam1 in [+1, -1, 0]:
                for lam2 in [+1, -1, 0]:
                    for tau in [+1, -1]:
                        for flip_tau in [False]:#, True]:

                            if not flip_tau:
                                dtau_lam1_tau = dtau[lam1][tau]
                                dtau_lam2_tau = dtau[lam2][tau]
                            else:
                                dtau_lam1_tau = dtau_flipped[lam1][tau]
                                dtau_lam2_tau = dtau_flipped[lam2][tau]

                            dsigmaZH_prefac = m['Z']*k/(constZH*s_hat**1.5)*(gZtaull[tau]**2 if not flip_tau else gZtaull[-tau]**2)*Cf
                            qx1_qbarx2      = qx1*qbarx2
                            qbarx1_qx2      = qbarx1*qx2 
                            dsigmaZH[tuple()] += dsigmaZH_prefac*(
                                  qx1_qbarx2*np.conjugate(dtau_lam1_tau)*np.conjugate(M_lambda_sigma_qqbar[lam1][tuple()])*M_lambda_sigma_qqbar[lam2][tuple()]*dtau_lam2_tau
                                + qbarx1_qx2*np.conjugate(dtau_lam1_tau)*np.conjugate(M_lambda_sigma_qbarq[lam1][tuple()])*M_lambda_sigma_qbarq[lam2][tuple()]*dtau_lam2_tau
                            )
                            for der in first_derivatives:
                                dsigmaZH[der] += dsigmaZH_prefac*(
                                     qx1_qbarx2*np.conjugate(dtau_lam1_tau)*(
                                        np.conjugate(M_lambda_sigma_qqbar[lam1][der])    *M_lambda_sigma_qqbar[lam2][tuple()]
                                       +np.conjugate(M_lambda_sigma_qqbar[lam1][tuple()])*M_lambda_sigma_qqbar[lam2][der])*dtau_lam2_tau
                                   + qbarx1_qx2*np.conjugate(dtau_lam1_tau)*(
                                        np.conjugate(M_lambda_sigma_qbarq[lam1][der])    *M_lambda_sigma_qbarq[lam2][tuple()]
                                       +np.conjugate(M_lambda_sigma_qbarq[lam1][tuple()])*M_lambda_sigma_qbarq[lam2][der])*dtau_lam2_tau
                                )
                            for der in second_derivatives:
                                dsigmaZH[der] += dsigmaZH_prefac*(
                                     qx1_qbarx2*np.conjugate(dtau_lam1_tau)*(
                                        np.conjugate(M_lambda_sigma_qqbar[lam1][(der[0],)])*M_lambda_sigma_qqbar[lam2][(der[1],)]
                                       +np.conjugate(M_lambda_sigma_qqbar[lam1][(der[1],)])*M_lambda_sigma_qqbar[lam2][(der[0],)])*dtau_lam2_tau
                                   + qbarx1_qx2*np.conjugate(dtau_lam1_tau)*(
                                        np.conjugate(M_lambda_sigma_qbarq[lam1][(der[0],)])*M_lambda_sigma_qbarq[lam2][(der[1],)]
                                       +np.conjugate(M_lambda_sigma_qbarq[lam1][(der[1],)])*M_lambda_sigma_qbarq[lam2][(der[0],)])*dtau_lam2_tau
                                )
    # Check(ed) that residual imaginary parts are tiny
    dsigmaZH  = {k:np.real(dsigmaZH[k])  for k in derivatives}
    return dsigmaZH

def getEvents( nTraining, eft=default_eft_parameters, make_weights = True, return_observers=None):

    training_features = _getEvents(nTraining)
    if make_weights:
        if return_observers:
            return training_features, _getWeights(training_features, eft=eft), None
        else:
            return training_features, _getWeights(training_features, eft=eft)
    else:
        if return_observers:
            return training_features
        else:
            return training_features, None

Nbins = 20
plot_options = {
    #'sqrt_s_hat': {'binning':[40,650,2250],      'tex':"#sqrt{#hat{s}}",},
    #'pT':         {'binning':[35,300,1000],      'tex':"p_{T}",},
    'sqrt_s_hat': {'binning':[30,0,1800],           'tex':"#sqrt{#hat{s}}",},
    'pT':         {'binning':[30,0,900],            'tex':"p_{T,Z}",},
    'y':          {'binning':[Nbins,-4,4],          'tex':"y",},
    'cos_theta':  {'binning':[Nbins,-1,1],          'tex':"cos(#Theta)",},
    'cos_theta_hat': {'binning':[Nbins,-1,1],       'tex':"cos(#hat{#theta})",},
    'phi_hat':    {'binning':[Nbins,-pi,pi],        'tex':"#hat{#phi}",},
    'fLL'         : {'binning':[Nbins,0,1],         'tex':'f_{LL}'          ,},
    'f1TT'        : {'binning':[Nbins,-1,1],        'tex':'f_{1TT}'         ,},
    'f2TT'        : {'binning':[Nbins, 0,4],        'tex':'f_{2TT}'         ,},
    'f1LT'        : {'binning':[Nbins,-1,1],        'tex':'f_{1LT}'         ,},
    'f2LT'        : {'binning':[Nbins,-1,1],        'tex':'f_{2LT}'         ,},
    'f1tildeLT'   : {'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{1LT}' ,},
    'f2tildeLT'   : {'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{2LT}' ,},
    'fTTprime'    : {'binning':[Nbins,-1,1],        'tex':'f_{TT}'     ,},
    'ftildeTTprime':{'binning':[Nbins,-1,1],        'tex':'#tilde{f}_{TT}',},
    }

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},
    {'color':ROOT.kBlue+2,      'point':make_eft(cHQ3 = .02),  'tex':"c_{HQ}^{(3)} = 0.03"},
    {'color':ROOT.kBlue-4,      'point':make_eft(cHQ3 = -.02), 'tex':"c_{HQ}^{(3)} = -0.03"},
    {'color':ROOT.kGreen+2,     'point':make_eft(cHW = 0.3),   'tex':"c_{HW} = 0.3"},
    {'color':ROOT.kGreen-4,     'point':make_eft(cHW = -0.3),  'tex':"c_{HW} = -0.3"},
    {'color':ROOT.kMagenta+2,   'point':make_eft(cHWtil = 0.3),   'tex':"c_{H#tilde{W}} = 0.3"},
    {'color':ROOT.kMagenta-4,   'point':make_eft(cHWtil = -0.3),  'tex':"c_{H#tilde{W}} = -0.3"},
]
