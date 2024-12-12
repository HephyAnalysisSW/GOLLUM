import math

# data structure

labels = ['htautau', 'ztautau', 'ttbar', 'diboson']
label_encoding = {0:"htautau", 1:"ztautau", 2:"ttbar", 3:"diboson", "htautau":0, "ztautau":1, "ttbar":2, "diboson":3}

feature_names = ["PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi","PRI_had_pt", "PRI_had_eta", "PRI_had_phi","PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi","PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi","PRI_n_jets","PRI_jet_all_pt","PRI_met", "PRI_met_phi", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_had_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality", "DER_lep_eta_centrality", ]

weight_index = len(feature_names)
label_index  = weight_index+1

# for constructing filenames
systematics     = ['tes', 'jes', 'met']
default_values  = (1, 1, 0)

# plot styles
import ROOT
plot_styles = {
        "htautau": {"fill_color": ROOT.kOrange+7,  "line_color": ROOT.kOrange+7, "line_width": 2},
        "ztautau": {"fill_color": ROOT.kGreen + 2, "line_color": ROOT.kBlack,    "line_width": 1},
        "ttbar":   {"fill_color": ROOT.kMagenta+2, "line_color": ROOT.kBlack,    "line_width": 1},
        "diboson": {"fill_color": ROOT.kBlue,      "line_color": ROOT.kBlack,    "line_width": 1},
        }

plot_options = {
    "PRI_lep_pt"                    :{'logY':True,  'tex':"p_{T}(l)",                 'binning':[30, 20, 400                    ]},
    "PRI_lep_eta"                   :{'logY':False, 'tex':"#eta(l)",                  'binning':[30, -2.5, 2.5                  ]},
    "PRI_lep_phi"                   :{'logY':False, 'tex':"#phi(l)",                  'binning':[30, -math.pi, math.pi          ]},
    "PRI_had_pt"                    :{'logY':True,  'tex':"p_{T}(#tau)",              'binning':[30, 25, 400                    ]},
    "PRI_had_eta"                   :{'logY':False, 'tex':"#eta(#tau)",               'binning':[30, -3, 3                      ]},
    "PRI_had_phi"                   :{'logY':False, 'tex':"#phi(#tau)",               'binning':[30, -math.pi, math.pi          ]},
    "PRI_jet_leading_pt"            :{'logY':True,  'tex':"p_{T}(j_{0})",             'binning':[30, 25, 400                    ]},
    "PRI_jet_leading_eta"           :{'logY':False, 'tex':"#eta(j_{0})",              'binning':[30, -5, 5                      ]},
    "PRI_jet_leading_phi"           :{'logY':False, 'tex':"#phi(j_{0})",              'binning':[30, -math.pi, math.pi          ]},
    "PRI_jet_subleading_pt"         :{'logY':True,  'tex':"p_{T}(j_{1})",             'binning':[30, 25, 400                    ]},
    "PRI_jet_subleading_eta"        :{'logY':False, 'tex':"#eta(j_{1})",              'binning':[30, -5, 5                      ]},
    "PRI_jet_subleading_phi"        :{'logY':False, 'tex':"#phi(j_{1})",              'binning':[30, -math.pi, math.pi          ]},
    "PRI_n_jets"                    :{'logY':True,  'tex':"N_{jet}",                  'binning':[15, 0, 15                      ]},
    "PRI_jet_all_pt"                :{'logY':True,  'tex':"H_{T}",                    'binning':[30, 0, 1000                    ]},
    "PRI_met"                       :{'logY':True,  'tex':"E_{T}^{miss}",             'binning':[30, 0, 300                     ]},
    "PRI_met_phi"                   :{'logY':False, 'tex':"#phi(E_{T}^{miss})",       'binning':[30, -math.pi, math.pi          ]},
    "DER_mass_transverse_met_lep"   :{'logY':True,  'tex':"m_{T}",                    'binning':[30, 0.0, 200                   ]},
    "DER_mass_vis"                  :{'logY':True,  'tex':"m_{vis.}(#tau,l)",         'binning':[30, 0, 500                     ]},
    "DER_pt_h"                      :{'logY':True,  'tex':"p_{T}(h)",                 'binning':[30, 0, 600                     ]},
    "DER_deltaeta_jet_jet"          :{'logY':False, 'tex':"|#eta(j_{0})-#eta(j_{1})|",'binning':[30, 0, 6                       ]},
    "DER_mass_jet_jet"              :{'logY':True,  'tex':"m(j_{0},j_{1})",           'binning':[30, 0, 3000                    ]},
    "DER_prodeta_jet_jet"           :{'logY':False, 'tex':"#eta(j_{0})#eta(j_{1})",   'binning':[30,-25, 25                     ]},
    "DER_deltar_had_lep"            :{'logY':False, 'tex':"#Delta R(#tau, l)",        'binning':[31, 0, 6.2                     ]},
    "DER_pt_tot"                    :{'logY':True,  'tex':"p_{T}(h+l+met+jets)",      'binning':[30, 0, 400                     ]},
    "DER_sum_pt"                    :{'logY':True,  'tex':"S_{T}",                    'binning':[30, 0, 1500                    ]},
    "DER_pt_ratio_lep_tau"          :{'logY':False, 'tex':"p_{T}(l)/p_{T}(#tau)",     'binning':[30, 0, 10                      ]},
    "DER_met_phi_centrality"        :{'logY':False, 'tex':"C(E_{T}^{miss})",          'binning':[30, -math.sqrt(2), math.sqrt(2)]},
    "DER_lep_eta_centrality"        :{'logY':False, 'tex':"C(l, jets)",               'binning':[30, 0, 1                       ]},
    }                                                                              
