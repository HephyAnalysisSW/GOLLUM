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
    "PRI_lep_pt"                    :{'logY':True,  'tex':"p_{T}(l)",                 'binning':[20, 20, 400                    ], 'y_ratio_range':[0.92, 1.08]},
    "PRI_lep_eta"                   :{'logY':False, 'tex':"#eta(l)",                  'binning':[20, -2.5, 2.5                  ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_lep_phi"                   :{'logY':False, 'tex':"#phi(l)",                  'binning':[20, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_had_pt"                    :{'logY':True,  'tex':"p_{T}(#tau)",              'binning':[20, 25, 400                    ], 'y_ratio_range':[0.8,  1.2 ]},
    "PRI_had_eta"                   :{'logY':False, 'tex':"#eta(#tau)",               'binning':[20, -3, 3                      ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_had_phi"                   :{'logY':False, 'tex':"#phi(#tau)",               'binning':[20, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_jet_leading_pt"            :{'logY':True,  'tex':"p_{T}(j_{0})",             'binning':[20, 25, 400                    ], 'y_ratio_range':[0.8,  1.2 ]},
    "PRI_jet_leading_eta"           :{'logY':False, 'tex':"#eta(j_{0})",              'binning':[20, -5, 5                      ], 'y_ratio_range':[0.8,  1.2 ]},
    "PRI_jet_leading_phi"           :{'logY':False, 'tex':"#phi(j_{0})",              'binning':[20, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_jet_subleading_pt"         :{'logY':True,  'tex':"p_{T}(j_{1})",             'binning':[20, 25, 400                    ], 'y_ratio_range':[0.7,  1.3 ]},
    "PRI_jet_subleading_eta"        :{'logY':False, 'tex':"#eta(j_{1})",              'binning':[20, -5, 5                      ], 'y_ratio_range':[0.9,  1.1 ]},
    "PRI_jet_subleading_phi"        :{'logY':False, 'tex':"#phi(j_{1})",              'binning':[20, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
    "PRI_n_jets"                    :{'logY':True,  'tex':"N_{jet}",                  'binning':[15, 0, 15                      ], 'y_ratio_range':[0.7,  1.3 ]},
    "PRI_jet_all_pt"                :{'logY':True,  'tex':"H_{T}",                    'binning':[20, 0, 1000                    ], 'y_ratio_range':[0.8,  1.2 ]},
    "PRI_met"                       :{'logY':True,  'tex':"E_{T}^{miss}",             'binning':[20, 0, 300                     ], 'y_ratio_range':[0.8,  1.2 ]},
    "PRI_met_phi"                   :{'logY':False, 'tex':"#phi(E_{T}^{miss})",       'binning':[20, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
    "DER_mass_transverse_met_lep"   :{'logY':True,  'tex':"m_{T}",                    'binning':[20, 0.0, 200                   ], 'y_ratio_range':[0.95, 1.05]},
    "DER_mass_vis"                  :{'logY':True,  'tex':"m_{vis.}(#tau,l)",         'binning':[20, 0, 500                     ], 'y_ratio_range':[0.9,  1.1 ]},
    "DER_pt_h"                      :{'logY':True,  'tex':"p_{T}(h)",                 'binning':[20, 0, 600                     ], 'y_ratio_range':[0.8,  1.2 ]},
    "DER_deltaeta_jet_jet"          :{'logY':False, 'tex':"|#eta(j_{0})-#eta(j_{1})|",'binning':[20, 0, 6                       ], 'y_ratio_range':[0.95, 1.05]},
    "DER_mass_jet_jet"              :{'logY':True,  'tex':"m(j_{0},j_{1})",           'binning':[20, 0, 3000                    ], 'y_ratio_range':[0.8,  1.2 ]},
    "DER_prodeta_jet_jet"           :{'logY':False, 'tex':"#eta(j_{0})#eta(j_{1})",   'binning':[20,-25, 25                     ], 'y_ratio_range':[0.8,  1.2 ]},
    "DER_deltar_had_lep"            :{'logY':False, 'tex':"#Delta R(#tau, l)",        'binning':[20, 0.2, 6.2                   ], 'y_ratio_range':[0.9,  1.1 ]},
    "DER_pt_tot"                    :{'logY':True,  'tex':"p_{T}(h+l+met+jets)",      'binning':[20, 0, 400                     ], 'y_ratio_range':[0.9,  1.1 ]},
    "DER_sum_pt"                    :{'logY':True,  'tex':"S_{T}",                    'binning':[20, 0, 1500                    ], 'y_ratio_range':[0.8,  1.2 ]},
    "DER_pt_ratio_lep_tau"          :{'logY':False, 'tex':"p_{T}(l)/p_{T}(#tau)",     'binning':[20, 0, 10                      ], 'y_ratio_range':[0.9,  1.1 ]},
    "DER_met_phi_centrality"        :{'logY':False, 'tex':"C(E_{T}^{miss})",          'binning':[20, -math.sqrt(2), math.sqrt(2)], 'y_ratio_range':[0.9,  1.1 ]},
    "DER_lep_eta_centrality"        :{'logY':False, 'tex':"C(l, jets)",               'binning':[20, 0, 1                       ], 'y_ratio_range':[0.95, 1.05]},
    }


colors = [
        ROOT.kBlue,           # 1
        ROOT.kRed,            # 2
        ROOT.kGreen + 2,      # 3
        ROOT.kOrange,         # 4
        ROOT.kMagenta,        # 5
        ROOT.kCyan,           # 6
        ROOT.kYellow + 2,     # 7
        ROOT.kPink + 10,      # 8
        ROOT.kViolet + 2,     # 9
        ROOT.kSpring + 5,     # 10
        ROOT.kTeal + 3,       # 11
        ROOT.kAzure + 6,      # 12
        ROOT.kOrange + 7,     # 13
        ROOT.kGray + 2,       # 14
        ROOT.kGreen - 9,      # 15
        ROOT.kBlue + 3,       # 16
        ROOT.kRed + 2,        # 17
        ROOT.kMagenta - 3,    # 18
        ROOT.kOrange - 4,     # 19
        ROOT.kSpring + 7,     # 20
        ROOT.kTeal - 5,       # 21
        ROOT.kCyan + 2,       # 22
        ROOT.kPink + 5,       # 23
        ROOT.kAzure + 10,     # 24
        ROOT.kYellow + 4,     # 25
        ROOT.kViolet - 4,     # 26
        ROOT.kGreen - 3,      # 27
        ROOT.kBlue - 5,       # 28
        ROOT.kRed - 7,        # 29
        ROOT.kMagenta + 4,    # 30
        ROOT.kOrange + 9,     # 31
        ROOT.kSpring + 12,    # 32
        ROOT.kCyan - 5,       # 33
        ROOT.kAzure - 6,      # 34
        ROOT.kYellow - 7      # 35
    ]

