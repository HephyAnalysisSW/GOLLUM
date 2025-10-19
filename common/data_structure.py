import math


# data structure
labels = ['tsch', 'ttch', 'tWch', 'TT']
label_encoding = {0:"tsch", 1:"ttch", 2:"tWch", 3:"TT", "tsch":0, "ttch":1, "tWch":2, "TT":3}

feature_names = [ 
        "top_pt", "top_eta", "top_phi", "top_m", "top_other_pt", "top_other_eta", "top_other_phi", "top_other_m", 
        "W_pt", "W_eta", "W_phi", "nu_pt", "nu_eta", "nu_phi", 
        "mT", "mlb0", "mlb1", "pT_bb", "cosTheta_lb_topRF", "FW3_R",
        "DEta_l_b0", "DEta_l_b1", "DEta_top_b0", "DEta_top_b1", "Cos_DPhi_top_b0", "Cos_DPhi_top_b1", "cos_DPhi_l_met" ]

weight_index = len(feature_names)
label_index  = weight_index+1

input_data = {
    'tsch': 
        [
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/t_sch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/t_sch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/t_sch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/t_sch',
        ],
    'ttch':
        [
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/TBar_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/TBar_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/TBar_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/TBar_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/T_tch_pow',
        ],
    'tWch':
        [
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/TBar_tWch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/TBar_tWch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/TBar_tWch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/TBar_tWch',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/T_tch_pow',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/T_tch_pow',
        ],
    'TT':
        [
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/TTSingleLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/TTSingleLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/TTSingleLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/TTSingleLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016_preVFP/singlelep-njet2p-met30/TTLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2016/singlelep-njet2p-met30/TTLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2017/singlelep-njet2p-met30/TTLep_pow_CP5',
            '/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/TT2lUnbinned_v8/UL2018/singlelep-njet2p-met30/TTLep_pow_CP5',
        ],
}


# for constructing filenames
systematics     = ['tes', 'jes', 'met']
default_values  = (1, 1, 0)

plot_styles = {'tsch': {'tex':'t (s-ch.)','fill_color': 807, 'line_color': 807, 'line_width': 2},
               'ttch': {'tex':'t (t-ch.)','fill_color': 418, 'line_color': 1, 'line_width': 1},
               'tWch': {'tex':'tW',       'fill_color': 618, 'line_color': 1, 'line_width': 1},
               'TT':   {'tex':'t#bar{t}', 'fill_color': 600, 'line_color': 1, 'line_width': 1}}

plot_options = {
"top_pt"               :{'logY':True,  'tex':"p_{T}(t) (GeV)",           'binning':[30, 0, 600                     ], 'y_ratio_range':[0.92, 1.08]},
"top_eta"              :{'logY':False, 'tex':"#eta(t)",                  'binning':[30, -3, 3                      ], 'y_ratio_range':[0.95, 1.05]},
"top_phi"              :{'logY':False, 'tex':"#phi(t)",                  'binning':[30, -math.pi, math.pi          ], 'y_ratio_range':[0.95, 1.05]},
"top_m"                :{'logY':True,  'tex':"M(t)",                     'binning':[30, 0, 600                     ], 'y_ratio_range':[0.8,  1.2 ]},
"top_other_pt"         :{'logY':True,  'tex':"p_{T}(t_2) (GeV)",         'binning':[30, 0, 600                     ], 'y_ratio_range':[0.95, 1.05]},
"top_other_eta"        :{'logY':False, 'tex':"#eta(t_2)",                'binning':[30, -3, 3                      ], 'y_ratio_range':[0.95, 1.05]},
"top_other_phi"        :{'logY':False, 'tex':"#phi(t_2)",                'binning':[30, -math.pi, math.pi          ], 'y_ratio_range':[0.8,  1.2 ]},
"top_other_m"          :{'logY':True,  'tex':"M(t_2)",                   'binning':[30, 0, 600                     ], 'y_ratio_range':[0.8,  1.2 ]},
"W_pt"                 :{'logY':True,  'tex':"p_{T}(W) (GeV)",           'binning':[30, 0, 600                     ], 'y_ratio_range':[0.95, 1.05]},
"W_eta"                :{'logY':True,  'tex':"#eta(W)",                  'binning':[30, -3, 3                      ], 'y_ratio_range':[0.7,  1.3 ]},
"W_phi"                :{'logY':False, 'tex':"#phi(W)",                  'binning':[30, -math.pi, math.pi          ], 'y_ratio_range':[0.9,  1.1 ]},
"nu_pt"                :{'logY':True,  'tex':"p_{T}(#nu) (GeV)",         'binning':[30, 0, 600                     ], 'y_ratio_range':[0.7,  1.3 ]},
"nu_eta"               :{'logY':False, 'tex':"#eta(#nu)",                'binning':[30, -3, 3                      ], 'y_ratio_range':[0.8,  1.2 ]},
"nu_phi"               :{'logY':False, 'tex':"#phi(#nu)",                'binning':[30, -math.pi, math.pi          ], 'y_ratio_range':[0.8,  1.2 ]},
"mT"                   :{'logY':True,  'tex':"M_{T}",                    'binning':[40, 0, 400                     ], 'y_ratio_range':[0.95, 1.05]},
"mlb0"                 :{'logY':True,  'tex':"M(l,b_0)",                 'binning':[30, 0, 600                     ], 'y_ratio_range':[0.9,  1.1 ]},
"mlb1"                 :{'logY':True,  'tex':"M(l,b_1)",                 'binning':[30, 0, 600                     ], 'y_ratio_range':[0.8,  1.2 ]},
"pT_bb"                :{'logY':True,  'tex':"p_T(bb)",                  'binning':[30, 0, 600                     ], 'y_ratio_range':[0.95, 1.05]},
"cosTheta_lb_topRF"    :{'logY':False, 'tex':"cos(#theta^#ast)",         'binning':[30, -1, 1                      ], 'y_ratio_range':[0.8,  1.2 ]},
"FW3_R"                :{'logY':False, 'tex':"FW3_R",                    'binning':[30, 0, 1                       ], 'y_ratio_range':[0.9,  1.1 ]},
"DEta_l_b0"            :{'logY':False, 'tex':"#Delta#eta(l,b_0)",        'binning':[30, 0, 3                      ], 'y_ratio_range':[0.8,  1.2 ]},
"DEta_l_b1"            :{'logY':False, 'tex':"#Delta#eta(l,b_1)",        'binning':[30, 0, 3                      ], 'y_ratio_range':[0.9,  1.1 ]},
"DEta_top_b0"          :{'logY':False, 'tex':"#Delta#eta(t,b_0)",        'binning':[30, 0, 3                      ], 'y_ratio_range':[0.9,  1.1 ]},
"DEta_top_b1"          :{'logY':False, 'tex':"#Delta#eta(t,b_1)",        'binning':[30, 0, 3                      ], 'y_ratio_range':[0.95, 1.05]},
"Cos_DPhi_top_b0"      :{'logY':False, 'tex':"cos(#Delta#phi(t,b_0))",   'binning':[30, -1, 1                      ], 'y_ratio_range':[0.95, 1.05]}, 
"Cos_DPhi_top_b1"      :{'logY':False, 'tex':"cos(#Delta#phi(t,b_1))",   'binning':[30, -1, 1                      ], 'y_ratio_range':[0.95, 1.05]},
"cos_DPhi_l_met"       :{'logY':False, 'tex':"cos(#Delta#phi(l,met))",   'binning':[30, -1, 1                      ], 'y_ratio_range':[0.95, 1.05]},
}



colors = [ 600,
           632,
           418,
           800,
           616,
           432,
           402,
           910,
           882,
           825,
           843,
           866,
           807,
           922,
           407,
           603,
           634,
           613,
           796,
           827,
           835,
           434,
           905,
           870,
           404,
           876,
           413,
           595,
           625,
           620,
           809,
           832,
           427,
           854,
           393]
