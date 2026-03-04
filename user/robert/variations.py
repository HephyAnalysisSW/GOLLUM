from collections import OrderedDict

variations = OrderedDict( {
    'nominal'      : {},
    'alphaS'       : {'remove_weight':None, 'weight_up': 'pdf_alphas_up', 'weight_down':'pdf_alphas_dn'}, 
    'ren'          : {'remove_weight':None, 'weight_up': 'scale_ren2p0_fac1p0', 'weight_down':'scale_ren0p5_fac1p0'}, 
    'fac'          : {'remove_weight':None, 'weight_up': 'scale_ren1p0_fac2p0', 'weight_down':'scale_ren1p0_fac0p5'}, 
    'isr'          : {'remove_weight':None, 'weight_up': 'shower_isr2p0_fsr1p0', 'weight_down':'shower_isr0p5_fsr1p0'}, 
    'fsr'          : {'remove_weight':None, 'weight_up': 'shower_isr1p0_fsr2p0', 'weight_down':'shower_isr1p0_fsr0p5'}, 

    'res0_2016APV' : {'sys_up': 'CMS_res_j_0_2016APV_up', 'sys_down':'CMS_res_j_0_2016APV_down', 'eras':['2016APV']},
    'res1_2016APV' : {'sys_up': 'CMS_res_j_1_2016APV_up', 'sys_down':'CMS_res_j_1_2016APV_down', 'eras':['2016APV']},
    'res2_2016APV' : {'sys_up': 'CMS_res_j_2_2016APV_up', 'sys_down':'CMS_res_j_2_2016APV_down', 'eras':['2016APV']},
    'res3_2016APV' : {'sys_up': 'CMS_res_j_3_2016APV_up', 'sys_down':'CMS_res_j_3_2016APV_down', 'eras':['2016APV']},
    'res4_2016APV' : {'sys_up': 'CMS_res_j_4_2016APV_up', 'sys_down':'CMS_res_j_4_2016APV_down', 'eras':['2016APV']},
    'res5_2016APV' : {'sys_up': 'CMS_res_j_5_2016APV_up', 'sys_down':'CMS_res_j_5_2016APV_down', 'eras':['2016APV']},

    'res0_2016'    : {'sys_up': 'CMS_res_j_0_2016_up', 'sys_down':'CMS_res_j_0_2016_down', 'eras':['2016']},
    'res1_2016'    : {'sys_up': 'CMS_res_j_1_2016_up', 'sys_down':'CMS_res_j_1_2016_down', 'eras':['2016']},
    'res2_2016'    : {'sys_up': 'CMS_res_j_2_2016_up', 'sys_down':'CMS_res_j_2_2016_down', 'eras':['2016']},
    'res3_2016'    : {'sys_up': 'CMS_res_j_3_2016_up', 'sys_down':'CMS_res_j_3_2016_down', 'eras':['2016']},
    'res4_2016'    : {'sys_up': 'CMS_res_j_4_2016_up', 'sys_down':'CMS_res_j_4_2016_down', 'eras':['2016']},
    'res5_2016'    : {'sys_up': 'CMS_res_j_5_2016_up', 'sys_down':'CMS_res_j_5_2016_down', 'eras':['2016']},

    'res0_2017'    : {'sys_up': 'CMS_res_j_0_2017_up', 'sys_down':'CMS_res_j_0_2017_down', 'eras':['2017']},
    'res1_2017'    : {'sys_up': 'CMS_res_j_1_2017_up', 'sys_down':'CMS_res_j_1_2017_down', 'eras':['2017']},
    'res2_2017'    : {'sys_up': 'CMS_res_j_2_2017_up', 'sys_down':'CMS_res_j_2_2017_down', 'eras':['2017']},
    'res3_2017'    : {'sys_up': 'CMS_res_j_3_2017_up', 'sys_down':'CMS_res_j_3_2017_down', 'eras':['2017']},
    'res4_2017'    : {'sys_up': 'CMS_res_j_4_2017_up', 'sys_down':'CMS_res_j_4_2017_down', 'eras':['2017']},
    'res5_2017'    : {'sys_up': 'CMS_res_j_5_2017_up', 'sys_down':'CMS_res_j_5_2017_down', 'eras':['2017']},

    'res0_2018'    : {'sys_up': 'CMS_res_j_0_2018_up', 'sys_down':'CMS_res_j_0_2018_down', 'eras':['2018']},
    'res1_2018'    : {'sys_up': 'CMS_res_j_1_2018_up', 'sys_down':'CMS_res_j_1_2018_down', 'eras':['2018']},
    'res2_2018'    : {'sys_up': 'CMS_res_j_2_2018_up', 'sys_down':'CMS_res_j_2_2018_down', 'eras':['2018']},
    'res3_2018'    : {'sys_up': 'CMS_res_j_3_2018_up', 'sys_down':'CMS_res_j_3_2018_down', 'eras':['2018']},
    'res4_2018'    : {'sys_up': 'CMS_res_j_4_2018_up', 'sys_down':'CMS_res_j_4_2018_down', 'eras':['2018']},
    'res5_2018'    : {'sys_up': 'CMS_res_j_5_2018_up', 'sys_down':'CMS_res_j_5_2018_down', 'eras':['2018']},

#    'jes_b'        : {'sys_up': 'CMS_scale_j_FlavorPureBottom_up',      'sys_down': 'CMS_scale_j_FlavorPureBottom_down'},
#    'jes_c'        : {'sys_up': 'CMS_scale_j_FlavorPureCharm_up',       'sys_down': 'CMS_scale_j_FlavorPureCharm_down'},
#    'jeq_g'        : {'sys_up': 'CMS_scale_j_FlavorPureGluon_up',       'sys_down': 'CMS_scale_j_FlavorPureGluon_down'},
#    'jes_q'        : {'sys_up': 'CMS_scale_j_FlavorPureQuark_up',       'sys_down': 'CMS_scale_j_FlavorPureQuark_down'},
#    'jes_abs'      : {'sys_up': 'CMS_scale_j_Regrouped_Absolute_up',    'sys_down': 'CMS_scale_j_Regrouped_Absolute_down'},
#    'jes_BBEC1'    : {'sys_up': 'CMS_scale_j_Regrouped_BBEC1_up',       'sys_down': 'CMS_scale_j_Regrouped_BBEC1_down'},
#    'jes_EC2'      : {'sys_up': 'CMS_scale_j_Regrouped_EC2_up',         'sys_down': 'CMS_scale_j_Regrouped_EC2_down'},
#    'jes_HF'       : {'sys_up': 'CMS_scale_j_Regrouped_HF_up',          'sys_down': 'CMS_scale_j_Regrouped_HF_down'},
#    'jes_relbal'   : {'sys_up': 'CMS_scale_j_Regrouped_RelativeBal_up', 'sys_down': 'CMS_scale_j_Regrouped_RelativeBal_down'},
#
#    'jes_abs_16'   : {'sys_up': 'CMS_scale_j_Regrouped_Absolute_2016_up',       'sys_down': 'CMS_scale_j_Regrouped_Absolute_2016_down',       'eras':['2016APV', '2016']},
#    'jes_BBEC1_16' : {'sys_up': 'CMS_scale_j_Regrouped_BBEC1_2016_up',          'sys_down': 'CMS_scale_j_Regrouped_BBEC1_2016_down',          'eras':['2016APV', '2016']},
#    'jes_EC2_16'   : {'sys_up': 'CMS_scale_j_Regrouped_EC2_2016_up',            'sys_down': 'CMS_scale_j_Regrouped_EC2_2016_down',            'eras':['2016APV', '2016']},
#    'jes_HF_16'    : {'sys_up': 'CMS_scale_j_Regrouped_HF_2016_up',             'sys_down': 'CMS_scale_j_Regrouped_HF_2016_down',             'eras':['2016APV', '2016']},
#    'jes_relsam_16': {'sys_up': 'CMS_scale_j_Regrouped_RelativeSample_2016_up', 'sys_down': 'CMS_scale_j_Regrouped_RelativeSample_2016_down', 'eras':['2016APV', '2016']},
#
#    'jes_abs_17'   : {'sys_up': 'CMS_scale_j_Regrouped_Absolute_2017_up',        'sys_down':'CMS_scale_j_Regrouped_Absolute_2017_down',       'eras':['2017']},
#    'jes_BBEC1_17' : {'sys_up': 'CMS_scale_j_Regrouped_BBEC1_2017_up',          'sys_down': 'CMS_scale_j_Regrouped_BBEC1_2017_down',          'eras':['2017']},
#    'jes_EC2_17'   : {'sys_up': 'CMS_scale_j_Regrouped_EC2_2017_up',            'sys_down': 'CMS_scale_j_Regrouped_EC2_2017_down',            'eras':['2017']},
#    'jes_HF_17'    : {'sys_up': 'CMS_scale_j_Regrouped_HF_2017_up',             'sys_down': 'CMS_scale_j_Regrouped_HF_2017_down',             'eras':['2017']},
#    'jes_relsam_17': {'sys_up': 'CMS_scale_j_Regrouped_RelativeSample_2017_up', 'sys_down': 'CMS_scale_j_Regrouped_RelativeSample_2017_down', 'eras':['2017']},
#
#    'jes_abs_18'   : {'sys_up': 'CMS_scale_j_Regrouped_Absolute_2018_up',       'sys_down': 'CMS_scale_j_Regrouped_Absolute_2018_down',       'eras':['2018']},
#    'jes_BBEC1_18' : {'sys_up': 'CMS_scale_j_Regrouped_BBEC1_2018_up',          'sys_down': 'CMS_scale_j_Regrouped_BBEC1_2018_down',          'eras':['2018']},
#    'jes_EC2_18'   : {'sys_up': 'CMS_scale_j_Regrouped_EC2_2018_up',            'sys_down': 'CMS_scale_j_Regrouped_EC2_2018_down',            'eras':['2018']},
#    'jes_HF_18'    : {'sys_up': 'CMS_scale_j_Regrouped_HF_2018_up',             'sys_down': 'CMS_scale_j_Regrouped_HF_2018_down',             'eras':['2018']},
#    'jes_relsam_18': {'sys_up': 'CMS_scale_j_Regrouped_RelativeSample_2018_up', 'sys_down': 'CMS_scale_j_Regrouped_RelativeSample_2018_down', 'eras':['2018']},
    
    'jes_total_EtaBin0'        : {'sys_up': 'CMS_scale_j_Total_EtaBin0_up',      'sys_down': 'CMS_scale_j_Total_EtaBin0_down'},
    'jes_total_EtaBin1'        : {'sys_up': 'CMS_scale_j_Total_EtaBin1_up',      'sys_down': 'CMS_scale_j_Total_EtaBin1_down'},
    'jes_total_EtaBin2'        : {'sys_up': 'CMS_scale_j_Total_EtaBin2_up',      'sys_down': 'CMS_scale_j_Total_EtaBin2_down'},
    'jes_total_EtaBin3'        : {'sys_up': 'CMS_scale_j_Total_EtaBin3_up',      'sys_down': 'CMS_scale_j_Total_EtaBin3_down'},
    'jes_total_EtaBin4'        : {'sys_up': 'CMS_scale_j_Total_EtaBin4_up',      'sys_down': 'CMS_scale_j_Total_EtaBin4_down'},
    'jes_total_EtaBin5'        : {'sys_up': 'CMS_scale_j_Total_EtaBin5_up',      'sys_down': 'CMS_scale_j_Total_EtaBin5_down'},

    'jme_uncl'     : {'sys_up': 'Uncl_up', 'sys_down': 'Uncl_down'},

    'Ele'          : {'removeweight':'lepEle_SF', 'weight_up': 'lepEle_SFUp', 'weight_down':'lepEle_SFDn'}, 
    'Mu'           : {'removeweight':'lepMu_SF', 'weight_up': 'lepMu_SFUp', 'weight_down':'lepMu_SFDn'}, 
    'btag_b'       : {'removeweight':'btagSF_fixedWP_SF', 'weight_up': 'btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFUp', 'weight_down':'btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFDn'}, 
    'btag_l'       : {'removeweight':'btagSF_fixedWP_SF', 'weight_up': 'btagSF_fixedWP_SF__CMS_eff_b_light_SFUp', 'weight_down':'btagSF_fixedWP_SF__CMS_eff_b_light_SFDn'}, 
    'pu'           : {'removeweight':'Pileup_SF', 'weight_up': 'Pileup_SFUp', 'weight_down':'Pileup_SFDn'}, 
    'l1pre'        : {'removeweight':'L1PreFiringWeight_Nom', 'weight_up':'L1PreFiringWeight_Up', 'weight_down':'L1PreFiringWeight_Dn'},
    
    })

syst_groups = {
    'MODELING': [ 'alphaS', 'ren', 'fac', 'isr', 'fsr', 
    ],
    'EXPERIMENTAL': [ 'Ele', 'Mu', 'btag_b', 'btag_l', 'l1pre', 'pu',
    ],
    'JER': [
        'res0_2016APV', 'res1_2016APV', 'res2_2016APV', 'res3_2016APV', 'res4_2016APV', 'res5_2016APV', 
        'res0_2016', 'res1_2016', 'res2_2016', 'res3_2016', 'res4_2016', 'res5_2016', 
        'res0_2017', 'res1_2017', 'res2_2017', 'res3_2017', 'res4_2017', 'res5_2017', 
        'res0_2018', 'res1_2018', 'res2_2018', 'res3_2018', 'res4_2018', 'res5_2018', 
    ],
    'JESTOTAL' : [
        'jes_total_EtaBin0', 'jes_total_EtaBin1', 'jes_total_EtaBin2',
        'jes_total_EtaBin3', 'jes_total_EtaBin4', 'jes_total_EtaBin5'
    ]

#    'JES1': ['jes_b', 'jes_c', 'jeq_g', 'jes_q', 'jes_abs',
#    ],
#    'JES2': [
#            'jes_BBEC1', 'jes_EC2', 'jes_HF', 'jes_relbal', 'jme_uncl',
#    ],
#    'JES3': [
#        'jes_abs_16', 'jes_BBEC1_16', 'jes_EC2_16', 'jes_HF_16', 'jes_relsam_16',
#        'jes_abs_17', 'jes_BBEC1_17', 'jes_EC2_17', 'jes_HF_17', 'jes_relsam_17',
#        'jes_abs_18', 'jes_BBEC1_18', 'jes_EC2_18', 'jes_HF_18', 'jes_relsam_18',
#    ]

    }
