# python scoreRun_generate.py --Ntoys 50 --mu 0.10 --postfix gen_mu_0p10
# python scoreRun_generate.py --Ntoys 50 --mu 0.25 --postfix gen_mu_0p25
# python scoreRun_generate.py --Ntoys 50 --mu 0.50 --postfix gen_mu_0p50
# python scoreRun_generate.py --Ntoys 50 --mu 0.75 --postfix gen_mu_0p75
# python scoreRun_generate.py --Ntoys 50 --mu 1.00 --postfix gen_mu_1p00
# python scoreRun_generate.py --Ntoys 50 --mu 1.25 --postfix gen_mu_1p25
# python scoreRun_generate.py --Ntoys 50 --mu 1.50 --postfix gen_mu_1p50
# python scoreRun_generate.py --Ntoys 50 --mu 1.75 --postfix gen_mu_1p75
# python scoreRun_generate.py --Ntoys 50 --mu 2.00 --postfix gen_mu_2p00
# python scoreRun_generate.py --Ntoys 50 --mu 2.25 --postfix gen_mu_2p25
# python scoreRun_generate.py --Ntoys 50 --mu 2.50 --postfix gen_mu_2p50
# python scoreRun_generate.py --Ntoys 50 --mu 2.75 --postfix gen_mu_2p75
# python scoreRun_generate.py --Ntoys 50 --mu 3.00 --postfix gen_mu_3p00

python scoreRun_generate.py --Ntoys 10000 #SPLIT200


# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_noSYS_mu_1p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_noSYS_mu_2p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_noSYS_mu_3p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 4 --postfix gen_noSYS_mu_4p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 5 --postfix gen_noSYS_mu_5p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-ttbar-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyJES_mu_1p0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyJES_mu_2p0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze tes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyJES_mu_3p0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze tes-met-bkg-ttbar-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyTES_mu_1p0 --jes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyTES_mu_2p0 --jes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-met-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyTES_mu_3p0 --jes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-met-bkg-ttbar-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyMET_mu_1p0 --jes 1.0 --tes 1.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyMET_mu_2p0 --jes 1.0 --tes 1.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-bkg-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyMET_mu_3p0 --jes 1.0 --tes 1.0 --ttbar 1.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-bkg-ttbar-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyBKG_mu_1p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --diboson 1.0 --freeze jes-tes-met-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyBKG_mu_2p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --diboson 1.0 --freeze jes-tes-met-ttbar-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyBKG_mu_3p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --diboson 1.0 --freeze jes-tes-met-ttbar-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyTTBAR_mu_1p0 --jes 1.0 --tes 1.0 --met 0.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyTTBAR_mu_2p0 --jes 1.0 --tes 1.0 --met 0.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-diboson
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyTTBAR_mu_3p0 --jes 1.0 --tes 1.0 --met 0.0 --bkg 1.0 --diboson 1.0 --freeze jes-tes-met-bkg-diboson
#
# python scoreRun_generate.py --Ntoys 50 --mu 1 --postfix gen_onlyDIBOSON_mu_1p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --freeze jes-tes-met-bkg-ttbar
# python scoreRun_generate.py --Ntoys 50 --mu 2 --postfix gen_onlyDIBOSON_mu_2p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --freeze jes-tes-met-bkg-ttbar
# python scoreRun_generate.py --Ntoys 50 --mu 3 --postfix gen_onlyDIBOSON_mu_3p0 --jes 1.0 --tes 1.0 --met 0.0 --ttbar 1.0 --bkg 1.0 --freeze jes-tes-met-bkg-ttbar
