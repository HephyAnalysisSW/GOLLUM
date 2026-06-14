import glob
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    name: str
    dataset: str
    xsec: float
    era: str
    key: str
    fracNegWeights: float | None = None
    local_path: str | None = None

    @property
    def is_disk(self):
        return self.local_path is not None

    def list_files(self, small=False):
        if not self.is_disk:
            raise RuntimeError(f"Sample {self.key} is not disk-backed.")
        files = sorted(glob.glob(os.path.join(self.local_path, "**", "*.root"), recursive=True))
        if small:
            return files[:3]
        return files


def _infer_era(dataset):
    if "UL16NanoAODAPV" in dataset:
        return "UL16APV"
    if "UL16NanoAOD" in dataset:
        return "UL16"
    if "UL17NanoAOD" in dataset:
        return "UL17"
    if "UL18NanoAOD" in dataset:
        return "UL18"
    return "unknown"


class _Kreator:
    def __init__(self):
        self.samples = []

    def makeMCComponent(self, name, dataset, user, pattern, xsec, **kwargs):
        era = _infer_era(dataset)
        key = f"{name}_{era}"
        sample = Sample(
            name=name,
            dataset=dataset,
            xsec=float(xsec),
            era=era,
            key=key,
            fracNegWeights=kwargs.get("fracNegWeights"),
        )
        self.samples.append(sample)
        return sample

    def makeLocalComponent(self, name, path, xsec, **kwargs):
        key = kwargs.get("key", name)
        sample = Sample(
            name=name,
            dataset=path,
            xsec=float(xsec),
            era=kwargs.get("era", "UL18"),
            key=key,
            local_path=path,
            fracNegWeights=kwargs.get("fracNegWeights"),
        )
        self.samples.append(sample)
        return sample


kreator = _Kreator()

# # ====== Z + Jets ======
# ## New FEWZ cross section 1921.8 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3, fracNegWeights=0.16)
DYJetsToLL_M50_LO =  kreator.makeMCComponent("DYJetsToLL_M50_LO", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)
DYJetsToLL_M10to50_LO =  kreator.makeMCComponent("DYJetsToLL_M10to50_LO", "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", "CMS", ".*root", 15810) 



# # ====== Z + Jets ======
# ## New FEWZ cross section 1921.8 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3, fracNegWeights=0.16)
DYJetsToLL_M50_LO =  kreator.makeMCComponent("DYJetsToLL_M50_LO", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)



# # ====== Z + Jets ======
# ## New FEWZ cross section 1921.8 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM", "CMS", ".*root", 1921.8*3, fracNegWeights=0.16)
DYJetsToLL_M50_LO =  kreator.makeMCComponent("DYJetsToLL_M50_LO", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)
DYJetsToLL_M50_LO_ext =  kreator.makeMCComponent("DYJetsToLL_M50_LO_ext", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)


# # ====== Z + Jets ======
# ## New FEWZ cross section 1921.8 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM", "CMS", ".*root", 1921.8*3, fracNegWeights=0.16)
DYJetsToLL_M50_LO =  kreator.makeMCComponent("DYJetsToLL_M50_LO", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)
DYJetsToLL_M50_LO_ext =  kreator.makeMCComponent("DYJetsToLL_M50_LO_ext", "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v1/NANOAODSIM", "CMS", ".*root", 1921.8*3)



## Cross sections from XSDB times k-factor 1.08 from ratio of FEWZ to inclusive DYJetsToLL_M50_LO

DYJetsToLL_M50_HT70to100      = kreator.makeMCComponent("DYJetsToLL_M50_HT70to100",      "/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       "CMS", ".*root", 140.0*1.08)
DYJetsToLL_M50_HT100to200      = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200",      "/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       "CMS", ".*root", 139.2*1.08)
DYJetsToLL_M50_HT200to400      = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400",      "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       "CMS", ".*root", 38.4*1.08)
DYJetsToLL_M50_HT400to600      = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600",      "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       "CMS", ".*root", 5.174*1.08)
DYJetsToLL_M50_HT600to800      = kreator.makeMCComponent("DYJetsToLL_M50_HT600to800",      "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       "CMS", ".*root", 1.258*1.08 )
DYJetsToLL_M50_HT800to1200     = kreator.makeMCComponent("DYJetsToLL_M50_HT800to1200",     "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",      "CMS", ".*root", 0.5598*1.08 )
DYJetsToLL_M50_HT1200to2500    = kreator.makeMCComponent("DYJetsToLL_M50_HT1200to2500",    "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",     "CMS", ".*root", 0.1305*1.08 )
DYJetsToLL_M50_HT2500toInf     = kreator.makeMCComponent("DYJetsToLL_M50_HT2500toInf",     "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",      "CMS", ".*root", 0.002997*1.08 )


DYJetsToLL_M50_HT70to100      = kreator.makeMCComponent("DYJetsToLL_M50_HT70to100",      "/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",       "CMS", ".*root", 140.0*1.08)
DYJetsToLL_M50_HT100to200      = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200",      "/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",       "CMS", ".*root", 139.2*1.08)
DYJetsToLL_M50_HT200to400      = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400",      "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",       "CMS", ".*root", 38.4*1.08)
DYJetsToLL_M50_HT400to600      = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600",      "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",       "CMS", ".*root", 5.174*1.08)
DYJetsToLL_M50_HT600to800      = kreator.makeMCComponent("DYJetsToLL_M50_HT600to800",      "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",       "CMS", ".*root", 1.258*1.08 )
DYJetsToLL_M50_HT800to1200     = kreator.makeMCComponent("DYJetsToLL_M50_HT800to1200",     "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",      "CMS", ".*root", 0.5598*1.08 )
DYJetsToLL_M50_HT1200to2500    = kreator.makeMCComponent("DYJetsToLL_M50_HT1200to2500",    "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",     "CMS", ".*root", 0.1305*1.08 )
DYJetsToLL_M50_HT2500toInf     = kreator.makeMCComponent("DYJetsToLL_M50_HT2500toInf",     "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",      "CMS", ".*root", 0.002997*1.08 )

DYJetsToLL_M50_HT70to100      = kreator.makeMCComponent("DYJetsToLL_M50_HT70to100",      "/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",       "CMS", ".*root", 140.0*1.08)
DYJetsToLL_M50_HT100to200      = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200",      "/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",       "CMS", ".*root", 139.2*1.08)
DYJetsToLL_M50_HT200to400      = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400",      "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",       "CMS", ".*root", 38.4*1.08)
DYJetsToLL_M50_HT400to600      = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600",      "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",       "CMS", ".*root", 5.174*1.08)
DYJetsToLL_M50_HT600to800      = kreator.makeMCComponent("DYJetsToLL_M50_HT600to800",      "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",       "CMS", ".*root", 1.258*1.08 )
DYJetsToLL_M50_HT800to1200     = kreator.makeMCComponent("DYJetsToLL_M50_HT800to1200",     "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",      "CMS", ".*root", 0.5598*1.08 )
DYJetsToLL_M50_HT1200to2500    = kreator.makeMCComponent("DYJetsToLL_M50_HT1200to2500",    "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",     "CMS", ".*root", 0.1305*1.08 )
DYJetsToLL_M50_HT2500toInf     = kreator.makeMCComponent("DYJetsToLL_M50_HT2500toInf",     "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",      "CMS", ".*root", 0.002997*1.08 )

DYJetsToLL_M50_HT70to100      = kreator.makeMCComponent("DYJetsToLL_M50_HT70to100",      "/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",       "CMS", ".*root", 140.0*1.08)
DYJetsToLL_M50_HT100to200      = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200",      "/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",       "CMS", ".*root", 139.2*1.08)
DYJetsToLL_M50_HT200to400      = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400",      "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",       "CMS", ".*root", 38.4*1.08)
DYJetsToLL_M50_HT400to600      = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600",      "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",       "CMS", ".*root", 5.174*1.08)
DYJetsToLL_M50_HT600to800      = kreator.makeMCComponent("DYJetsToLL_M50_HT600to800",      "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",       "CMS", ".*root", 1.258*1.08 )
DYJetsToLL_M50_HT800to1200     = kreator.makeMCComponent("DYJetsToLL_M50_HT800to1200",     "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",      "CMS", ".*root", 0.5598*1.08 )
DYJetsToLL_M50_HT1200to2500    = kreator.makeMCComponent("DYJetsToLL_M50_HT1200to2500",    "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",     "CMS", ".*root", 0.1305*1.08 )
DYJetsToLL_M50_HT2500toInf     = kreator.makeMCComponent("DYJetsToLL_M50_HT2500toInf",     "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",      "CMS", ".*root", 0.002997*1.08 )

#DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/250904_134742", 6087.44921467)
#DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/250904_134749", 6135.542164)
#DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/250904_134756", 991.8578552)
#DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/250904_134803", 504.7171968)
#DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/250904_134809", 351.967252)
#DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/250904_134816", 173.9668168)
#DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/250904_134823", 172.266490027)
#DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos = kreator.makeLocalComponent("DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/250904_134829", 78.4195807947)
#DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/251014_153049", 1908.0157)
#DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/251016_144321", 172.69619)
DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_120_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_120_Photos/251124_092852", 1912.8516)
DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_120_200_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_120_200_Photos/251124_092858", 28.816713)
DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/260109_163747", 2.9652782)
DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/260109_163754", 0.18982734)
DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/260109_163801", 0.046906194)
DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/260109_163807", 0.010329357)
DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/260109_163813", 0.0071970617)
DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne = kreator.makeLocalComponent("DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne", "/eos/vbc/group/cms/robert.schoefbeck/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/260109_163820", 0.00086687414)


all_samples = tuple(kreator.samples)
samples_by_key = {sample.key: sample for sample in all_samples}


def _matches_token(sample, token):
    return token in {
        sample.key,
        sample.name,
        sample.dataset,
        sample.dataset.strip("/"),
    }


def get_sample(token):
    matches = [sample for sample in all_samples if _matches_token(sample, token)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No DY sample matching '{token}'. Known keys include: {', '.join(sorted(samples_by_key)[:10])}")

    keys = ", ".join(sample.key for sample in matches)
    raise KeyError(f"Ambiguous DY sample '{token}'. Use one of: {keys}")


def list_sample_keys(pattern=None):
    keys = sorted(samples_by_key)
    if pattern:
        rx = re.compile(pattern)
        keys = [key for key in keys if rx.search(key)]
    return keys
