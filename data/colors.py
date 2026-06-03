# https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/#categorical-data-eg-1d-stackplots
# https://cms-analysis.docs.cern.ch/guidelines/plotting/examples/#stack-plot-with-cmsstyle
import cmsstyle
#cmsstyle.p10.kBlue
#cmsstyle.p10.kYellow
#cmsstyle.p10.kRed
#cmsstyle.p10.kAsh
#cmsstyle.p10.kViolet
#cmsstyle.p10.kBrown
#cmsstyle.p10.kOrange
#cmsstyle.p10.kGreen
#cmsstyle.p10.kGray
#cmsstyle.p10.kCyan
import ROOT

colors = {
 'TTLep_pow':   cmsstyle.p10.kRed,
 'DrellYan':    cmsstyle.p10.kBlue,
 'SingleTop':   cmsstyle.p10.kYellow,
 'TTSemi_pow':  cmsstyle.p10.kOrange,
 'EtaS':        ROOT.kBlue,
 'EtaP':        ROOT.kRed,
}

# sams as cmsstyle.p10 colors in the order above
# but with hex codes
cmap_petroff10_mpl = ['#3f90da',
                      '#ffa90e',
                      '#bd1f01',
                      '#94a4a2',
                      '#832db6',
                      '#a96b59',
                      '#e76300',
                      '#b9ac70',
                      '#717581',
                      '#92dadd']

def get_color( sample_name ):
    for k, c in colors.items():
        if sample_name.startswith( k ):
            return c

