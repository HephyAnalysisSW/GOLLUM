import cmsstyle
# https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/#categorical-data-eg-1d-stackplots
# https://cms-analysis.docs.cern.ch/guidelines/plotting/examples/#stack-plot-with-cmsstyle
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

colors = {
 'TTLep_pow':   cmsstyle.p10.kRed,
 'DrellYan':    cmsstyle.p10.kBlue,
 'SingleTop':   cmsstyle.p10.kYellow,
 'TTSemi_pow':  cmsstyle.p10.kOrange,
}

def get_color( sample_name ):
    for k, c in colors.items():
        if sample_name.startswith( k ):
            return c

