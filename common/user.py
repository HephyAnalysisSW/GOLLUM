import os

data_directory         = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/"

if os.environ['USER'] in ['robert.schoefbeck']:
    # Where the plots go:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/Challenge/"
    # For model output:
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/Challenge/models/"
    derived_data_directory = "/eos/vbc/group/cms/robert.schoefbeck/Higgs_uncertainty/data/"

elif os.environ['USER'] in ['dennis.schwarz']:
    plot_directory         = "/groups/hephy/cms/dennis.schwarz/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/models/"

elif os.environ['USER'] in ['claudius.krause']:
    raise RuntimeError( "Hello Claudius. Specify your directories in common/user.py." )

else:

    raise RuntimeError( "HELLO NEW USER! Configure your directories in common/user.py! Look in the file how others did it." )

    plot_directory  = "./plots/"
    model_directory = "./models/"
    data_directory  = "./data/"
