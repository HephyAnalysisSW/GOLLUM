import os

data_directory         = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/"

if os.environ['USER'] in ['robert.schoefbeck']:
    # Where the plots go:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/Challenge/"
    # For model output:
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/Challenge/models/"
    # Maybe we let this later point to the input data.

elif os.environ['USER'] in ['claudius.krause']:
    raise RuntimeError( "Hello Claudius. Specify your directories in common/user.py." )

else:

    raise RuntimeError( "HELLO NEW USER! Configure your directories in common/user.py! Look in the file how others did it." )

    plot_directory  = "./plots/"
    model_directory = "./models/"
    data_directory  = "./data/"
