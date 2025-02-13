import os

# default
data_directory         = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/"
derived_data_directory = "/scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/"
derived_test_data_directory = "/scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/test_data/"
tmp_mem_directory      = "/dev/shm/%s/"%(os.environ['USER'])

if os.environ['USER'] in ['robert.schoefbeck']:
    # Where the plots go:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/Challenge/"
    # For model output:
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/Challenge/models/"
    output_directory       = "/scratch-cbe/users/robert.schoefbeck/Challenge/output/"

elif os.environ['USER'] in ['dennis.schwarz']:
    plot_directory         = "/groups/hephy/cms/dennis.schwarz/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/models/"
    output_directory       = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/"

elif os.environ['USER'] in ['ang.li']:
    plot_directory         = "/groups/hephy/cms/ang.li/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/ang.li/HiggsChallenge/models/"
    output_directory       = "/groups/hephy/cms/ang.li/HiggsChallenge/output/"

elif os.environ['USER'] in ['cristina.giordano']:
    plot_directory         = "/groups/hephy/cms/cristina.giordano/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/cristina.giordano/HiggsChallenge/models/"
    output_directory       = "/groups/hephy/cms/cristina.giordano/HiggsChallenge/output/"

elif os.environ['USER'] in ['lisa.benato']:
    plot_directory         = "/groups/hephy/cms/lisa.benato/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/lisa.benato/HiggsChallenge/models/"
    output_directory       = "/groups/hephy/cms/lisa.benato/HiggsChallenge/output/"

elif os.environ['USER'] in ['claudius.krause']:
    raise RuntimeError( "Hello Claudius. Specify your directories in common/user.py." )

else:

    raise RuntimeError( "HELLO NEW USER! Configure your directories in common/user.py! Look in the file how others did it." )

    plot_directory  = "./plots/"
    model_directory = "./models/"
    data_directory  = "./data/"
