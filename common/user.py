import os

# default
training_data_dir      = "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/"

try:
    user = os.environ['USER']
except:
    user = "noUserFound"

tmp_mem_directory      = "/dev/shm/%s/"%(user)

if user in ['robert.schoefbeck']:
    # Where the plots go:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/SBIPDF/"
    # For model output:
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/SBIPDF/models/"
    cache_directory        = "/groups/hephy/cms/robert.schoefbeck/SBIPDF/caches/"
    output_directory       = "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

    
elif user in ['dennis.schwarz']:
    plot_directory         = "/groups/hephy/cms/dennis.schwarz/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/models/"
    output_directory       = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/"

elif user in ['ang.li']:
    plot_directory         = "/groups/hephy/cms/ang.li/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/ang.li/HiggsChallenge/models/"
    cache_directory        = "/groups/hephy/cms/ang.li/HiggsChallenge/caches/"
    output_directory       = "/groups/hephy/cms/ang.li/HiggsChallenge/output/"

elif user in ['cristina.giordano']:
    plot_directory         = "/groups/hephy/cms/cristina.giordano/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/cristina.giordano/HiggsChallenge/models/"
    cache_directory        = "/groups/hephy/cms/cristina.giordano/HiggsChallenge/caches/"
    output_directory       = "/groups/hephy/cms/cristina.giordano/HiggsChallenge/output/"

elif user in ['lisa.benato']:
    plot_directory         = "/groups/hephy/cms/lisa.benato/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/lisa.benato/HiggsChallenge/models/"
    cache_directory        = "/groups/hephy/cms/lisa.benato/HiggsChallenge/caches/"
    output_directory       = "/groups/hephy/cms/lisa.benato/HiggsChallenge/output/"

elif user in ['maryam.shooshtari']:
    plot_directory         = "/groups/hephy/cms/maryam.shooshtari/www/HiggsChallenge/"
    model_directory        = "/groups/hephy/cms/maryam.shooshtari/HiggsChallenge/models/"
    cache_directory        = "/groups/hephy/cms/maryam.shooshtari/HiggsChallenge/caches/"
    output_directory       = "/groups/hephy/cms/maryam.shooshtari/HiggsChallenge/output/"

elif user in ['claudius.krause']:
    plot_directory         = "/groups/hephy/mlearning/HiggsChallenge/claudius/plots/"
    model_directory        = "/groups/hephy/mlearning/HiggsChallenge/claudius/models/"
    cache_directory        = "/groups/hephy/mlearning/HiggsChallenge/claudius/caches/"
    output_directory       = "/groups/hephy/mlearning/HiggsChallenge/claudius/output/"

elif user in ['sergio.sanchez.cruz']:
    # Where the plots go:
    plot_directory         = "/groups/hephy/cms/sergio.sanchez.cruz/www/SBIPDF/"
    # For model output:
    model_directory        = "/groups/hephy/cms/sergio.sanchez.cruz/SBIPDF/models/"
    cache_directory        = "/groups/hephy/cms/sergio.sanchez.cruz/SBIPDF/caches/"
    output_directory       = "/scratch-cbe/users/sergio.sanchez.cruz/SBIPDF/output/"

else:

    raise RuntimeError( "HELLO NEW USER! Configure your directories in common/user.py! Look in the file how others did it." )

    plot_directory  = "./plots/"
    model_directory = "./models/"
    cache_directory = "./caches/"
    data_directory  = "./data/"
