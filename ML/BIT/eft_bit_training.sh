# submit --memory 35 --title bit --walltime 02-00:00:00 --queue medium  "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_allWC" --output=eft_bit_training_15072026/ 

submit --memory 16 --title bit_allWC_2016 "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_allWC" --output=eft_bit_training_15072026/ 

submit --memory 16 --title bit_ML4EFT_2016 "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_ML4EFTWC" --output=eft_bit_training_15072026/

submit --memory 16 --title bit_nonML4EFT "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_nonML4EFTWC" --output=eft_bit_training_15072026/

# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_ctG_only"

# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_ct_nonML4EFT"
# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_ct_ML4EFT"

# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_cQ_nonML4EFT_block01"
# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_cQ_nonML4EFT_block02"

# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_cQ_ML4EFT_block01"
# submit --memory 35 --title bit8h "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft.yaml --every 0 --job bit_TT01j2l_EFT_2016_cQ_ML4EFT_block02"

submit --memory 16 --title bit_ND_allWC_2016 "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft_ND.yaml --every 0 --job bit_TT01j2l_EFT_2016_allWC" --output=eft_bit_training_15072026/

submit --memory 16 --title bit_ND_ML4EFT_2016 "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft_ND.yaml --every 0 --job bit_TT01j2l_EFT_2016_ML4EFTWC" --output=eft_bit_training_15072026/

submit --memory 16 --title bit_ND_nonML4EFT "python /users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM/ML/BIT/eft_bit_training.py configs/unbinned_v7_eft/bit_training/unbinned_2016_eft_ND.yaml --every 0 --job bit_TT01j2l_EFT_2016_nonML4EFTWC" --output=eft_bit_training_15072026/
