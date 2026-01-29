submit --memory 20 --title tfmc "python tfmc_training.py ../../configs/Eta_unbinned/Eta_unbinned_2016.yaml    --plot-directory v2 --overwrite --job tfmc_EtaSandP_2016 --plot --norm-plot" 
submit --memory 20 --title tfmc "python tfmc_training.py ../../configs/Eta_unbinned/Eta_unbinned_2016APV.yaml --plot-directory v2 --overwrite --job tfmc_EtaSandP_2016APV --plot --norm-plot"
submit --memory 20 --title tfmc "python tfmc_training.py ../../configs/Eta_unbinned/Eta_unbinned_2017.yaml    --plot-directory v2 --overwrite --job tfmc_EtaSandP_2017 --plot --norm-plot"
submit --memory 20 --title tfmc "python tfmc_training.py ../../configs/Eta_unbinned/Eta_unbinned_2018.yaml    --plot-directory v2 --overwrite --job tfmc_EtaSandP_2018 --plot --norm-plot"
