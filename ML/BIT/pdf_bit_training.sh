#submit --memory 10 --nNodes 8 "python pdf_bit_training.py ../../configs/unbinned/unbinned_2016.yaml  --job bit_pod_TTLep_pow_2016"
#submit --memory 10 --nNodes 8 "python pdf_bit_training.py ../../configs/unbinned/unbinned_2016APV.yaml  --job bit_pod_TTLep_pow_2016APV"
submit --memory 25 --nNodes 8 --queue medium --walltime 02-00:00:00 "python pdf_bit_training.py ../../configs/unbinned/unbinned_2017.yaml  --job bit_pod_TTLep_pow_2017"
submit --memory 30 --nNodes 8 --queue medium --walltime 02-00:00:00 "python pdf_bit_training.py ../../configs/unbinned/unbinned_2018.yaml  --job bit_pod_TTLep_pow_2018"
