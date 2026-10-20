## Binned rotated
#
#cfg_base="binned_2016_7"
#cfg_version="binned_2016_v5"
#cfg_path="configs/binned_v5/2016/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos --freezePOI 0.0001
#
#fitresult="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_2016_7_binned_2016_v5_rotate_freezePOI0.0001_fit.json"
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
#
## Binned rotated, no syst
#
#cfg_base="binned_2016_no_syst_6"
#cfg_version="binned_2016_v5"
#cfg_path="configs/binned_v5/2016/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
##python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
## Unbinned rotated
#
#cfg_base="unbinned_2016_7"
#cfg_version="unbinned_2016_v5"
#cfg_path="configs/unbinned_v5/2016/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 --overwrite fit #--freezePOI 0.0001
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
## Unbinned rotated, no syst
#
#cfg_base="unbinned_2016_no_syst_6"
#cfg_version="unbinned_2016_v5"
#cfg_path="configs/unbinned_v5/2016/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 --overwrite fit
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
#
## Unbinned rotated, Delphes
#
#cfg_base="unbinned_delphes_6"
#cfg_version="unbinned_v5D"
#cfg_path="configs/unbinned_v5D/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 --overwrite fit #--freezePOI 0.0001
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
## Binned rotated, Delphes
#
#cfg_base="binned_delphes_6"
#cfg_version="binned_delphes_v5D"
#cfg_path="configs/binned_v5D/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json
#
#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001
#
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
#
###################
## DELPHES RUN II #
#

#
# Binned rotated, Delphes, 3
#
cfg_base="binned_delphes_3_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

python orth/orthogonalize_Fisher.py ${cfg_path} 
python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001

python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py  --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 

#
# Binned rotated, Delphes, 4
#
cfg_base="binned_delphes_4_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

python orth/orthogonalize_Fisher.py ${cfg_path} 
python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001

python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py  --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

##################

#
# Binned rotated, Delphes, 5
#
cfg_base="binned_delphes_5_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

python orth/orthogonalize_Fisher.py ${cfg_path} 
python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001

python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py ${cfg_path} --Q 175 --postfit ${fitresult} --rotate ${rotate} 
python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

#
# Binned rotated, Delphes, 6
#
cfg_base="binned_delphes_6_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

python orth/orthogonalize_Fisher.py ${cfg_path} 
python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001

python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py ${cfg_path} --Q 175 --postfit ${fitresult} --rotate ${rotate} 
python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#
# Binned rotated, Delphes, 6 NOT normalized
#
#cfg_base="binned_delphes_6_RunII_unnorm"
#cfg_version="binned_delphes_RunII_v5D"
#cfg_path="configs/binned_v5D/${cfg_base}.yaml"
#outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#
#fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
#rotate=${outdir}/eigen_basis_binned_delphes_6_RunII_${cfg_version}.json # using the normalized rotation; we only change the lumi uncertainty
#
##python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #--freezePOI 0.0001
#
#python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
#python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
##python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

# Binned rotated, Delphes, 7, 7-1
cfg_base="binned_delphes_7_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

#python orth/orthogonalize_Fisher.py ${cfg_path} 
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos --freezePOI 0.01

fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
fitresult=/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_7_RunII_binned_delphes_RunII_v5D_rotate_freezePOI0.01_fit.json
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

###
### Binned rotated, Delphes, 7, ATLASpdf21
###
##cfg_base="binned_delphes_7_RunII_ATLASpdf21"
##cfg_version="binned_delphes_RunII_v5D"
##cfg_path="configs/binned_v5D/${cfg_base}.yaml"
##outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
##rotate=${outdir}/eigen_basis_binned_delphes_7_RunII_${cfg_version}.json
##
##python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos 
##fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
##python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
##python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

#
# Binned rotated, Delphes, 8, 8-1 and 8-2
#
cfg_base="binned_delphes_8_RunII"
cfg_version="binned_delphes_RunII_v5D"
cfg_path="configs/binned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

#python orth/orthogonalize_Fisher.py ${cfg_path} 

rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos #8
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos --freezePOI 0.001 #8-1
#python fit/Likelihood.py ${cfg_path} --overwrite fit --rotate ${rotate} --minos --freezePOI 0.1 #8-2

fitresult=/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_8_RunII_binned_delphes_RunII_v5D_rotate_fit.json # 8
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
fitresult=/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_8_RunII_binned_delphes_RunII_v5D_rotate_freezePOI0.001_fit.json # 8-1
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
fitresult=/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_8_RunII_binned_delphes_RunII_v5D_rotate_freezePOI0.1_fit.json # 8-2
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit.py --Q 175 ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 


python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/postfitsys/binned_fromfit.py ${fitresult} --config ${cfg_path} --rotate ${rotate} 
#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

#
# Unbinned rotated, Delphes, 6
#
cfg_base="unbinned_delphes_6_RunII"
cfg_version="unbinned_v5D"
cfg_path="configs/unbinned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"
#/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_delphes_6_RunII_unbinned_v5D_rotate_fit.json
fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

python orth/orthogonalize_Fisher.py ${cfg_path} 
python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 --overwrite fit #--freezePOI 0.0001
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 
python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

# Unbinned rotated, Delphes, 7, 7-1
cfg_base="unbinned_delphes_7_RunII"
cfg_version="unbinned_v5D"
cfg_path="configs/unbinned_v5D/${cfg_base}.yaml"
outdir="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/"

#python orth/orthogonalize_Fisher.py ${cfg_path} 
rotate=${outdir}/eigen_basis_${cfg_base}_${cfg_version}.json

# "7"
python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 #--overwrite fit
fitresult=${outdir}/${cfg_base}_${cfg_version}_rotate_fit.json
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

# "7-1"
python fit/Likelihood.py ${cfg_path} --rotate ${rotate} --verbosity 2 --overwrite fit --freezePOI 0.01
fitresult="/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_delphes_7_RunII_unbinned_v5D_rotate_freezePOI0.01_fit.json"
python plot/pdf/pdf_postfit.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 

#python plot/pdf/pdf_postfit_decompose.py ${cfg_path} --postfit ${fitresult} --rotate ${rotate} 


#paper commands

#python plot/pdf/pdf_postfit.py configs/binned_v5D/binned_delphes_6_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//binned_delphes_6_RunII_binned_delphes_RunII_v5D_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_binned_delphes_6_RunII_binned_delphes_RunII_v5D.json
#python plot/pdf/pdf_postfit.py configs/binned_v5D/binned_delphes_7_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_7_RunII_binned_delphes_RunII_v5D_rotate_freezePOI0.01_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_binned_delphes_7_RunII_binned_delphes_RunII_v5D.json
#python plot/pdf/pdf_postfit.py configs/unbinned_v5D/unbinned_delphes_6_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//unbinned_delphes_6_RunII_unbinned_v5D_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_unbinned_delphes_6_RunII_unbinned_v5D.json
#python plot/pdf/pdf_postfit.py configs/unbinned_v5D/unbinned_delphes_7_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_delphes_7_RunII_unbinned_v5D_rotate_freezePOI0.01_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_unbinned_delphes_7_RunII_unbinned_v5D.json

# no sys
python plot/pdf/pdf_postfit.py configs/binned_v5D/binned_delphes_6_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_6_RunII_binned_delphes_RunII_v5D_nosyst_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_binned_delphes_6_RunII_binned_delphes_RunII_v5D.json
python plot/pdf/pdf_postfit.py configs/unbinned_v5D/unbinned_delphes_6_RunII.yaml --Q 175 --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_delphes_6_RunII_unbinned_v5D_nosyst_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_unbinned_delphes_6_RunII_unbinned_v5D.json

python plot/pdf/pdf_postfit.py configs/binned_v5D/binned_delphes_6_RunII.yaml --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_delphes_6_RunII_binned_delphes_RunII_v5D_nosyst_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_binned_delphes_6_RunII_binned_delphes_RunII_v5D.json
python plot/pdf/pdf_postfit.py configs/unbinned_v5D/unbinned_delphes_6_RunII.yaml --postfit /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_delphes_6_RunII_unbinned_v5D_nosyst_rotate_fit.json --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output//eigen_basis_unbinned_delphes_6_RunII_unbinned_v5D.json
