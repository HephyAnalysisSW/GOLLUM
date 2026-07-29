#! /bin/bash

#
# script that creates the model folder structure
# when given a reference folder, it symlinks all the models
# into the new folder
#
# usage: you have a new config version but want to keep using
# the valid models from a previous version to avoid retraining
#

# receive model folder as first argument
# these should be expanded to full path
MODEL_FOLDER="$(realpath "$1")"
REFERENCE_FOLDER="${2:+$(realpath "$2")}"

for ERA in "2016" "2016APV" "2017" "2018"
do

    ERA_FOLDER="SR_${ERA}"

    # allow binned and unbinned
    MODEL_FOLDERS=("ICH" "ICPH")
    if [[ $MODEL_FOLDER == *"unbinned"* ]]; then
        MODEL_FOLDERS=("BIT" "ICP" "PNN" "Scaler" "TFMC")
    fi
    
    for MODEL_TYPE in ${MODEL_FOLDERS[@]}
    do
        FOLDER="${MODEL_FOLDER}/${ERA_FOLDER}/${MODEL_TYPE}"
        mkdir -p "$FOLDER"
        if [[ -n "$REFERENCE_FOLDER" ]]; then
            pushd "$FOLDER" > /dev/null
            for REF_MODEL_NAME in $(ls "${REFERENCE_FOLDER}/${ERA_FOLDER}/${MODEL_TYPE}")
            do
                ln -s "${REFERENCE_FOLDER}/${ERA_FOLDER}/${MODEL_TYPE}/${REF_MODEL_NAME}"
            done
            popd > /dev/null
        fi

    done
done

