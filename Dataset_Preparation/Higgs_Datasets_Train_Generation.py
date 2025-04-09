
import logging                                                                                                                                                                                              
import numpy as np
from Higgs_Datasets import Data
import pandas as pd

# 设置日志配置，确保输出到控制台
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the main function.")

    input_directory = '/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data'
    logger.info(f"Initializing Data instance with input_directory: {input_directory}")

    try:
        data_instance = Data(input_directory)
        logger.info("Data instance created successfully.")
    except Exception as e:
        logger.error(f"Failed to create Data instance: {e}")
        return

    try:
        tes = 0.9991431
        jes = 1.0058473
        soft_met = 1.279198
        ttbar_scale = 0.98807263
        diboson_scale = 0.39890692
        bkg_scale = 1.0008626 

        logger.info("Calling get_syst_train_set with systematics parameters.")

        hdf5_filename = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/trial/trial1.h5"

        # 调用 get_syst_train_set 方法保存一个包含所有信息的数据集
        syst_train_set = data_instance.get_syst_train_set(
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            dopostprocess=True,
            save_to_hdf5=True,
            hdf5_filename=hdf5_filename
        )

        if syst_train_set is not None:
            logger.info("Systematic training set created and saved successfully.")
        else:
            logger.warning("Systematic training set is None.")

    except Exception as e:
        logger.error(f"Error in getting training dataset with systematics: {e}")

    logger.info("Main function completed.")

if __name__ == "__main__":
    main()
