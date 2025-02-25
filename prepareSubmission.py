import os,glob
from common.logger import get_logger
import yaml
import shutil

import argparse

if __name__ == '__main__':
  # Argument parser setup
  parser = argparse.ArgumentParser(description="Prepare for submission.")
  parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
  parser.add_argument("-c", "--config", help="Path to the config file.")
  parser.add_argument("--ntuple", help="Path to the pre-saved ntuples.")

  args = parser.parse_args()

  logger = get_logger(args.logLevel, logFile = None)

  logger.warning("Please make sure to execute this script directly under HEPHY-uncertainty folder.")

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  cfg_new = cfg.copy()

  for t in cfg['Tasks']:
    for s in cfg['Selections']:
      os.makedirs(os.path.join('models',t,s,'model_path'))

      # copy the model
      original_model_path = cfg[t][s]['model_path']
      new_model_path = os.path.join('models',t,s,'model_path')
      model_file_list = glob.glob(os.path.join(original_model_path,'*.index'))
      model_index_list = [os.path.basename(imodel).replace('.index','') for imodel in model_file_list]
      latest_model_idx = max(model_index_list)
      logger.info("Copying {} to {}".format(os.path.join(original_model_path,latest_model_idx+'.index'),new_model_path))
      shutil.copyfile(os.path.join(original_model_path,latest_model_idx+'.index'),os.path.join(new_model_path,latest_model_idx+'.index'))
      logger.info("Copying {} to {}".format(os.path.join(original_model_path,latest_model_idx+'.data-00000-of-00001'),new_model_path))
      shutil.copyfile(os.path.join(original_model_path,latest_model_idx+'.data-00000-of-00001'),os.path.join(new_model_path,latest_model_idx+'.data-00000-of-00001'))
      logger.info("Copying {} to {}".format(os.path.join(original_model_path,'config.pkl'),new_model_path))
      shutil.copyfile(os.path.join(original_model_path,'config.pkl'),os.path.join(new_model_path,'config.pkl'))
      logger.info("Copying {} to {}".format(os.path.join(original_model_path,'checkpoint'),new_model_path))
      shutil.copyfile(os.path.join(original_model_path,'checkpoint'),os.path.join(new_model_path,'checkpoint'))
      cfg_new[t][s]['model_path'] = new_model_path

      # copy the calibration
      if 'calibration' in cfg[t][s]:
        os.makedirs(os.path.join('models',t,s,'calibration'))
        original_cal_path = cfg[t][s]['calibration']
        new_cal_path = os.path.join('models',t,s,'calibration')
        logger.info("Copying {} to {}".format(original_cal_path,new_cal_path))
        shutil.copyfile(original_cal_path,os.path.join(new_cal_path,os.path.basename(original_cal_path)))
        cfg_new[t][s]['calibration'] = os.path.join(new_cal_path,os.path.basename(original_cal_path))

      # copy the icp file
      if 'icp_file' in cfg[t][s]:
        os.makedirs(os.path.join('models',t,s,'icp_file'))
        original_icp_path = cfg[t][s]['icp_file']
        new_icp_path = os.path.join('models',t,s,'icp_file')
        logger.info("Copying {} to {}".format(original_icp_path,new_icp_path))
        shutil.copyfile(original_icp_path,os.path.join(new_icp_path,os.path.basename(original_icp_path)))
        cfg_new[t][s]['icp_file'] = os.path.join(new_icp_path,os.path.basename(original_icp_path))

  # copy the pre-saved ntuples and CSI files
  os.makedirs("data")
  logger.info("Copying {} to {}".format(args.ntuple,"data"))
  shutil.copytree(args.ntuple,"data/tmp_data")

  # save the new config file
  with open('config_submission.yaml', 'w') as yaml_file:
    yaml.dump(cfg_new, yaml_file, default_flow_style=False)

  logger.info("config_submission.yaml saved.")
