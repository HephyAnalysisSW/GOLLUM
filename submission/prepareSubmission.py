import os,glob
import yaml
import shutil
import subprocess
import sys

import argparse

def copyAndReplaceOffsetInflate(original_file, new_file, offset, inflate):
    with open(original_file, "r") as src, open(new_file, "w") as dest:
        for line in src:
            if "####REPLACEOFFSET####" in line:
                line = line.replace("####REPLACEOFFSET####", f"offset = {offset}")
            if "####REPLACEINFLATE####" in line:
                line = line.replace("####REPLACEINFLATE####", f"inflate = {inflate}")
            dest.write(line)

if __name__ == '__main__':
  # Argument parser setup
  parser = argparse.ArgumentParser(description="Prepare for submission.")
  parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
  parser.add_argument("-c", "--config", help="Path to the config file.")
  parser.add_argument("--offset", type=float, default=0.0)
  parser.add_argument("--inflate", type=float, default=1.0)
  parser.add_argument("--version", type=str, default="v0")
  parser.add_argument("--ntuple", help="Path to the pre-saved ntuples.")

  args = parser.parse_args()
  from common.logger import get_logger

  logger = get_logger(args.logLevel, logFile = None)

  logger.warning("Please make sure to execute this script directly under main HEPHY-uncertainty folder. Excuting this in subfolders will not work.")

  # copy the Model.p
  logger.info("Copying {} to {}".format("submission/Model.py","Model.py"))
  # shutil.copyfile("submission/Model.py","Model.py")
  logger.info("Set offset = {}, inflate = {}".format(args.offset, args.inflate))
  copyAndReplaceOffsetInflate(original_file="submission/Model.py", new_file="model.py", offset=args.offset, inflate=args.inflate)

  # copy the example.py
  logger.info("Copying {} to {}".format("submission/example.py","example.py"))
  shutil.copyfile("submission/example.py","example.py")

  # copy the runModel_internal_test.py
  logger.info("Copying {} to {}".format("submission/runModel_internal_test.py","runModel_internal_test.py"))
  shutil.copyfile("submission/runModel_internal_test.py","runModel_internal_test.py")

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  cfg_new = cfg.copy()

  for t in cfg['Tasks']:
    for s in cfg['Selections']:
      os.makedirs(os.path.join('models',t,s,'model_path'))

      # copy the model
      original_model_path = cfg[t][s]['model_path']
      new_model_path = os.path.join('models',t,s,'model_path')
      with open(os.path.join(original_model_path,"checkpoint")) as ckpf:
        ckp_path = yaml.safe_load(ckpf)['model_checkpoint_path']

      #model_file_list = glob.glob(os.path.join(original_model_path,'*.index'))
      #model_index_list = [int(os.path.basename(imodel).replace('.index','')) for imodel in model_file_list]
      #latest_model_idx = max(model_index_list)
      #logger.info("ML model for {} {}: Latest index {}".format(t,s,latest_model_idx))

      logger.info("ML model for {} {}: copying from {}".format(t,s,ckp_path))
      latest_model_idx = os.path.basename(ckp_path)

      # copy the .index
      logger.info("Copying {} to {}".format(ckp_path+'.index',new_model_path))
      shutil.copyfile(ckp_path+'.index',os.path.join(new_model_path,str(latest_model_idx)+'.index'))

      # copy the .data
      logger.info("Copying {} to {}".format(ckp_path+'.data-00000-of-00001',new_model_path))
      shutil.copyfile(ckp_path+'.data-00000-of-00001',os.path.join(new_model_path,str(latest_model_idx)+'.data-00000-of-00001'))

      # copy the config.pkl
      logger.info("Copying {} to {}".format(os.path.join(original_model_path,'config.pkl'),new_model_path))
      shutil.copyfile(os.path.join(original_model_path,'config.pkl'),os.path.join(new_model_path,'config.pkl'))

      # write the new checkpoint metadata
      logger.info("Creating checkpoint in {}".format(new_model_path))
      shutil.copyfile(os.path.join(original_model_path,'checkpoint'),os.path.join(new_model_path,'checkpoint'))
      with open(os.path.join(new_model_path,"checkpoint"), "w") as ckpf_new:
        ckpf_new.write('model_checkpoint_path: "{}"\n'.format(str(latest_model_idx)))

      # update the new config with the local paths
      cfg_new[t][s]['model_path'] = new_model_path

      # copy the calibration
      if 'calibration' in cfg[t][s]:
        os.makedirs(os.path.join('models',t,s,'calibration'))
        original_cal_path = cfg[t][s]['calibration']
        new_cal_path = os.path.join('models',t,s,'calibration')

        logger.info("Copying {} to {}".format(original_cal_path,new_cal_path))
        shutil.copyfile(original_cal_path,os.path.join(new_cal_path,os.path.basename(original_cal_path)))

        # update the new config with the local paths
        cfg_new[t][s]['calibration'] = os.path.join(new_cal_path,os.path.basename(original_cal_path))

      # copy the icp file
      if 'icp_file' in cfg[t][s]:
        os.makedirs(os.path.join('models',t,s,'icp_file'))
        original_icp_path = cfg[t][s]['icp_file']
        new_icp_path = os.path.join('models',t,s,'icp_file')

        logger.info("Copying {} to {}".format(original_icp_path,new_icp_path))
        shutil.copyfile(original_icp_path,os.path.join(new_icp_path,os.path.basename(original_icp_path)))

        # update the new config with the local paths
        cfg_new[t][s]['icp_file'] = os.path.join(new_icp_path,os.path.basename(original_icp_path))

  if "Poisson" in cfg:
    for s in cfg['Poisson']:
      # copy the model if we select on MVA
      if 'model_path' in cfg["Poisson"][s]:
          original_model_path = cfg["Poisson"][s]['model_path']
          new_model_path = os.path.join('models',"Poisson",s, 'model_path')
          os.makedirs( new_model_path )
          with open(os.path.join(original_model_path,"checkpoint")) as ckpf:
            ckp_path = yaml.safe_load(ckpf)['model_checkpoint_path']

          logger.info("Poisson ML model for {}: copying from {}".format(s,ckp_path))
          latest_model_idx = os.path.basename(ckp_path)

          # copy the .index
          logger.info("Copying {} to {}".format(ckp_path+'.index',new_model_path))
          shutil.copyfile(ckp_path+'.index',os.path.join(new_model_path,str(latest_model_idx)+'.index'))

          # copy the .data
          logger.info("Copying {} to {}".format(ckp_path+'.data-00000-of-00001',new_model_path))
          shutil.copyfile(ckp_path+'.data-00000-of-00001',os.path.join(new_model_path,str(latest_model_idx)+'.data-00000-of-00001'))

          # copy the config.pkl
          logger.info("Copying {} to {}".format(os.path.join(original_model_path,'config.pkl'),new_model_path))
          shutil.copyfile(os.path.join(original_model_path,'config.pkl'),os.path.join(new_model_path,'config.pkl'))

          # write the new checkpoint metadata
          logger.info("Creating checkpoint in {}".format(new_model_path))
          shutil.copyfile(os.path.join(original_model_path,'checkpoint'),os.path.join(new_model_path,'checkpoint'))
          with open(os.path.join(new_model_path,"checkpoint"), "w") as ckpf_new:
            ckpf_new.write('model_checkpoint_path: "{}"\n'.format(str(latest_model_idx)))

          # update the new config with the local paths
          cfg_new["Poisson"][s]['model_path'] = new_model_path

      # copy the IC file
      os.makedirs(os.path.join('models',"Poisson",s,'IC'))
      original_ic_path = cfg["Poisson"][s]['IC']
      new_ic_path = os.path.join('models',"Poisson",s,'IC')

      logger.info("Copying {} to {}".format(original_ic_path,new_ic_path))
      shutil.copyfile(original_ic_path,os.path.join(new_ic_path,os.path.basename(original_ic_path)))

      # update the new config with the local paths
      cfg_new["Poisson"][s]['IC'] = os.path.join(new_ic_path,os.path.basename(original_ic_path))

#      # copy the icp files
      for p in ['htautau', 'ztautau', 'ttbar', 'diboson']:
        os.makedirs(os.path.join('models',"Poisson",s,'ICP_'+p))
        original_icp_path = cfg["Poisson"][s]['ICP'][p]
        new_icp_path = os.path.join('models',"Poisson",s,'ICP_'+p)

        logger.info("Copying {} to {}".format(original_icp_path,new_icp_path))
        shutil.copyfile(original_icp_path,os.path.join(new_icp_path,os.path.basename(original_icp_path)))

        # update the new config with the local paths
        cfg_new["Poisson"][s]["ICP"][p] = os.path.join(new_icp_path,os.path.basename(original_icp_path))

  # copy the CSI files
  os.makedirs("data/tmp_data")
  CSI_list = glob.glob(os.path.join(args.ntuple,'*.pkl'))
  for icsi in CSI_list:
    logger.info("Copying {} to {}".format(icsi,"data/tmp_data"))
    shutil.copyfile(icsi,os.path.join("data/tmp_data",os.path.basename(icsi)))

  # save the new config file
  with open('config_submission.yaml', 'w') as yaml_file:
    yaml.dump(cfg_new, yaml_file, default_flow_style=False)

  logger.info("config_submission.yaml saved.")

  # logger.info("Now tar the whole directory ;)")
  # subprocess.call(['tar', '-czf', '../submission.tar', '.'])
  zip_path = f'../submission_{args.version}.zip'
  if os.path.isfile(zip_path):
    logger.info(f"[ERROR] ZIP already exists: {zip_path}")
    logger.info(f"Do nothing and quit.")
    sys.exit()

  logger.info("Now zip the whole directory ;)")
  #subprocess.call(['zip', '-r', zip_path, '.','-x ".git/*" ".gitignore" ".*" "*/.*"'])
  #subprocess.run(['zip', '-r', zip_path, '.','-x ".git/*" ".gitignore" ".*" "*/.*"'])
  exclude_patterns = [".git/*", ".gitignore", ".*", "*/.*"]  # Add more patterns as needed
  exclude_patterns += ["README.md", "user/*", "submission/*", "toy_generator/*", "Workflow/configs/*"]
  exclude_args = sum([["-x", pattern] for pattern in exclude_patterns], [])
  subprocess.run(['zip', '-r', zip_path, '.',*exclude_args], check=True)

  txt_path = zip_path.replace(".zip", ".txt")
  with open(txt_path, "w") as f:
    f.write(f"config = {args.config}\n")
    f.write(f"offset = {args.offset}\n")
    f.write(f"inflate = {args.inflate}\n")

  logger.info(f"ZIP: {zip_path}")
  logger.info(f"TXT: {txt_path}")
