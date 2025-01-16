import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '.')
import importlib
import yaml
import h5py
import numpy as np
import networks.Models as ms
from data_loader.data_loader_2 import H5DataLoader
import common.user as user

class Inference:
  def __init__(self,cfg_path):
    with open(cfg_path) as f:
      self.cfg = yaml.safe_load(f)
      print("Config loaded from {}".format(cfg_path))
    assert "Tasks" in self.cfg, "Section Tasks not defined in config!"
    assert "Selections" in self.cfg, "Section Selections not defined in config!"

    self.training_data = {}
    self.toys = {}
    self.models = {}
    self.teststat = {}
    self.selections = self.cfg['Selections']
    #self.n_split = self.cfg['Data']['n_split']
    #self.max_n_batch = self.cfg['Data']['max_n_batch']
    self.loadmodels()
    # FIXME: loading data and toys here might be a bit waste of memory...
    #self.load_training_data()
    #self.load_toy()

    self.h5s = {}

    # Loading cross section uncertainties
    self.alpha_bkg      = self.cfg['Parameters']['alpha_bkg']
    self.alpha_tt       = self.cfg['Parameters']['alpha_tt']
    self.alpha_diboson  = self.cfg['Parameters']['alpha_diboson']

  #def load_training_data(self):
  #  import common.datasets as datasets
  #  for s in self.selections:
  #    self.training_data[s] = datasets.get_data_loader( selection=s, n_split=self.n_split)
  #    print("Training data loaded for selection: {}".format(s))

  def training_data_loader(self,selection,n_split):
      import common.datasets as datasets
      d = datasets.get_data_loader( selection=selection, n_split=n_split)
      print("Training data loaded for selection: {}".format(selection))
      return d

  #def load_toy(self):
  #  #import common.selections
  #  for s in self.selections:
  #    toy_path = os.path.join(self.cfg['Data']['Toy']['dir'],s,self.cfg['Data']['Toy']['filename'])
  #    self.toys[s] = H5DataLoader(
  #        file_path          = toy_path,
  #        batch_size         = self.cfg['Data']['Toy']['batch_size'],
  #        n_split            = self.cfg['Data']['Toy']['n_split'],
  #        #selection_function = getattr(common.selections,s),
  #        selection_function = None,
  #    )
  #    print("Toy loaded for selection {} from {}.".format(s,toy_path))

  def load_toy_file(self,filename,batch_size,n_split):
      assert os.path.exists(filename), "Toy file {} does not exist!".format(filename)
      t = H5DataLoader(
          file_path          = filename,
          batch_size         = batch_size,
          n_split            = n_split,
          #selection_function = getattr(common.selections,s),
          selection_function = None,
      )
      print("Toy loaded from {}.".format(filename))
      return t

  def loadH5(self,filename,selection):
    h5f = h5py.File(filename)
    # check whether the model path matches
    for t in self.cfg['Tasks']:
      assert h5f.attrs[t+"_module"] == self.cfg[t][selection]['module'], "Task {} selection {}: inconsistent module! H5: {} -- Config: {}".format(t,selection,h5f.attrs[t+"_module"],self.cfg[t][s]['module'])
      assert h5f.attrs[t+"_model_path"] == self.cfg[t][selection]["model_path"], "Task {} selection {}: inconsistent model path! H5 {} -- Config {}".format(t,h5f.attrs[t+"_model_path"],self.cfg[t][s]["model_path"])
    return h5f

  def loadMLresults(self, name, filename, selection, ignore_done=False):
    h5_filename = os.path.join( user.output_directory, 'tmp_data', filename+'_'+selection+'.h5')
    assert os.path.exists(h5_filename), "File {} does not exist! Trying running the save mode first.".format(h5_filename)
    if (not ignore_done) and (name in self.h5s) and (selection in self.h5s[name]):
      #print("ML results {} with {} is already loaded. Skipping...".format(name,selection))
      pass
    else:
      h5f = self.loadH5(h5_filename, selection)
      if not name in self.h5s:
        self.h5s[name] = {}
      # self.h5s[name][selection] = h5f
      self.h5s[name][selection] = {}
      self.h5s[name][selection]["MultiClassifier_predict"] = h5f["MultiClassifier_predict"][:]
      self.h5s[name][selection]["htautau_DeltaA"] = h5f["htautau_DeltaA"][:]
      self.h5s[name][selection]["ztautau_DeltaA"] = h5f["ztautau_DeltaA"][:]
      self.h5s[name][selection]["ttbar_DeltaA"] = h5f["ttbar_DeltaA"][:]
      self.h5s[name][selection]["diboson_DeltaA"] = h5f["diboson_DeltaA"][:]
      self.h5s[name][selection]["Weight"] = h5f["Weight"][:]
      print("ML results {} with {} loaded from {}".format(name,selection,filename+selection+'.h5'))



  def loadmodels(self):
    for t in self.cfg['Tasks']:
      assert t in self.cfg, "{t} not defined in config!"
      self.models[t] = {}
      for s in self.selections:
        assert s in self.cfg[t], "{s} not define in {t} in the config!"
        m = ms.getModule(self.cfg[t][s]["module"])
        self.models[t][s] = m.load(self.cfg[t][s]["model_path"])
        print("Task {} selection {}: Module {} loaded with model path {}.".format(t,s,self.cfg[t][s]["module"],self.cfg[t][s]["model_path"]))
        #setattr(self,t,m.load(cfg[t]["model_path"]))

  def dSigmaOverDSigmaSM_h5( self, name, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_jes=0, nu_tes=0, nu_met=0):
      # Multiclassifier
      p_mc = self.h5s[name][selection]["MultiClassifier_predict"]

      # htautau
      DA_pnn_htautau = self.h5s[name][selection]["htautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
      nu_A_htautau = self.models['htautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
      p_pnn_htautau = np.exp( np.dot(DA_pnn_htautau, nu_A_htautau))

      # ztautau
      DA_pnn_ztautau = self.h5s[name][selection]["ztautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
      nu_A_ztautau = self.models['ztautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
      p_pnn_ztautau = np.exp( np.dot(DA_pnn_ztautau, nu_A_ztautau))

      # ttbar
      DA_pnn_ttbar = self.h5s[name][selection]["ttbar_DeltaA"] # <- this should be Nx9, 9 numbers per event
      nu_A_ttbar = self.models['ttbar'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
      p_pnn_ttbar = np.exp( np.dot(DA_pnn_ttbar, nu_A_ttbar))

      # diboson
      DA_pnn_diboson = self.h5s[name][selection]["diboson_DeltaA"] # <- this should be Nx9, 9 numbers per event
      nu_A_diboson = self.models['diboson'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
      p_pnn_diboson= np.exp( np.dot(DA_pnn_diboson, nu_A_diboson))

      # RATES
      f_bkg_rate = (1+self.alpha_bkg)**nu_bkg
      f_tt_rate = (1+self.alpha_tt)**nu_tt
      f_diboson_rate = (1+self.alpha_diboson)**nu_diboson

      #return (mu*p_mc[:,0]/(p_mc[:,1:].sum(axis=1)) + 1)*p_pnn_jes
      return ((mu*p_mc[:,0]*p_pnn_htautau + p_mc[:,1]*f_bkg_rate*p_pnn_ztautau + p_mc[:,2]*f_tt_rate*f_bkg_rate*p_pnn_ttbar + p_mc[:,3]*f_diboson_rate*f_bkg_rate*p_pnn_diboson) / p_mc[:,:].sum(axis=1))

  def save(self):
    """
    Save the ML prediction in a h5 file with the following information:
    Label, Weight, ML results (could be prediction from classifier or parameters from PNN)
    """

    for s in self.selections:
        for obj in self.cfg['Save']:

            # The data we run on
            if obj == "Toy":
                os.makedirs( os.path.join(user.output_directory, 'tmp_data'), exist_ok=True)
                obj_fn = os.path.join(user.output_directory, 'tmp_data', self.cfg['Toy_name']+'_'+s+'.h5')
                if os.path.exists(obj_fn):
                    print ("Warning! Temporary file %s exists. Will overwrite."%obj_fn )
            else:
                obj_fn = os.path.join(user.output_directory, 'tmp_data', obj+'_'+s+'.h5')

            with h5py.File(obj_fn, "w") as h5f:
              datasets = {
                  "Label": [],
                  "Weight": [],
                  }

              # Save general information
              h5f.attrs["selection"] = s
              for t in self.cfg['Tasks']:
                if not "save" in self.cfg[t]:
                  continue
                for iobj in self.cfg[t]['save']:
                  datasets[t+'_'+iobj] = []

                h5f.attrs[t+"_module"] = self.cfg[t][s]["module"]
                h5f.attrs[t+"_model_path"] = self.cfg[t][s]["model_path"]

              # Save ML results
              if obj=="TrainingData":
                data_input = self.training_data_loader(s,self.cfg['Save'][obj]['n_split'])
              else:
                toy_path = os.path.join(self.cfg['Save'][obj]['dir'],s,self.cfg['Toy_name']+'.h5')
                data_input = self.load_toy_file(toy_path,self.cfg['Save'][obj]['batch_size'],self.cfg['Save'][obj]['n_split'])
              for i_batch, batch in enumerate(data_input):
                features, weights, labels = data_input.split(batch)
                if obj!="TrainingData":
                  nevts = features.shape[0]
                  labels = np.array([-1]*nevts)
                  #weights = np.array([1]*nevts)

                datasets["Label"].append(labels)
                datasets["Weight"].append(weights)

                for t in self.cfg['Tasks']:
                  if not "save" in self.cfg[t]:
                    continue
                  for iobj in self.cfg[t]["save"]:
                    if iobj=="predict":
                      pred = self.models[t][s].predict(features)
                      datasets[t+'_'+iobj].append(pred)
                    elif iobj=='DeltaA':
                      DA = self.models[t][s].get_DeltaA(features)
                      datasets[t+'_'+iobj].append(DA)
                    else:
                      raise Exception("save type not recognized! Currently supported: predict, DeltaA")
                if self.cfg['Save'][obj]['max_n_batch']>-1 and i_batch>=self.cfg['Save'][obj]['max_n_batch']:
                  break

              for ds in datasets:
                datasets[ds] = np.concatenate(datasets[ds],axis=0)
                h5f.create_dataset(ds, data=datasets[ds],
                    compression="gzip",  # Use gzip compression
                    compression_opts=4   # Compression level (1: fastest, 9: smallest))
                )
              print("Saved temporary results in {}".format(obj_fn))

  def penalty(self, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
        return nu_bkg**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2

  def predict(self, mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met ):
    import time
    # perform the calculation
    uTerm = 0
    for selection in self.selections:
      #print("Predicting for region {}".format(selection))

      # Load ML result for training data
      self.loadMLresults( name='TrainingData', filename=self.cfg['Predict']['TrainingData'], selection=selection)

      # Load ML result for toy
      if self.cfg['Predict']['use_toy']:
          self.loadMLresults( name='Toy', filename=self.cfg['Toy_name'], selection=selection)

      # dSoDS for training data
      weights = self.h5s['TrainingData'][selection]["Weight"]
      dSoDS_sim = self.dSigmaOverDSigmaSM_h5( 'TrainingData',selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_jes=nu_jes, nu_tes=nu_tes, nu_met=nu_met )
      incS = (weights[:]*(1-dSoDS_sim)).sum()

      # dSoDS for toys
      if self.cfg['Predict']['use_toy']:
        weights_toy = self.h5s['Toy'][selection]["Weight"]
        dSoDS_toy = self.dSigmaOverDSigmaSM_h5( 'Toy',selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_jes=nu_jes, nu_tes=nu_tes, nu_met=nu_met )
      else:
        dSoDS_toy = dSoDS_sim
        weights_toy = weights

      penalty = self.penalty(nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)

      uTerm += -2 *(incS+(weights_toy[:]*np.log(dSoDS_toy)).sum())+penalty

    return uTerm

  def clossMLresults(self):
      for n in list(self.h5s):
        for s in list(self.h5s[n]):
          # self.h5s[n][s].close()
          del self.h5s[n][s]
        del self.h5s[n]
