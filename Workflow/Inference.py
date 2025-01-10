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

class Inference:
  def __init__(self,cfg_path):
    with open(cfg_path) as f:
      self.cfg = yaml.safe_load(f)
      print("Config loaded from {}".format(cfg_path))
    assert "Data" in self.cfg, "Section Data not defined in config!"
    assert "Tasks" in self.cfg, "Section Tasks not defined in config!"

    self.data = {}
    self.toys = {}
    self.models = {}
    self.teststat = {}
    self.selections = self.cfg['Data']['selections']
    self.n_split = self.cfg['Data']['n_split']
    self.max_n_batch = self.cfg['Data']['max_n_batch']
    self.loadmodels()
    # FIXME: loading data and toys here might be a bit waste of memory...
    self.loaddata()
    self.loadtoy()
    self.h5s = {}

  def loaddata(self):
    import common.datasets as datasets
    for s in self.selections:
      self.data[s] = datasets.get_data_loader( selection=s, n_split=self.n_split)
      print("Data loaded for selection: {}".format(s))

  def loadtoy(self):
    for s in self.selections:
      toy_path = os.path.join(self.cfg['Data']['Toy']['dir'],s,self.cfg['Data']['Toy']['filename'])
      self.toys[s] = H5DataLoader(
          file_path          = toy_path, 
          batch_size         = self.cfg['Data']['Toy']['batch_size'],
          n_split            = self.cfg['Data']['Toy']['n_split'],
          selection_function = None,
      ) 
      print("Toy loaded for selection {} from {}.".format(s,toy_path))

  def loadH5(self,filename):
    h5f = h5py.File(filename)
    # check whether the model path matches
    for t in self.cfg['Tasks']:
      assert h5f.attrs[t+"_module"] == self.cfg[t]['module'], "Task {}: inconsistent module! H5: {} -- Config: {}".format(t,h5f.attrs[t+"_module"],self.cfg[t]['module'])
      assert h5f.attrs[t+"_model_path"] == self.cfg[t]["model_path"], "Task {}: inconsistent model path! H5 {} -- Config {}".format(t,h5f.attrs[t+"_model_path"],self.cfg[t]["model_path"])
    return h5f

  def loadMLresults(self,name,filename=None,ignore_done=False):
    if (ignore_done) or (not name in self.h5s):
      self.h5s[name] = {}
      for s in self.selections:
        if filename is None:
          filename = self.cfg["Predict"]["sim_path"]
        filename = filename+s+'.h5'
        h5f = self.loadH5(filename)
        self.h5s[name][s] = h5f
        print("ML results loaded from {}".format(filename))


  def loadmodels(self):
    for t in self.cfg['Tasks']:
      assert t in self.cfg, "{t} not defined in config!"
      m = ms.getModule(self.cfg[t]["module"])
      self.models[t] = m.load(self.cfg[t]["model_path"])
      print("Task {}: Module {} loaded with model path {}.".format(t,self.cfg[t]["module"],self.cfg[t]["model_path"]))
      #setattr(self,t,m.load(cfg[t]["model_path"]))

  def dSigmaOverDSigmaSM( self, features, mu=1, nu_jes=0 ):
      # FIXME: this ONLY works with multiclassifier and JES. Will make it more flexible when other uncertainties come
      p_mc = self.models['MultiClassifier'].predict(features)
      p_pnn_jes = self.models['JES'].predict(features, nu=(nu_jes,))
      return (mu*p_mc[:,0]/(p_mc[:,1:].sum(axis=1)) + 1)*p_pnn_jes

  def dSigmaOverDSigmaSM_h5( self, name, selection, mu=1, nu_ztautau=0, nu_tt=0, nu_diboson=0, nu_jes=0, nu_tes=0, nu_met=0):
      # FIXME: this ONLY works with multiclassifier and JES. Will make it more flexible when other uncertainties come
      # Multiclassifier
      p_mc = self.h5s[name][selection]["MultiClassifier_predict"]

      # JES
      DA_pnn_jes = self.h5s[name][selection]["JES_DeltaA"]
      nu_A = self.models['JES'].nu_A((nu_jes,))
      p_pnn_jes = np.exp( np.dot(DA_pnn_jes, nu_A))

      # TES
      # to be implemented

      # MET
      # to be implemented

      # RATES
      # FIXME: hardcode alphas for now
      alpha_ztautau = 0
      alpha_tt = 0
      alpha_diboson = 0

      f_ztautau_rate = (1+alpha_ztautau)**nu_ztautau
      f_tt_rate = (1+alpha_tt)**nu_tt
      f_diboson_rate = (1+alpha_diboson)**nu_diboson 

      #return (mu*p_mc[:,0]/(p_mc[:,1:].sum(axis=1)) + 1)*p_pnn_jes
      return ((mu*p_mc[:,0] + p_mc[:,1]*f_ztautau_rate + p_mc[:,2]*f_tt_rate + p_mc[:,3]*f_diboson_rate) / p_mc[:,:].sum(axis=1))*p_pnn_jes


  def testStat(self, mu, nu_jes):
    for s in self.selections:
      print("Calculating test statistics for selection {}".format(s))
      self.teststat[s] = 0
      for i_batch, batch in enumerate(self.data[s]):
        features, _, _ = self.data[s].split(batch)
        dSoDS = self.dSigmaOverDSigmaSM( features, mu=mu, nu_jes = nu_jes )
        ts_batch = np.log(dSoDS).sum()
        self.teststat[s] += ts_batch

        if self.max_n_batch>-1 and i_batch>=self.max_n_batch:
          break
    return self.teststat

  def save(self, filename, isData):
    """
    Save the ML prediction in a h5 file with the following information:
    Label, Weight, ML results (could be prediction from classifier or parameters from PNN)
    filename: base name of the h5 file
    isData: whether the events come from data. If set to True, will assign all lables to -1 and all weights to 1.
    """
    for s in self.selections:
      with h5py.File(filename+s+'.h5', "w") as h5f:
        datasets = {
            "Label": [],
            "Weight": [],
            }

        h5f.attrs["selection"] = s
        for t in self.cfg['Tasks']:
          if not "save" in self.cfg[t]:
            continue
          for obj in self.cfg[t]['save']:
            datasets[t+'_'+obj] = []
          
          h5f.attrs[t+"_module"] = self.cfg[t]["module"]
          h5f.attrs[t+"_model_path"] = self.cfg[t]["model_path"]

        data_input = self.data[s] if not isData else self.toys[s]
        for i_batch, batch in enumerate(data_input):
          features, weights, labels = data_input.split(batch)
          if isData:
            nevts = features.shape[0]
            labels = np.array([-1]*nevts)
            weights = np.array([1]*nevts)

          datasets["Label"].append(labels)
          datasets["Weight"].append(weights)

          for t in self.cfg['Tasks']:
            if not "save" in self.cfg[t]:
              continue
            for obj in self.cfg[t]["save"]:
              if obj=="predict":
                pred = self.models[t].predict(features)
                datasets[t+'_'+obj].append(pred)
              elif obj=='DeltaA':
                DA = self.models[t].get_DeltaA(features)
                datasets[t+'_'+obj].append(DA)
              else:
                raise Exception("save type not recognized! Currently supported: predict, DeltaA")
          if self.max_n_batch>-1 and i_batch>=self.max_n_batch:
            break

        for obj in datasets:
          datasets[obj] = np.concatenate(datasets[obj],axis=0)
          h5f.create_dataset(obj, data=datasets[obj])
        print("Saved ML results in {}".format(filename+s+'.h5'))

  def penalty(self, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
        return nu_ztautau**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2

  def predict(self, selection, mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met, isData):
    self.loadMLresults(name='sim')
    # calculate the toy h5 file on the fly, commented out for now
    #self.save("toy",isData)

    # perform the calculation
    assert selection in self.selections, "Selection {} not available!".format(selection)

    weights = self.h5s['sim'][selection]["Weight"]
    dSoDS_sim = self.dSigmaOverDSigmaSM_h5( 'sim',selection, mu=mu, nu_ztautau=nu_ztautau, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_jes=nu_jes, nu_tes=nu_tes, nu_met=nu_met )
    incS = (weights[:]*(1-dSoDS_sim)).sum()
    penalty = self.penalty(nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)

    # Handle toys
    if isData:
      self.save("toy",isData)
      self.loadMLresults(name='toy',filename='toy')
      weights_toy = self.h5s['toy'][selection]["Weight"]
      dSoDS_toy = self.dSigmaOverDSigmaSM_h5( 'toy',selection, mu=mu, nu_ztautau=nu_ztautau, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_jes=nu_jes, nu_tes=nu_tes, nu_met=nu_met )
    else:
      dSoDS_toy = dSoDS_sim
      weights_toy = weights

    uTerm = -2 *(incS+(weights_toy[:]*np.log(dSoDS_toy)).sum())+penalty

    return uTerm

  def clossMLresults(self):
      for n in list(self.h5s):
        for s in list(self.h5s[n]):
          self.h5s[n][s].close()
          del self.h5s[n][s]
        del self.h5s[n]
