import importlib
import yaml
import numpy as np
import networks.Models as ms

class Inference:
  def __init__(self,cfg_path):
    with open(cfg_path) as f:
      self.cfg = yaml.safe_load(f)
      print("Config loaded from {}".format(cfg_path))
    assert "Data" in self.cfg, "Section Data not defined in config!"
    assert "Tasks" in self.cfg, "Section Tasks not defined in config!"

    self.data = {}
    self.models = {}
    self.teststat = {}
    self.selections = self.cfg['Data']['selections']
    self.n_split = self.cfg['Data']['n_split']
    self.max_n_batch = self.cfg['Data']['max_n_batch']
    self.loadmodels()
    self.loaddata()

  def loaddata(self):
    import common.datasets as datasets
    for s in self.selections:
      self.data[s] = datasets.get_data_loader( selection=s, n_split=self.n_split)
      print("Data loaded for selection: {}".format(s))

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

