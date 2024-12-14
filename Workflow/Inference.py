import importlib
import yaml
import h5py
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
    self.MLloaded=False

  def loaddata(self):
    import common.datasets as datasets
    for s in self.selections:
      self.data[s] = datasets.get_data_loader( selection=s, n_split=self.n_split)
      print("Data loaded for selection: {}".format(s))

  def loadMLresults(self):
    self.h5s = {}
    for s in self.selections:
      h5f = h5py.File(self.cfg["Predict"]["sim_path"]+s+'.h5')
      # check whether the model path matches
      for t in self.cfg['Tasks']:
        assert h5f.attrs[t+"_module"] == self.cfg[t]['module'], "Task {}: inconsistent module! H5: {} -- Config: {}".format(t,h5f.attrs[t+"_module"],self.cfg[t]['module'])
        assert h5f.attrs[t+"_model_path"] == self.cfg[t]["model_path"], "Task {}: inconsistent model path! H5 {} -- Config {}".format(t,h5f.attrs[t+"_model_path"],self.cfg[t]["model_path"])
      self.h5s[s] = h5f
      print("ML results loaded from {}".format(self.cfg["Predict"]["sim_path"]+s+'.h5'))
      self.MLloaded = True


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

  def dSigmaOverDSigmaSM_h5( self, h5f, mu=1, nu_jes=0 ):
      # FIXME: this ONLY works with multiclassifier and JES. Will make it more flexible when other uncertainties come
      p_mc = h5f["MultiClassifier_predict"]
      DA_pnn_jes = h5f["JES_DeltaA"]
      bias_pnn_jes = self.models['JES'].get_bias()
      nu_A = self.models['JES'].nu_A((nu_jes,))
      p_pnn_jes = np.exp( bias_pnn_jes + np.dot(DA_pnn_jes, nu_A))
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

        for i_batch, batch in enumerate(self.data[s]):
          features, weights, labels = self.data[s].split(batch)
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


  def predict(self, mu, nu_jes, isData):
    if not self.MLloaded:
      self.loadMLresults()
    # calculate the toy h5 file on the fly, commented out for now
    #self.save("toy",isData)

    # perform the calculation
    teststat_h5 = {}
    for s in self.selections:
      dSoDS = self.dSigmaOverDSigmaSM_h5( self.h5s[s], mu=mu, nu_jes = nu_jes )
      testStat = np.log(dSoDS).sum()
      teststat_h5[s] = testStat

    return teststat_h5

  def clossMLresults(self):
    if self.MLloaded:
      for s in self.h5s:
        self.h5s[s].close()
    self.MLloaded = False
