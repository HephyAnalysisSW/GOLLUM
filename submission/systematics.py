#!/usr/bin/env python
# -*- coding: utf-8 -*-

__doc__ = """
This module contains the systematics functions for the FAIR Challenge.
Originally written by David Rousseau, and Victor Estrade.
"""
__version__ = "4.0"
__author__ = "David Rousseau, and Victor Estrade "


import copy
import pandas as pd
import numpy as np
from derived_quantities import DER_data

import logging
logger = logging.getLogger("UNC")

# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:
    """
    A simple 4-vector class to ease calculation, work easy peasy on numpy vector of 4 vector
    """

    px = 0.0
    py = 0.0
    pz = 0.0
    e = 0.0

    def __init__(self, apx=0.0, apy=0.0, apz=0.0, ae=0.0):
        """
        Constructor with 4 coordinates

        Parameters:
            apx (float): x coordinate
            apy (float): y coordinate
            apz (float): z coordinate
            ae (float): energy coordinate

        Returns:
            None
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError(
                "Energy is too small! Energy: {}, p: {}".format(self.e, self.p())
            )

    def copy(self):
        """
        Copy the current V4 object

        Parameters:
            None

        Returns:
            copy (V4): a copy of the current V4 object
        """
        return copy.deepcopy(self)

    def p2(self):
        """
        Compute the squared norm of the 3D momentum

        Parameters:
            None

        Returns:
            p2 (float): squared norm of the 3D momentum
        """
        return self.px**2 + self.py**2 + self.pz**2

    def p(self):
        """
        Compute the norm of the 3D momentum

        Parameters:
            None

        Returns:
            p (float): norm of the 3D momentum

        """
        return np.sqrt(self.p2())

    def pt2(self):
        """
        Compute the squared norm of the transverse momentum

        Parameters:
            None

        Returns:
            pt2 (float): squared norm of the transverse momentum
        """
        return self.px**2 + self.py**2

    def pt(self):
        """
        Compute the norm of the transverse momentum

        Parameters:
            None

        Returns:
            pt (float): norm of the transverse momentum
        """

        return np.sqrt(self.pt2())

    def m(self):
        """
        Compute the mass

        Parameters:
            None

        Returns:
            m (float): mass
        """

        return np.sqrt(np.abs(self.e**2 - self.p2()))  # abs is needed for protection

    def eta(self):
        """
        Compute the pseudo-rapidity

        Parameters:
            None

        Returns:
            eta (float): pseudo-rapidity
        """

        return np.arcsinh(self.pz / self.pt())

    def phi(self):
        """
        Compute the azimuthal angle

        Parameters:
            None

        Returns:
            phi (float): azimuthal angle
        """

        return np.arctan2(self.py, self.px)

    def deltaPhi(self, v):
        """
        Compute the azimuthal angle difference with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaPhi (float) - azimuthal angle difference
        """

        return (self.phi() - v.phi() + 3 * np.pi) % (2 * np.pi) - np.pi

    def deltaEta(self, v):
        """
        Compute the pseudo-rapidity difference with another V4 object

        Parameters:
            v (V4): the other V4 object

        Returns:
            deltaPhi (float): azimuthal angle difference

        """
        return self.eta() - v.eta()

    def deltaR(self, v):
        """
        Compute the delta R with another V4 object

        Parameters:
            v (V4): the other V4 object

        Returns:
            deltaEta (float): pseudo-rapidity difference
        """

        return np.sqrt(self.deltaPhi(v) ** 2 + self.deltaEta(v) ** 2)

    def eWithM(self, m=0.0):
        """
        Compute the energy with a given mass

        Parameters:
            m (float): mass

        Returns:
            e (float): energy with a given mass

        """

        return np.sqrt(self.p2() + m**2)

    def __str__(self):

        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (
            self.px,
            self.py,
            self.pz,
            self.e,
            self.pt(),
            self.eta(),
            self.phi(),
            self.m(),
        )

    def scale(self, factor=1.0):
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs(factor * self.e)

    def scaleFixedM(self, factor=1.0):
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)

    def setPtEtaPhiM(self, pt=0.0, eta=0.0, phi=0.0, m=0.0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt * np.cos(phi)
        self.py = pt * np.sin(phi)
        self.pz = pt * np.sinh(eta)
        self.e = self.eWithM(m)

    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e

    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px = self.px + other.px
            self.py = self.py + other.py
            self.pz = self.pz + other.pz
            self.e = self.e + other.e
        except AttributeError:
            return NotImplemented
        return self

    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px = copy.px + other.px
            copy.py = copy.py + other.py
            copy.pz = copy.pz + other.pz
            copy.e = copy.e + other.e
        except AttributeError:
            return NotImplemented
        return copy


def ttbar_bkg_weight_norm(weights, detailedlabel, systBkgNorm):
    """
    Apply a scaling to the weight. For ttbar background

    Args:
        * weights (array-like): The weights to be scaled
        * detailedlabel (array-like): The detailed labels
        * systBkgNorm (float): The scaling factor

    Returns:
        array-like: The scaled weights
    """
    weights[detailedlabel == "ttbar"] = weights[detailedlabel == "ttbar"] * systBkgNorm
    return weights


def diboson_bkg_weight_norm(weights, detailedlabel, systBkgNorm):
    """
    Apply a scaling to the weight. For Diboson background

    Args:
        * weights (array-like): The weights to be scaled
        * detailedlabel (array-like): The detailed labels
        * systBkgNorm (float): The scaling factor


    Returns:
        array-like: The scaled weights

    """
    weights[detailedlabel == "diboson"] = (
        weights[detailedlabel == "diboson"] * systBkgNorm
    )
    return weights


def all_bkg_weight_norm(weights, label, systBkgNorm):
    """
    Apply a scaling to the weight.

    Args:
        weights (array-like): The weights to be scaled
        label (array-like): The labels
        systBkgNorm (float): The scaling factor

    Returns:
        array-like: The scaled weights

    """
    weights[label == 0] = weights[label == 0] * systBkgNorm
    return weights


# ==================================================================================
# Manipulate the 4-momenta
# ==================================================================================
def mom4_manipulate(data, systTauEnergyScale, systJetEnergyScale, soft_met, seed=31415):
    """
    Manipulate primary inputs : the PRI_had_pt PRI_jet_leading_pt PRI_jet_subleading_pt and recompute the others values accordingly.

    Args:
        * data (pandas.DataFrame): The dataset to be manipulated
        * systTauEnergyScale (float): The factor applied to PRI_had_pt
        * systJetEnergyScale (float): The factor applied to all jet pt
        * soft_met (float): The additional soft MET energy
        * seed (int): The random seed

    Returns:
        pandas.DataFrame: The manipulated dataset

    """
    logger.info("Starting mom4_manipulate function")

    try:
        vmet = V4()
        vmet.setPtEtaPhiM(data["PRI_met"], 0.0, data["PRI_met_phi"], 0.0)
        logger.info("Initialized vmet with PRI_met and PRI_met_phi")
    except Exception as e:
        logger.error(f"Error initializing vmet: {e}")
    # met_sumet=data["PRI_met_sumet"]

    if systTauEnergyScale != 1.0:
        try:
            data["PRI_had_pt"] *= systTauEnergyScale
            logger.info(f"Applied systTauEnergyScale to PRI_had_pt")
        except Exception as e:
            logger.error(f"Error applying systTauEnergyScale to PRI_had_pt: {e}")
        try:
            vtau = V4()
            vtau.setPtEtaPhiM(
                data["PRI_had_pt"], data["PRI_had_eta"], data["PRI_had_phi"], 0.8
            )
            logger.info("Initialized vtau with PRI_had_pt, PRI_had_eta, PRI_had_phi")
        except Exception as e:
            logger.error(f"Error initializing vtau: {e}")
        try:
            vtauDeltaMinus = vtau.copy()
            vtauDeltaMinus.scaleFixedM((1.0 - systTauEnergyScale) / systTauEnergyScale)
            vmet += vtauDeltaMinus
            vmet.pz = 0.0
            vmet.e = vmet.eWithM(0.0)
        except Exception as e:
            logger.error(f"Error during tau scaling and updating vmet: {e}")
    if systJetEnergyScale != 1.0:
        try:
            data["PRI_jet_leading_pt"] = np.where(
                data["PRI_n_jets"] > 0,
                data["PRI_jet_leading_pt"] * systJetEnergyScale,
                data["PRI_jet_leading_pt"],
            )
            logger.info("Applied systJetEnergyScale to PRI_jet_leading_pt")
        except Exception as e:
            logger.error(f"Error applying systJetEnergyScale to PRI_jet_leading_pt: {e}")

        try:
            data["PRI_jet_subleading_pt"] = np.where(
                data["PRI_n_jets"] > 1,
                data["PRI_jet_subleading_pt"] * systJetEnergyScale,
                data["PRI_jet_subleading_pt"],
            )
            logger.info("Applied systJetEnergyScale to PRI_jet_subleading_pt")
        except Exception as e:
            logger.error(f"Error applying systJetEnergyScale to PRI_jet_subleading_pt: {e}")

        try:
            data["PRI_jet_all_pt"] *= systJetEnergyScale
            logger.info("Applied systJetEnergyScale to PRI_jet_all_pt")
        except Exception as e:
            logger.error(f"Error applying systJetEnergyScale to PRI_jet_all_pt: {e}")

        try:
            vj1 = V4()
            vj1.setPtEtaPhiM(
                data["PRI_jet_leading_pt"].where(data["PRI_n_jets"] > 0, other=0.0),
                data["PRI_jet_leading_eta"].where(data["PRI_n_jets"] > 0, other=0.0),
                data["PRI_jet_leading_phi"].where(data["PRI_n_jets"] > 0, other=0.0),
                0.0,
            )
            logger.info("Initialized vj1 with PRI_jet_leading_pt, PRI_jet_leading_eta, PRI_jet_leading_phi")
        except Exception as e:
            logger.error(f"Error initializing vj1: {e}")

        try:
            vj1DeltaMinus = vj1.copy()
            vj1DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
            vmet += vj1DeltaMinus
            vmet.pz = 0.0
            vmet.e = vmet.eWithM(0.0)
            logger.info("Applied jet1 scaling and updated vmet")
        except Exception as e:
            logger.error(f"Error during jet1 scaling and updating vmet: {e}")
        try:
            eta = data["PRI_jet_subleading_eta"].where(data["PRI_n_jets"] > 1, other=0.0)
            eta = np.clip(eta, -10, 10)
            vj2 = V4()
            vj2.setPtEtaPhiM(
                data["PRI_jet_subleading_pt"].where(data["PRI_n_jets"] > 1, other=0.0),
                eta,
                data["PRI_jet_subleading_phi"].where(data["PRI_n_jets"] > 1, other=0.0),
                0.0,
            )
            logger.info("Initialized vj2 with PRI_jet_subleading_pt, PRI_jet_subleading_eta, PRI_jet_subleading_phi")
        except Exception as e:
            logger.error(f"Error initializing vj2: {e}")

        try:
            vj2DeltaMinus = vj2.copy()
            vj2DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
            vmet += vj2DeltaMinus
            vmet.pz = 0.0
            vmet.e = vmet.eWithM(0.0)
            logger.info("Applied jet2 scaling and updated vmet")
        except Exception as e:
            logger.error(f"Error during jet2 scaling and updating vmet: {e}")

    if soft_met > 0:
        try:
            logger.info("Initializing random state with seed.")
            seed = int(seed)
            random_state = np.random.RandomState(seed=seed)
            logger.info("Getting dataset size.")
            SIZE = data.shape[0]
            logger.info(f"Dataset size: {SIZE}")
            logger.info("Creating V4 instance for soft MET term.")
            v4_soft_term = V4()
            logger.info("V4 instance created for soft MET term.")
            logger.info("Generating random px for soft MET term.")
            v4_soft_term.px = random_state.normal(0.0, soft_met, size=SIZE).astype(np.float64)
            logger.info(f"Generated px: Mean = {v4_soft_term.px.mean()}, Std = {v4_soft_term.px.std()}")
            logger.info("Generating random py for soft MET term.")
            v4_soft_term.py = random_state.normal(0.0, soft_met, size=SIZE).astype(np.float64)
            logger.info(f"Generated py: Mean = {v4_soft_term.py.mean()}, Std = {v4_soft_term.py.std()}")
            logger.info("Initializing pz to zeros for soft MET term.")
            v4_soft_term.pz = np.zeros(SIZE, dtype=np.float64)
            logger.info(f"pz initialized with shape: {v4_soft_term.pz.shape}, dtype: {v4_soft_term.pz.dtype}")
            logger.info("Calculating energy component e with zero mass for soft MET term.")
            v4_soft_term.e = v4_soft_term.eWithM(0.0)
            logger.info(f"Energy component e calculated with shape: {v4_soft_term.e.shape if isinstance(v4_soft_term.e, np.ndarray) else 'scalar'}")
            logger.info("Adding soft MET term to vmet.")
            vmet = vmet + v4_soft_term
            logger.info("Successfully added soft MET term to vmet.")
            logger.info("Added soft MET term to vmet")
        except Exception as e:
            logger.error(f"Error adding soft MET term: {e}")
    try:
        data["PRI_met"] = vmet.pt()
        data["PRI_met_phi"] = vmet.phi()
        logger.info("Updated PRI_met and PRI_met_phi from vmet")
    except Exception as e:
        logger.error(f"Error updating PRI_met and PRI_met_phi: {e}")

    DECIMALS = 3
    try:
        data["PRI_had_pt"] = data["PRI_had_pt"].round(decimals=DECIMALS)
        data["PRI_had_eta"] = data["PRI_had_eta"].round(decimals=DECIMALS)
        data["PRI_had_phi"] = data["PRI_had_phi"].round(decimals=DECIMALS)
        data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
        data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
        data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
        data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
        data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
        data["PRI_jet_leading_pt"] = data["PRI_jet_leading_pt"].round(decimals=DECIMALS)
        data["PRI_jet_leading_eta"] = data["PRI_jet_leading_eta"].round(decimals=DECIMALS)
        data["PRI_jet_leading_phi"] = data["PRI_jet_leading_phi"].round(decimals=DECIMALS)
        data["PRI_jet_subleading_pt"] = data["PRI_jet_subleading_pt"].round(decimals=DECIMALS)
        data["PRI_jet_subleading_eta"] = data["PRI_jet_subleading_eta"].round(decimals=DECIMALS)
        data["PRI_jet_subleading_phi"] = data["PRI_jet_subleading_phi"].round(decimals=DECIMALS)
        data["PRI_jet_all_pt"] = data["PRI_jet_all_pt"].round(decimals=DECIMALS)
        logger.info("Rounded all relevant features to DECIMALS places")
    except Exception as e:
        logger.error(f"Error rounding feature values: {e}")

    return data


def make_unweighted_set(data_set):
    keys = ["htautau", "ztautau", "ttbar", "diboson"]
    unweighted_set = {}
    for key in keys:
        unweighted_set[key] = data_set["data"][data_set["detailedlabel"] == key].sample(
            frac=1, random_state=31415
        )

    return unweighted_set

def postprocess(data):
    """
    Select the events with the following conditions:
    * PRI_had_pt > 26
    * PRI_jet_leading_pt > 26
    * PRI_jet_subleading_pt > 26
    * PRI_lep_pt > 20

    This is applied to the dataset after the systematics are applied

    Args:
        data (pandas.DataFrame): The manipulated dataset

    Returns:
        pandas.DataFrame: The postprocessed dataset
    """
    # apply higher threshold on had pt (dropping events)
    data = data.drop(data[data.PRI_had_pt < 26].index)

    #need to reindex
    data.reset_index(drop=True, inplace=True)

    # apply threshold on leading and subleading jets if they exist
    # note that it is assumed that the systematics transformation is monotonous in pt
    # so that leading and subleading jet should never be swapped

    # if subleading jet pt below high threshold, do so it never existed
    mask = data['PRI_jet_subleading_pt'].between(0, 26)
    data.loc[mask, 'PRI_jet_all_pt'] -= data['PRI_jet_subleading_pt']
    data.loc[mask, 'PRI_jet_subleading_pt'] = -25
    data.loc[mask, 'PRI_jet_subleading_eta'] = -25
    data.loc[mask, 'PRI_jet_subleading_phi'] = -25
    data.loc[mask, 'PRI_n_jets'] -= 1

    # if leading jet pt below high threshold, do so it never existed
    mask = data['PRI_jet_leading_pt'].between(0, 26)
    data.loc[mask, 'PRI_jet_all_pt'] -= data['PRI_jet_leading_pt']
    data.loc[mask, 'PRI_jet_leading_pt'] = -25
    data.loc[mask, 'PRI_jet_leading_eta'] = -25
    data.loc[mask, 'PRI_jet_leading_phi'] = -25
    data.loc[mask, 'PRI_n_jets'] -= 1



    # apply low threshold on lepton pt (does nothing)
    data = data.drop(data[data.PRI_lep_pt < 20].index)

    return data

def systematics(
    data_set=None,
    tes=1.0,
    jes=1.0,
    soft_met=0.0,
    seed=31415,
    dopostprocess=True,
):
    """
    Apply systematics to the dataset

    Args:
        * data_set (dict): The dataset to apply systematics to
        * tes (float): The factor applied to PRI_had_pt
        * jes (float): The factor applied to all jet pt
        * soft_met (float): The additional soft MET energy
        * seed (int): The random seed
        * ttbar_scale (float): The scaling factor for ttbar background
        * diboson_scale (float): The scaling factor for diboson background
        * bkg_scale (float): The scaling factor for other backgrounds

    Returns:
        dict: The dataset with applied systematics
    """
    logger.info("Entering systematics function.")
    logger.info(f"Input parameters - tes: {tes}, jes: {jes}, soft_met: {soft_met}, seed: {seed}, dopostprocess: {dopostprocess}")
    data_set_new = data_set.copy()
    logger.info("Dataset copied successfully. Checking initial data structure and types.")
    for key, value in data_set_new.items():
        if isinstance(value, pd.DataFrame):
            logger.info(f"DataFrame '{key}' - Shape: {value.shape}, Dtypes: {value.dtypes}")
        elif isinstance(value, np.ndarray):
            logger.info(f"Numpy array '{key}' - Dtype: {value.dtype}, Shape: {value.shape}")

    # modify primary features according to tes, jes softmet
    logger.info("Calling mom4_manipulate to apply systematic variations to primary features.")
    data_syst = data_set["data"].copy()

    try:
        data_syst = mom4_manipulate(
            data=data_set["data"].copy(),
            systTauEnergyScale=tes,
            systJetEnergyScale=jes,
            soft_met=soft_met,
            seed=seed,
        )
        logger.info("Systematic variations applied to primary features successfully.")
    except Exception as e:
        logger.error(f"Error applying systematic variations with mom4_manipulate: {e}")
        raise

    if dopostprocess:
        logger.info("Postprocessing the dataset (applying thresholds and possibly removing events).")
        try:
            data_syst = postprocess(data_syst)
            logger.info("Postprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            raise

    logger.info("Building the resulting dataset dictionary.")
    data_syst_set = {}
    for key in data_set_new.keys():
        if key not in ["data","settings"]:
            data_syst_set[key] = data_syst.pop(key)
            logger.info(f"Moved key '{key}' from data_syst to data_syst_set.")
    # compute DERived features
    logger.info("Calculating derived features using DER_data.")
    try:
        data_syst_set["data"] = DER_data(data_syst)
        logger.info("Derived features calculated successfully.")
    except Exception as e:
        logger.error(f"Error calculating derived features with DER_data: {e}")
        raise
    if "settings" in data_set_new.keys():
        data_syst_set["settings"] = data_set_new["settings"]
        logger.info("Added 'settings' to data_syst_set.")

    logger.info("Exiting systematics function. Returning modified dataset.")
    return data_syst_set


def get_bootstrapped_dataset(
    test_set,
    mu=1.0,
    seed=11415,
    ttbar_scale=None,
    diboson_scale=None,
    bkg_scale=None,
    poisson=True,
):
    """
    Generate a bootstrapped dataset

    Args:
        * test_set (dict): The original test dataset
        * mu (float): The scaling factor for htautau background
        * seed (int): The random seed
        * ttbar_scale (float): The scaling factor for ttbar background
        * diboson_scale (float): The scaling factor for diboson background
        * bkg_scale (float): The scaling factor for other backgrounds

    Returns:
        pandas.DataFrame: The bootstrapped dataset
    """
    bkg_norm = {
        "ztautau": 1.0,
        "diboson": 1.0,
        "ttbar": 1.0,
        "htautau": 1.0,
    }

    if bkg_scale is not None:
        bkg_scale_ = bkg_scale
    else:
        bkg_scale_ = 1.0

    if ttbar_scale is not None:
        bkg_norm["ttbar"] = ttbar_scale * bkg_scale_

    if diboson_scale is not None:
        bkg_norm["diboson"] = diboson_scale * bkg_scale_

    if bkg_scale is not None:
        bkg_norm["ztautau"] = bkg_scale_

    bkg_norm["htautau"] = mu

    pseudo_data = []
    Seed = seed
    for i, key in enumerate(test_set.keys()):
        Seed = Seed + i

        if poisson:
            random_state = np.random.RandomState(seed=Seed)
            new_weights = random_state.poisson(bkg_norm[key] * test_set[key]["weights"])
        else:
            new_weights = bkg_norm[key] * test_set[key]["weights"]

        temp_data = test_set[key][new_weights > 0]


        temp_data.loc[:, "weights"] = new_weights[new_weights > 0]

        temp_data.loc[:, "labels"] = 1 if key == "htautau" else 0

        temp_data["detailed_labels"] = np.nan

        temp_data.loc[:, "detailed_labels"] = 0 if key == "htautau" else temp_data.loc[:, "detailed_labels"]

        temp_data.loc[:, "detailed_labels"] = 1 if key == "ztautau" else temp_data.loc[:, "detailed_labels"]

        temp_data.loc[:, "detailed_labels"] = 2 if key == "ttbar" else temp_data.loc[:, "detailed_labels"]

        temp_data.loc[:, "detailed_labels"] = 3 if key == "diboson" else temp_data.loc[:, "detailed_labels"]

        pseudo_data.append(temp_data)

    pseudo_data = pd.concat(pseudo_data)

    pseudo_data.reset_index(drop=True, inplace=True)

    unweighted_data = repeat_rows_by_weight(pseudo_data.copy(), seed=seed)

    return unweighted_data


def get_systematics_dataset(
    data_set,
    tes=1.0,
    jes=1.0,
    soft_met=0.0,
    dopostprocess=False,
    save_to_hdf5=False,
    hdf5_filename=None,
    seed = 31415,
):
    from systematics import systematics
    logger.info("Entered get_syst_train_set function.")

    # Convert train_set['data'] to DataFrame if it is a numpy array
    columns = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']
    if isinstance(data_set['data'], np.ndarray):
        logger.info("Converting train_set['data'] to DataFrame.")
        data_set['data'] = pd.DataFrame(data_set['data'], columns=columns)

    # Call systematics function
    syst_test_set = systematics(
        data_set,
        tes,
        jes,
        soft_met,
        dopostprocess=dopostprocess,
        seed = seed,
    )

    # Save to HDF5 files if required
    if save_to_hdf5:
        # Extract all data, labels, detailed_labels, and weights
        data = syst_test_set['data'].copy()

        # Save to the specified HDF5 file
        with h5py.File(hdf5_filename, "w") as hf:
            if isinstance(data, pd.DataFrame):
                data_array = data.to_numpy(dtype='float64')
                logger.info(f"Saving 'data' to file '{hdf5_filename}' with dtype: {data_array.dtype}")
                hf.create_dataset('data', data=data_array, compression="gzip")
                del data_array  # Release memory

        logger.info(f"Systematic test set saved to {hdf5_filename}")

    return syst_test_set


# Assuming 'data_set' is a DataFrame with a 'weights' column
def repeat_rows_by_weight(data_set,seed=11415):

    # Ensure 'weights' column is integer, as fractional weights don't make sense for row repetition
    data_set["weights"] = data_set["weights"].astype(int)

    # Repeat rows based on the 'weights' column
    repeated_data_set = data_set.loc[data_set.index.repeat(data_set["weights"])]

    # Reset index to avoid duplicate indices
    repeated_data_set.reset_index(drop=True, inplace=True)

    repeated_data_set = repeated_data_set.sample(frac=1, random_state=seed).reset_index(drop=True)

    repeated_data_set.drop(columns="weights", inplace=True)

    return repeated_data_set
