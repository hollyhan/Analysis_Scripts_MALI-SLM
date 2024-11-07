#!/usr/bin/env pytho/
'''
Script to plot delay time in coupled MALI-SLM simulations in 
comparison to standalone MALI simulations from one or more 
landice globalStats files.
Holly Han, 9/27/2024
'''

import os
import sys
import netCDF4
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import time
import numpy as np
from netCDF4 import Dataset
from optparse import OptionParser
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Path (including filename) to region mask file if needed in analyzing regionalStats
fpath_region = "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"

# Density of ice in kg/m^3
rhoi = 910.0  # kg/m^3
rhoo = 1000.0 #1028.0 to make it consistent with the value used in the SLM 
rhow = 1000.0
Aocn_const = 3.62E+14 # m2 calculated from the SLM
AocnBeta_const = 3.62E+14 # m2 area of the ocean not grounded by ice
timeStride = 5 # this should be the ratio of the MALI output interval to the SLM output interval. If analyzing ONLY standalone runs, keep the value at '1'
init_year_index_mali = 15 # time index in the standalone MALI run to represent the initial year of interest (e.g. 2015 has index 15 if run started at 2000.)
init_year_index_slm = 3

analyze_whole_domain = False
analyze_regional_domain = True

pct_mass_loss = 10 # fraction of grounded ice mass that will have melted to calculate dynamic timescale

ic_file = "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/hist_04/output/output_state_2000.nc"
regions_file = "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
region_number = 9 # ISMIP6 basin number to analyze. 9 For the Amundsen Sea sector

# dictionary for each run including run path, legend, etc. 

uncoupled_ctrlAE = {
    "name": "uncoupled ctrlAE",
    "legname": "uncoupled ctrlAE",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/standalone/ctrlAE_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/AIS_4to20km_r01_20220907_obs_TF_1995-2017_8km_x_60m.nc",
    "color": 'k',
    "style": "-",
    "coupled": False
}

coupled_WA_ctrlAE = {
    "name": "coupled-WA ctrlAE",
    "legname": "coupled-WA ctrlAE",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/ctrlAE_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/AIS_4to20km_r01_20220907_obs_TF_1995-2017_8km_x_60m.nc",
    "color": 'k',
    "style": "--",
    "coupled": True
}

coupled_EA_ctrlAE = {
    "name": "coupled-EA ctrlAE",
    "legname": "coupled-EA  ctrlAE",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_EAIS/ctrlAE_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/AIS_4to20km_r01_20220907_obs_TF_1995-2017_8km_x_60m.nc",
    "color": 'k',
    "style": "-.",
    "coupled": True
    
}

uncoupled_expAE01 = {
    "name": "uncoupled expAE01",
    "legname": "uncoupled expAE01",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/standalone/expAE01_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/AIS_4to20km_r01_20220907_TF_NorESM1-M_RCP26-repeat_2300.nc",
    "color": 'b',
    "style": "-",
    "coupled": False
}

coupled_WA_expAE01 = {
    "name": "coupled-WA expAE01",
    "legname": "coupled-WA expAE01",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/expAE01_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/AIS_4to20km_r01_20220907_TF_NorESM1-M_RCP26-repeat_2300.nc",
    "color": 'b',
    "style": "--",
    "coupled": True
}

coupled_EA_expAE01 = {
    "name": "coupled-EA expAE01",
    "legname": "coupled-EA  expAE01",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_EAIS/expAE01_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/AIS_4to20km_r01_20220907_TF_NorESM1-M_RCP26-repeat_2300.nc",
    "color": 'b',
    "style": "-.",
    "coupled": True
    
}

uncoupled_expAE02 = {
    "name": "uncoupled expAE02",
    "legname": "uncoupled expAE02",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/standalone/expAE02_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE02_04/AIS_4to20km_r01_20220907_TF_CCSM4_RCP85_2300.nc",
    "color": '#abd9e9', # sky blue
    "style": "-",
    "coupled": False
    
}

coupled_WA_expAE02 = {
    "name": "coupled-WA expAE02",
    "legname": "coupled-WA expAE02",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/expAE02_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE02_04/AIS_4to20km_r01_20220907_TF_CCSM4_RCP85_2300.nc",
    "color": '#abd9e9', 
    "style": "--",
    "coupled": True
}

coupled_EA_expAE02 = {
    "name": "coupled-EA expAE02",
    "legname": "coupled-EA expAE02",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_EAIS/expAE02_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE02_04/AIS_4to20km_r01_20220907_TF_CCSM4_RCP85_2300.nc",
    "color": '#abd9e9',
    "style": "-.",
    "coupled": True
}

uncoupled_expAE03 = {
    "name": "uncoupled MALI expAE03",
    "legname": "uncoupled expAE03",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/standalone/expAE03_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE03_04/AIS_4to20km_r01_20220907_TF_HadGEM2-ES_RCP85_2300.nc",
    "color": '#e69f00', #gold
    "style": "-",
    "coupled": False
}

coupled_WA_expAE03 = {
    "name": "coupled-WA expAE03",
    "legname": "coupled-WA expAE03",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/expAE03_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE03_04/AIS_4to20km_r01_20220907_TF_HadGEM2-ES_RCP85_2300.nc",
    "color": '#e69f00', 
    "style": "--",
    "coupled": True
}

coupled_EA_expAE03 = {
    "name": "coupled-EA expAE03",
    "legname": "coupled-EA expAE03",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_EAIS/expAE03_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE03_04/AIS_4to20km_r01_20220907_TF_HadGEM2-ES_RCP85_2300.nc",
    "color": '#e69f00',
    "style": "-.",
    "coupled": True
}

uncoupled_expAE05 = {
    "name": "uncoupled MALI expAE05",
    "legname": "uncoupled expAE05",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/standalone/expAE05_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE05_04/AIS_4to20km_r01_20220907_TF_UKESM1-0-LL_SSP585_2300.nc",
    "color": 'm',
    "style": "-",
    "coupled": False
}

coupled_WA_expAE05 = {
    "name": "coupled-WA expAE05",
    "legname": "coupled-WA expAE05",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_WAIS/expAE05_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE05_04/AIS_4to20km_r01_20220907_TF_UKESM1-0-LL_SSP585_2300.nc",
    "color": 'm',
    "style": "--",
    "coupled": True
}

coupled_EA_expAE05 = {
    "name": "coupled-EA expAE05",
    "legname": "coupled-EA expAE05",
    "path": "/pscratch/sd/h/hollyhan/ismip6-2300_run_outputs_20240901/coupled_EAIS/expAE05_04",
    "path_TF": "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE05_04/AIS_4to20km_r01_20220907_TF_UKESM1-0-LL_SSP585_2300.nc",
    "color": 'm',
    "style": "-.",
    "coupled": True
}


# group the runs into coupled and uncoupled runs
#runs_uncoupled =  [uncoupled_ctrlAE, uncoupled_expAE01, uncoupled_expAE02, uncoupled_expAE03, uncoupled_expAE05] #[] [
#runs_coupled_WA = [coupled_WA_ctrlAE, coupled_WA_expAE01, coupled_WA_expAE02, coupled_WA_expAE03, coupled_WA_expAE05] #[]
#runs_coupled_EA = [coupled_EA_ctrlAE, coupled_EA_expAE01, coupled_EA_expAE02, coupled_EA_expAE03, coupled_EA_expAE05]# [ ]
runs_uncoupled =  [uncoupled_ctrlAE, uncoupled_expAE01, uncoupled_expAE02, uncoupled_expAE03, uncoupled_expAE05] #[] [
runs_coupled_WA = [coupled_WA_ctrlAE, coupled_WA_expAE01, coupled_WA_expAE02, coupled_WA_expAE03, coupled_WA_expAE05] #[]
runs_coupled_EA = [coupled_EA_ctrlAE, coupled_EA_expAE01, coupled_EA_expAE02, coupled_EA_expAE03, coupled_EA_expAE05]# [ ]

# experiments = [
#      #("ctrlAE", runs_uncoupled[0], runs_coupled_WA[0], runs_coupled_EA[0]),
#      ("expAE01", runs_uncoupled[0], runs_coupled_WA[0], runs_coupled_EA[0]),
#      ("expAE02", runs_uncoupled[1], runs_coupled_WA[1], runs_coupled_EA[1]),
#      ("expAE03", runs_uncoupled[2], runs_coupled_WA[2], runs_coupled_EA[2]),
#      ("expAE05", runs_uncoupled[3], runs_coupled_WA[3], runs_coupled_EA[3]),
# ]
# List of all experiments (WA and EA coupled runs)
experiments = [
      ("ctrlAE", runs_uncoupled[0], runs_coupled_WA[0], runs_coupled_EA[0]),
      ("expAE01", runs_uncoupled[1], runs_coupled_WA[1], runs_coupled_EA[1]),
      ("expAE02", runs_uncoupled[2], runs_coupled_WA[2], runs_coupled_EA[2]),
      ("expAE03", runs_uncoupled[3], runs_coupled_WA[3], runs_coupled_EA[3]),
      ("expAE05", runs_uncoupled[4], runs_coupled_WA[4], runs_coupled_EA[4]),
  ]

# Separate list for 'path_TF' values ("experiment_name", "path to file")
# Make sure the experiment names are the same as those defined in the list "experiments" above
path_to_thermal_forcing_files = [
    #("ctrlAE", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/AIS_4to20km_r01_20220907_obs_TF_1995-2017_8km_x_60m.nc"),
    ("expAE01", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/AIS_4to20km_r01_20220907_TF_NorESM1-M_RCP26-repeat_2300.nc"),
    ("expAE02", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE02_04/AIS_4to20km_r01_20220907_TF_CCSM4_RCP85_2300.nc"),
    ("expAE03", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE03_04/AIS_4to20km_r01_20220907_TF_HadGEM2-ES_RCP85_2300.nc"),
    ("expAE05",  "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE05_04/AIS_4to20km_r01_20220907_TF_UKESM1-0-LL_SSP585_2300.nc"),
]

path_to_smb_forcing_files = [
    #("ctrlAE", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/AIS_4to20km_r01_20220907_RACMO2.3p2_ANT27_smb_climatology_1995-2017_minus1_bare_land.nc"),
    ("expAE01", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/AIS_4to20km_r01_20220907_smb_NorESM1-M_RCP26-repeat_2300_minus1_bare_land.nc"),
    ("expAE02", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE02_04/AIS_4to20km_r01_20220907_smb_CCSM4_RCP85_2300_minus1_bare_land.nc"),
    ("expAE03", "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE03_04/AIS_4to20km_r01_20220907_smb_HadGEM2-ES_RCP85_2300_minus1_bare_land.nc"),
    ("expAE05",  "/pscratch/sd/h/hollyhan/production_runs_ismip62300_AIS_4to20km_20240415/landice/ismip6_run/ismip6_ais_proj2300/expAE05_04/AIS_4to20km_r01_20220907_smb_UKESM1-0-LL_SSP585_2300_minus1_bare_land.nc"),
]
# Set global NumPy print options for precision and suppress scientific notation
#np.set_printoptions(precision=6, suppress=True)

# Extract data for each run
def extract_data_for_run(runs):
    for run in runs:
        print("Processing run: " + run['path'])
        run['data'] = outputData(run)
        run['data_regional'] = regionalStats(run)
        run['data_gs'] = globalStats(run)


class slm_outputs:
    def __init__(self, run, varname):
        # --------
        # Analysis from the sea-level model output file
        # --------
        self.run = run
        print('reading in the file:')
        print(os.path.join(self.run['path'], varname))
        f = open(os.path.join(self.run['path'], varname), 'r')
        data = f.readlines()
        self.data = np.loadtxt(os.path.join(self.run['path'], varname))
        self.yrs = np.linspace(2015,2300,58) #needs to be manually changed
        # self.yrs = np.linspace(2015,2070,11) #needs to be manually changed
        self.label = self.run['legname']+f' ({varname})'
        self.changeDt = np.zeros(len(data),)
        self.changeTotal = np.zeros(len(data),)
        self.changeDt[0] = 0
        self.changeTotal[0] = 0
        for i in range(len(self.changeDt)-1):
            self.changeDt[i+1] = (float(data[i+1]) - float(data[i]))
            self.changeTotal[i+1] = (float(data[i+1]) - float(data[0]))
        f.close()
        
def grounded(cellMask):
    # return ((cellMask&32)//32) & ~((cellMask&4)//4)
    return ((cellMask & 32) // 32) * np.logical_not((cellMask & 4) // 4)


def xtime2numtimeMy(xtime):
    """Define a function to convert xtime character array to numeric time values using local arithmetic"""
    # First parse the xtime character array into a string
    xtimestr = netCDF4.chartostring(
        xtime)  # convert from the character array to an array of strings using the netCDF4 module's function

    numtime = np.zeros((len(xtimestr),))
    ii = 0
    dayOfMonthStart = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    for stritem in xtimestr:
        itemarray = stritem.strip().replace('_', '-').replace(':', '-').split(
            '-')  # Get an array of strings that are Y,M,D,h,m,s
        results = [int(i) for i in itemarray]
        numtime[ii] = results[0] + (dayOfMonthStart[results[1] - 1] - 1 + results[2]) / 365.0  # decimal year
        ii += 1
    return numtime


class regionalStats:
    def __init__(self, run):
        # --------
        # Analysis from regional stats file
        # Contents heavily copied over from 'plot_regionalStats.py' in MPAS-Tools
        # https://github.com/MPAS-Dev/MPAS-Tools/tree/master/landice/output_processing_li
        # --------   
        
        # NOTE! the regionalStats files have initial timestamp of 2015, so values at 2015 are at zero-th time index.
        f = netCDF4.Dataset(run['path'] + '/' + 'regionalStats.nc', 'r')
        self.nRegions = len(f.dimensions['nRegions'])
        massUnit = 'Gt'
        self.dt = f.variables['deltat'][:]/(3600.0*24.0*(365.0)) # in yr
        self.dtnR = np.tile(self.dt.reshape(len(self.dt),1), (1,self.nRegions))  # repeated per region with dim of nt,nRegions
        self.xtimes = f.variables['xtime'][:]
        self.yr = xtime2numtimeMy(self.xtimes)
        self.vol = f.variables['regionalIceVolume'][:] * rhoi / 1.0e12 # in Gt
        self.VAF = f.variables['regionalVolumeAboveFloatation'][:] * rhoi / 1.0e12 # in Gt
        self.VAFchange = self.VAF[:,:] - self.VAF[0,:]
        self.grdVol =  f.variables['regionalGroundedIceVolume'][:] * rhoi / 1.0e12 # in Gt
        self.grdVolchange = self.grdVol[:,:] - self.grdVol[0,:]
        self.floatVol =  f.variables['regionalFloatingIceVolume'][:] * rhoi / 1.0e12 # in Gt
        self.floatVolchange = self.floatVol[:,:] - self.floatVol[0,:]
        self.grdSMB = f.variables['regionalSumGroundedSfcMassBal'][:] / 1.0e12 # in Gt
        self.cumGrdSMB = np.cumsum(self.grdSMB*self.dtnR, axis=0)
        self.GLflux = f.variables['regionalSumGroundingLineFlux'][:] / 1.0e12 # in Gt
        self.GLMigflux = f.variables['regionalSumGroundingLineMigrationFlux'][:] / 1.0e12 # in Gt
        self.cumGLMigflux = np.cumsum(self.GLMigflux*self.dtnR, axis=0)
        self.areaTot = f.variables['regionalIceArea'][:]/1000.0**2
        self.areaGrd = f.variables['regionalGroundedIceArea'][:,:]/1000.0**2
        self.areaGrdChange = self.areaGrd[:,:] - self.areaGrd[0,:]
        self.areaFlt = f.variables['regionalFloatingIceArea'][:]/1000.0**2
        # sum of Grd ice components
        self.grdSum = self.grdSMB - self.GLflux - self.GLMigflux # note negative sign on two GL terms - they are both positive grounded to floating
        self.cumGrdSum = np.cumsum(self.grdSum*self.dtnR, axis=0)

        # find dynamical time for grounded ice mass to have lost X-percentage loss of intial grounded ice mass
        target_mass = self.grdVol[0,region_number] - (self.grdVol[0,region_number] * pct_mass_loss * 0.01)
        index = (np.argmin(np.abs(self.grdVol[:,region_number] - target_mass)))
        self.dynamic_time_region = self.yr[index] - self.yr[0]
        self.areaGrdChange_region = self.areaGrdChange[:,region_number]
        print(f'initial and final grounded ice mass in the Amundsen Sea Sector are {self.grdVol[0,region_number]} Gt and {self.grdVol[-1,region_number]} Gt')
        print(f'{abs(self.grdVolchange[-1,region_number]/self.grdVol[0,region_number])*100} percent total mass is lost between 2015-2300')
        print(f'{pct_mass_loss} percent mass loss set as the thresold, yielding target mass: {target_mass} Gt')
        print(f'the grounded ice mass loss after dynamic time is {self.grdVol[index,region_number]}Gt after dynamic  year {self.dynamic_time_region}')
              
        print('=====FINISHED CALCULATING DYNAMICAL TIME FOR THE SIMULATION====')
        
        if 'regionaNames' in f.variables:
            rNamesIn = f.variables['regionNames'][:]
        else:
            fn = Dataset(fpath_region, 'r')
            rNamesIn = fn.variables['regionNames'][:]

        # Process region names
        rNamesOrig = list()
        for r in range(self.nRegions):
            thisString = rNamesIn[r, :].tobytes().decode('utf-8').strip()  # convert from char array to string
            rNamesOrig.append(''.join(filter(str.isalnum, thisString)))  # this bit removes non-alphanumeric chars

        # Antarctic data from:
        # Rignot, E., Bamber, J., van den Broeke, M. et al. Recent Antarctic ice mass loss from radar interferometry
        # and regional climate modelling. Nature Geosci 1, 106-110 (2008). https://doi.org/10.1038/ngeo102
        # Table 1: Mass balance of Antarctica in gigatonnes (10^12 kg) per year by sector for the year 2000
        # https://www.nature.com/articles/ngeo102/tables/1
        # Note: May want to switch to input+, net+
        # Note: Some ISMIP6 basins combine multiple Rignot basins.  May want to separate if we update our regions.
        ISMIP6basinInfo = {
                'ISMIP6BasinAAp': {'name': 'Dronning Maud Land', 'input': [60,9], 'outflow': [60,7], 'net': [0, 11], 'shelfMelt': [57.5]},
                'ISMIP6BasinApB': {'name': 'Enderby Land', 'input': [39,5], 'outflow': [40,2], 'net': [-1,5], 'shelfMelt': [24.6]},
                'ISMIP6BasinBC': {'name': 'Amery-Lambert', 'input': [73, 10], 'outflow': [77,4], 'net': [-4, 11], 'shelfMelt': [35.5]},
                'ISMIP6BasinCCp': {'name': 'Phillipi, Denman', 'input': [81, 13], 'outflow': [87,7], 'net':[-7,15], 'shelfMelt': [107.9]},
                'ISMIP6BasinCpD': {'name': 'Totten', 'input': [198,37], 'outflow': [207,13], 'net': [-8,39], 'shelfMelt': [102.3]},
                'ISMIP6BasinDDp': {'name': 'Mertz', 'input': [93,14], 'outflow': [94,6], 'net': [-2,16], 'shelfMelt': [22.8]},
                'ISMIP6BasinDpE': {'name': 'Victoria Land', 'input': [20,1], 'outflow': [22,3], 'net': [-2,4], 'shelfMelt': [22.9]},
                'ISMIP6BasinEF': {'name': 'Ross', 'input': [61+110,(10**2+7**2)**0.5], 'outflow': [49+80,(4**2+2^2)**0.5], 'net': [11+31,(11*2+7**2)**0.5], 'shelfMelt': [70.3]},
                'ISMIP6BasinFG': {'name': 'Getz', 'input': [108,28], 'outflow': [128,18], 'net': [-19,33], 'shelfMelt': [152.9]},
                'ISMIP6BasinGH': {'name': 'Amundsen Sea Embayment', 'input': [177,25], 'outflow': [237,4], 'net': [-61,26], 'shelfMelt': [290.9]}, 
                'ISMIP6BasinHHp': {'name': 'Bellingshausen', 'input': [51,16], 'outflow': [86,10], 'net': [-35,19], 'shelfMelt': [76.3]},
                'ISMIP6BasinHpI': {'name': 'George VI', 'input': [71,21], 'outflow': [78,7], 'net': [-7,23], 'shelfMelt': [152.3]},
                'ISMIP6BasinIIpp': {'name': 'Larsen A-C', 'input': [15,5], 'outflow': [20,3], 'net': [-5,6], 'shelfMelt': [32.9]},
                'ISMIP6BasinIppJ': {'name': 'Larsen E', 'input': [8,4], 'outflow': [9,2], 'net': [-1,4], 'shelfMelt': [4.3]},
                'ISMIP6BasinJK': {'name': 'FRIS', 'input': [93+142, (8**2+11**2)**0.5], 'outflow': [75+145,(4**2+7**2)**0.5], 'net': [18-4,(9**2+13**2)**0.5], 'shelfMelt': [155.4]},
                'ISMIP6BasinKA': {'name': 'Brunt-Stancomb', 'input': [42+26,(8**2+7**2)**0.5], 'outflow': [45+28,(4**2+2**2)**0.5], 'net':[-3-1,(9**2+8**2)**0.5], 'shelfMelt': [10.4]}
                }

        # Parse region names to more usable names, if available
        self.rNames = [None]*self.nRegions
        for r in range(self.nRegions):
            if rNamesOrig[r] in ISMIP6basinInfo:
                self.rNames[r] = ISMIP6basinInfo[rNamesOrig[r]]['name']
            else:
                self.rNames[r] = rNamesOrig[r]
                
        
class globalStats:
    def __init__(self, run):
        # --------
        # Analysis from global stats file
        # --------                                                                                                                                             
        f = netCDF4.Dataset(run['path'] + '/' + 'globalStats.nc', 'r')
        self.nt = len(f.dimensions['Time'])
        self.yrs = np.zeros((self.nt,))
        # yrs = f.variables['daysSinceStart'][:] / 365.0
        self.xtimes = f.variables['xtime'][:]

        self.yrs = xtime2numtimeMy(self.xtimes)
        # self.dyrs = self.yrs[1:] - self.yrs[0:-1]
        self.VAF = f.variables['volumeAboveFloatation'][:] / 1.0e12 * rhoi  # Gt
        self.melt = f.variables['totalFloatingBasalMassBal'][:] / -1.0e12  # Gt
        # Clean a few melt blips
        self.melt[self.melt > 3000.0] = np.NAN
        self.meltrate = f.variables['avgSubshelfMelt'][:]
        self.meltrate[self.meltrate > 500.0] = np.NAN
        self.GA = f.variables['groundedIceArea'][:] / 1000.0 ** 2  # km^2
        self.GLflux = f.variables['groundingLineFlux'][:] / 1.0e12  # Gt/y
        self.GLflux[0] = np.NAN  # remove the zero in the first time level
        self.floatArea = f.variables['floatingIceArea'][:] / (1000.0 ** 2)  # km2
        self.floatVol = f.variables['floatingIceVolume'][:] / (1000.0 ** 3)  # km3
        self.floatThk = self.floatVol / self.floatArea * 1000.0  # m
        self.grdArea = f.variables['groundedIceArea'][:]  # m2
        self.grdVol = f.variables['groundedIceVolume'][:] / (1000.0 ** 3)  # km3
        self.grdThk = self.grdVol / self.grdArea * 1000.0  # m
        self.totalVol = f.variables['totalIceVolume'][:] / (1000.0 **3) #km3
        self.VAFrate = np.zeros(self.VAF.shape)
        self.VAFrate[1:] = (self.VAF[1:] - self.VAF[:-1]) / (self.yrs[1:] - self.yrs[:-1])
        self.GAloss = self.GA[0] - self.GA[:]
        
        
def plot_regionalStats():
    fig_reg, ax_reg = plt.subplots(4, 4, figsize=(14, 12), num=1)
    fig_reg.suptitle('Grounded Ice Mass Change Since 2015 in Regional Basins', fontsize=16)
    # Loop through each experiment (e.g., expAE02, expAE05)
    for experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA in experiments:
      
        nRegions = uncoupled_run['data_regional'].nRegions
        rNames = uncoupled_run['data_regional'].rNames

        grdVolchange_uncoupled = uncoupled_run['data_regional'].grdVolchange
        grdVolchange_coupled_WA = coupled_run_WA['data_regional'].grdVolchange
        grdVolchange_coupled_EA = coupled_run_EA['data_regional'].grdVolchange

        # Plot the grounded ice mass for uncoupled, WA, and EA
        for reg in range(nRegions):
            if reg == 0:
                axX = ax_reg.flatten()[reg]
            else:
                ax_reg.flatten()[reg].sharex(axX)
               
            ax_reg.flatten()[reg].set_title(f"{reg+1}. {rNames[reg]}")
            ax_reg.flatten()[reg].plot(uncoupled_run['data_regional'].yr, grdVolchange_uncoupled[:, reg], color=uncoupled_run['color'], linestyle=uncoupled_run['style'], label=uncoupled_run['legname'], linewidth=1)
            ax_reg.flatten()[reg].plot(coupled_run_WA['data_regional'].yr, grdVolchange_coupled_WA[:, reg], color=coupled_run_WA['color'], linestyle=coupled_run_WA['style'], label=coupled_run_WA['legname'], linewidth=1)
            ax_reg.flatten()[reg].plot(coupled_run_EA['data_regional'].yr, grdVolchange_coupled_EA[:, reg], color=coupled_run_EA['color'], linestyle=coupled_run_EA['style'], label=coupled_run_EA['legname'], linewidth=1)
            
            #ax_reg.flatten()[reg].set_xlabel('Year')
            #ax_reg.flatten()[reg].set_ylabel('Mass change (Gt)')
            ax_reg.flatten()[reg].grid(True)
    
    #plt.subplots_adjust(hspace=0.8, wspace=0.3)  # Increase hspace to add more vertical space
    fig_reg.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    fig_reg.text(0.54, 0.04, 'Year', ha='center', fontsize=14)  # X-label
    fig_reg.text(0.01, 0.5, 'Mass change (Gt)', va='center', rotation='vertical', fontsize=14)  # Y-label
    #plt.legend(loc='best', fontsize='small')

    #fig_reg.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_reg.savefig('regional_grnd_mass_change.png', dpi=300)
    plt.show()

class outputData:
    def __init__(self, run):
        # --------
        # Analysis from output file
        # --------
        
        fpath = run['path'] + '/' + 'output_state_2015.nc'        
        if os.path.exists(fpath):
            print('reading in output_state_2*.nc files') 
            DS = xr.open_mfdataset(run['path'] + '/' + 'output_state_2*.nc', combine='nested', concat_dim='Time',
                decode_timedelta=False)
        else:
            print('reading in output_state_all.nc file')
            DS = xr.open_mfdataset(run['path'] + '/' + 'output_state_all.nc', combine='nested', concat_dim='Time',
                decode_timedelta=False)   
            
        self.yrs_mali = DS['daysSinceStart'].load() / 365.0 + 2000.0  # '2000' is the initial model year registered in the output_state file.
        if (timeStride != 1):
            inds = np.arange(0, DS.dims['Time'], timeStride, dtype='i')
            self.yrs = self.yrs_mali[inds]
        else:
            self.yrs = self.yrs_mali
        
        nt = len(self.yrs)
        nCells = DS.dims['nCells']
        cellMask = DS['cellMask']
        bed = DS['bedTopography']
        bed0 = bed.isel(Time=0).load()
        thickness = DS['thickness']
        thickness0 = thickness.isel(Time=0).load()
        areaCell = DS['areaCell'][0, :].load()
        latCell = DS['latCell'][0,:].load()
        lonCell = DS['lonCell'][0,:].load() - np.pi # shift the range to [-pi pi]
        DS.close()
        
        # save thickness and bedtopo attributes for plotting TF
        self.thk0 = thickness0
        self.bed0= bed0
        
        # calculate the area distortion (scale) factor for the polar stereographic projection
        self.k = np.zeros(len(latCell),)
        k0 = (1 - np.sin(-71 * np.pi / 180)) / 2 # setting 71 degree S as the standard parallel
        # print("===k0 value ===", k0) 
        lat0 = -90 * np.pi / 180 #center of the latitude  #standard parallel in radians where distortion should be zero (i.e., k=1)
        lon0 = 0 # center of the longitude
        # expression for 'k' from p. 157 eqn. 21-4 in https://pubs.usgs.gov/pp/1395/report.pdf. 
        #p. 142 provides a table showing 'k' for the stereographic projection
        self.k = 2 * k0 / (1 + np.sin(lat0) * np.sin(latCell) + np.cos(lat0) * np.cos(latCell) * np.cos(lonCell - lon0))

        # calculate grounded ice volume change through MALI output interval
        nt_mali = len(self.yrs_mali)
        self.grdVol = np.zeros(nt_mali)
        for t in range(nt_mali):
            bedt = bed[t, :].load()
            cellMaskt = cellMask[t, :].load()
            thicknesst = thickness[t, :].load()
            areaCellt = areaCell.sum()            
            self.grdVol[t] = (areaCell * grounded(cellMaskt) * thicknesst / (self.k**2)).sum()
        self.grdVolchange = self.grdVol - self.grdVol[init_year_index_mali] # index '0' represents 2000. We want initial year to be 2015
        
        # calculate sea-level change correcting for bedrock deformation following Goelzer et al. 2020
        # this process is necessary for 
        # initialize arrays for the number of SLM timesteps
        self.den = np.zeros((nt,))
        self.vaf = np.zeros((nt,))
        self.pov = np.zeros((nt,))
        self.vaf_z0 = np.zeros((nt,))
        self.pov_z0 = np.zeros((nt,))
        
        i = 0
        # Initialize flags to track whether a message has been printed
        gmsle_missing_warned = False
        ocean_area_missing_warned = False
        ocean_beta_missing_warned = False
        for t in inds:
            sl_indx = int(t/timeStride)
            # get path to SLM output files
            fname_gmslc = os.path.join(run['path']+'OUTPUT_SLM', 'gmsle_deltaSL_Ocean_fixed')
            fname_Aocn = os.path.join(run['path']+'OUTPUT_SLM', 'ocean_area')
            fname_AocnBeta = os.path.join(run['path'], 'oceanBeta_area')
            
            if os.path.exists(fname_gmslc):
                gmsle_change = slm_outputs(run, 'gmsle_deltaSL_Ocean_fixed').changeTotal  # to correct for GMSLE
                z0 = gmsle_change[sl_indx] 
            else: 
                if run['coupled'] and not gmsle_missing_warned:
                    print("'gmsle_change' file doesn't exist even though it should. Setting z0 to zeros")
                    gmsle_missing_warned = True  # Set the flag to True to prevent further prints
                z0 = 0.0
                
            if os.path.exists(fname_Aocn):
                Aocn_slm = slm_outputs(run, 'ocean_area').data  # glboal ocean area calculted in the SLM
                self.Aocn = Aocn_slm[sl_indx]
            else: 
                if run['coupled'] and not gmsle_missing_warned:
                    print(f"'ocean_area' file doesn't exist even though it should. using the pre-defined constant ocean area value {Aocn_const}")
                    ocean_area_missing_warned = True
                self.Aocn = Aocn_const
                
            if os.path.exists(fname_AocnBeta):
                AocnBeta_slm = slm_outputs(run, 'oceanBeta_area').data  # global ocean area calculated in the SLM
                self.AocnBeta = AocnBeta_slm[sl_indx]
            else: 
                if run['coupled'] and not gmsle_missing_warned:
                    print(f"'oceanBeta_area' file doesn't exist even though it should. using the pre-defined constant oceanbeta value {AocnBeta_const}")
                    ocean_beta_missing_warned = True
                self.AocnBeta = AocnBeta_const
            
            #print(f'======At year {self.yrs[i]}, Ocean area is {self.Aocn/1e6}km2, OceanBeta area is {self.AocnBeta/1e6}m2=======')
            bedt = bed[t, :].load()
            cellMaskt = cellMask[t, :].load()
            thicknesst = thickness[t, :].load()
            areaCellt = areaCell.sum()
            
            self.vaf[i] = (areaCell * grounded(cellMaskt) / (self.k**2) *
                           np.maximum(thicknesst + (rhoo / rhoi) * np.minimum(np.zeros((nCells,)), bedt), 0.0)).sum() # eqn. (1) Goelzer et al., 2020
            self.pov[i] = (areaCell / (self.k**2) * np.maximum(0.0 * bedt, -1.0 * bedt)).sum() # eqn. (8)
            
            self.den[i] = (areaCell / (self.k**2) * grounded(cellMaskt) *
              (rhoi / rhow - rhoi / rhoo)).sum() # eqn. (10) Goelzer et al., 2021
            self.vaf_z0[i] = (areaCell * grounded(cellMaskt) / (self.k**2) *
                           np.maximum(thicknesst + (rhoo / rhoi) * np.minimum(np.zeros((nCells,)), bedt+z0), 0.0)).sum() # eqn. (13) Goelzer et al., 2020,
            self.pov_z0[i] = (areaCell / (self.k**2) * np.maximum(0.0 * bedt, -1.0 * bedt+z0)).sum() # eqn. (14) 
        
            i += 1
            
        # density correction
        self.SLCden = -1.0 * (self.den / self.Aocn - self.den[0] / self.Aocn) # in m, eqn. 11

        # calculate change in sea level from volume above floatation
        self.SLC_VAF = (self.vaf - self.vaf[init_year_index_slm]) * (rhoi / rhoo) / self.Aocn * -1.0 # make sure Aocn is taken from the coupled result if exists.
        # corrected GMSLE values
        # correct for GMSLE using the ocean area, adding the z0 factor 
        self.SLEaf_z0 = (self.vaf_z0 / self.Aocn) * (rhoi / rhoo) # eqn. 2
        self.SLCaf_z0 = -1.0 * (self.SLEaf_z0 - self.SLEaf_z0[init_year_index_slm]) # in m, eqn. 3
        self.SLCpov_z0 = -1.0 * (self.pov_z0 / self.Aocn - self.pov_z0[init_year_index_slm] / self.Aocn) # in m, eqn. 9
        self.SLCcorr_z0 = self.SLCaf_z0 + self.SLCpov_z0 # + self.SLCden. For now, density correction is ommitted because we are using fresh water density only. 

        # correct for GMSLE using the oceanBeta area, adding the z0 factor
        self.SLEaf_z0_AocnBeta = (self.vaf_z0 / self.AocnBeta) * (rhoi / rhoo) # eqn. 2
        self.SLCaf_z0_AocnBeta = -1.0 * (self.SLEaf_z0_AocnBeta - self.SLEaf_z0_AocnBeta[init_year_index_slm]) # in m, eqn. 3
        self.SLCpov_z0_AocnBeta = -1.0 * (self.pov_z0 / self.AocnBeta - self.pov_z0[init_year_index_slm] / self.AocnBeta) # in m, eqn. 9
        self.SLCcorr_z0_AocnBeta = self.SLCaf_z0_AocnBeta + self.SLCpov_z0_AocnBeta # + self.SLCden. For now, density correction is ommitted because we are using fresh water density only. 

        # correct for GMSLE using the ocean area, no z0 factor
        self.SLEaf = (self.vaf / self.Aocn) * (rhoi / rhoo) # eqn. 2
        self.SLCaf = -1.0 * (self.SLEaf - self.SLEaf[init_year_index_slm]) # in m, eqn. 3
        self.SLCpov = -1.0 * (self.pov / self.Aocn - self.pov[init_year_index_slm] / self.Aocn) # in m, eqn. 9
        self.SLCcorr = self.SLCaf + self.SLCpov # + self.SLCden. For now, density correction is ommitted because we are using fresh water density only. 


class calculate_delay_grnd_mass:
    # calculate difference in timing at which the same amount of ice volume has
    # retreated in given uncoupled and coupled simulations
    def __init__(self, uncoupled_run, coupled_run, years_array, whole_domain=True, regions=True):
        # uncoupled_run: run data extracted from uncoupled simulation
        # coupled_run: run data extracted from coupled simulation
        # years_array: array of years at which delay effects are assessed
        # whole_domain: whether to calculate delay in ice retreat across the whole Antarctic domain
        # regions: whether to calculate delay in ice retreat in a specific region (e.g. Amundsen Sea sector)
        
        self.yrs = years_array     
        if whole_domain:
            print("Calculating delay time in years across the whole Antarctica")
            # Initialize an array to store calculated delay effects
            self.delay_time = np.zeros(len(years_array))
            nearest_matches = []

            # get years from each run
            t_coupled_raw = coupled_run['data'].yrs_mali
            # find indicies for the years nearest to years of interest
            ind_t_coupled = find_closest_indices_time(t_coupled_raw, years_array)
            # refine time array
            t_coupled = t_coupled_raw[ind_t_coupled]
            print(f"======years to analyze are {t_coupled} from the coupled run=======")

            # do the same thing for the uncoupled run
            t_uncoupled_raw = uncoupled_run['data'].yrs_mali
            ind_t_uncoupled = find_closest_indices_time(t_uncoupled_raw, years_array)
            t_uncoupled = t_uncoupled_raw[ind_t_uncoupled]
            print(f"======years to analyze are {t_uncoupled} from the uncoupled run=======")

            # grounded ice volume change in the coupled simulation at target years
            self.dgrdVol_coup = coupled_run['data'].grdVolchange[ind_t_coupled]
            
            # Find the index in the uncoupled run where the grounded ice mass change is closest to the coupled mass change
            nearest_indices = []
            for coup_vol in self.dgrdVol_coup:
                nearest_index = np.argmin(np.abs(uncoupled_run['data'].grdVolchange - coup_vol))
                nearest_indices.append(nearest_index)
            
            mass_nearest_uncoup = uncoupled_run['data'].grdVolchange[nearest_indices]
            print(self.dgrdVol_coup)
            print(mass_nearest_uncoup)
            
            # Find the year and the corresponding grounded ice mass change in the uncoupled run
            year_nearest_uncoup = t_uncoupled_raw[nearest_indices]
           
            # calculate difference in timing in melting the same amount of grounded ice volume
            self.delay_time = t_coupled - year_nearest_uncoup # uncoupled_year

            # get grounded ice change in the uncoupled simulation at the target years
            self.dgrdVol_uncoup = uncoupled_run['data'].grdVolchange[ind_t_uncoupled]
            
            # difference in grounded ice volume/mass change in uncoupled vs. coupled simulations at the target years
            self.diff_in_dgrdVol = self.dgrdVol_uncoup - self.dgrdVol_coup
            self.diff_in_dgrdVol_gt = self.diff_in_dgrdVol * rhoi * 1e-12 # in Gt
            
            # make sure the arrays are numpy arrays
            self.delay_time = np.array(self.delay_time)
            
            print('==================================')
            print(f"At years: {self.yrs}")
            print(f'coupled_grdVolChange value at years: {self.dgrdVol_coup * rhoi * 1e-12} Gt')
            print(f'diff in grounded mass change {self.diff_in_dgrdVol_gt} Gt')
            print(f'diff in grounded mass change in percentage between the nearest values: {abs(self.diff_in_dgrdVol / self.dgrdVol_coup)*100.0} percent')
            print(f"Delay in years = {self.delay_time} years (Coupled: {t_coupled}, Uncoupled: {year_nearest_uncoup})")
            print('==================================')
            
            # Plot for debug purpose
#             plt.subplots(figsize=(10, 6))
#             plt.plot(self.yrs, self.dgrdVol_uncoup * rhoi * 1e-12,label=f"{uncoupled_run['name']}", color=uncoupled_run['color'], linestyle='-')
#             plt.plot(self.yrs, self.dgrdVol_coup * rhoi * 1e-12,label=f"{coupled_run['name']}", color=coupled_run['color'], linestyle='--')

#             plt.xlabel('Year')
#             plt.ylabel('ice mass (Gt)')
#             plt.title('Grounded ice Mass change')
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

        if regions:
            print("Calculating delay time in years in a region")

            # get raw time array
            t_uncoupled_raw_reg = uncoupled_run['data_regional'].yr
            t_coupled_raw_reg = coupled_run['data_regional'].yr

            ind_t_uncoupled_reg = find_closest_indices_time(t_uncoupled_raw_reg, years_array)
            ind_t_coupled_reg = find_closest_indices_time(t_coupled_raw_reg, years_array)

            # refined time array
            t_uncoupled_reg = t_uncoupled_raw_reg[ind_t_uncoupled_reg]
            t_coupled_reg = t_coupled_raw_reg[ind_t_coupled_reg]

            # Initialize an array to store calculated delay effects
            self.delay_time_region = np.zeros(len(ind_t_coupled_reg))

            # get raw grounded ice volume change
            dgrdVol_uncoupled_raw_reg = uncoupled_run['data_regional'].grdVolchange[:,region_number] # already in Gt
            dgrdVol_coupled_raw_reg = coupled_run['data_regional'].grdVolchange[:,region_number] # already in Gt

            # get grounded ice change in the uncoupled simulation at the target years
            self.dgrdVol_uncoup_region = dgrdVol_uncoupled_raw_reg[ind_t_uncoupled_reg]
 
            # find dgrdVol at the years and region of interest 
            self.dgrdVol_coup_region = dgrdVol_coupled_raw_reg[ind_t_coupled_reg]
         
            # Find the index in the uncoupled run where the grounded ice mass change is closest to the coupled mass change
            nearest_indices_reg = []
            for coup_vol in self.dgrdVol_coup_region:
                nearest_index = np.argmin(np.abs(dgrdVol_uncoupled_raw_reg - coup_vol))
                nearest_indices_reg.append(nearest_index)
            
            mass_nearest_uncoup_reg = dgrdVol_uncoupled_raw_reg[nearest_indices_reg]                  
                
            # Find the year and the corresponding grounded ice mass change in the uncoupled run
            year_nearest_uncoup_reg = t_uncoupled_raw_reg[nearest_indices_reg]
            
            # calculate difference in timing in melting the same amount of grounded ice volume
            self.delay_time_region = t_coupled_reg - year_nearest_uncoup_reg

            # difference in grounded ice volume/mass change in uncoupled vs. coupled simulations at the target years
            self.diff_in_dgrdVol_region = self.dgrdVol_uncoup_region - self.dgrdVol_coup_region
            
            # Convert masked arrays to numpy array
            self.diff_in_dgrdVol_region = self.diff_in_dgrdVol_region.filled(np.nan)
            self.dgrdVol_uncoup_region = self.dgrdVol_uncoup_region.filled(np.nan)
            self.dgrdVol_coup_region = self.dgrdVol_coup_region.filled(np.nan)

            print('==================================')
            print(f"At years: {self.yrs}")   
            print(f'coupled_grdVolChange value at years: {self.dgrdVol_coup_region} Gt')
            print(f'diff in grounded mass change {self.diff_in_dgrdVol_region} Gt')
            print(f'diff in grounded mass change in percentage between the nearest values: {abs(self.diff_in_dgrdVol_region / self.dgrdVol_coup_region)*100.0} percent')
            print(f"Delay in years = {self.delay_time_region} years (Coupled: {t_coupled_reg}, Uncoupled: {year_nearest_uncoup_reg})")
            print('==================================')
            
            # Plot for debug purpose
#             plt.subplots(figsize=(10, 6))
#             plt.plot(self.yrs, self.dgrdVol_uncoup_region,label=f"{uncoupled_run['name']}", color=uncoupled_run['color'], linestyle='-')
#             plt.plot(self.yrs, self.dgrdVol_coup_region,label=f"{coupled_run['name']}", color=coupled_run['color'], linestyle='--')

#             plt.xlabel('Year')
#             plt.ylabel('ice mass (Gt)')
#             plt.title('Regional Grounded ice Mass change')
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()
    


# Function to plot unnormalized and normalized delay time for grounded ice mass changes
def plot_delay_grnd_mass(save_fig=True):
 
    years_to_analyze = np.array(range(2100,2301,50))
    x1 = np.arange(len(years_to_analyze))  # X positions for the groups (years)
    width = 0.08  # Bar width
    num_experiments = len(experiments)
    
    if analyze_whole_domain:
        fig1, ax1 = plt.subplots(figsize=(11, 6))
    if analyze_regional_domain:
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        fig3, ax3 = plt.subplots(figsize=(11, 6))
            
    # Loop over each experiment
    for i, (experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA) in enumerate(experiments):
        print(f'=== For experiment {experiment_name}:')
        rNames = uncoupled_run['data_regional'].rNames
        
        # Offset positions for each experiment
        pos1 = x1 - (num_experiments / 2) * width + i * width  # Shift bars horizontally for each experiment

        # extract delay time
        delay_time_extractor_WA = calculate_delay_grnd_mass(uncoupled_run, coupled_run_WA, years_to_analyze, whole_domain=analyze_whole_domain, regions=analyze_regional_domain)
        delay_time_extractor_EA = calculate_delay_grnd_mass(uncoupled_run, coupled_run_EA, years_to_analyze, whole_domain=analyze_whole_domain, regions=analyze_regional_domain)
        
        if analyze_whole_domain:
            delay_time_WA = delay_time_extractor_WA.delay_time
            delay_time_EA = delay_time_extractor_EA.delay_time
            ax1.bar(pos1, delay_time_WA, width, label=f'{experiment_name} coupled-WA', color=coupled_run_WA['color'])
            ax1.bar(pos1, delay_time_EA, width, label=f'{experiment_name} coupled-EA', color=coupled_run_EA['color'], hatch='//', edgecolor='black')

        if analyze_regional_domain:
            dynamic_time_region = uncoupled_run['data_regional'].dynamic_time_region
            delay_time_region_WA = delay_time_extractor_WA.delay_time_region
            delay_time_region_EA = delay_time_extractor_EA.delay_time_region
            delay_time_region_WA_norm = delay_time_region_WA / dynamic_time_region
            delay_time_region_EA_norm = delay_time_region_EA / dynamic_time_region
            ax2.bar(pos1, delay_time_region_WA, width, label=f'{experiment_name} coupled-WA', color=coupled_run_WA['color'])
            #ax2.bar(pos1, delay_time_region_EA, width, label=f'{experiment_name} coupled-EA', color=coupled_run_WA['color'], hatch='//', edgecolor='black')

            ax3.bar(pos1, delay_time_region_WA_norm, width, label=f'{experiment_name} coupled-WA', color=coupled_run_WA['color'])
            
    if analyze_whole_domain:
        # Customize unnormalized delay time plot
        ax1.set_xticks(x1 + (num_experiments * width) / 2 - width)  # Center the tick marks between the two bars for each year
        ax1.set_xticklabels([f'{year}' for year in years_to_analyze])  # Label x-axis with actual years (e.g., 2100, 2200, etc.)
        ax1.set_xlabel('Year',fontsize=13)
        ax1.set_ylabel('Delay time (yr)',fontsize=13)
        ax1.set_title('Delay time in Antarctica ',fontsize=13)
        ax1.legend(loc='upper left')
        #ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=12) 
        plt.tight_layout()
        
    if analyze_regional_domain:
        ax2.set_xticks(x1 + (num_experiments * width) / 2 - width / 2)  # Center the tick marks between the two bars for each year
        ax2.set_xticklabels([f'{year}' for year in years_to_analyze])   # Label x-axis with actual years (e.g., 2100, 2200, etc.)
        ax2.set_xlabel('Year',fontsize=13)
        ax2.set_ylabel('Delay time (yr)',fontsize=13)
        ax2.set_title(f'Delay time  in {rNames[region_number]}',fontsize=13)
        ax2.legend(loc='upper left')
        #ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=12) 
        plt.tight_layout()
        
        ax3.set_xticks(x1 + (num_experiments * width) / 2 - width / 2)  # Center the tick marks between the two bars for each year
        ax3.set_xticklabels([f'{year}' for year in years_to_analyze])   # Label x-axis with actual years (e.g., 2100, 2200, etc.)
        ax3.set_xlabel('Year',fontsize=13)
        ax3.set_ylabel('Normalized delay time',fontsize=13)
        ax3.set_title(f'Normalized delay time in {rNames[region_number]}',fontsize=13)
        ax3.legend(loc='upper left')
        #ax3.grid(True)
        ax3.tick_params(axis='both', which='major', labelsize=12) 
        plt.tight_layout()
        
    # Show plots
    plt.show()
    
    if save_fig:
        if analyze_whole_domain:
            fig1.savefig('delay_grnd_mass_antarctica.png', dpi=300)
        if analyze_regional_domain:        
            fig2.savefig('delay_grnd_mass_region.png', dpi=300)
            fig3.savefig('delay_grnd_mass_region_normalized.png', dpi=300)

        
def find_closest_indices_time(time_array, target_times):
    # Convert to numpy arrays if they aren't already
    time_array = np.array(time_array)
    target_times = np.array(target_times)

    # Create an empty list to store the indices
    indices_array = []

    # Loop over each target time
    for target_time in target_times:
        # Find the index in time_array1 where the value is closest to the target_time
        index = np.argmin(np.abs(time_array - target_time))
        indices_array.append(index)

    return np.array(indices_array)
    
# Function to plot grounded ice mass over time for each experiment
def plot_grounded_ice_mass(experiments):
    fig, ax_grdVol = initialize_figure()
    
    # Loop through each experiment (e.g., expAE02, expAE05)
    for experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA in experiments:
        # Extract the grounded ice volume for the full time series. No need to interpolate for coupled runs because we can just use MALI outputs directly.
        print(f'===calling plot grounded ice mass======= : experiment nam  f{experiment_name}')
        time_uncoupled = uncoupled_run['data'].yrs_mali
        grdVol_uncoupled = uncoupled_run['data'].grdVol * rhoi * 1e-12  # Convert volume to mass (Gt)
        grdVol_coupled_WA = coupled_run_WA['data'].grdVol * rhoi * 1e-12  # Convert volume to mass (Gt)
        grdVol_coupled_EA = coupled_run_EA['data'].grdVol * rhoi * 1e-12  # Convert volume to mass (Gt)
                
        # Plot the grounded ice mass for uncoupled, WA, and EA
        ax_grdVol.plot(time_uncoupled, grdVol_uncoupled, label=f'{experiment_name} Uncoupled', color=uncoupled_run['color'], linestyle=uncoupled_run['style'])
        ax_grdVol.plot(time_uncoupled, grdVol_coupled_WA, label=f'{experiment_name} Coupled-WA', color=coupled_run_WA['color'], linestyle=coupled_run_WA['style'])
        ax_grdVol.plot(time_uncoupled, grdVol_coupled_EA, label=f'{experiment_name} Coupled-EA', color=coupled_run_EA['color'], linestyle=coupled_run_EA['style'])
    
    # Customize the plot
    ax_grdVol.set_xlabel('Year',fontsize=16)
    ax_grdVol.set_ylabel('Grounded Ice Mass (Gt)',fontsize=16)
    ax_grdVol.set_title('Grounded Ice Mass Over Time',fontsize=16)
    ax_grdVol.legend(loc='best')
    ax_grdVol.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.savefig('grnd_mass.png', dpi=300)


# Function to calculate delay time for sea-level thresholds relative to the uncoupled run
def calculate_delay_sea_level(time_uncoupled, slc_uncoupled_interp, slc_coupled_interp, thresholds):
    delay_time = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        # Find the year when the uncoupled run reaches the threshold
        uncoupled_year = time_uncoupled[np.argmax(slc_uncoupled_interp >= threshold)]
        
        # Find the year when the coupled run reaches the threshold
        coupled_year = time_uncoupled[np.argmax(slc_coupled_interp >= threshold)]
        
        # Calculate the delay time
        delay_time[idx] = coupled_year - uncoupled_year
        
        print(f"Threshold: {threshold}m, Coupled Year: {coupled_year.values}, Uncoupled Year: {uncoupled_year.values}, Delay: {delay_time[idx]} years")
    return delay_time


def plot_sea_level_change(experiments):
    fig, ax = initialize_figure()
    
    # Prepare for delay time plotting
    fig_delay, ax_delay = plt.subplots(figsize=(10, 6))
    bar_width = 0.3  # Bar width

    # Loop through each experiment
    for i, (experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA) in enumerate(experiments):
        # Extract the sea-level change data for the full time series
        time_uncoupled = uncoupled_run['data'].yrs_mali
        slc_uncoupled = uncoupled_run['data'].SLC_VAF
        
        # Interpolate uncoupled run to the coupled WA and EA time steps
        slc_uncoupled_interp = np.interp(time_uncoupled, coupled_run_WA['data'].yrs, slc_uncoupled)
        
        time_coupled_WA = coupled_run_WA['data'].yrs
        slc_coupled_WA = coupled_run_WA['data'].SLCcorr
        slc_coupled_WA_interp = np.interp(time_uncoupled, time_coupled_WA, slc_coupled_WA)
        
        time_coupled_EA = coupled_run_EA['data'].yrs
        slc_coupled_EA = coupled_run_EA['data'].SLCcorr
        slc_coupled_EA_interp = np.interp(time_uncoupled, time_coupled_EA, slc_coupled_EA)
        
        #print('++++++++ difference in coupled vs. uncoupled sea-level change at 2300 +++++++++')
        #print(f'+++++++++uncoupled minus coupled-WA {slc_uncoupled_interp - slc_coupled_WA_interp}++++++++')
        #print(f'+++++++++uncoupled minus coupled-EA {slc_uncoupled_interp - slc_coupled_EA_interp}++++++++')
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # Calculate thresholds as percentages of the max SLR in the Coupled-WA run
        max_slc_coupled_WA = np.max(slc_coupled_WA_interp)
        thresholds = [max_slc_coupled_WA * p for p in [0.25, 0.5, 0.75, 1.0]]
            
        # Calculate delay times for WA and EA
        print(f"Calculating delay time for given sea-level threshold for coupled simulation: {coupled_run_WA['name']}")
        delay_time_coupled_WA = calculate_delay_sea_level(time_uncoupled, slc_uncoupled_interp, slc_coupled_WA_interp, thresholds)
        print(f"Calculating delay time for given sea-level threshold for coupled simulation: {coupled_run_EA['name']}")
        delay_time_coupled_EA = calculate_delay_sea_level(time_uncoupled, slc_uncoupled_interp, slc_coupled_EA_interp, thresholds)
        
        # Offset positions for each experiment
        x = np.arange(len(thresholds)) + i * bar_width  # Shift bars horizontally for each experiment

        # Plot the WA delay (solid base)
        ax_delay.bar(x, delay_time_coupled_WA, bar_width, label=f'{experiment_name} WA', color=coupled_run_WA['color'], alpha=0.8)

        # Overlay the EA delay (hatched on top of WA)
        ax_delay.bar(x, delay_time_coupled_EA, bar_width, label=f'{experiment_name} EA', color=coupled_run_EA['color'], alpha=0.8, hatch='//', edgecolor='black')

        # Plot sea-level change time series
        ax.plot(time_uncoupled[0:300], slc_uncoupled_interp[0:300], label=f'{experiment_name} Uncoupled', color=uncoupled_run['color'], linestyle=uncoupled_run['style'])
        ax.plot(time_uncoupled[0:300], slc_coupled_WA_interp[0:300], label=f'{experiment_name} Coupled-WA', color=coupled_run_WA['color'], linestyle=coupled_run_WA['style'])
        ax.plot(time_uncoupled[0:300], slc_coupled_EA_interp[0:300], label=f'{experiment_name} Coupled-EA', color=coupled_run_EA['color'], linestyle=coupled_run_EA['style'])
        
    # Customize the delay time plot
    ax_delay.set_xticks(np.arange(len(thresholds)) + (len(experiments) - 1) * bar_width / 2)
    ax_delay.set_xticklabels([f'{int(p*100)}%' for p in [0.25, 0.5, 0.75, 1.0]])  # Label x-axis with percentages
    ax_delay.set_xlabel('Percentage of Max Sea-Level Rise in Coupled-WA Run',fontsize=16)
    ax_delay.set_ylabel('Delay time (years)',fontsize=16)
    ax_delay.set_title('Delay in Reaching Sea-Level Rise Thresholds (Relative to Coupled-WA)',fontsize=16)
    ax_delay.legend(loc='upper left')
    ax_delay.tick_params(axis='both', which='major', labelsize=16) 
    # ax_delay.grid(True)

    # Customize the sea-level change plot
    ax.set_xlabel('Year',fontsize=16)
    ax.set_ylabel('Sea-Level Change (m)',fontsize=16)
    ax.set_title('Sea-Level Change Over Time',fontsize=16)
    #ax.legend(loc='best')
    ax.grid(True)
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
    # Save the figures
    fig.savefig('sea_level_change_over_time.png', dpi=300)
    fig_delay.savefig('normalized_delay_sea_level_thresholds.png', dpi=300)


    
# def plot_sea_level_change(experiments): # this plot shows delay for each SLR threshold (in meters)
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Define the sea-level rise thresholds to evaluate
#     thresholds = [0.5, 1.0, 1.5]
    
#     # Prepare for delay time plotting
#     fig_delay, ax_delay = plt.subplots(figsize=(10, 6))
#     width = 0.3  # Bar width
#     x = np.arange(len(thresholds))  # X positions for the thresholds
    
#     # Loop through each experiment (e.g., expAE02, expAE05)
#     for i, (experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA) in enumerate(experiments):
#         # Extract the sea-level change data for the full time series
#         time_uncoupled = uncoupled_run['data'].yrs_mali
#         slc_uncoupled = uncoupled_run['data'].SLC_VAF
        
#         # Interpolate uncoupled run to the coupled WA and EA time steps
#         slc_uncoupled_interp = np.interp(time_uncoupled, coupled_run_WA['data'].yrs, slc_uncoupled)
        
#         time_coupled_WA = coupled_run_WA['data'].yrs
#         slc_coupled_WA = coupled_run_WA['data'].SLCcorr
#         slc_coupled_WA_interp = np.interp(time_uncoupled, time_coupled_WA, slc_coupled_WA)
        
#         time_coupled_EA = coupled_run_EA['data'].yrs
#         slc_coupled_EA = coupled_run_EA['data'].SLCcorr
#         slc_coupled_EA_interp = np.interp(time_uncoupled, time_coupled_EA, slc_coupled_EA)
        
#         # Plot the sea-level change for uncoupled, WA, and EA
#         ax.plot(time_uncoupled, slc_uncoupled_interp, label=f'{experiment_name} Uncoupled', color=uncoupled_run['color'], linestyle='-')
#         ax.plot(time_uncoupled, slc_coupled_WA_interp, label=f'{experiment_name} Coupled WA', color=coupled_run_WA['color'], linestyle='--')
#         ax.plot(time_uncoupled, slc_coupled_EA_interp, label=f'{experiment_name} Coupled EA', color=coupled_run_EA['color'], linestyle='-.')

#         # Calculate delay times for WA and EA using the independent function
#         print(f"Calculating delay time for given sea-level thresold for coupled simuation: {coupled_run_WA['name']}")
#         delay_time_coupled_WA = calculate_delay_sea_level(time_uncoupled, slc_uncoupled_interp, slc_coupled_WA_interp, thresholds)
#         print(f"Calculating delay time for given sea-level thresold for coupled simuation: {coupled_run_EA['name']}")
#         delay_time_coupled_EA = calculate_delay_sea_level(time_uncoupled, slc_uncoupled_interp, slc_coupled_EA_interp, thresholds)
        
#         # Offset positions for each experiment
#         pos = x + i * width  # Shift bars horizontally for each experiment

#         # Plot the WA delay (solid base)
#         ax_delay.bar(pos, delay_time_coupled_WA, width, label=f'{experiment_name} WA delay', color=coupled_run_WA['color'], alpha=0.8)

#         # Overlay the EA delay (hatched on top of WA)
#         ax_delay.bar(pos, delay_time_coupled_EA, width, label=f'{experiment_name} EA delay', color=coupled_run_EA['color'], hatch='//', edgecolor='black')
    
#     # Customize the main sea-level change plot
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Sea-level change (m)')
#     ax.set_title('Sea-level change since year 2000')
#     ax.legend(loc='best')
#     ax.grid(True)
    
#     # Show the main sea-level change plot
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('SLC.png', dpi=300)
    
#     # Customize the delay time plot
#     ax_delay.set_xticks(x + width / 2)  # Center the tick marks between the bars
#     ax_delay.set_xticklabels([f'{threshold}m' for threshold in thresholds])  # Label x-axis with thresholds
#     ax_delay.set_xlabel('Sea-level rise threshold (m)')
#     ax_delay.set_ylabel('Delay time (years)')
#     ax_delay.set_title('Delay in Reaching Sea-level Thresholds for All Experiments')
#     ax_delay.legend(loc='upper left')
    
#     # Show the delay time plot
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('delay_sea_level.png', dpi=300)


# def plot_sea_level_change(experiments):
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Loop through each experiment (e.g., expAE02, expAE05)
#     for experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA in experiments:
#         # Extract the grounded ice volume for the full time series
#         time_uncoupled = uncoupled_run['data'].yrs_mali
#         slc_uncoupled = uncoupled_run['data'].SLC_VAF # ideally, i would take this from globalstats function and not have to interpolate
#         slc_uncoupled_interp = np.interp(time_uncoupled, coupled_run_WA['data'].yrs, slc_uncoupled)
#         print(time_uncoupled)
#         print(len(time_uncoupled))
#         time_coupled_WA = coupled_run_WA['data'].yrs
#         slc_coupled_WA = coupled_run_WA['data'].SLCcorr
#         # interpolate values in case SLM output interval and MALI output interval are different. 
#         # np.interp will do nothing if time_uncoupled and time_coupled_WA are identical.
#         slc_coupled_WA_interp = np.interp(time_uncoupled, time_coupled_WA, slc_coupled_WA)

#         time_coupled_EA = coupled_run_EA['data'].yrs
#         slc_coupled_EA = coupled_run_EA['data'].SLCcorr
#         slc_coupled_EA_interp = np.interp(time_uncoupled, time_coupled_EA, slc_coupled_EA)
        
#         # Plot the grounded ice mass for uncoupled, WA, and EA
#         ax.plot(time_uncoupled, slc_uncoupled_interp, label=f'{experiment_name} Uncoupled', color=uncoupled_run['color'], linestyle='-')
#         ax.plot(time_uncoupled, slc_coupled_WA_interp, label=f'{experiment_name} Coupled WA', color=coupled_run_WA['color'], linestyle='--')
#         ax.plot(time_uncoupled, slc_coupled_EA_interp, label=f'{experiment_name} Coupled EA', color=coupled_run_EA['color'], linestyle='-.')
    
#     # Customize the plot
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Sea-level change (m)')
#     ax.set_title('Sea-level change since year 2000')
#     ax.legend(loc='best')
#     ax.grid(True)
    
#     # Show the plot
#     plt.tight_layout()
#     #plt.gca().invert_yaxis()  # Flipping the y-axis
#     plt.show()
#     plt.savefig('SLC.png', dpi=300)
    

# Helper function to create consistent figure and axis properties
def initialize_figure():
    return plt.subplots(figsize=(8, 6))
    
    # Function to calculate effective circumference and spatial resolution
def calculate_spatial_resolution(latitudes_degrees, equatorial_circumference, longitude_points_list):
    effective_circumferences = equatorial_circumference * np.cos(np.radians(latitudes_degrees))
    spatial_resolutions = {}
    
    for longitude_points in longitude_points_list:
        spatial_res = effective_circumferences / longitude_points
        spatial_resolutions[longitude_points] = spatial_res
    
    return effective_circumferences, spatial_resolutions

def plot_spatial_resolution():  
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the latitudes of interest (e.g., every 10 degrees from 0° to 90°)
    latitudes_degrees = np.arange(0, 91, 10)

    # Earth's equatorial circumference in km
    equatorial_circumference = 40075  # km

    # Define the grid configurations (GL nodes in latitude)
    longitude_points_list = [1024, 2048, 4096]

    # Define color-blind friendly colors for the markers
    colors = ['#E69F00', '#56B4E9', '#009E73']

    # Calculate effective circumference and spatial resolutions
    effective_circumferences, spatial_resolutions = calculate_spatial_resolution(
        latitudes_degrees, equatorial_circumference, longitude_points_list
    )

    for i, (longitude_points, spatial_res) in enumerate(spatial_resolutions.items()):
        plt.plot(latitudes_degrees, spatial_res, marker='o', color=colors[i], label=str(longitude_points // 2))

    ax.set_title('Spatial Resolution Across Latitudes for Different Grid Configurations')
    ax.set_xlabel('Latitude (°)',fontsize=14)
    ax.set_ylabel('Spatial Resolution (km)',fontsize=14)
    ax.legend(title='GL nodes in latitude',fontsize=16)
    ax.grid(True)

    # Show and save the plot
    plt.show()
    plt.savefig('SLM_grid_resoluion_along_latitude.png', dpi=300)


class extract_tot_smb_forcing():
    # take total surface mass balance
    # where SMB forcing represents anomalies at the surface 
    # of the Antarctic Ice Sheet compared to the reference period of 1995–2014
    def __init__(self, experiment_name, years_array):
        # experiment_name: name of experiment to analyze the TF field
        # years_array: array of years to find average TF values

        if analyze_regional_domain:
            # First, read in the region mask file
            regions = netCDF4.Dataset(regions_file, 'r')
            regionCellMasks = regions.variables["regionCellMasks"][:, region_number]
            regions.close()

        # Get path to the appropriate TF file            
        for exp_name, path_SMB in path_to_smb_forcing_files: 
            if experiment_name == exp_name:
                fpath_SMB = path_SMB
                print(f'Found a forcing file at: {fpath_SMB}')
                break

        if fpath_SMB is None:
            raise ValueError("Error: SMB forcing file can't be acquired." 
                         "Check 'experiment_name' and 'exp_name' defined in"
                         " 'path_to_smb_forcing_files'")

        print("Opening the SMB forcing file for experiment") 
        print(f"'{fpath_SMB}'") 
        DS = xr.open_dataset(fpath_SMB)

        smb0 = DS['sfcMassBal'].isel(Time=5).load() #SMB at year 2000
        xtime = DS['xtime'].load()
        nCells = DS.dims['nCells']

        # Extract only the year part from 'xtime'
        xtime_str = np.array([''.join(x.astype(str)) for x in xtime.values])  # Convert char arrays to strings
        years_SMB = np.array([int(t[:4]) for t in xtime_str])  # Extract the first 4 characters (the year)
        years_SMB = np.array(years_SMB) # Don't know why, but np.array needs to be re-done here.

        # Find indices of xtime in the smb forcing file that corresponds to the years of interest
        indices_years = [np.argmin(np.abs(years_SMB - year)) for year in years_array]
        years_SMB = years_SMB[indices_years]

        # debug purpose: check if years are correct
        #print(indices_years)
        #print(years_TF)

        # get initial bedtopo and thickness # Possible change!? change this to the initial condition file as opp
        print(f"Opening the initial condition file from {ic_file}")
        DS2 = xr.open_mfdataset(ic_file, combine='nested', concat_dim='Time',
            decode_timedelta=False)
        thk0 = DS2['thickness'].isel(Time=0).load()
        areaCell = DS2['areaCell'][0,:].load()
        print("Done reading the initial condition file")

        # create a ice mask based on initial thickness
        ice_mask = thk0 > 1.

        # Initialize lists to store the average thermal forcing for each year for the current experiment
        self.tot_smb = []
        self.tot_smb_region = []
           
        for idx, year in enumerate(years_SMB):
            year_idx = indices_years[idx]
            print(f"Processing year {year} at index {year_idx}")

            # load a year at a time
            smb = DS['sfcMassBal'].isel(Time=year_idx)            
            # convert kg/m^2/s to m/yr
            #smb = smb * (60. * 60. * 24. * 365.) / rhoi
            # convert kg/m^2/s to Gt/yr
            smb = (smb * (60. * 60. * 24. * 365.) / 1.e12) * areaCell
            # mask out regions with no ice
            smb_masked = smb.where(ice_mask, other=np.nan)
            # Calcualte total smb
            smb_tot = np.nansum(smb_masked)
            # Store the results for this experiment and year
            self.tot_smb.append(smb_tot)
            if analyze_regional_domain:
                # Mask out only the region of interest, in our case, Amundsen sea sector
                smb_region_masked = smb_masked.where(regionCellMasks == 1, other=np.nan)
                smb_tot_region = np.nansum(smb_region_masked)
                self.tot_smb_region.append(smb_tot_region) 
        
        # Convert lists to numpy arrays
        self.tot_smb = np.array(self.tot_smb)
        if analyze_regional_domain:
            self.tot_smb_region = np.array(self.tot_smb_region)

def plot_TF():
    # plot thermal forcing time series
    years_to_analyze_smb = np.array(range(2000,2300,20))#(range(2020,2301,10))

    fig_TF, ax_TF = initialize_figure()
    fig_TF_region, ax_TF_region = initialize_figure()
    
    for i, (experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA) in enumerate(experiments):
        rNames = uncoupled_run['data_regional'].rNames
        TF_extractor = extract_tot_smb_forcing(experiment_name, years_to_analyze_smb)
        tot_smb = smb_extractor.tot_smb
        tot_smb_region = smb_extractor.tot_smb_region

        ax_smb.plot(years_to_analyze_smb, tot_smb, label=f'{experiment_name} whole mesh', 
                 color=uncoupled_run['color'], linestyle='-')#, marker='o')

        if analyze_regional_domain:
            # plot smb
            ax_smb_region.plot(years_to_analyze_smb, tot_smb_region, label=f'{experiment_name}', 
                        color=coupled_run_WA['color'], linestyle='-')#, marker='o')
            # ax_smb_region.scatter(years_to_analyze_smb, tot_smb_region, label=f'{experiment_name} Coupled-WA',
                        # facecolors=coupled_run_WA['color'], edgecolors= coupled_run_WA['color'], s=30, marker='o')
                                             
           
    ax_smb.set_xlabel('Year')
    ax_smb.set_ylabel('SMB (Gt/yr)')
    ax_smb.set_title('Total Surface Mass balance (Gt/yr)')
    ax_smb.legend(loc='best')
    ax_smb.grid(True)
    fig_smb.savefig('time_VS_smb_tot.png', dpi=300)

    if analyze_regional_domain:    
        ax_smb_region.set_xlabel('Year')
        ax_smb_region.set_ylabel('SMB (Gt/yr)')
        ax_smb_region.set_title(f'Total Surface Mass balance (Gt/yr) in {rNames[region_number]}')
        ax_smb_region.legend(loc='best')
        ax_smb_region.grid(True)
        fig_smb_region.savefig('time_VS_smb_tot_region.png', dpi=300)

    plt.show()

    
class extract_avg_thermal_forcing():
    # take average of thermal forcing from ocean forcing
    def __init__(self, experiment_name, years_array):
        # experiment_name: name of experiment to analyze the TF field
        # years_array: array of years to find average TF values

        if analyze_regional_domain:
            # First, read in the region mask file
            regions = netCDF4.Dataset(regions_file, 'r')
            regionCellMasks = regions.variables["regionCellMasks"][:, region_number]
            regions.close()

        # Get path to the appropriate TF file            
        for exp_name, path_TF in path_to_thermal_forcing_files: 
            if experiment_name == exp_name:
                fpath_TF = path_TF
                print(f'Found a forcing file at: {fpath_TF}')
                break

        if fpath_TF is None:
            raise ValueError("Error: Thermal forcing file can't be acquired." 
                         "Check 'experiment_name' and 'exp_name' defined in"
                         " 'path_to_thermal_forcing_files'")
           
        print("Opening the thermal forcing file for experiment") 
        DS = xr.open_mfdataset(fpath_TF, combine='nested', concat_dim='Time',
             decode_timedelta=False)

        TFocean0 = DS['ismip6shelfMelt_3dThermalForcing'].isel(Time=5).load() #TF at year 2000 at which our melt param are tuned to
        zbnds = DS['ismip6shelfMelt_zOcean'].isel(Time=0).load()
        xtime = DS['xtime'].load()
        nLayers = np.array(DS.dims['nISMIP6OceanLayers'])
        nCells = DS.dims['nCells']
        
        # Extract only the year part from 'xtime'
        xtime_str = np.array([''.join(x.astype(str)) for x in xtime.values])  # Convert char arrays to strings
        years_TF = np.array([int(t[:4]) for t in xtime_str])  # Extract the first 4 characters (the year)
        years_TF = np.array(years_TF) # Don't know why, but np.array needs to be re-done here.

        indices_years = [np.argmin(np.abs(years_TF - year)) for year in years_array]
        years_TF = years_TF[indices_years]
        
        # debug purpose: check if years are correct
        #print(indices_years)
        #print(years_TF)
              
        # get initial bedtopo and thickness # Possible change!? change this to the initial condition file as opp
        print(f"Opening the initial condition file from {ic_file}")
        DS2 = xr.open_mfdataset(ic_file, combine='nested', concat_dim='Time',
            decode_timedelta=False)
        bed0_uncoup = DS2['bedTopography'].isel(Time=0).load()
        thk0_uncoup = DS2['thickness'].isel(Time=0).load()

        # mask out floating ice
        grdthk0 = thk0_uncoup.where(thk0_uncoup <= (rhow / rhoi) * bed0_uncoup)
        
        # get the depth of top boundary of the ocean layers
        layerTop = zbnds + 30
        
        # mask out the top 3-6 layers since the TF in ocean surface doesn't really play a role in ice-shelf melting
        mask_layers = layerTop > -180  # Adjust the depth threshold as needed
        
        # create a complete mask
        mask = (grdthk0 != 0) & (layerTop < bed0_uncoup) & (mask_layers)

        # Initialize lists to store the   for each year for the current experiment
        self.avg_TF = []
        self.avg_TF_masked= []
        self.avg_TF_region = []
        self.avg_TF_masked_region = []            
        for idx, year in enumerate(years_TF):
            
            year_idx = indices_years[idx]
            print(f"Processing year {year} at index {year_idx}")
            
            # load a year at a time
            TFocean = DS['ismip6shelfMelt_3dThermalForcing'].isel(Time=year_idx)
            
            # set TF w.r.t. the initial year (year 2000)
            TFocean = TFocean - TFocean0
            
            # Apply mask to set masked-out values to NaN (instead of loading into memory with .load())
            TFocean_masked = TFocean.where(~mask, other=np.nan)

            # Mask out only the region of interest, in our case, Amundsen sea sector  
            # keep values where regionCellMasks == 1, set NaN where it's 0
            TFocean_unmasked_region = TFocean.where(regionCellMasks[:, np.newaxis] == 1, other=np.nan) 
            TFocean_masked_region = TFocean_masked.where(regionCellMasks[:, np.newaxis] == 1, other=np.nan) 

            # Calculate the average thermal forcing for this year
            TFocean_avg = TFocean.mean(dim=['nCells', 'nISMIP6OceanLayers'], skipna=True)
            TFocean_masked_avg = TFocean_masked.mean(dim=['nCells', 'nISMIP6OceanLayers'], skipna=True)

            TFocean_avg_region = TFocean_unmasked_region.mean(dim=['nCells', 'nISMIP6OceanLayers'], skipna=True)
            TFocean_masked_avg_region = TFocean_masked_region.mean(dim=['nCells', 'nISMIP6OceanLayers'], skipna=True)

            # Store the results for this experiment and year
            self.avg_TF.append(TFocean_avg.values)
            self.avg_TF_masked.append(TFocean_masked_avg.values) 
            self.avg_TF_region.append(TFocean_avg_region.values)
            self.avg_TF_masked_region.append(TFocean_masked_avg_region.values) 

        # Convert lists to numpy arrays
        self.avg_TF = np.array(self.avg_TF)
        self.avg_TF_region = np.array(self.avg_TF_region)
        self.avg_TF_masked = np.array(self.avg_TF_masked) # This is the field we want eventually
        self.avg_TF_masked_region = np.array(self.avg_TF_masked_region) # This is the field we want eventually

        # Normalize by the max TF value
        max_TF = max(self.avg_TF)  # Find the maximum thermal forcing for this experiment
        self.normalized_avg_TF = [value / max_TF for value in self.avg_TF]

        max_TF_masked = max(self.avg_TF_masked)
        self.normalized_avg_TF_masked = [value / max_TF_masked for value in self.avg_TF_masked]

        max_TF_region = max(self.avg_TF_region)
        self.normalized_avg_TF_region = [value / max_TF_region for value in self.avg_TF_region]

        max_TF_masked_region = max(self.avg_TF_masked_region)
        self.normalized_avg_TF_masked_region = [value / max_TF_masked_region for value in self.avg_TF_masked_region]


def plot_TF_and_delay_time():
   
    avg_TF_combined = []
    delay_time_combined = []

    # Years to analyze for thermal forcing vs. delay time
    years_to_analyze_TF_vs_delay = np.array(range(2000,2301,20))

    # Initialize figures and axes
    if analyze_whole_domain:
        fig_TF_vs_delay, ax_TF_vs_delay = initialize_figure()
        fig_time_vs_TF, ax_time_vs_TF = initialize_figure()    
        fig_TF_vs_delay, ax_TF_vs_delay = initialize_figure()

    if analyze_regional_domain:
            fig_time_vs_TF_region, ax_time_vs_TF_region = initialize_figure()
            fig_delay_reg, ax_delay_reg = initialize_figure()
            fig_delay_vs_grdDArea_reg, ax_delay_vs_grdDArea_reg = initialize_figure()
            fig_TF_vs_delay_region, ax_TF_vs_delay_region = initialize_figure()
            fig_normTF_vs_normDT_region_commonYrs, ax_normTF_vs_normDT_region_commonYrs = initialize_figure()
            fig_normTF_vs_normDT_region, ax_normTF_vs_normDT_region = initialize_figure()
            fig_TF_vs_normDT_region, ax_TF_vs_normDT_region = initialize_figure() 
            fig_time_vs_NormDT_region, ax_time_vs_NormDT_region = initialize_figure()
            fig_time_vs_DT_region, ax_time_vs_DT_region = initialize_figure()   
            
    for i, (experiment_name, uncoupled_run, coupled_run_WA, coupled_run_EA) in enumerate(experiments):
        rNames = uncoupled_run['data_regional'].rNames
        # Get averaged TF field for Antarctica and in the basin of interest defined in 'regions_number'
        # Extract thermal forcing once and store the result
        avg_TF_extractor = extract_avg_thermal_forcing(experiment_name, years_to_analyze_TF_vs_delay)
        avg_TF = avg_TF_extractor.avg_TF_masked
        avg_TF_region = avg_TF_extractor.avg_TF_masked_region
        avg_TF_normalized = avg_TF_extractor.normalized_avg_TF_masked
        avg_TF_normalized_region = avg_TF_extractor.normalized_avg_TF_masked_region

        # calculate delay time
        delay_time_extractor_WA = calculate_delay_grnd_mass(uncoupled_run, coupled_run_WA, 
                                                            years_to_analyze_TF_vs_delay, 
                                                            whole_domain=analyze_whole_domain, 
                                                            regions=analyze_regional_domain)
        delay_time_extractor_EA = calculate_delay_grnd_mass(uncoupled_run,
                                                            coupled_run_EA,
                                                            years_to_analyze_TF_vs_delay,
                                                            whole_domain=analyze_whole_domain,
                                                            regions=analyze_regional_domain)

        if analyze_whole_domain: 
                
            delay_time_WA = delay_time_extractor_WA.delay_time
            delay_time_EA = delay_time_extractor_EA.delay_time
            ax_time_vs_TF.plot(years_to_analyze_TF_vs_delay, avg_TF, label=f'{experiment_name} whole mesh', 
                        color=uncoupled_run['color'], linestyle='-')#, marker='o')
            ax_TF_vs_delay.scatter(avg_TF, delay_time_WA, label=f'{experiment_name} Coupled-WA',
                        facecolors=coupled_run_WA['color'], edgecolors= coupled_run_WA['color'], s=30, marker='o')
            ax_TF_vs_delay.scatter(avg_TF, delay_time_EA, label=f'{experiment_name} Coupled-EA',
                        facecolors='none', edgecolors= coupled_run_WA['color'], s=30, marker='^')

        if analyze_regional_domain:
            # calculate dynamic timescale for region of interest    
            dynamic_time_region = uncoupled_run['data_regional'].dynamic_time_region
            delay_time_region_WA = delay_time_extractor_WA.delay_time_region
            delay_time_region_EA = delay_time_extractor_EA.delay_time_region
            delay_time_region_WA_norm = delay_time_region_WA / dynamic_time_region
            delay_time_region_EA_norm = delay_time_region_EA / dynamic_time_region
            
            # plot time vs. avg_TF in the region
            ax_time_vs_TF_region.plot(years_to_analyze_TF_vs_delay, avg_TF_region, label=f'{experiment_name} ({rNames[region_number]})', 
                           color=uncoupled_run['color'], linestyle='-')#, marker='o')
            # calculate changes in grounded ice area
            grdthkAreaChange_reg = uncoupled_run['data_regional'].areaGrdChange_region
            years_from_regionalStats = uncoupled_run['data_regional'].yr
            interp_func_grdArea = interp1d(years_from_regionalStats, grdthkAreaChange_reg, kind='cubic', fill_value="extrapolate")
            grdthkAreaChange_reg_interped  = interp_func_grdArea(years_to_analyze_TF_vs_delay)
            
            # some debug statement
            # print(f'grounded ice area change at initial and final time is {grdthkAreaChange_reg[0]} and {grdthkAreaChange_reg[-1]}')
            # print(f'this is at years {years_from_regionalStats[0]} and {years_from_regionalStats[-1]} from regionalstats years.')
            # print(f'and yr to analyze are {years_to_analyze_TF_vs_delay[0]} and {years_to_analyze_TF_vs_delay[-1]}')
            
            # plot delay time vs. grounded ice area change
            ax_delay_vs_grdDArea_reg.scatter(-1*grdthkAreaChange_reg_interped, delay_time_region_WA_norm, label=f'{experiment_name} Coupled-WA',
                        facecolors=coupled_run_WA['color'], edgecolors= coupled_run_WA['color'], s=30, marker='o')
            
            # plot normalized delay time vs. time
            ax_time_vs_NormDT_region.scatter(years_to_analyze_TF_vs_delay, delay_time_region_WA_norm, label=f'{experiment_name} Coupled-WA',
                        facecolors=coupled_run_WA['color'], edgecolors= coupled_run_WA['color'], s=30, marker='o')
                                             
            # only plot coupled-WA or coupled-EA depeding on which region to plot
            ax_delay_reg.scatter(avg_TF_normalized_region, delay_time_region_WA, label=f'{experiment_name} Coupled-WA',
                        facecolors=coupled_run_WA['color'], edgecolors= coupled_run_WA['color'], s=30, marker='o')
                 
            # plot normalized at the same TF values
            # Define a common grid of normalized TF values for alignment
            common_normalized_TF = np.linspace(0, 1, 15)  # 11 points from 0 to 1
            # Interpolate delay time to match common normalized TF grid
            interp_func = interp1d(avg_TF_normalized_region, delay_time_region_WA_norm, kind='linear', bounds_error=False, fill_value='extrapolate')
            norm_delay_time_interpolated = interp_func(common_normalized_TF)   
            
            ax_TF_vs_delay_region.scatter(avg_TF_region, delay_time_region_WA, label=f'{experiment_name} Coupled-WA',
                          marker='o', linestyle='-', color=coupled_run_WA['color'])
            
            ax_normTF_vs_normDT_region_commonYrs.scatter(common_normalized_TF, norm_delay_time_interpolated, label=f'{experiment_name} Coupled-WA',
                          marker='o', linestyle='-', color=coupled_run_WA['color'])
            
            ax_TF_vs_normDT_region.scatter(avg_TF_region, delay_time_region_WA_norm, label=f'{experiment_name} Coupled-WA',
                          marker='o', linestyle='-', color=coupled_run_WA['color'])
            
            ax_normTF_vs_normDT_region.scatter(avg_TF_normalized_region, delay_time_region_WA_norm, label=f'{experiment_name} Coupled-WA',
                          marker='o', linestyle='-', color=coupled_run_WA['color'])
            ax_time_vs_DT_region.scatter(years_to_analyze_TF_vs_delay, delay_time_region_WA, label=f'{experiment_name}', 
                           color=uncoupled_run['color'], linestyle='-')
            
            
            avg_TF_combined.extend(common_normalized_TF)
            delay_time_combined.extend(norm_delay_time_interpolated)
            


    if analyze_whole_domain:
        ax_time_vs_TF.set_xlabel('Year')
        ax_time_vs_TF.set_ylabel('Thermal Forcing anomaly (°C)')
        ax_time_vs_TF.set_title('Average Thermal Forcing over Time (Antarctica)')
        ax_time_vs_TF.legend(loc='best')
        ax_time_vs_TF.grid(True)
        fig_time_vs_TF.savefig('time_VS_TF_Antarctica.png', dpi=300)
        
        ax_TF_vs_delay.set_xlabel('Thermal forcing anomaly (°C)')
        ax_TF_vs_delay.set_ylabel('Delay time (yr)')
        ax_TF_vs_delay.set_title('Thermal forcing vs. Delay time (whole domain)')
        ax_TF_vs_delay.legend(loc='upper left')
        ax_TF_vs_delay.grid(True)
        fig_TF_vs_delay.savefig('delay_time_VS_TF_Antarctica_noLegend.png', dpi=300)

        
    if analyze_regional_domain:    
        ax_time_vs_TF_region.set_xlabel('Time (yr)')
        ax_time_vs_TF_region.set_ylabel('Thermal Forcing anomaly (°C)')
        ax_time_vs_TF_region.set_title(f'Average thermal forcing anomaly over time ({rNames[region_number]})')
        ax_time_vs_TF_region.legend(loc='best')
        ax_time_vs_TF_region.grid(True)
        fig_time_vs_TF_region.savefig(f'time_VS_TF_{rNames[region_number]}.png', dpi=300)
        
        ax_delay_vs_grdDArea_reg.set_xlabel('Loss of grounded area (km^2)')
        ax_delay_vs_grdDArea_reg.set_ylabel('Delay time (yr)')
        ax_delay_vs_grdDArea_reg.set_title(f'Grounded ice area loss vs. Delay time ({rNames[region_number]})')
        ax_delay_vs_grdDArea_reg.grid(True)
        ax_delay_vs_grdDArea_reg.legend(loc='upper left')
        fig_delay_vs_grdDArea_reg.savefig(f'GroundedAreaChange_vs_DT_{rNames[region_number]}.png', dpi=300)
        
        ax_time_vs_NormDT_region.set_xlabel('Time (yr)')
        ax_time_vs_NormDT_region.set_ylabel('Normalized delay time')
        ax_time_vs_NormDT_region.set_title(f'Time vs. Normalized delay time ({rNames[region_number]})')
        ax_time_vs_NormDT_region.grid(True)
        ax_time_vs_NormDT_region.legend(loc='upper left')
        fig_time_vs_NormDT_region.savefig(f'Time_vs_NormDT_{rNames[region_number]}.png', dpi=300)
        
        ax_delay_reg.set_xlabel('Normalized thermal forcing anomaly')
        ax_delay_reg.set_ylabel('Delay time (yr)')
        ax_delay_reg.set_title(f'Normalizes TF vs. Delay Time ({rNames[region_number]})')
        #ax_delay_reg.legend(loc='upper left')
        ax_delay_reg.grid(True)
        #fig_delay_reg.savefig(f'NormTF_VS_delayTime_{rNames[region_number]}_{pct_mass_loss}pct.png', dpi=300)

        ax_TF_vs_delay_region.set_xlabel('Thermal forcing anomaly (°C)')
        ax_TF_vs_delay_region.set_ylabel('Delay time (yr)')
        ax_TF_vs_delay_region.set_title(f'Average TF vs. Delay time ({rNames[region_number]})')
        ax_TF_vs_delay_region.grid(True)
        ax_TF_vs_delay_region.legend(loc='upper left')
        #fig_TF_vs_delay_region.savefig(f'TF_vs_delayTime_{rNames[region_number]}_{pct_mass_loss}pct.png', dpi=300)
                
        ax_TF_vs_normDT_region.set_xlabel('Thermal forcing anomaly (°C)')
        ax_TF_vs_normDT_region.set_ylabel('Normalized delay time')
        ax_TF_vs_normDT_region.set_title(f'Average TF vs. Normalized delay time ({rNames[region_number]})')
        ax_TF_vs_normDT_region.grid(True)
        ax_TF_vs_normDT_region.legend(loc='upper left')
        #fig_TF_vs_normDT_region.savefig(f'TF_vs_NormDT_{rNames[region_number]}_{pct_mass_loss}pct.png', dpi=300)

        ax_normTF_vs_normDT_region.set_xlabel('Normalized thermal forcing anomaly')
        ax_normTF_vs_normDT_region.set_ylabel('Normalized delay time')
        ax_normTF_vs_normDT_region.set_title(f'Normalized TF vs. Normalized delay time ({rNames[region_number]})')
        ax_normTF_vs_normDT_region.grid(True)
        ax_normTF_vs_normDT_region.legend(loc='upper left')
        #fig_normTF_vs_normDT_region.savefig(f'NormTF_VS_NormDT_{rNames[region_number]}_{pct_mass_loss}pct.png', dpi=300)
        
        ax_time_vs_DT_region.set_xlabel('Time (yr)')
        ax_time_vs_DT_region.set_ylabel('Delay time (yr)')
        ax_time_vs_DT_region.set_title(f'Time vs. Delay Time ({rNames[region_number]})')
        ax_time_vs_DT_region.grid(True)
        ax_time_vs_DT_region.legend(loc='upper left')
        fig_time_vs_DT_region.savefig(f'Time_vs_DT_{rNames[region_number]}.png', dpi=300)
        
        avg_TF_combined = np.array(avg_TF_combined)
        delay_time_combined = np.array(delay_time_combined)
        
        #positive_indices = avg_TF_combined > 0
        #avg_TF_combined = avg_TF_combined[positive_indices]
        #delay_time_combined = delay_time_combined[positive_indices]

        # Perform curve fitting
        power_params, _ = curve_fit(power_law_model, avg_TF_combined, delay_time_combined, p0=[1,1], maxfev=2000)
        a, b = power_params
        x_fit = np.linspace(min(avg_TF_combined), max(avg_TF_combined), 100)
        y_fit = power_law_model(x_fit, *power_params)  
        # Print the fitted parameters
        print("Power law parameters:")
        print("a =", a)
        print("b =", b)
        ax_normTF_vs_normDT_region_commonYrs.set_xlabel('Normalized thermal forcing anomaly')
        ax_normTF_vs_normDT_region_commonYrs.set_ylabel('Normalized delay time')
        ax_normTF_vs_normDT_region_commonYrs.plot(x_fit, y_fit, 'k-', label=f'Power Law Fit: $y = {a:.2f} x^{b:.2f}$')
        #ax_normTF_vs_normDT_region_commonYrs.fill_between(x_fit, min(delay_time_region_WA_norm), max(delay_time_region_WA_norm), color='gray', alpha=0.2)
        ax_normTF_vs_normDT_region_commonYrs.set_title(f'Normalized TF anomaly vs. Normalized delay time ({rNames[region_number]})')
        ax_normTF_vs_normDT_region_commonYrs.grid(True)
        ax_normTF_vs_normDT_region_commonYrs.legend(loc='upper left')
        #fig_normTF_vs_normDT_region_commonYrs.savefig(f'NormTF_VS_NormDT_region_commonYrs_{pct_mass_loss}pct_curveFit.png', dpi=300)

    plt.show()

# Define power law model
def power_law_model(x, a, b):
    return a * x**b

# Process all runs
extract_data_for_run(runs_uncoupled)
extract_data_for_run(runs_coupled_WA)
extract_data_for_run(runs_coupled_EA)

# Call the function to plot grounded ice mass for all experiments
plot_grounded_ice_mass(experiments)
#plot_delay_grnd_mass(save_fig=True)
plot_sea_level_change(experiments)
#plot_regionalStats()
#plot_spatial_resolution()
#plot_TF_and_delay_time()
#plot_smb()