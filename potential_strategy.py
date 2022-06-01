"""
Script to optimize potential irrigation strategy for each year
over a set of historic weather years
"""

# imports
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import differential_evolution
from aquacrop.core import *
from aquacrop.classes import *
import time
import argparse

# set random seed
time.sleep(1)
np.random.seed(int(time.time()))


# allocate params
COST = 1. # cost of irrigation water $/ha-mm
MAXIRR = 10000 # max irrigation application per season ha-mm
CYEAR = 2000 # year chosen to ensure constant Co2 concentration every season




# functions to create and run model
def create_maize_model(smts,year1,year2,wdf,
              crop = CropClass('Maize','05/01'),
              soil= SoilClass('ClayLoam'),
              init_wc = InitWCClass(wc_type='Pct',value=[70]),
              co2conc=369.41,max_irr_season=10_000):

    irr_mngt = IrrMngtClass(IrrMethod=5,SMT=list(smts),MaxIrrSeason=max_irr_season)
    model = AquaCropModel('{0}/05/01'.format(year1),'{0}/12/31'.format(year2),wdf,soil,
                          crop,IrrMngt=irr_mngt,InitWC=init_wc,CO2conc=co2conc)
    
    model.initialize()
    # model.step()
    
    return model

# run full seasonal simulation
def run_model_till_end(model):
    gs_init = int(model.InitCond.GrowthStage)*1
        
    while (not model.ClockStruct.ModelTermination):

        if model.InitCond.TAW>0:
            dep = model.InitCond.Depletion/model.InitCond.TAW
        else:
            dep=0

        gs = int(model.InitCond.GrowthStage)-1


        if gs<0 or gs>2:
            depth=0
        else:
            if 1-dep< model.IrrMngt.SMT[gs]:
                depth = min(dep*model.InitCond.TAW,model.IrrMngt.MaxIrr)
            else:
                depth=0

        model.ParamStruct.IrrMngt.depth = depth

        model.step()
#         print(model.InitCond.DAP)

            

    return model



#fuunc to calculate profit
def calc_profit(model):
    
    out = model.Outputs.Final
    
    yld = float(out['Yield (tonne/ha)'].mean())
    tirr = float(out['Seasonal irrigation (mm)'].mean())

    profit = yld*180 - tirr*COST -1728
    
    return profit



#func to optimize smts
def optimize_sim(func):

    max_bound = np.ones(4)
    min_bound = np.zeros(4)
    bounds = [(min_bound[i], max_bound[i]) for i in range(4)]

    res = differential_evolution(func,bounds=bounds,disp=True,workers=8,tol=0.0001,maxiter=50)


    return res.fun,res.x


def run_and_return_profit(smts,year,maxirr):


    model=create_maize_model(smts.reshape(-1),year,year,wdf,max_irr_season=maxirr)

    model = run_model_till_end(model,)
    prof = calc_profit(model)

    return -prof


###### main code #######



# weather dataframe
wdf = prepare_weather(get_filepath('champion_climate.txt'))
wdf.Date.min(),wdf.Date.max()
sim_start='05/01'
sim_end = '12/31'



for year in range(1982,2019):
    rew_func = partial(run_and_return_profit,year=year,maxirr=MAXIRR)

    _p,smt = optimize_sim(rew_func)

    model=create_maize_model(smt,year,year,wdf,max_irr_season=MAXIRR)
    model = run_model_till_end(model,)
        
    pd.DataFrame([year,smt,_p]).T.to_csv('outputs/potential_results.csv',index=None, mode='a', header=False)
    