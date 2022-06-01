"""
Script to re-optimize irrigation every growth stage based on historical data

Including for different irrigation caps

This script will only run for one combination of 
(year, irrigation cap, forecast, etc.). Running multiple  combinations
as done in the article can be achieved using argparse and reading in an input 
file with all combinations:

parser = argparse.ArgumentParser()
parser.add_argument("rep", nargs='?', default="1")
args = parser.parse_args()
rep = int(args.rep)

adapt_inputs = pd.read_csv('./inputs.csv')
row = adapt_inputs.iloc[rep-1]
MAXIRR = row.cap
CYEAR = row.year
...

"""

import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import differential_evolution
from aquacrop.core import *
from aquacrop.classes import *
import copy
import time


#random seed
time.sleep(1)
np.random.seed(int(time.time()))

# allocate params
COST = 1.  # irrigation cost
MAXIRR = 10000  # max seasonal irrigtion cap
CYEAR = 1989  # year for optimization
FORECAST = 0  # using forecasts
    

# functions to save and load model params and current condition
def save_model(model):
    
    clock_copy = copy.deepcopy(model.ClockStruct)
    out_copy = copy.deepcopy(model.Outputs)

    init_cond_dict = dict()
    for att in dir(model.InitCond):
        if att[0]!='_':
            val = model.InitCond.__getattribute__(att)
            init_cond_dict[att]=val
            
    return [clock_copy,out_copy,init_cond_dict,copy.deepcopy(model.weather)]

def load_model(model,params):
    
    clock_copy,out_copy,InitCond_copy,weather = params
    
    model.ClockStruct=copy.deepcopy(clock_copy)
    model.Outputs=copy.deepcopy(out_copy)
    model.weather=copy.deepcopy(weather)
    
    model.InitCond=InitCondClass(12)
    for att in InitCond_copy.keys():
        if att[0]!='_':
            val = InitCond_copy[att]
            model.InitCond.__setattr__(att,val)
            
    
    return model



# functions to create and run model
def create_maize_model(smts,year1,year2,wdf,
              crop = CropClass('Maize','05/01'),
              soil= SoilClass('ClayLoam'),
              init_wc = InitWCClass(wc_type='Pct',value=[70]),
              co2conc=369.41,max_irr_season=MAXIRR):

    irr_mngt = IrrMngtClass(IrrMethod=5,SMT=list(smts),MaxIrrSeason=max_irr_season)
    model = AquaCropModel('{0}/05/01'.format(year1),'{0}/12/31'.format(year2),wdf,soil,
                          crop,IrrMngt=irr_mngt,InitWC=init_wc,CO2conc=co2conc)
    
    model.initialize()
    # model.step()
    
    return model

# function to run model till end of growth stage
def run_model_till_end(model,stop_at_gs=False):
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
        
        if stop_at_gs and (int(model.InitCond.GrowthStage)!=gs_init):
            break
            
            

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
    """
    Note: 
    Change the tol, and maxiter params to ensure you find global optimum
    Change workers to fit your PC capabilities
    article used tol=0.0001 and maxiter=50
    """
    
    max_bound = np.ones(4)
    min_bound = np.zeros(4)
    bounds = [(min_bound[i], max_bound[i]) for i in range(4)]

    res = differential_evolution(func,bounds=bounds,disp=True,workers=8,tol=1,maxiter=5)


    return res.fun,res.x



# func to minimize/maximise
def run_and_return_profit(smts,params,weather_list):
    """
    for each candidate smt strategy: 
        go through each year in dataset, 
        load model and set smt strat to the candidate one, 
        run model and save profit
        return mean profit for maximise 
    """
    total=[]
    for year in range(1982,2019):
        model=create_maize_model(smts.reshape(-1),year,year,wdf)
        model = load_model(model,params)
        model.weather = weather_list[year-1982]
        model = run_model_till_end(model,)
        total.append(calc_profit(model))
    
    prof = np.mean(total)
    
    if model.Outputs.Final['Seasonal irrigation (mm)'].mean()>MAXIRR:
        return 100_000
    else:
        return -prof


# run test year till end of season
def run_test(smts,year1,year2,params):
    model=create_maize_model(smts.reshape(-1),year1,year2,wdf,max_irr_season=MAXIRR)
    model = load_model(model,params)
    model = run_model_till_end(model)
    print(model.InitCond.IrrCum)
    out = model.Outputs.Final
    yld = float(out['Yield (tonne/ha)'].mean())
    tirr = float(out['Seasonal irrigation (mm)'].mean())

    return calc_profit(model),yld,tirr


# advance test simulation 7 days then re-optimize smts
def run_till_gs_and_reopt(model,cy):

    # run till next growth satge
    model = run_model_till_end(model,True)

    # save model
    params = save_model(model)

    # optimize smts
    rew_func = partial(run_and_return_profit,
                       params=params,weather_list=weather_list)

    profit,smts = optimize_sim(rew_func)
    
    model.IrrMngt.SMT=smts.flatten()

    return model,smts






###### main code #######





# weather list for re-optimization
wdf = prepare_weather(get_filepath('champion_climate.txt'))
wdf.Date.min(),wdf.Date.max()
sim_start='05/01'
sim_end = '12/31'
weather_list = []
for year in range(1982,2019):
    new_start = pd.to_datetime('{0}/'.format(year)+sim_start)
    new_end = pd.to_datetime('{0}/'.format(year)+sim_end)
    
    assert wdf.Date.iloc[0] <= new_start
    assert wdf.Date.iloc[-1] >= new_end

    # remove weather data outside of simulation dates
    weather_df = wdf[wdf.Date>=new_start]
    weather_df = weather_df[weather_df.Date<=new_end]
    
    weather_list.append(weather_df.values)



# main code
results_arr=[]
for rep in range(1):
    year_res=[]
    year_res.append(rep)
    year_res.append(MAXIRR)
    year_res.append(COST)
    year_res.append(FORECAST)
    year_res.append(CYEAR)

    # start with optimal fixed strategy depending on maximum irriagtion cap

    if MAXIRR == 10_000:
        smt = np.array([0.50467352, 0.61434077, 0.3622174,  0.0])
    elif MAXIRR == 75:
        smt = np.array([0.11808814, 0.16865024, 0.3310347,  0.0])
    elif MAXIRR == 100:
        smt = np.array([0.13505919, 0.07850071, 0.8145998,  0.0])
    elif MAXIRR == 125:
        smt = np.array([0.05127158, 0.03720932, 0.80038886, 0.0])
    elif MAXIRR == 150:
        smt = np.array([0.03125578, 0.2730232,  0.70472257, 0.0])
    elif MAXIRR == 175:
        smt = np.array([0.01883106, 0.33505548, 0.78613462, 0.0])
    elif MAXIRR == 200:
        smt = np.array([0.40570058, 0.18266749, 0.79365051, 0.0])
    elif MAXIRR == 250:
        smt = np.array([0.49465802, 0.44268202, 0.34538826, 0.0])
    elif MAXIRR == 300:
        smt = np.array([0.49573032, 0.40311588, 0.37505826, 0.0])
    elif MAXIRR == 350:
        smt = np.array([0.46857963, 0.44267616, 0.37422431, 0.0])
    elif MAXIRR == 400:
        smt = np.array([0.50554841, 0.48620426, 0.39177135, 0.0])
    elif MAXIRR == 450:
        smt = np.array([0.50466926, 0.61632605, 0.36221305, 0.0])
    elif MAXIRR == 500:
        smt = np.array([0.50411676, 0.61622206, 0.36222928, 0.0])
    elif MAXIRR == 550:
        smt = np.array([0.50411802, 0.61422673, 0.36221792, 0.0])


    year_res.append(smt)

    #create and save test model
    model=create_maize_model(smt,CYEAR,CYEAR,wdf)
    params = save_model(model)

    # evaluate fixed strategy on test year
    test1_res,yld,tirr = run_test(smt,CYEAR,CYEAR,params)
    print('test1_{0}'.format(CYEAR),test1_res,smt)
    year_res.append(test1_res)
    year_res.append(yld)
    year_res.append(tirr)
    print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)


    for d in range(1,3):
        # run the test year till next growth stage, re-optimize strategy

        model,smt=run_till_gs_and_reopt(model,CYEAR)
        print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)

        year_res.append(model.InitCond.DAP)
        year_res.append(model.InitCond.GrowthStage)
        year_res.append(smt)

        params = save_model(model)
        test2_res,yld,tirr = run_test(smt,CYEAR,CYEAR,params)
        print('test2_{0}'.format(CYEAR),test2_res,smt)
        year_res.append(test2_res)
        year_res.append(yld)
        year_res.append(tirr)


                
    results_arr.append(year_res)
    break

pd.DataFrame(results_arr).to_csv('outputs/gs_adaptation_results.csv', mode='a', header=False)
