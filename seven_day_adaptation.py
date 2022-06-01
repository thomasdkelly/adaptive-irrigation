"""
Script to re-optimize irrigation every 7 days based on historical data 
and perfect weekly weather forecast

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


COST = 1.  # irrigation cost
MAXIRR = 10000  # max seasonal irrigtion cap
CYEAR = 1989  # year for optimization
FORECAST = 0  # using forecasts

NUM_REOPT = 17 # number of in-season reoptimizations
DAYS = 7 # days between reoptimisations


    

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



# function to run model a given N number of steps
def run_model_till_end(model,N=1000):
    steps = N*1
        
    while (not model.ClockStruct.ModelTermination) and steps>0:

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
        steps-=1

    return model



#func to calculate profit
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

# func to minimize/maximise if perfect weekly forecast included
def run_and_return_profit_forecast(smts,params,weather_list):
    """
    for each candidate smt strategy: 
        go through each year in dataset, 
        load model current state of model and set smt strat to the candidate one, 
        run model and save profit
        return mean profit for maximise 
    """
    total=[]
    for year in range(1982,2019):
        model=create_maize_model(smts.reshape(-1),year,year,wdf)
        model = load_model(model,params)
        model.weather[model.ClockStruct.TimeStepCounter+7:] = weather_list[year-1982][model.ClockStruct.TimeStepCounter+7:]
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
def run_7_days_and_reopt(model,cy,forecast=0):

    # run for 7 days
    model = run_model_till_end(model,DAYS)

    # save model
    params = save_model(model)

    # optimize smts with or without perfect forecast
    if forecast==1:
        rew_func = partial(run_and_return_profit_forecast,
                        params=params,weather_list=weather_list)
        
    else:
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



start_year=1982
results_arr=[]
for rep in range(1): # if doing multiple repeats
    year_res=[]
    year_res.append(rep)
    year_res.append(MAXIRR)
    year_res.append(COST)
    year_res.append(FORECAST)
    year_res.append(CYEAR)

    # starting optimal fixed strategy
    smt = np.array([0.50467352, 0.61434077, 0.3622174,  0.0])

    #create and save test model
    year_res.append(smt)
    model=create_maize_model(smt,CYEAR,CYEAR,wdf)
    params = save_model(model)

    # evaluate fixed strategy on test year
    test1_res,yld,tirr = run_test(smt,CYEAR,CYEAR,params)
    print('test1_{0}'.format(CYEAR),test1_res,smt)
    year_res.append(test1_res)
    year_res.append(yld)
    year_res.append(tirr)
    print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)


    for d in range(1,NUM_REOPT+1):
        # run the test year model forward 7 days, re-optimize strategy

        print('Day {0}'.format(d*DAYS))

        model,smt=run_7_days_and_reopt(model,CYEAR,FORECAST)

        print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)

        # save results
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
        

pd.DataFrame(results_arr).to_csv('outputs/seven_day_results.csv', mode='a', header=False)
