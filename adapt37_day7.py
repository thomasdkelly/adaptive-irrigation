import numpy as np
import pandas as pd
from functools import partial

from scipy.optimize import differential_evolution

from aquacrop.core import *
from aquacrop.classes import *
import copy
import time
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

#args, row of input dataframe
parser = argparse.ArgumentParser()
parser.add_argument("a", nargs='?', default="42")
args = parser.parse_args()
rep = int(args.a)
# rep=1

#random seed
time.sleep(rep)
np.random.seed(int(time.time()))

# allocate params
adapt_inputs = pd.read_csv('./adapt37_day7_inputs.csv')
row = adapt_inputs.iloc[rep-1]

MAXIRR = row.cap
RISK = row.risk
CYEAR = row.year
FORECAST = row.forecast

# MAXIRR = 10_000
# RISK = 0
# CYEAR = 1982
# FORECAST = 1
NUM_REOPT=17
DAYS = 7
COST=1.


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

def run_model_till_end(model,N=1000):
    steps = N*1
        
    while (not model.ClockStruct.ModelTermination) and steps>0:

        if model.InitCond.TAW>0:
            dep = model.InitCond.Depletion/model.InitCond.TAW
        else:
            dep=0

        gs = int(model.InitCond.GrowthStage)-1

#         if N<100:
#             if 1-dep< model.IrrMngt.SMT[3]:
#                 depth = min(dep*model.InitCond.TAW,model.IrrMngt.MaxIrr)
#             else:
#                 depth=0            
            
        if gs<0 or gs>2:
            depth=0
        #         gs=model.InitCond.DAP//(25)
        else:
            if 1-dep< model.IrrMngt.SMT[gs]:
                depth = min(dep*model.InitCond.TAW,model.IrrMngt.MaxIrr)
            else:
                depth=0


        model.ParamStruct.IrrMngt.depth = depth

        model.step()
        steps-=1

    return model



#fuunc to calculate profit
def calc_profit(model):
    
    out = model.Outputs.Final
    
    yld = float(out['Yield (tonne/ha)'].mean())
    tirr = float(out['Seasonal irrigation (mm)'].mean())

    profit = yld*180 - tirr*COST -1728
    
    return profit



#func to optimize smts
def optimize_sim_pswarm(func):
    


    max_bound = np.ones(4)
    min_bound = np.zeros(4)
    bounds = [(min_bound[i], max_bound[i]) for i in range(4)]

    res = differential_evolution(func,bounds=bounds,disp=True,workers=8,tol=0.0001,maxiter=50)


    return res.fun,res.x

# func to optimize
def run_and_return_profit(smts,year1,year2,params,weather_list):
    total=[]
    i=0
    # for year in range(year1,year1+30):
    for year in range(1982,2019):
        model=create_maize_model(smts.reshape(-1),year2,year2,wdf)
        model = load_model(model,params)
        model.weather = weather_list[year-1982]
        #model.IrrMngt.SMT = smts.reshape(-1)
        model = run_model_till_end(model,)
        total.append(calc_profit(model))
        i+=1

    #print(total/i)
    
    risk_prem = 0.5 * RISK * np.var(total) / np.abs(np.mean(total))
    CE = np.mean(total) - risk_prem
    
    if model.Outputs.Final['Seasonal irrigation (mm)'].mean()>MAXIRR:
        return 100_000
    else:
        return -CE

def run_and_return_profit_forecast(smts,year1,year2,params,weather_list):
    total=[]
    i=0
    # for year in range(year1,year1+30):
    for year in range(1982,2019):
        model=create_maize_model(smts.reshape(-1),year2,year2,wdf)
        model = load_model(model,params)
        model.weather[model.ClockStruct.TimeStepCounter+7:] = weather_list[year-1982][model.ClockStruct.TimeStepCounter+7:]
        #model.IrrMngt.SMT = smts.reshape(-1)
        model = run_model_till_end(model,)
        total.append(calc_profit(model))
        i+=1

    #print(total/i)
    
    risk_prem = 0.5 * RISK * np.var(total) / np.abs(np.mean(total))
    CE = np.mean(total) - risk_prem
    
    if model.Outputs.Final['Seasonal irrigation (mm)'].mean()>MAXIRR:
        return 100_000
    else:
        return -CE


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


# advance test simulation N days then re-optimize smts
def run_10_days_and_reopt(model,cy,forecast=0):

    # run for 10 days
    model = run_model_till_end(model,DAYS)

    # save model
    params = save_model(model)


    # optimize smts
    if forecast==1:
        rew_func = partial(run_and_return_profit_forecast,year1=start_year,
                        year2 = cy,
                        params=params,weather_list=weather_list)
        
    else:
        rew_func = partial(run_and_return_profit,year1=start_year,
                        year2 = cy,
                        params=params,weather_list=weather_list)
        

    profit,smts = optimize_sim_pswarm(rew_func)
    
    model.IrrMngt.SMT=smts.flatten()
    #print(smts,model.InitCond.DAP)

    return model,smts


# main code
start_year=1982
results_arr=[]
for _ in range(1):
    for current_year in [CYEAR]:
        year_res=[]
        year_res.append(rep)
        year_res.append(MAXIRR)
        year_res.append(RISK)
        year_res.append(COST)
        year_res.append(FORECAST)
        year_res.append(current_year)

        if COST == 1:
            smt = np.array([0.50467352, 0.61434077, 0.3622174,  0.0])
    
        # create model
        #model=create_maize_model(smt,current_year,current_year,wdf)

        #params = save_model(model)
    
        # optimize smts
        #rew_func = partial(run_and_return_profit,year1=start_year,
         #                  year2 = current_year,
         #                  params=params,weather_list=weather_list)

       # _p,smt = optimize_sim_pswarm(rew_func)
        
        year_res.append(smt)
        model=create_maize_model(smt,current_year,current_year,wdf)

        params = save_model(model)

        test1_res,yld,tirr = run_test(smt,current_year,current_year,params)
        print('test1_{0}'.format(current_year),test1_res,smt)
        year_res.append(test1_res)
        year_res.append(yld)
        year_res.append(tirr)
        print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)

        # for d in range(1,3):
        for d in range(1,NUM_REOPT+1):
            print('Day {0}'.format(d*DAYS))

            model,smt=run_10_days_and_reopt(model,current_year,FORECAST)

            # model,smt=run_10_days_and_reopt(model,current_year)
            print(model.InitCond.GrowthStage,model.InitCond.DAP,model.InitCond.IrrCum)

            year_res.append(model.InitCond.DAP)
            year_res.append(model.InitCond.GrowthStage)
            year_res.append(smt)

            params = save_model(model)
            test2_res,yld,tirr = run_test(smt,current_year,current_year,params)
            print('test2_{0}'.format(current_year),test2_res,smt)
            year_res.append(test2_res)
            year_res.append(yld)
            year_res.append(tirr)


                
        results_arr.append(year_res)
        break

pd.DataFrame(results_arr).to_csv('outputs/gs_adapt37_days7.csv', mode='a', header=False)
