# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:09:06 2022

@author: Heeeg
"""
from cProfile import label
import csv
from ctypes.wintypes import PLARGE_INTEGER
from gettext import find
from inspect import trace
from lib2to3.pgen2.token import PLUSEQUAL
from locale import normalize
from os import sep
from sqlite3 import complete_statement
from statistics import mean
from tkinter import BROWSE
from turtle import end_fill
from matplotlib.cbook import ls_mapper
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import matplotlib
import scipy
import numpy as np
import pyLPD.MLtools as pmlt
from scipy import optimize
from scipy import signal
from scipy import constants as C

#%%
filepath = 'D:/wh/Documents/Lab/Data/data5.14/tek0049.csv'
with open(filepath) as f:
    # f.readlines()
    reader = csv.reader(f)
    for i in range(21):
        i+=1
        header_row = next(reader)
        if header_row != [] and header_row != ['', '', '']:
            print(header_row)
    
    # for ind, header_values in enumerate(header_row):
    #     print(ind, header_values)
        
    times, ch2s, ch3s = [], [], []
    #iterator convert to list
    readers = list(reader)
    leng = len(readers)
    print('Data_length:',leng)
        
    for row in readers[21:]:
        if len(row) == 3:
            t_1 = float(row[0])
            ch2 = float(row[1]) #mzi
            ch3 = float(row[2]) #trans
            # if t_1 == '0':
            #     print(ch2)
            times.append(t_1)
            ch2s.append(ch2)
            ch3s.append(ch3)
#%%
a = np.array(times)
times_inis = np.where(a == 2.096e-3)
times_ends = np.where(a == 2.146e-3)
# b = list(enumerate(times_inis))
# print(times_inis)
if len(times_inis[0]) > 1 or len(times_ends[0]) > 1:
    time_ini = times_inis[0][1]
    time_end = times_ends[0][1]
else:
    time_ini = times_inis[0][0]
    time_end = times_ends[0][0]
print(time_ini)
print(time_end)
print('range:', time_end-time_ini)

# time_ini = 89171
# time_end = 99171
times_cut = times[time_ini:time_end]
ch2s_cut = ch2s[time_ini:time_end] #voltage
ch3s_cut = ch3s[time_ini:time_end]
# print(ch3s_cut)
# c1 = []
# for ind, val in enumerate(ch3s_cut):
#     c1.append(ind)
# print(c1[::-1])  
#%%
def comp_max(x1, x2):
    if x1 >= x2:
        return x1
    else:
        return x2
    
def comp_min(x1, x2):
    if x1 <= x2:
        return x1
    else:
        return x2

def lorentz(x, A, cen, widp, wid):
    return A * (1 - widp / ((x-cen)**2 + wid**2))

def sinfit(x, A0, A, x0, T):
    return A0 + A * np.cos((x-x0) / T * 2*np.pi)
#%%
threshold = 0.95
trace_length = len(ch3s_cut)
trace_Q_baseline = scipy.signal.savgol_filter(ch3s_cut, 11, 1)
for i in range(1, trace_length):
    x1 = trace_Q_baseline[i]
    x2 = trace_Q_baseline[i-1]*(1-0.5/trace_length)
    trace_Q_baseline[i] = comp_max(x1, x2)

for i in range(trace_length-2, -1, -1):
    x1 = trace_Q_baseline[i]
    x2 = trace_Q_baseline[i+1]*(1-0.5/trace_length)
    trace_Q_baseline[i] = comp_max (x1, x2)
    
trace_Q_trunc = ch3s_cut / trace_Q_baseline
# cutoff_pos = round(trace_length/100)
# trace_Q_trunc[0:cutoff_pos+1]
# trace_Q_trunc[1:cutoff_pos:1]
#%%
trace_MZI = ch2s_cut
mzi_arry = np.array(ch2s)

# data_raw = pd.read_csv(filepath, 
#                        header=19, 
#                        na_values='--'
#                        )
# data_raw.rename(columns={'CH1':'mzi', 'CH2':'trans'}, inplace=True)
# print(data_raw.head())

# start3 = data_raw.TIME.iloc(time_ini, 'TIME')
# end3 = data_raw.TIME.iloc(time)

# start1 = round(trace_length/4)
# end_1 = round(3 * trace_length/4)
mzi_min, mzi_max = pmlt.envPeak(mzi_arry, delta=0.15, sg_order=0)
mzi_max = mzi_max[time_ini:time_end]
mzi_min = mzi_min[time_ini:time_end]
mzi_nor = ((trace_MZI-mzi_min) / (mzi_max-mzi_min)) / 2
# print(len(mzi_nor), len(trace_MZI))

MZI_peak = max(mzi_nor)
MZI_baseline = np.median(mzi_nor)
MZI_pos = np.where(mzi_nor == MZI_peak)
MZI_amp = MZI_peak - MZI_baseline

for i in np.arange(MZI_pos[0][0], trace_length):
    if mzi_nor[i] < MZI_baseline:
        MZI_period = 4 * (i-MZI_pos[0][0]+1)
        break
    else:
        continue
#%%
while True:
    transmission = min(trace_Q_trunc)
    peakpos = list(trace_Q_trunc).index(transmission)
    if transmission > threshold:
        break
    Q_baseline = trace_Q_baseline[peakpos]
    
    for i in range(peakpos, len(ch3s_cut)):
        trace_Q_basestop = trace_Q_baseline[i]*(1+transmission)/2
        if ch3s_cut[i] > trace_Q_basestop:
            if int(i)-peakpos <= 3:
                lw = 6
                break
            else:
                lw = 2 * (int(i)-peakpos)
                break
        else:
            continue
         
    print(peakpos)
    peak_base_max = ch3s_cut[peakpos]
    print(peak_base_max)
    step_lw = peakpos+10*lw
    peakends = range(peakpos, step_lw)
    if step_lw > trace_length:
        peakends = range(peakpos, trace_length)
    
    for peakend in peakends:
        peak_base_max = comp_max(peak_base_max, ch3s_cut[peakend])
        step = peak_base_max-(1-threshold)*Q_baseline
        
        if ch3s_cut[peakend] < step:
            break
    
    peak_base_max = ch3s_cut[peakpos]
    peak_step = peakpos-10*lw
    
    if peak_step < 1:
        peak_step = 1 
    peakstarts = np.arange(peakpos, peak_step, -1)
    for peakstart in peakstarts:
        peak_base_max = comp_max(peak_base_max, ch3s_cut[peakstart])
        step = peak_base_max-(1-threshold)*Q_baseline
        
        if ch3s_cut[peakstart] < step:
            peakstart = peakstart
            break

    trace_Q_trunc[peakstart:peakend] = 1
    tofit = ch3s_cut[peakstart:peakend:1]
    x = np.arange(peakstart, peakend)
# for ind, val in enumerate(times_cut):
#     x.append(ind)

    popt,_ = optimize.curve_fit(lorentz, x, tofit, 
                                p0=[Q_baseline, peakpos, (1-transmission)*(lw**2)/4, lw/2])    
#%%    
    MZI_period_local = MZI_period
    step1 = peakpos-2*MZI_period_local
    MZIstart = max(step1, 1)
    step2 = peakpos+2*MZI_period_local
    MZIend = min(step2, trace_length)
    
    trace_MZI_tofit = mzi_nor[MZIstart:MZIend]
    normalize_fit = trace_MZI_tofit - np.mean(trace_MZI_tofit)
    trace_MZI_phasor = signal.hilbert(normalize_fit)
    phasor_len = len(trace_MZI_phasor)
    phasor_start = round(phasor_len/4)
    phasor_end = round(3 * phasor_len/4)
    print(phasor_start)
    print(phasor_end)
    trace_MZI_phasor = trace_MZI_phasor[phasor_start:phasor_end]
    # print(trace_MZI_phasor)
    phase1 = []
    for i in range(2, len(trace_MZI_phasor)):
        pha = trace_MZI_phasor[i] / trace_MZI_phasor[i-1]
        phase1.append(pha)
        
    MZI_period_local = 2*np.pi / mean(np.angle(phase1))
    
    MZI_max = max(trace_MZI_tofit)
    MZI_pos = np.where(trace_MZI_tofit == MZI_max)

    x = np.arange(MZIstart, MZIend)
    popt_MZI,_ = optimize.curve_fit(sinfit, x, trace_MZI_tofit,
                                 p0=[MZI_baseline, MZI_amp, MZIstart-peakpos+MZI_pos[0][0], MZI_period_local])





print(popt)
print(popt_MZI)

ax4 = plt.figure()
x11 = np.arange(0, len(ch3s_cut))
plt.scatter(x11, ch3s_cut, s=10, c='royalblue', alpha=0.5, label='Data')
plt.plot(lorentz(x11, *popt), 'r--')
plt.scatter(x11, mzi_nor, s=5, c='orange', alpha=0.5, label='Data')
plt.plot(sinfit(x11, *popt_MZI), 'b--')
# plt.plot(lorentz(x11, A=0.5313, cen=6.1352e3, widp=1.6042e6, wid=1.3833e3), 'b--')
plt.show()
# p0 = [0.1*0.016644, 0.016644, 0.00001**2]
# plsq = optimize.leastsq(residuals, p0, args=(ch3s_cut, times_cut))

# ax2 = plt.figure()
# plt.plot(times_cut, ch3s_cut, c='green', linewidth=0.7)
# plt.plot(times_cut, func(times_cut, plsq[0]), c='red')

# plt.show()
#%%