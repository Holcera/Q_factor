# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:09:06 2022

@author: Heeeg
"""
import sys
print(sys.path)
import csv
import math
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLPD.MLtools as pmlt
from read_csv_all import read_csv
from ctypes.wintypes import PLARGE_INTEGER
from gettext import find
from inspect import trace
from lib2to3.pgen2.token import PLUSEQUAL
from sqlite3 import complete_statement, paramstyle
from turtle import end_fill
from matplotlib import rcParams
from matplotlib.cbook import ls_mapper
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
from scipy import optimize, signal
from scipy import constants as C

#%%
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20

def cm2inch(*tupl):
    inch = 2
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
#%%
# indexCsv = input('\nData_Index:')
# filepath = 'F:/tek'+indexCsv+'.csv'
filepath = r"C:\Users\WANG\OneDrive\KEIO\Wang\OSC\W07.CSV"
#%%
data_raw = read_csv(filepath)

for i in range(len(data_raw.columns)):
    data_raw.rename(columns={data_raw.columns[i]: f"ch{i}"}, inplace=True) #CH0:time

time = data_raw.ch0.tolist()
trans = data_raw.ch1.tolist()
MZI = data_raw.ch2.tolist()
mid = trans

#%% nor
# trans_min, trans_max = pmlt.envPeak(np.array(trans), delta=0.05, sg_order=0)
# trans_nor =trans / trans_max
# trans = trans_nor

# mid_min, mid_max = pmlt.envPeak(np.array(mid), delta=0.05, sg_order=0)
# mid_nor =mid / mid_max
# mid = mid_nor
#%%
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7))
ax1.set(facecolor='#FFFFCC')

x0 = pd.array(range(0, len(trans)))
trans = pd.array(trans, dtype=float)
line1, = ax1.plot(x0, trans, '.', label='trans_Data')
ax1.set_xlim(x0[0], x0[-1])
ax1.set_ylim(trans.min(), trans.max())
ax1.legend(loc=3)
ax1.set_title('Drag mouse to select')

ax2.set(facecolor='#FFFFCC')
line2, = ax2.plot(x0, trans, '.', label='trans_Data')
ax2.set_xlim(x0[0], x0[-1])
ax2.set_ylim(trans.min(), trans.max())
ax2.legend(loc=3)

#%%
def onselect(xmin, xmax):
    # global rangeList
    # rangeList = []
    indmin, indmax = np.searchsorted(x0, (xmin, xmax))
    indmax = min(len(x0) - 1, indmax)
    
    # for i in range(int(xmin), int(xmax)):
    #       rangeList.append(i)
    
    thisx = x0[indmin:indmax]
    thisy = trans[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw()
  
def onselect2(xmin, xmax):
    global rangeList
    rangeList = []
    indmin, indmax = np.searchsorted(x0, (xmin, xmax))
    indmax = min(len(x0) - 1, indmax)
    
    for i in range(int(xmin), int(xmax)):
          rangeList.append(i)
#   rangeList = append([indmin,indmax])
    thisx = x0[indmin:indmax]
    thisy = trans[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw()
  
def get_pos(event):
    if rangeList != []:
        rangeStart = rangeList[0]
        rangeEnd = rangeList[-1]
        # print('Cutstart:', rangeStart)
        # print('Cutend:', rangeEnd)
        # print('rangelen:', rangeEnd-rangeStart)
        give_pos(rangeStart, rangeEnd)
    else:
        print('No valid values were obtained')

def give_pos(ini, end1):
    # trans_cut = []
    # for i in trans[ini:end1]:
    #     trans_cut.append(i)
    a = np.array(time)
    times_inis = np.where(a == time[ini])
    times_ends = np.where(a == time[end1])
    if len(times_inis[0]) > 1 or len(times_ends[0]) > 1:
        time_ini = times_inis[0][1]
        time_end = times_ends[0][1]
    else:
        time_ini = times_inis[0][0]
        time_end = times_ends[0][0]
    
    global trans_cut
    global ch2s_cut
    trans_cut = mid[time_ini:time_end]
     
    
    ch2s_cut = ch2s[time_ini:time_end]
    get_mzipara(time_ini, time_end)
    
    get_shift()
    # print(time_ini)
    # print(time_end)
    # print(times[ini])   
    # print(times[end1])
    # get_shift(trans_cut)
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
    return A*(1-widp/((x-cen)**2+wid**2))

def lorentz_fano(x, A, cen, widp, wid, f0):
    return (A*((x-cen+f0)**2+wid**2-widp))/((x-cen)**2+wid**2)

def sinfit(x, A0, A, x0, T):
    return A0+A*np.cos((x-x0)/T*2*C.pi)
#%%
# lw1 = 6
def get_shift():
    # print(trans_cut)
    global trans_filter
    global trans_shift
    trans_len = len(trans_cut)
    trans_filter = scipy.signal.savgol_filter(trans_cut, 21, 1)
    
    for i in range(1, trans_len):
        x1 = trans_filter[i]
        x2 = trans_filter[i-1]*(1-0.5/trans_len)
        trans_filter[i] = comp_max(x1, x2)

    for i in range(trans_len-2, -1, -1):
        x1 = trans_filter[i]
        x2 = trans_filter[i+1]*(1-0.5/trans_len)
        trans_filter[i] = comp_max (x1, x2)
    
    trans_shift = trans_cut / trans_filter
    para_loop_get()
# cutoff_pos = round(trans_len/100)
# trans_shift[0:cutoff_pos+1]
# trans_shift[1:cutoff_pos:1]
    # print(trans_filter)
    # print(trans_shift)
    # print(trans_cut)
def para_loop_get(threshold=0.80):
    trans_len = len(trans_cut)
    while True:
        trans = min(trans_shift)
        peakpos = list(trans_shift).index(trans)
        if trans > threshold:
            break
        
        Q_baseline = trans_filter[peakpos]
        for i in range(peakpos, len(trans_cut)):
            # print(peakpos)
            # print(trans_cut[peakpos:])
            trace_Q_basestop = trans_filter[i]*(1+trans)/2
            if trans_cut[i] > trace_Q_basestop:
                if int(i)-peakpos <= 3:
                    lw1 = 6
                    break
                else:
                    lw1 = 2 * (int(i)-peakpos)
                    break
            else:
                continue
    
        peak_base_max = trans_cut[peakpos]
        # print(peakpos)
        # print(peak_base_max)
        step_lw = peakpos+10*lw1
        peakends = range(peakpos, step_lw)
        
        if step_lw > trans_len:
            peakends = range(peakpos, trans_len)
        
        for peakend in peakends:
            peak_base_max = comp_max(peak_base_max, trans_cut[peakend])
            step = peak_base_max-(1-threshold)*Q_baseline
            
            if trans_cut[peakend] < step:
                break
    
        peak_base_max = trans_cut[peakpos]
        peak_step = peakpos-10*lw1
        
        if peak_step < 1:
            peak_step = 1 
        
        peakstarts = np.arange(peakpos, peak_step, -1)
        
        for peakstart in peakstarts:
            peak_base_max = comp_max(peak_base_max, trans_cut[peakstart])
            step = peak_base_max-(1-threshold)*Q_baseline
            
            if trans_cut[peakstart] < step:
                break

        trans_shift[peakstart:peakend] = 1
        tofit = trans_cut[peakstart:peakend:1]
        x = np.arange(peakstart, peakend)
        # for ind, val in enumerate(times_cut):
        #     x.append(ind)
        pfit,_ = optimize.curve_fit(lorentz, x, tofit, 
                                    p0=[Q_baseline, peakpos, (1-trans)*(lw1**2)/4, lw1/2], 
                                    maxfev=4500)
        
        if Q_fano:
            pfit,_ = optimize.curve_fit(lorentz_fano, x, tofit, 
                                        p0=[pfit[0], pfit[1], abs(pfit[2]), pfit[3], 0], 
                                        maxfev=6500)
        
        
        mzi_fit(peakpos)
        
    print('Params:A, cen, widp, wid:\n', *pfit)
    
    if Q_fano:
        plot_fit_fano(pfit, trans_cut)
    else:
        plot_fit(pfit, trans_cut, *pfit_MZI)
       
    return pfit

def get_mzipara(time_ini, time_end):
    global MZI_period
    global MZI_amp
    global MZI_base
    global mzi_nor
    
    mzi_arry = np.array(ch2s)
    trans_len = len(trans_cut)
    print(mzi_arry)
    mzi_min, mzi_max = pmlt.envPeak(mzi_arry, 
                                    delta=0.004, 
                                    sg_order=0,
                                    smooth=0.05
                                    )
    mzi_max = mzi_max[time_ini:time_end]
    mzi_min = mzi_min[time_ini:time_end]
    mzi_nor = ((ch2s_cut-mzi_min) / (mzi_max-mzi_min)) / 600 #nor
    
    MZI_peak = max(mzi_nor)
    MZI_base = np.median(mzi_nor)
    MZI_pos = np.where(mzi_nor == MZI_peak)
    MZI_amp = MZI_peak - MZI_base
    
    for i in range(MZI_pos[0][0], trans_len):
        if mzi_nor[i] < MZI_base:
            MZI_period = 4 * (i-MZI_pos[0][0]+1)
            break
        else:
            continue

def mzi_fit(peakpos):
    global pfit_MZI
    
    trans_len = len(trans_cut)
    
    MZI_period1 = MZI_period
    step1 = peakpos-2*MZI_period1
    MZIstart = max(step1, 1)
    step1 = peakpos+2*MZI_period1
    MZIend = min(step1, trans_len)
    
    ch2s_cut_tofit = mzi_nor[MZIstart:MZIend]
    nor_fit = ch2s_cut_tofit - np.mean(ch2s_cut_tofit)
    ch2s_cutphase = signal.hilbert(nor_fit)
    pha_len = len(ch2s_cutphase)
    pha_start = round(pha_len/4)
    pha_end = round(3 * pha_len/4)
    # print(pha_start)
    # print(pha_end)
    ch2s_cutphase = ch2s_cutphase[pha_start:pha_end]
    # print(ch2s_cutphase)
    phase1 = []
    for i in range(2, len(ch2s_cutphase)):
        pha = ch2s_cutphase[i] / ch2s_cutphase[i-1]
        phase1.append(pha)
        
    MZI_period1 = 2*np.pi / np.mean(np.angle(phase1))
    
    MZI_max = max(ch2s_cut_tofit)
    MZI_pos = np.where(ch2s_cut_tofit == MZI_max)

    x = np.arange(MZIstart, MZIend)
    pfit_MZI,_ = optimize.curve_fit(sinfit, x, ch2s_cut_tofit,
                                 p0=[MZI_base, MZI_amp, MZIstart-peakpos+MZI_pos[0][0], MZI_period1],
                                 maxfev=2000)
    
    
def plot_fit(pfit, trans_cut, *pfit_MZI):
    plt.figure(figsize=(8, 6))
    x11 = np.arange(0, len(trans_cut))
    freq_1 = (len(x11)/pfit_MZI[3]*20) #MHz
    freq_delta = freq_1 / len(x11)
    # print(len(x11))
    print(pfit)
    # pfit_save = input("\nSave pfit?\n(y/n):" )
    # if pfit_save == 'y':
    #     data_fit = pd.DataFrame()
    #     data_fit['fit'] = pfit
    #     data_fit.to_csv(r'C:\Users\H\Documents\Lab_BUPT\Data\Convert\fit_para.csv', index=False)
    
    x11_1 = int(len(x11)/200)
    freq_tol = [] 
    x11_4 = []
        
    for i in range(0, x11_1+1):
        x11_3 = i*200
        freq_2 = round(x11_3*freq_delta, 0)
        x11_4.append(x11_3)
        freq_tol.append(freq_2)
    
    
    plt.scatter(x11, trans_cut, s=30, c='darkslateblue', alpha=0.5, label='Cavity Exp.')
    plt.plot(lorentz(x11, *pfit), c='red', label='Lorentz fitting', lw=1)
    plt.scatter(x11, mzi_nor, s=15, c='orange', alpha=0.7, label='MZI Exp.')
    plt.plot(sinfit(x11, *pfit_MZI), 'k--', label='Sine fitting') 
    
    # plt.title(label='λ0=1550.11, Q_load=1.072e8, Q_0=1.424e8')
    plt.legend(loc='center right', frameon=False, fontsize=19)
    plt.xticks(x11_4, freq_tol)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('Frequency (MHz)')
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    # plt.tight_layout
    
    
    plt.show()
    calculate_Q(times, pfit, *pfit_MZI)   

def plot_fit_fano(pfit, trans_cut):
    plt.figure()
    x11 = np.arange(0, len(trans_cut))
    plt.scatter(x11, trans_cut, s=10, c='royalblue', alpha=0.7, label='trans_Data')
    plt.plot(lorentz_fano(x11, *pfit), 'r--', label='trans_Fit')
    plt.scatter(x11, mzi_nor, s=5, c='orange', alpha=0.5, label='mzi_Data')
    plt.plot(sinfit(x11, *pfit_MZI), 'b--', label='mzi_Fit')    
    plt.legend(loc=7, bbox_to_anchor=(1, 0.5))
    plt.show()
    calculate_Q(times, pfit, *pfit_MZI)
    
# def plot_fit_mzi(pfit_MZI, trans_cut):
#     plt.figure()
#     x11 = np.arange(0, len(trans_cut))
#     plt.plot(x11, ch2s_cut, 'g')
#     plt.plot(sinfit(x11, *pfit_MZI), 'b--')
    
#     plt.show()
        
def calculate_Q(times, pfit, *pfit_MZI):
    f_hz = 50 #Hz
    PZT_perV = 10e9 #GHz
    vpp = 3
    lambda_cen = 1550e-9
    mzi_D1 = 20 #MHz
    freq_cen = C.c / lambda_cen

    try:
        k0 = abs(pfit[3]) + math.sqrt((pfit[3]**2 - pfit[2]))
    except ValueError:
        print('\nFitting Error!!!!!!')
        k0 = abs(pfit[3]) + math.sqrt(abs((pfit[3]**2 - pfit[2])))

    k_l = 2 * abs(pfit[3])
    # print(k_l)
    Q_load = freq_cen / (k_l/pfit_MZI[3]*mzi_D1*1e6)
    Q_0 = freq_cen / (k0/pfit_MZI[3]*mzi_D1*1e6)
    # else:
    #     ts=[]
    #     for i in range(len(times)-1):
    #         t=times[i+1]-times[i]
    #         ts.append(t)
    
        # t_mean = np.mean(ts)
        # print('time_interval:''%e'%t_mean)
        # print(t_mean)
    
        delta_t_l = t_mean * (k_l-1)
        delta_t_0 = t_mean * (k0-1)
        delta_f = 2*C.pi * vpp * PZT_perV * delta_t_l * (0.5 * int(f_hz))
        delta_f_0 = 2*C.pi * vpp * PZT_perV *delta_t_0 * (0.5 * int(f_hz))
        # print(delta_f)
        Q_load = freq_cen / float(delta_f)
        Q_0 = freq_cen / float(delta_f_0)

    print('\nQ_load:''%e'%Q_load)
    print('Q_0:''%e'%Q_0)
    
#%%
# f_hz, PZT_perV, vpp, lambda_cen = input('Scan频率, PZT_perV(HZ), Vpp, 中心波长(单位SI):').split()
# f_hz = int(f_hz)
# PZT_perV = int(float(PZT_perV))
# vpp = int(vpp)
# lambda_cen = float(lambda_cen)
    
button_pos = plt.axes([0.43, 0.02, 0.14, 0.04])
b1 = Button(button_pos, 'Get Q')
b1.on_clicked(get_pos)
span1 = SpanSelector(ax1, onselect, 'horizontal', useblit=True)
span2 = SpanSelector(ax2, onselect2, 'horizontal', useblit=True)

plt.show()
#%%