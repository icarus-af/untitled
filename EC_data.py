# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:01:06 2023

@author: icarus
"""

from patsy import dmatrices

import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import preprocessing

from scipy.signal import find_peaks

import pandas as pd
import numpy as np

import dash
from dash import html, dash_table
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from uncertainties import ufloat
from uncertainties.umath import *

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

from scipy.interpolate import UnivariateSpline
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import matplotlib as mpl


# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Times New Roman']
# rcParams['font.sans-serif'] = ['Lato']
rcParams['axes.labelpad'] = 15
plt.rcParams['font.size'] = 15
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["font.weight"] = "bold"

colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF' , '#FF5252']
black, red, yellow, green, blue, pink = '#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'

## 6 cores ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

## 6 cores ['preto', 'vermelho', 'amarelo', 'verde', 'azul', 'rosa']

#%%

def getcm(n, cmap):
    """
    Gets a sequence of n colors from cmap
    """    
    if n == 6:

        colors = ['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF']

    else:
        
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    
    return colors

def plot(item, colors=['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'], Emin=-100000, Emax=100000, b=None, scan=1, sep=',', tech='cv'):
    """
    Reads an {item}.txt file filtered by {scan} column and outputs {peak_o, peak_r, ioveri, dEp}
    Different conditions based on {tech} = 'cv' or 'swv'
    if b           = 'b' then blank is considered on analysis
    Emax and Emin  = potential ranges for peak detection
    """
    
    # Read data for Cyclic Voltammetry analysis
    if tech == 'cv':
    
        df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['WE(1).Potential (V)']
        i = df['WE(1).Current (A)']*1e6
        i_f = i
          
        if b is not None:
            
            df_b = pd.read_csv(f"{item}-b.txt", sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).Current (A)']*1e6
            i_f = i - i_b
            
    # Read data for Squared Wave Voltammetry analysis
    
    elif tech == 'swv':
        
        df = pd.read_csv(f'{item}-a.txt', sep=sep)
        
        if 'Scan' in df.columns:
        
            df = df[df['Scan'] == scan]
        
        E = df['Potential applied (V)']
        i = df['WE(1).δ.Current (A)']*1e6
        i_f = i
          
        if b is not None:
            
            df_b = pd.read_csv(f'{item}-b.txt', sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b['WE(1).δ.Current (A)']*1e6
            i_f = i - i_b
            
    # Calculate pertinent data analysis
           
    peak_o = i_f[E < Emax][E > Emin].max()
    peak_r = i_f[E < Emax][E > Emin].min()
    ioveri = abs(peak_o/peak_r)
    
    E_o = E[i_f[E < Emax][E > Emin].idxmax()]
    E_r = E[i_f[E < Emax][E > Emin].idxmin()]
    dEp = abs(E_o-E_r)
    
    # First figure (0) of the full voltammogram
    
    plt.figure(0)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=.7)
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    # plt.title('')
    # plt.ylim(-210,140)
    plt.legend(fontsize=10, frameon=False)
    
    # plt.savefig('20230808.png', dpi=200, bbox_inches='tight')
    
    # If blank is pertinent: figure (1) of the blank
    
    if b is not None:
    
        plt.figure(1)
        
        plt.plot(E, i_b, color=colors[item], label=f'{item}-branco', alpha=.7)
        
        plt.xlabel('E / V vs Ag/AgCl')
        plt.ylabel('I / $\mu$A')
        # plt.title('')
        # plt.ylim(-210,140)
        plt.legend(fontsize=10, frameon=False)
        
        # plt.savefig('02082023-b.png', dpi=200, bbox_inches='tight')
    plt.figure(2)
    
    print('PEAKPEAK')
    print(E_o)
    print('-'*90)
    print(peak_o)
    print('-'*90)
    peaks, _ = find_peaks(i_f, prominence=1)
    print(peaks)
    print('-'*90)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=1)
    plt.axvline(x=E[peaks[0]], linewidth=2, color='red')
    
    ### plt.axvline(x=E_r, ymin=peak_r-0.5*peak_r, ymax=peak_r+0.5*peak_r,) ###
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    plt.title('peak detection')
    
    return peak_o, peak_r, ioveri, dEp

def plot(item, colors=['#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'], b=None, scan=1, sep=',', tech='cv', specialpotential=None, specialcurrent=None):
    """
    Reads an {item}.txt file filtered by {scan} column and outputs {peak_o, peak_r, ioveri, dEp}
    Different conditions based on {tech} = 'cv' or 'swv'
    if b           = 'b' then blank is considered on analysis
    Emax and Emin  = potential ranges for peak detection
    """
    
    # Read data for Cyclic Voltammetry analysis
    
    if tech == 'cv': 
        current = 'WE(1).Current (A)'
    if tech == 'swv':
        current = 'WE(1).δ.Current (A)'
    if specialcurrent is not None:
        current = specialcurrent
        
    potential='WE(1).Potential (V)'
    if specialpotential is not None:
        potential = specialpotential
    
    df = pd.read_csv(f'{item}-a.txt', sep=sep)
    
    E = df[potential]
    i = df[current]*1e6
    i_f = i 
    
    if 'Scan' in df.columns:
    
        df = df[df['Scan'] == scan]
    
        E = df[potential]
        i = df[current]*1e6
        i_f = i 

          
        if b is not None:
            
            df_b = pd.read_csv(f"{item}-b.txt", sep=sep)
            
            if 'Scan' in df_b.columns:
            
                df_b = df_b[df_b['Scan'] == scan]
                
            i_b = df_b[current]*1e6
            i_f = i - i_b
            
    # Calculate pertinent data analysis
    
    dy = np.gradient(i_f, E)
    
    peak_indices = np.concatenate([find_peaks(i, prominence=1)[0], find_peaks(-i, prominence=1)[0]])
           
    peak_x = [ E[item] for item in peak_indices ]
    peak_y = [ i_f[item] for item in peak_indices ]
    
    if len(peak_indices) != 1:
    
        ioveri = abs(peak_y[0]/peak_y[1])
        E_o = peak_x[0]
        E_r = peak_x[1]
        dEp = abs(E_o-E_r)
    
    # First figure (0) of the full voltammogram
    
    plt.figure(0)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=.7)
    
    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    # plt.title('')
    # plt.ylim(-210,140)
    plt.legend(fontsize=10, frameon=False)
    
    # plt.savefig('20230808.png', dpi=200, bbox_inches='tight')
    
    # If blank is pertinent: figure (1) of the blank
    
    if b is not None:
    
        plt.figure(1)
        
        plt.plot(E, i_b, color=colors[item], label=f'{item}-branco', alpha=.7)
        
        plt.xlabel('E / V vs Ag/AgCl')
        plt.ylabel('I / $\mu$A')
        # plt.title('')
        # plt.ylim(-210,140)
        plt.legend(fontsize=10, frameon=False)
        
        # plt.savefig('02082023-b.png', dpi=200, bbox_inches='tight')
    plt.figure(2)
    
    plt.plot(E, i_f, color=colors[item], label=f'{item}', alpha=1)
    plt.scatter(peak_x, peak_y, linewidth=2, color=red)

    plt.xlabel('E / V vs Ag/AgCl')
    plt.ylabel('I / $\mu$A')
    plt.title('peak detection')
    
    return E, i


#%%
def peaks(reps, cmap, Emin=-100000, Emax=100000, b=None, scan=1, sep=',', tech='cv'):
    """
    Runs a loop of plots() functions and prints the results accordingly
    colors         = use getcm() to determine colors
    """
    
    colors = getcm(reps, cmap)
    
    t = stats.t.ppf(1-0.025, reps-1)
    
    if b is not None:
        
        peaks = np.array( [plot(item, colors=colors, Emin=Emin, Emax=Emax, tech=tech, sep=sep, scan=scan, b='b') for item in np.arange(0,reps)] ).T
    
    else: 
        
        peaks = np.array( [plot(item, colors=colors, Emin=Emin, Emax=Emax, tech=tech, sep=sep, scan=scan) for item in np.arange(0,reps)] ).T
    
    anodic = peaks[0]
    
    mean_a = anodic.mean()
    std_a = anodic.std(ddof=1)
    ci_a = std_a*t/(reps**(0.5))
    err_a = round(ci_a/mean_a*100, 0)
    
    print('PEAK ANODIC CURRENTS:')
    print('-'*40)
    print('Array with all peaks: ')
    print(anodic)
    print('Mean = ', mean_a)
    print('Std = ', std_a)
    print('CI = ', mean_a, '+-', ci_a)
    print('Error = ', err_a, '%')
    
    cathodic = peaks[1]
    
    mean_c = cathodic.mean()
    std_c = cathodic.std(ddof=1)
    ci_c = std_c*t/(reps**(0.5))
    err_c = round(ci_c/mean_c*100, 0)
    
    print('-'*40)
    print('PEAK CATHODIC CURRENTS:')
    print('-'*40)
    print('Array with all peaks: ')
    print(cathodic)
    print('Mean = ', mean_c)
    print('Std = ', std_c)
    print('CI = ', mean_c, '+-', ci_c)
    print('Error = ', err_c, '%')
    
    ioveri = peaks[2]
    
    mean_ioveri = ioveri.mean()
    std_ioveri = ioveri.std(ddof=1)
    ci_ioveri = std_ioveri*t/(reps**(0.5))
    err_ioveri = round(ci_ioveri/mean_ioveri*100, 0)
    
    print('-'*40)
    print('ANODIC/CATHODIC CURRENT RATIO:')
    print('-'*40)
    print('Array with all peaks: ')
    print(ioveri)
    print('Mean = ', mean_ioveri)
    print('Std = ', std_ioveri)
    print('CI = ', mean_ioveri, '+-', ci_ioveri)
    print('Error = ', err_ioveri, '%')
    
    dEp = peaks[3]
    
    mean_dEp = dEp.mean()
    std_dEp = dEp.std(ddof=1)
    ci_dEp = std_dEp*t/(reps**(0.5))
    err_dEp = round(ci_dEp/mean_dEp*100, 0)
    
    print('-'*40)
    print('ANODIC/CATHODIC CURRENT RATIO:')
    print('-'*40)
    print('Array with all peaks: ')
    print(dEp)
    print('Mean = ', mean_dEp)
    print('Std = ', std_dEp)
    print('CI = ', mean_dEp, '+-', ci_dEp)
    print('Error = ', err_dEp, '%')
    
    print('='*50)
    print('='*50)
    print('SUMMARY:')
            
    print('i_a:     ', round(mean_a, 4), '+-', round(ci_a, 4), 'uA', f"        ({err_a})%")
    print('i_c:     ', round(mean_c, 4), '+-', round(ci_c, 4), 'uA', f"        ({err_c})%")
    print('i_a/i_c: ', round(mean_ioveri, 4), '+-', round(ci_ioveri, 4), f"        ({err_ioveri})%")
    print('dEp:     ', round(mean_dEp,4), '+-', round(ci_dEp, 4), 'V', f"        ({err_dEp})%")
    
    print('='*50)
    print('='*50)
    
    return peaks

#%%

def reg_model(y, x, method='OLS', err=None):
    """
    Fits and report a linear model ({method} = 'OLS' or 'WLS') for y(x) returning (model, coefs, r2, r2_adj)
    """
    
    x = sm.add_constant(x)
    
    if method == 'OLS':
    
        model = sm.OLS(y, x).fit()
        
    elif method == 'WLS':
        
        weights =  1/(err**2)
        model = sm.WLS(y, x, weights=weights).fit()
        
    coefs = np.array(model.params)    
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    
    print(model.summary())
    
    return model, coefs, r2, r2_adj

def linreg_plot(x, y, coefs, xlabel='x', ylabel='y', title=''):
    """
    Create a scatter plot with a linear regression line.
    
    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.
    coefs (array-like): Linear regression coefficients [intercept, slope].
    xlabel (str, optional): X-axis label (default: 'x').
    ylabel (str, optional): Y-axis label (default: 'y').
    title (str, optional): Plot title.
    
    Example:
    linreg_plot(x_data, y_data, [intercept, slope], xlabel='Time', ylabel='Distance', title='Linear Regression')
    plt.show()
    """
    x = np.array(x)
    y = np.array(y)
    
    xp = np.linspace(x.min(), x.max(), 100)
    yp = coefs[0] + coefs[1]*xp
    
    plt.scatter(x, y, color=black)
    plt.plot(xp, yp, color=red)
    
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    
    plt.title(f'{title}')
    
    return

#%% OUTLIER TEST

def grubbs_stat(y):
    std_dev = np.std(y, ddof=1)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations/ std_dev
    print("Grubbs Statistics Value : {:.4f}".format(Gcal))
    return Gcal, max_ind

def calculate_critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    print("Grubbs Critical Value: {:.4f}".format(critical_value))
    return critical_value

def check_G_values(Gs, Gc, inp, max_index):
    TEST = 0
    if Gs > Gc:
        print('Position: ', max_index)
        print('{} - OUTLIER. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        TEST = 1
    else:
        print('{}. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        TEST = 0        
    return TEST

def ESD_Test(input_series, alpha, max_outliers):
    n = 0
    for iterations in range(max_outliers):
        Gcritical = calculate_critical_value(len(input_series), alpha)
        Gstat, max_index = grubbs_stat(input_series)
        check = check_G_values(Gstat, Gcritical, input_series, max_index)
        input_series = np.delete(input_series, max_index)
        n += check
    print(f'Found {n} outliers!')
    
    return n

def remove_outliers(data, n):
    for item in np.arange(n):
        posmax = np.argmax(data)
        data = np.delete(data, posmax)
    return data

def boxplots(my_dict):
    
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    
    return fig
