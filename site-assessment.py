'''
This code is to help you plot the model output from the Global Wind Atlas.

Feel free to play around and modify things. 

Check out the source code for wind_stats to understand how things are working. 
https://github.com/jules-ch/wind-stats

Ryan Tonkin
'''

import wind_stats as ws
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
import pandas as pd
from windrose import WindroseAxes, WindAxes
import matplotlib.gridspec as gridspec

density = 1.225 #kg/m3

def nudge(speed):
    '''
    Ensures velocity != 0 by nudging all the speeds by a small amount (improves weibull fit)
    '''

    nudge_factor = 0.01
    speed += nudge_factor
    return speed

def power_density(speed):
    '''
    Calc. power density. 
    in: speed - wind velocity in m/s
    '''
    return 0.5*density*speed**3

def load_wind_atlas_file(fs):
    '''
    Loads a wind atlas file and returns a data structure with all 
    the relevent weibull, roughness and frequency data

    in: fs - file string for GWC file
    out: mod - data structure
    '''
    fp = open(fs, 'r')
    mod = ws.GWAReader.load(fp)
    fp.close()
    return mod

def return_weibull_pars(mod, roughness, height, sector):
    '''
    Returns the weibull parameters for a given elevelation, roughness and sector (wind direction)

    in: roughness - effective roughness length of the terrain
        height - model height above sea level
        sector - direction of wind flow (bucket)
    out: A - weibull scale par
         k - weibull shape par
         f - frequency
    '''

    loc = return_indices(mod,roughness, height, sector)

    return float(mod.A[loc]), float(mod.k[loc]), float(mod.frequency[loc[0],loc[2]])

def return_indices(mod,roughness, height, sector):
    '''
    Returns the data structure indices for a particular roughness, height and sector
    
    in: mod - GWA model data structure
        roughness - effective roughness length of the terrain
        height - model height above sea level
        sector - direction of wind flow (bucket)
    out: (ri, hi, si) - roughness index, height index, sector index
    '''
    try:
        ri = np.where(mod.roughness.data==roughness)[0][0]
    except:
        print(f'There is something wrong with your roughness value, please select a value from {mod.roughness.values}')
        exit()
    try:
        hi = np.where(mod.height.data==height)[0][0]
    except:
        print(f'There is something wrong with your height value, please select a value from {mod.height.values}')
        exit()
    try:
        si = np.where(mod.sector.data==sector)[0][0]
    except:
        print(f'There is something wrong with your sector value, please select a value from {mod.sector.values}')
        exit()

    return (ri, hi, si)

def plot_sector_dists(mod, roughness = 0., height = 10., plot_color="tab:green", mean_color = "tab:red"):
    '''
    Plots the weibull distribution for each sector for a given roughness and height.

    in: mod - GWD model data structure
        roughness - desired roughness
        height - height value to plot
    '''

    # get weibull pars and plot them
    fig,axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 8), sharex=True)

    max = 25 # m/s 
    Nx = 100 
    x = np.linspace(0, max, Nx)
    for sec, ax in zip(mod.sector.values,axs.ravel()):
        
        A, k, f = return_weibull_pars(mod, roughness, height, sec)

        params = (1,k,0,A) #scipy format
        mean = stats.exponweib.stats(*params, moments='m')
        title_str = f'Sector = {sec} deg.'
        
        ax.plot([mean,mean],[0,stats.exponweib.pdf(mean,*params)], color = mean_color, label = 'Mean')
        ax.plot(x, stats.exponweib.pdf(x, *params), color=plot_color, label = 'Weibull')
        textstr = '\n'.join((
                    r'mean$=%.2f$ m/s' %mean,
                    r'freq$=%.2f$  %%' %f,
                    r'A$=%.2f$ m/s' %A,
                    r'k$=%.2f$' %k))


        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', edgecolor = 'gray', facecolor='white', alpha=0.75)
        ax.text(0.585, 0.95, textstr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
        if sec > mod.sector.values[7]: ax.set_xlabel("Velocity (m/s)")

        ax.grid(True)
        ax.set_ylabel("Probability Density")
        ax.set_xlim(0,25)
        ax.set_ylim(bottom=0)
        ax.set_title(title_str, fontsize = 10)
        if sec == mod.sector.values[3]: ax.legend(loc='lower right')

    fig.suptitle(f'Weibull Distributions at: {mod.coordinates} | roughness: {roughness} m | height: {height} m ', fontsize=16) # note these coordinate values are slightly rounded. 

    plt.tight_layout()
    plt.show()

def plot_data(ax, vel, bins=6, Nx=100, bar_color="tab:blue", plot_color="tab:green", mean_color = "tab:red"):
    '''
    Plots wind speed data and fits a weibull curve to it.

    in: ax - plotting axis 
        vel - velocity data array
        bins - number of bins to discretised the data into
        Nx - number of sample points for the plotting of the weibull distribution
        other - plotting pars
    out: ax - modified plotting axis
         mean - mean of the fitted weibull dist.
         params - weibull parameters (a,k,shift,A)
    '''

    hist, bins = np.histogram(vel, bins=bins, density=True)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    
    params = stats.exponweib.fit(vel, floc=0, f0=1)
    mean = stats.exponweib.stats(*params, moments='m')
    x = np.linspace(0, bins[-1], Nx)
    ax.plot([mean,mean],[0,stats.exponweib.pdf(mean,*params)], color = mean_color, label = 'Data: Mean')
    ax.plot(x, stats.exponweib.pdf(x, *params), color=plot_color, label = 'Data: Weibull')
    ax.bar(center, hist, align="center", width=width, 
        color=bar_color, edgecolor = 'k', alpha = 0.5, label = 'Data')
    
    return ax, mean, params

def plot_total_model(ax, model, roughness = 0., height = 10., plot_color="tab:green", mean_color = "tab:red", max_vel = 25):
    '''
    Plots the total weibull distribution (i.e for all directions) for wind speed from the GWA model.
    
    in: ax - matplotlib axis
        model - GWA model results
        roughness - desired roughness
        height - height value to plot
        other plotting vars
    out: ax - modified matplotlib axis
         mean - global mean velocity (i.e., of all directions at a spot)
         params - wiebull distribution parameters for the velocities
    '''

    A, k, f = ws.get_weibull_parameters(model, roughness_length=roughness, height=height)
    params = (1,k,0,A) #put into scipy format
    mean = stats.exponweib.stats(*params, moments='m')

    x = np.linspace(0, max_vel, 100)
    ax.plot([mean,mean],[0,stats.exponweib.pdf(mean,*params)], color = mean_color, label = 'Model: Mean')
    ax.plot(x, stats.exponweib.pdf(x, *params), color=plot_color, label = 'Model: Weibull')
    
    return ax, mean, params

def compare_data_model(df, model, roughness = 0., height = 10, bins=6, Nx=100, bar_color="tab:blue" ):
    '''
    Plots a comparison of the velocity data and the GWA model results
    
    in: df - velocity data imported from the CSV
        model - model results from GWA
        roughness - desired roughness
        height - height above sea level
        bins - number of bins for the histogram
        Nx - number of samples for the plotting of the distributions
        other - plotting vars
    '''
    vel = df['Speed(m/s)']
    fig,ax = plt.subplots(figsize=(12,8))
    ax, mean, params = plot_data(ax, vel,bins, Nx, bar_color, plot_color = "tab:blue", mean_color="tab:red")
    ax, mean_model, params_model = plot_total_model(ax, model, roughness, height, plot_color="tab:green", mean_color = "tab:orange")

    size = 18
    ax.set_xlabel("Wind Speed (m/s)", fontsize = size)
    ax.set_ylabel("Probability Density", fontsize = size)
    ax.legend(fontsize=size) 
    ax.set_title(f'Comparison of model results at {model.coordinates} with wind data at ({data_lat}, {data_lon})', fontsize = size)
    ax.tick_params(axis='both', which='major', labelsize=size)  
    ax.tick_params(axis='both', which='minor', labelsize=size)
    plt.tight_layout()
    
    # Calculate average power denisity in the wind using the weibull parameters 
    Pd_data = calculate_average_power_density(params[3],params[1])
    Pd_model = calculate_average_power_density(params_model[3],params_model[1])

    print(f'Wind Data Raw: The mean velocity is : {round(df["Speed(m/s)"].mean(), 2)} m/s and the mean power density is: {round(df["Pd(W/m2)"].mean(), 2)} W/m2')
    print(f'Wind Data Fit: The mean velocity is : {round(mean, 2)} m/s and the mean power density is: {round(Pd_data, 2)} W/m2')
    print(f'Model Results: The mean velocity is : {round(mean_model, 2)} m/s and the mean power density is: {round(Pd_model, 2)} W/m2')
    
    plt.show()
    return (mean,Pd_data),(mean_model,Pd_model)

def read_met_data(fs):
    '''
    reads met data from a standard (unmodified) CLIFLO .csv file.

    in: fs - file string
    out: df - pandas data frame
    '''
    from pathlib import Path
    global data_lat, data_lon   # lazy coding but makes it easy later on
    
    file = Path(fs)
    if file.exists() and file.is_file and file.suffix not in ['.csv']:
        print("It doesn't look like your file is a .csv. Make sure you download your met data as a csv from CLIFLO.")
        exit()
    
    df = pd.read_csv(fs, nrows=1, skiprows=1)
    data_lat, data_lon = round(df['Lat(dec_deg)'][0],3),round(df['Long(dec_deg)'][0],3)

    df = pd.read_csv(fs, skiprows=8)
    df = df.iloc[:-6]
    
    df['Speed(m/s)'] = df['Speed(m/s)'].apply(nudge)
    df['Pd(W/m2)'] = df['Speed(m/s)'].apply(power_density)

    return df

def plot_data_roses(df, bins = [0,4,8,12,16,20], type = None):
    '''
    Plots different types of wind roses using the WindRose package
    
    in: df - met data (pandas dataframe) 
    out ax - modified axis
    '''

    if type == None:
        type = ['bar', 'contour']

    if 'bar' in type:   
        ax = WindroseAxes.from_ax()
        ax.bar(df['Dir(DegT)'],df['Speed(m/s)'], bins = bins, nsector = 12, normed=True, opening=0.95)
        ax.yaxis.set_major_formatter('{x:.1f} %')
        ax.set_legend()
    
    if 'contour' in type:
        ax = WindroseAxes.from_ax()
        ax.contourf(df['Dir(DegT)'],df['Speed(m/s)'], bins = bins,  nsector = 12, normed=True)
        ax.yaxis.set_major_formatter('{x:.1f} %')
        ax.set_legend()

    return ax

def plot_model_rose(velocity_data, model, roughness, height,bins=[0,4,8,12,16,20]):
    '''
    Plots the wind rose from the GWA model results.

    in: velocity_data - velocity data from CLIFLO
        model - GWA model results
        roughness - your chosen roughness
        height - height about ground
        bins - velocity bins
    '''
    A, k, f = ws.get_weibull_parameters(model, roughness_length=roughness, height=height)

    # fiddle with model results to make them fit WindroseAxes weird plotting methods...
    f = np.append(f,f[0])
    sec = (np.append(model.sector.values,model.sector.values[0])+90)*np.pi/180
    sec = np.flip(sec)

    ax = plot_data_roses(velocity_data,bins, type='contour')
    
    # get frequencies for each wind direction if required
    table = ax._info['table']
    wd_freq = np.sum(table, axis=0)

    ax.plot(sec,f, label = 'Model', color='k', linewidth = 3)
    lines = ax.get_lines()
    ax.set_title("Comparison of CLIFLO Data with GWA Results")
      
    fig, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    rect = ax2[0].get_position()
    ax2[0].remove()
    rect2 = ax2[1].get_position()
    ax2[1].remove()

    ax3 = WindroseAxes(fig, rect)
    fig.add_axes(ax3)
    ax3.contourf(velocity_data['Dir(DegT)'],velocity_data['Speed(m/s)'], bins = bins,  nsector = 12, normed=True)
    ax3.yaxis.set_major_formatter('{x:.1f} %')
    ax3.set_legend(bbox_to_anchor=(-0.21, 0))
    
    fig.add_subplot(1,2,2,projection = 'polar')
    ax4 = fig.gca()
    ax4.set_theta_direction(-1)
    ax4.set_theta_zero_location('N')
    sec2 = (np.append(model.sector.values,model.sector.values[0]))*np.pi/180

    ax4.plot(sec2,f, label = 'Model', color='k', linewidth = 3)
    ax4.legend(loc='lower left')
    ax4.set_xticks(ax4.get_xticks())
    ax4.set_xticklabels(['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W'])

    
    y_lim4 = ax4.get_ylim()
    y_lim3 = ax3.get_ylim()

    if y_lim4[1]>=y_lim3[1]:
        ax3.set_ylim(y_lim4)
        rng = np.arange(0, y_lim4[1], 2)
    else:
        ax4.set_ylim(y_lim3)
        rng = np.arange(0, y_lim3[1], 2)

    ax4.set_yticks(rng)
    ax4.set_yticklabels(rng)
    ax3.set_yticks(rng)
    ax3.set_yticklabels(rng)
    ax3.yaxis.set_major_formatter('{x:.1f} %')
    ax4.yaxis.set_major_formatter('{x:.1f} %')

    plt.show()
    
def calculate_average_power_density(A, k):
    '''
    Calculate the average power density of the wind (assuming a weibull distribution)
    in: A - weibull scale parameter
        k - weibull shape parameter
    out (float) mean power density given weibull distribution (W/m2)

    See: Serban, A., Paraschiv, L. S., Paraschiv, S., Assessment of wind energy potential based on
        Weibull and Rayleigh distribution models, Energy Reports, 6, 250-276.
    '''
    
    return 0.5*density*A**3*special.gamma((k+3)/k)

def print_incon(model,df):
    '''
    Prints the data structures.
    
    in: model - GWA model file
        df - pandas data structure from CLIFLO csv  file.
    '''
    print("")
    print('################################ Model Results ################################')
    print(model)
    print('###############################################################################')
    print('')
    print('############################################## Met Data ##############################################')
    print(df)
    print('######################################################################################################')
    print('')
    
def __main__():
    '''
    Runs all the plotting and data processing scripts
    '''
    
    model_fs = 'gwa3_gwc_mxirhhqr.lib'        # Note: must be a .lib file in the format downloaded from the GWA.
    data_fs = 'station_list.csv'      # Note: must be a .CSV file in the format from CLIFLO.

    # load data and model results
    model = load_wind_atlas_file(model_fs)
    df = read_met_data(data_fs)
    # print out of the model and data. Make sure things look alright. 
    print_incon(model,df)
    
    roughness = 0.0         # Select the (closest) roughness of the terrain 
    height = 10             # Select hub height
    bins = [0,4,8,12,16,20] # Histogram bins

    
    plot_sector_dists(model,roughness,height)
    data_means, model_means = compare_data_model(df, model, roughness, height, bins=35)
    
    plot_data_roses(df, bins)
    plt.show()

    plot_model_rose(df, model, roughness, height, bins)

__main__()