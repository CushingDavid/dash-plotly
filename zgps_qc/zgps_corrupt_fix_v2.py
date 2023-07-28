
## Import required libraries: 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import *
from tkinter import ttk 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

## User defined function for tkinter output
## from functions_tkinter_output_buttons import *
## EDIT the day range here for calculating the model
day_min = 1
day_max = 10
##Read files in the directory: 
files = glob.glob(os.path.join("./V1G1 Files/V1G1_zDGPS*"))

df = pd.DataFrame()
## For loop to read in the rest of the att files:    
for i in range(0,len(files)):
    data = pd.read_csv(files[i], skiprows=9, sep=",", header=None)
    df_temp = pd.DataFrame(data)
    day =  files[i][-3:]
    df_temp['day'] = day
    df = pd.concat([df,df_temp], axis=0)

### set up the column headers :
cols =["time","proc_lat","proc_lon","proc_geo_ht","proc_sph_ht","raw_lat","raw_lon","raw_geo_ht","raw_sph_ht","pitch","roll","heave","day"]
df.columns =cols
## Define a day column in df
df['day'] = df['day'].astype(int)
df['proc_lat'] = df['proc_lat'].astype(float)
df['proc_lon'] = df['proc_lon'].astype(float)
df['proc_geo_ht'] = df['proc_geo_ht'].astype(float)
df['proc_sph_ht'] = df['proc_sph_ht'].astype(float)

## Set the time to datetime - don't worry about the date, as we have a day column
df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S")
## Drop all the processed spikes from the df for model building .... 
df1 = df[ (df['proc_geo_ht'] <= 2000) & (df['proc_geo_ht'] >= -2000)]

## Lst of all the days for GUI
days = (df1['day'].unique()).tolist()
## Model building:
df_mod = df1.loc[(df1['day'] >= day_min) & (df1['day'] <= day_max)]
X = df_mod.drop(['time', 'proc_lat', 'proc_lon', 'proc_sph_ht', 'proc_geo_ht', 'day' ],axis= 1)
y = df_mod['proc_geo_ht']

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
  
# fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
  
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
  
# predicting on test data-set
y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
# evaluating the model on training dataset
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
r2_train = r2_score(y_train, y_train_predicted)
  
# evaluating the model on test dataset
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)
  
print("The model performance for the training set")
print("-------------------------------------------")
print("RMSE of training set is {}".format(rmse_train))
print("R2 score of training set is {}".format(r2_train))
  
print("\n")
 
print("The model performance for the test set")
print("-------------------------------------------")
print("RMSE of test set is {}".format(rmse_test))
print("R2 score of test set is {}".format(r2_test))

##################################################################################################
### Make these functions external ####
### function to output df into fwf format 
from tabulate import tabulate

def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)

pd.DataFrame.to_fwf = to_fwf

### Functions for tkinter output buttons: 
def apply_poly_reg_model2():
    btn = tk.Label(window)
    btn.grid(row=0, column=0, padx=20, pady=10)

    day = n.get()
    day = int(day)
    df_temp = df.loc[(df['day'] == day)]
    X = df_temp.drop(['time', 'proc_lat', 'proc_lon', 'proc_sph_ht', 'proc_geo_ht', 'day' ],axis= 1)
    y = df_temp['proc_geo_ht']
    y_predict = poly_model.predict(poly_features.fit_transform(X))
    fig = plt.figure(figsize=(15, 12))
    plt.subplot(1, 2, 1)  # row 1, column 2, count 1
    plt.scatter(y, y_predict, s=0.4, c='red')
    # plt.scatter(y_predict, s=0.4, c='blue') 
    plt.xlabel(f'Day {day} Real Data' )
    plt.ylabel(f'Day {day} model prediction')   
    plt.xlim(y.min(), y.max())
    plt.ylim(y_predict.min(), y_predict.max())
    
    # row 1, column 2, count 2
    plt.subplot(1, 2, 2)
    plt.plot(y)
    plt.plot(y_predict, alpha=0.5) 
    plt.xlabel(f'Sample' )
    plt.ylabel(f'Day {day}  Proc Geo Ht')   
    #plt.xlim(y.min(), y.max())
    plt.ylim(y_predict.min(), y_predict.max())
    plt.legend([f"Day {day} orig", f"Day {day} model"])
    # space between the plots
    plt.tight_layout(4)
    plt.show()

    # specify the window as master
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, ipadx=40, ipady=20, columnspan=3)

    # navigation toolbar
    #toolbarFrame = tk.Frame(master=window)
    #toolbarFrame.grid(row=4,column=0)
    #toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

#### Output the day to fwf file:
def output_to_fwf_file():
    day = n.get()
    day = int(day)
    df_temp = df.loc[(df['day'] == day)]
    X = df_temp.drop(['time', 'proc_lat', 'proc_lon', 'proc_sph_ht', 'proc_geo_ht', 'day' ],axis= 1)
    y = df_temp['proc_geo_ht']
    df_temp['proc_geo_ht_fix'] = poly_model.predict(poly_features.fit_transform(X))
    df_temp['proc_geo_ht_fix'] = df_temp['proc_geo_ht_fix'].round(2)
    df_temp['time'] = df_temp['time'].dt.strftime("%H:%M:%S")
    df_temp.drop(['day'], axis=1, inplace=True)
    df_temp.to_fwf(f"V1G1_zDGPS_2022_{day}_model.att")

###################################################################################################### 
# Creating tkinter window
window = tk.Tk()
window.title('Combobox')
window.geometry('1200x800')

## Set up the rows and columns ....
for x in range(3):
   window.columnconfigure(x, weight=1, minsize=30)
for x in range(4):   
   window.rowconfigure(x, weight=1, minsize=30)

window.columnconfigure(1, weight=2, minsize=30) 
window.rowconfigure(3, weight=75, minsize=30)

# label text for title
ttk.Label(window, text = "ZGPS FIX FOR ANY DAY", 
          background = 'plum', foreground ="white", 
          font = ("Times New Roman", 20)).grid(row = 0, column = 1)
  
# label
ttk.Label(window, text = "Select the Julian Day :",
          font = ("Times New Roman", 12)).grid(column = 0,
          row = 1, padx = 10, pady = 25)
  
# Combobox creation:
n = tk.IntVar()
daycombo = ttk.Combobox(window, state="readonly", textvariable = n)
# Adding combobox drop down list
# daycombo['values'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
daycombo.config(values = days)
daycombo.grid(column = 1, row = 1)
## Set the first day as the default
daycombo.current()
# day = n.get()
# day = int(day)

## Button ..... for any function that we want ....
btn = Button(window, text = 'Graph of the model', command = apply_poly_reg_model2)
btn.grid(column=0, row = 2, padx=2, pady=2, ipadx=1, ipady=1, sticky='nesw') 

## Button ..... for any function that we want ....
btn = Button(window, text = 'Output the day back to text format', command = output_to_fwf_file)
btn.grid(column=1, row = 2, padx=2, pady=2, ipadx=1, ipady=1, sticky='nesw') 

# Button for closing
exit_button = Button(window, text="Exit", command=window.destroy, width = 10)
exit_button.grid(column=2, row = 2, padx=2, pady=2, ipadx=1, ipady=1, sticky='nesw')

window.mainloop()

