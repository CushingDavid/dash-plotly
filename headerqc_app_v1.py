##################################################################################
##                          Spark Header QC Dashboard - v2                      ##
##                               David Cushing                                  ##
##################################################################################

## Import the python libraries needed. 
## pip install the packages as required

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly
## import chart_studio.plotly as py
import os, glob 

pd.options.display.max_columns = None
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import jupyter_dash as jd


### Read in sequence 050 as  the start of our df: 
file = './qc_headers/p001j00_rf_hyd_hdr_1072050.att'
## 23 coulmns
cols = ('SHOT', 'FFID', 'GUNFLAG', 'TYPE', 'OFFSET', 'SPCDELTA', 'SHT_X', 'SHT_Y', 'REC_X', 'REC-Y', 'RELEV', 'SELEV', 'SDEPTH', 'RDATUM', 'SDATUM', 'SWEVEL', 'TIDE', 'MSL_TIDE', 'WDS', 'WDR', 'W_DEPTH', 'WBT', 'SLINE')
## Define the column widths:
colspec = [(8, 12), (20, 24), (34, 36), (46, 48), (57, 60), (61, 72), (73, 86), (87, 101), (102, 115), (116, 130), (132, 139), (145, 151), (157, 163), (176, 178), (188, 190), (198, 203), (205, 211), (217, 223), (227, 238), (239, 250), (256, 263), (270, 275), (279, 287)]
### Read in the header file - infer doesn't work too well....: 
df = pd.read_fwf(file, colspecs=colspec, skiprows=1, names=cols )
# df.head()

files = glob.glob(os.path.join("./qc_headers/p*.att"))
## print(files)
## For loop to read in the rest of the att files:    
for i in range(1,len(files)):
    data = pd.read_fwf(files[i], colspecs=colspec, skiprows=1, names=cols )
    df_temp = pd.DataFrame(data)
    df = pd.concat([df,df_temp], axis=0)
## print(df)

## Function to output the df to fwf file: 
from tabulate import tabulate

def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)

pd.DataFrame.to_fwf = to_fwf

## Create ACQSEQ column 
df['ACQSEQ'] = df['SLINE'].astype(str).apply(lambda x:x[-3:]).astype(int)
df = df.sort_values(by=['ACQSEQ', 'FFID'])
df1 = df.dropna()

#### FFID and SHOT min and max: 
dfc = df1.groupby('ACQSEQ')['FFID']
df1 = df1.assign(FFID_MIN=dfc.transform(min), FFID_MAX=dfc.transform(max))
dfc = df.groupby('ACQSEQ')['FFID']
df = df.assign(FFID_MIN=dfc.transform(min), FFID_MAX=dfc.transform(max))
dfc = df1.groupby('ACQSEQ')['SHOT']
df1 = df1.assign(SP_MIN=dfc.transform(min), SP_MAX=dfc.transform(max))
dfc = df.groupby('ACQSEQ')['SHOT']
df = df.assign(SP_MIN=dfc.transform(min), SP_MAX=dfc.transform(max))

## Output the whole attribute dataset to file: 
df.to_fwf("4693tot4_attributes_extra_headers.att")
df1.to_fwf("4693tot4_attributes_removed_NAN.att")

## 
proj_title = '2022057 / 4693tot4 Gengibre'
acq_min = df1['ACQSEQ'].min()
acq_max = df['ACQSEQ'].max()

###### Visualisation Functions and Dashboard ########

### WBT 3D Function: 
def display_wbt3d():
    data1 = df1
    minx = data1['SHT_X'].min()
    miny = data1['SHT_Y'].min()
    maxx = data1['SHT_X'].max()
    maxy = data1['SHT_Y'].max()
    minz = data1['WBT'].min()
    maxz = data1['WBT'].max()
    z=pd.Series(data1['WBT'])
    x=pd.Series(data1['SHT_X'])
    y=pd.Series(data1['SHT_Y'])
    # 
    # print(x)
    chart1 = go.Figure(data = [go.Mesh3d(
                   z=z,
                   x=x,
                   y=y, 
                   opacity=0.25,
                   color='purple'
                   )])
    chart1.update_layout(title='WBT: Shot X-Y', autosize=False,
                  width=1000, height=1000,
                  scene = dict(
                        xaxis = dict(nticks=8, range=[minx,maxx],),
                        yaxis = dict(nticks=8, range=[miny,maxy],),
                        zaxis = dict(nticks=8, range=[maxz,minz],),
                        camera_eye = {"x": 1.5, "y": -1.5, "z": 0.5},
                        ),
                  margin=dict(l=35, r=20, b=35, t=40))
    return chart1


## 3D WBT with Gun Flag Display: 
def display_wbtgun3d():
    data1 = df1
    minx = data1['SHT_X'].min()
    miny = data1['SHT_Y'].min()
    maxx = data1['SHT_X'].max()
    maxy = data1['SHT_Y'].max()
    minz = data1['WBT'].min()
    maxz = data1['WBT'].max()
    z=pd.Series(data1['WBT'])
    x=pd.Series(data1['SHT_X'])
    y=pd.Series(data1['SHT_Y'])
    color = data1['GUNFLAG']
    # 
    # print(x)
    chart1 = px.line_3d(
                   z=z,
                   x=x,
                   y=y, 
                   color=color
                   )
    chart1.update_layout(title='WBT with GUNFLAG: Shot X-Y (Click legend to swap guns)', autosize=False,
                  width=1000, height=1000,
                  scene = dict(
                        xaxis = dict(nticks=8, range=[minx,maxx],),
                        yaxis = dict(nticks=8, range=[miny,maxy],),
                        zaxis = dict(nticks=8, range=[maxz,minz],),
                        camera_eye = {"x": 1.5, "y": -1.5, "z": 0.5},
                        ),
                  margin=dict(l=35, r=20, b=35, t=40))
    return chart1

app = JupyterDash(__name__, external_stylesheets=[dbc.themes.PULSE])

app.layout = html.Div([
    ## Header
    dbc.Col([
        html.H2(f'SPARK Header QC for {proj_title}', className="bg-primary text-white p-2 mb-2 text-center"),
        # html.H3(f'For Sequences: {acq_min} - {acq_max}'),
    ], style={'textAlign': 'center'}),

    #html.Br(),
    dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H5(f'Pick A Sequence, Sucker:', style={'color': 'white', 'margin-left':'7px'}),
                    html.Br(),
                    dcc.Dropdown(id='ACQSEQ_dropdown', options=[{'label': ACQSEQ, 'value': str(ACQSEQ)}
                                            for ACQSEQ in df1['ACQSEQ'].drop_duplicates().sort_values()]),
                    html.Br(),
                    html.Br(),
                    html.H4(id='sline', style={'color': 'white', 'margin-left':'7px'} ),
                    html.H4(id='ffidrange', style={'color': 'white', 'margin-left':'7px'} ),
                    html.H4(id='sprange', style={'color': 'white', 'margin-left':'7px'}),
                    html.Br(),                        
                ], width=2, className="bg-primary text-black"),
                dbc.Col([
                        dcc.Graph(id='SPCDELTA_line'),
                        #html.Br(),
                        dcc.Graph(id='WBT_GUN'),
                        #html.Br(),
                        dcc.Graph(id='OFFSET')
                        ], width=5),
                dbc.Col([
                        dcc.Graph(id='W_DEPTH'),
                        #html.Br(),
                        dcc.Graph(id='TIDE'),
                        #html.Br(),
                        dcc.Graph(id='TYPE_bar')
                        ], width=5)
            ])
        ], label='Sequence-by-Sequence Header QC'),
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                        dcc.Graph(figure=display_wbt3d())
                ], width = 6),
                dbc.Col([
                        dcc.Graph(figure=display_wbtgun3d())
                ], width = 6)    
            ])
        ], label='Global Header QC')
    ])
])     

## Callback for line information: 
@app.callback(Output('ffidrange', 'children'), 
              Output('sprange', 'children'),
              Output('sline', 'children'),
              Input('ACQSEQ_dropdown', 'value'))
def line_stats(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ffidmin = data['FFID'].min() 
    ffidmax = data['FFID'].max()
    spmin = data['SHOT'].min() 
    spmax = data['SHOT'].max()
    sline = data['SLINE'].max()
    return f"FFID Range: {ffidmin} - {ffidmax}", f"SP Range : {spmin} - {spmax}", f"SLINE : {sline}"

## Callback for SPCDELTA line plot:
@app.callback(Output('SPCDELTA_line', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_spc_delta(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    # data = df1[[int(ACQSEQ), 'FFID', 'SPCDELTA']]
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    # Line plot for SPCDELTA
    chart = px.line(data,
                   x='FFID',
                   y='SPCDELTA',
                   range_x=['FFID_MIN','FFID_MAX'], range_y=[6000,11000], 
                   title=f'SPC_DELTA for Sequence: {ACQSEQ}'
                   )
    chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_traces(line_color='RebeccaPurple')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    chart.update_layout(autosize=False, height=400, width =960)
    return chart

## Callback for WBT with GUN line plot:
@app.callback(Output('WBT_GUN', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_wbt_gun(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ymin = data['WBT'].min() - 50
    ymax = data['WBT'].max() + 50
    xmax = data['FFID'].max()
    xmin = data['FFID'].min()
    chart = px.scatter(data,
                   x='FFID',
                   y='WBT',
                   color="GUNFLAG", 
                   range_x=[xmin,xmax], 
                   range_y=[ymin,ymax], 
                   title=f'WBT for Each Gun - Sequence: {ACQSEQ}'
                   )
    #chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    chart.update_layout(autosize=False, height=400)
    return chart

## Callback for OFFSET line plot with GUNFLAG
@app.callback(Output('OFFSET', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_offset(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ymin = data['OFFSET'].min() - 10
    ymax = data['OFFSET'].max() + 10
    chart = px.line(data,
                   x='FFID',
                   y='OFFSET',
                   color="GUNFLAG",
                   color_discrete_sequence=['RebeccaPurple', 'MediumPurple'],
                   range_x=['FFID_MIN','FFID_MAX'], range_y=[ymin,ymax], 
                   title=f'OFFSET for each GUNFLAG - Sequence: {ACQSEQ}'
                   )
    #chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    chart.update_layout(autosize=False, height=400)
    return chart

## Callback for TIDE Scatter plot
@app.callback(Output('TIDE', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_tide_msl(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ymin = data['MSL_TIDE'].min() - 0.1
    ymax = data['MSL_TIDE'].max() + 0.1
    xmax = data['FFID'].max()
    xmin = data['FFID'].min()
    chart = px.scatter(data,
                   x='FFID',
                   y='MSL_TIDE',
                   color="SWEVEL", 
                   range_x=[xmin,xmax], range_y=[ymin,ymax], 
                   title=f'MSL TIDE with SWEVEL - Sequence: {ACQSEQ}'
                   )
    #chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    chart.update_layout(autosize=False, height=400)
    return chart

## Callback for the W_DEPTH line plot
@app.callback(Output('W_DEPTH', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_wdepth(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ymin = data['W_DEPTH'].min() - 50
    ymax = data['W_DEPTH'].max() + 50
    chart = px.line(data,
                   x='FFID',
                   y=['WDS', 'WDR', 'W_DEPTH'],
                   color_discrete_map={"WDS": "MediumPurple", "WDR": "gold", "W_DEPTH":"RebeccaPurple"}, 
                   range_x=['FFID_MIN','FFID_MAX'], range_y=[ymin,ymax], 
                   title=f'W_DEPTH / WDR / WDS - Sequence: {ACQSEQ}'
                   )
    #chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    chart.update_layout(autosize=False, height=400)
    return chart

## Callback for the TYPE bar chart
@app.callback(Output('TYPE_bar', 'figure'), Input('ACQSEQ_dropdown', 'value'))
def display_shot_type(ACQSEQ):
    if not ACQSEQ:
       raise PreventUpdate
    data = df1[df1['ACQSEQ'] == int(ACQSEQ)]
    ymin = 0
    ymax = 2
    chart = px.bar(data,
                   x='FFID',
                   y='TYPE',
                   color="GUNFLAG", 
                   range_x=['FFID_MIN','FFID_MAX'], range_y=[ymin,ymax],
                   color_continuous_scale=[(0, "Gold"), (1, "RebeccaPurple")], 
                   title=f'SHOT TYPE with GUNFLAG - Sequence: {ACQSEQ}'
                   )
    #chart.add_hrect(y0=6000, y1=7000, line_width=0, fillcolor="red", opacity=0.2)
    chart.update_layout(plot_bgcolor='lavender')
    chart.update_xaxes(linewidth = 2, linecolor ='black')
    chart.update_yaxes(linewidth = 2, linecolor = 'black')
    chart.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    chart.update_layout(autosize=False, height=400)
    return chart


if __name__=='__main__':
    app.run_server(mode='inline', port = 8225)




