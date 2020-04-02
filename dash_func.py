import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import re
import pycountry_convert as pc
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html

def spliteKeyWord(country):
    '''
    Functions to get the English Country Name
    '''
    #regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    regex = r"[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, country, re.UNICODE)
    return ' '.join(matches)

def get_filtered_data(df, filters):
    '''
    Based on the filters to get the needed data.
    Use DFS to get all data
    ----params----------
    --inputs--
    df: pd.DataFrame source dataframe
    filters: list(dict{'filtername': filter_values}), list of dict-filters
    --returns--
    bool_filters: list of bool array
    attr_filters: list of applied filters
    '''
    #store all results in bool_filters and attr_filters
    bool_filters = []
    attr_filters = []
    # initilizations
    attr = ''
    bool_init = np.array([True]*len(df.index))
    index = 0 # index keep track of which filter is applied now, indicator of dfs exit.
    # start dfs to get bool masks/bool_filters and legends/attr_filtes
    dfs_bool(df, filters, bool_filters, attr_filters, index, attr, bool_init)
    return bool_filters, attr_filters

def dfs_bool(df, filters, bool_filters, attr_filters, index, attr, bool_init):
    '''
    Based on the filters to get the needed data.
    Use DFS to get all data
    ----params----------
    --inputs--
    df: pd.DataFrame source dataframe
    filters: list(dict{'filtername': filter_values}), list of dict-filters.
    bool_filters: list(np.array(True, False....)) store all bool arries 
    att_filters: np.array(str,...) store the filters, so can be used as the legend latter.
    index: int  current index of filters
    bool_init: np.array(True, False) current bool values, True mean the corresponding rows will be selected, vice verse.
    '''
    if index == len(filters):
        bool_filters.append(bool_init)
        attr_filters.append(attr)
        return 
    filter_name, values = list(filters[index].keys())[0], list(filters[index].values())[0]
    for v in values:
        #print(f'{filter_name}:{v}')
        #print(f'{filter_name}:{values}')
        bool_next = bool_init & np.array((df[filter_name]==v).to_list())
        attr_next = attr + v
        dfs_bool(df, filters, bool_filters, attr_filters, index + 1, attr_next, bool_next)
    return

def generate_table(df, bool_filters, start_date, end_date, maxsize=20):
    '''
    Generate Table based on given filters and date ranges.
    The table is limitted by maxsize.
    ==============
    params:
    df: pd 
    bool_filters: list
    attr_filters: list
    start_date: datetime
    end_date: datetime
    =================
    return:
    data: html.Table filtered data
    '''
    bool_init = np.array([False]*len(df.index))
    start_date = dt.strptime(start_date.split('T')[0], '%Y-%m-%d')
    end_date = dt.strptime(end_date.split('T')[0], '%Y-%m-%d')
    print(start_date)
    for b in bool_filters:
        bool_init = bool_init | b
    date_time = np.array(df.index.to_list())
    bool_init = bool_init & (date_time >= start_date) & (date_time <= end_date)
    dataframe = df[bool_init]
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe),maxsize))
        ])
    ])

def generate_data_for_figure(pd_no_dup, metric, bool_filters, attr_filters):
    '''
    Generate data for figure based on given filters and date ranges.
    ==============
    params:
    pd_no_dup: pd 
    metric: str 
    bool_filters: list of bool mask for corresponding filters
    attr_filters: list of legend of corresponding data
    =================
    return:
    data: dict for figure, filtered data
    '''
    data = [ dict(
                    x=pd_no_dup[b].index,
                    y=pd_no_dup[b][metric],
                    #text=df[df['continent'] == i]['country'],
                    mode='lines+markers',
                    opacity=0.7,
                    marker={
                        'size': 9,
                        'line': {'width': 1}
                    },
                    name=a
                ) for b,a in zip(bool_filters,attr_filters)
            ]
    return data

def get_attrs(df, attr):
    '''
    Get values for specific filters
    ==============
    params:
    df: pd 
    attr: str name of the metric
    =================
    return:
    attr_list: list of dict values for given attr
    
    '''
    if attr == 'Metrics':
        attrs = list(df.columns.to_list())[3:]
    elif attr == 'Continents':
           attrs = ['North America',
        'South America', 'Asia','Australia',
        'Africa','Europe','Antarctica']
    else:
        attrs = df[attr].unique()
    attr_list = []
    for c in attrs:
        attr_list.append({'label': c, 'value': c})
    return attr_list  