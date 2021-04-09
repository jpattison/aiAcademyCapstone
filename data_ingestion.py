"""
Ingests and manipulates data for models

"""

import pandas as pd
import os
import numpy as np
import csv

def standardiseColumns(df):
    # Provided 
    cols = set(df.columns.tolist())
    if 'StreamID' in cols:
        df = df.rename(columns={"StreamID": "stream_id"})
    
    if 'TimesViewed' in cols:
        df = df.rename(columns={"TimesViewed": "times_viewed"})
        
    if 'total_price' in cols:    
        df = df.rename(columns={"total_price": "price"})

    return df


def readJsonData(directory):
    print("read from {0}.".format(directory))

    all_files = []
    if not os.path.isdir(directory):
        return False
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filelocation = directory+'/'+filename
            
            jsonPandas = pd.read_json(filelocation)
            jsonPandas = standardiseColumns(jsonPandas)
            all_files.append(jsonPandas)
    
    df = pd.concat(all_files, ignore_index=True)
    df['invoice_date']=pd.to_datetime(df[['year', 'month', 'day']])
    df['invoice'] = df['invoice'].str.replace('\D+', '')
    return df
        



def limitTopTenRevenue(df):
    # As per instructions limit top 10
    topTenCountries = readJsonData('cs-train').groupby(['country'])['price'].sum().sort_values(ascending=False).head(10).index.values
    
    
    topTenDf = df[df['country'].isin(topTenCountries)]
    return topTenDf


def createDataFrame(data_location):
    df = readJsonData(data_location)

    # limit df to analyse top ten countries by revenue

    return limitTopTenRevenue(df)


# The following data performs feature engineering on original dataframe

def daterange(start_date, end_date, inclusive=False):
    # Useful for iterating between two dates

    delta = 0
    if inclusive:
        delta = 1

    for n in range(int((end_date - start_date).days + delta)):
        yield n, start_date + np.timedelta64(n, 'D')  

def convertToTimeSeries(df, country=None):
    # Convert dataframe into time series    

    if country:
        df = df[df['country']==country]
    
    idx = pd.date_range(min(df.invoice_date), max(df.invoice_date))
    
    by_date = df.groupby('invoice_date')
    
    ts = by_date[['price', 'times_viewed']].sum()
    
    ts['interactions'] = by_date.size()
    ts['unique_invoices'] = by_date[['invoice']].nunique()
    ts['unique_streams'] = by_date[['stream_id']].nunique()
    ts = ts.reindex(idx, fill_value=0)
    ts = ts.rename(columns={"price": "revenue"})
    
    
    return ts
         
    
def featureEngineerTs(ts, is_training):
    """
    
    Create new columns from time series dataframe to enable revenue predictions  

    target_revenue: The summed revenue of the next 30 days that model aims to predicted

    past_X_revenue: 4 columns representing the previous revenue of the last 7, 31, 60, 150 days
    
    For the next four descriptions X = [30, 150]
    past_X_mean_interactions: Mean number of rows in the time series in last X days
    past_X_mean_view: Mean views over last X days
    past_X_mean_invoices: Mean number of invoices last X days
    past_X_mean_streams: Mean number of streams last X days


    """
    


    targetDayFuture = 30 + 1 # We calculate up to 30 days in the future
    pastRevDates = [7, 31, 60, 150]
    pastFeatureDates = [30, 150]
    
    
    start_date = min(ts.index)
    end_date = max(ts.index)
    all_dates = ts.index.values
    
    # Target for future revenues
    y = np.zeros(all_dates.size)
    
    # Past revenues for features
    pastRevs = {}
    for i in pastRevDates:
        pastRevs[i] = np.zeros(all_dates.size)
                               
    # Past features
    
    average_interactions = {}
    average_views = {}
    average_invoices = {}                       
    average_streams = {}
    
    for j in pastFeatureDates:    
        average_interactions[j] = np.zeros(all_dates.size)
        average_views[j] = np.zeros(all_dates.size)
        average_invoices[j] = np.zeros(all_dates.size)                       
        average_streams[j] = np.zeros(all_dates.size)                                                                               
    
    for d, current_date in daterange(start_date, end_date, inclusive=True):
        plusFuture = current_date + np.timedelta64(targetDayFuture, 'D')
        tomorrow = current_date + np.timedelta64(1, 'D') # aim to predict next 30 days, make sure not to include today
        mask = np.in1d(all_dates, np.arange(tomorrow,plusFuture,dtype='datetime64[D]'))
        y[d] = ts[mask]['revenue'].sum()
        
        # Calculate past revenues for features
        for i in pastRevDates:
            initDate = current_date - np.timedelta64(i, 'D')
            mask = np.in1d(all_dates, np.arange(initDate,tomorrow,dtype='datetime64[D]'))
            pastRevs[i][d] = ts[mask]['revenue'].sum()
            
        for j in pastFeatureDates:

            initDate = current_date - np.timedelta64(j, 'D')
            mask = np.in1d(all_dates, np.arange(initDate,tomorrow,dtype='datetime64[D]'))      
            average_interactions[j][d] = ts[mask]['interactions'].mean()
            average_views[j][d] = ts[mask]['times_viewed'].mean()
            average_invoices[j][d] = ts[mask]['unique_invoices'].mean()
            average_streams[j][d] = ts[mask]['unique_streams'].mean()    
        
        
    
    for i in pastRevDates: 
        ts['past_{0}_revenue'.format(i)] = pastRevs[i]
    
    for j in pastFeatureDates:
        
        ts['past_{0}_mean_interactions'.format(j)] =average_interactions[j]
        ts['past_{0}_mean_views'.format(j)] =average_views[j]
        ts['past_{0}_mean_invoices'.format(j)] =average_invoices[j]
        ts['past_{0}_mean_streams'.format(j)] =average_streams[j]

    ts['target_revenue'] = y
    
    
    # Calculate previous values
    
    
    dates = ts.index
    if is_training:
        dates = dates[0:-30]
        ts = ts[ts.index.isin(dates)]
        # remove the last 30 days from training set for safety
    return ts[['target_revenue', 'past_7_revenue', 'past_31_revenue', 'past_60_revenue', 'past_150_revenue', 
               'past_30_mean_interactions', 'past_150_mean_interactions', 
               'past_30_mean_views', 'past_150_mean_views',
               'past_30_mean_invoices', 'past_150_mean_invoices',
               'past_30_mean_streams', 'past_150_mean_streams']]


def df_saver(dictex, directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filelocation = directory+'/'+filename
            
            jsonPandas = pd.read_json(filelocation)
            jsonPandas = standardiseColumns(jsonPandas)
            all_files.append(jsonPandas)    
    
    
    for key, val in dictex.items():
        filelocation = os.path.join(directory, "data_{0}.csv".format(str(key)))
        val.to_csv(filelocation, index_label='invoice_date')

    key_location = os.path.join(directory, "keys.txt")
    
    with open(key_location, "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))

def df_loader(directory):
    """Reading data from keys"""

    key_location = os.path.join(directory, "keys.txt")
    if not os.path.exists(key_location):
        return False
    with open(key_location, "r") as f:
        keys = eval(f.read())

    dictex = {}    
    for key in keys:
        dataframe_location = os.path.join(directory, "data_{}.csv".format(str(key)))
        dictex[key] = pd.read_csv(dataframe_location, parse_dates=True, index_col='invoice_date')

    return dictex


def getTimeSeriesDf(invoice_files, is_training, saved_df_location, load_dataframes=True):
    # Retrieves time series models from training data.
    # Can provide a url of exiting saved models to skip creation
    # Returns a dictionary including a dataframe combining all countries in addition a dataframe for each dictionary

    if load_dataframes and os.path.isdir(saved_df_location):
        
        df_dict = df_loader(saved_df_location)
        if df_dict:
            # confirm exists
            return df_dict
    
    df = createDataFrame(invoice_files)
    
    dataframes = {}

    ts_all = convertToTimeSeries(df)
    tts_all = featureEngineerTs(ts_all, is_training)
   
    dataframes['all'] = tts_all

    for country in df.country.unique():
        print(country)
        ts_country = convertToTimeSeries(df, country)
        tts_country = featureEngineerTs(ts_country, is_training)
        dataframes[country] = tts_country
        

    # save models:
    df_saver(dataframes, saved_df_location)

    return dataframes



