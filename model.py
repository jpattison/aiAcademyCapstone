from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

import time, os, joblib
import numpy as np
import data_ingestion as di

from logger import update_train_log, update_predict_log

list_columns_features = ['past_30_mean_interactions', 'past_30_mean_views', 'past_30_mean_invoices', 'past_30_mean_streams', 'past_150_revenue', 'past_60_revenue']

# Definitions of the two models we compare
ridgeRegressionModel = {
    'pipe': Pipeline([("scaler", StandardScaler()),
                 ("rg", Ridge(alpha=1.0))]),
    'model_name': "ridge_regression",
    'model_version': 0.1,
    'model_version_note': "RF on AAVAIL revenue data"
}

boostRegressionPipe = {
    'pipe': Pipeline([("scaler", StandardScaler()),
                 ("gbr", GradientBoostingRegressor(random_state=0))]),
    'model_name': "gradient_boosting_regression",
    'model_version': 0.1,
    'model_version_note': "Gradient Boosting Regression on AAVAIL churn"
}


# Enable choosing model through text input
def get_model_details(model_key='boost'):
    if model_key=='ridge':
        model_details = ridgeRegressionModel
    elif model_key == 'boost' or model_key==None:
        model_details = boostRegressionPipe
    else:
        esc = "invalid model name"
        raise Exception(esc)
    return model_details


def get_model_path(model_details, country, test=False):
    # For saving / loading
    if test:
        return os.path.join("models","test.joblib")
    
    return os.path.join("models", "model-{0}-{1}-{2}.joblib".format(model_details['model_name'], country, model_details['model_version']))


def train_model(training_directory, model_key=None, test=False):

    model_details=get_model_details(model_key)


   # Train for each country for provided model
    df_directory = 'cs-train-dataframes'
    if test:
        df_directory = 'cs-train-dataframes-testing'

    df_dict = di.getTimeSeriesDf(training_directory, True, df_directory, load_dataframes=True)
   
    pipe = model_details['pipe']

    countries = list(df_dict.keys())
    if test:
        countries = countries[0:1]

    # If test subset to make faster

    rmse_training_report = {}

    for country in countries:

        ## start timer for runtime
        time_start = time.time()

        df = df_dict[country]

        # Create X, Y test
        
        X = df[list_columns_features]
        y = df['target_revenue']

        if test:
            n_samples = int(np.round(0.9 * X.shape[0]))
            subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples, replace=False).astype(int)
            mask = np.in1d(np.arange(y.size),subset_indices)
            y=y[mask]
            X=X[mask]  

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25, shuffle=True)

        
        # Train on X train
        pipe.fit(X_train, y_train)

        # Record evaluation report
        y_pred = pipe.predict(X_test)
        
        eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))

        rmse_training_report[country] = eval_rmse
        # Train on whole data set

        pipe.fit(X, y)

        
        save_path = get_model_path(model_details, country, test)
        if test:
            print("... saving test version of model")
            joblib.dump(pipe, save_path)
        else:

            joblib.dump(pipe, save_path)

        
        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)

        # update the log file
        update_train_log(X.shape, eval_rmse, runtime,
                    model_details, country, test=test)

        

    return rmse_training_report



def load_model(model_details, country, test=False):
    """
    load a model
    """
    model_path = get_model_path(model_details, country, test)
   
    if not os.path.exists(model_path):
        exc = "Model '{}' cannot be found did you train the full model?".format(model_path)
        raise Exception(exc)


    model = joblib.load(model_path)
    return(model)



def model_predict(day, month, year, country, model_key, test=False):
    # Read data from production and make a prediction
    # Convert query into suitable dataframe


    ## start timer for runtime
    time_start = time.time()

    model_details=get_model_details(model_key)


    tsDict = di.getTimeSeriesDf('cs-production', is_training=False, saved_df_location='cs-production-dataframes', load_dataframes=True) 
    
    if country==None:
        country='all'
    ts = tsDict[country]
    X = ts[list_columns_features]

    predict_date = np.datetime64('{0}-{1}-{2}'.format(year, month, day))

    model = load_model(model_details, country, test)

    # check if date in dates
    all_dates = ts.index.values

    if predict_date not in all_dates:
        exc = "Date '{}' is not in production dataset".format(predict_date)
        #print(all_dates)
        raise Exception(exc)

    # grab single row relating to date

    query = X.loc[[predict_date]]
    # feed into prediction

    y_pred = model.predict(query)
    # return response

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    
    query_date = np.datetime_as_string(predict_date, timezone='UTC')
    
    update_predict_log(y_pred, query_date, country,
                       runtime, model_details, test=test)
    
    return({'y_pred':y_pred})

