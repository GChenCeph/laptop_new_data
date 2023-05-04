import numpy as np
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

def truncated_svd_impute(data, k, tol=1e-6, max_iter=200):
    imputer = SimpleImputer(strategy='mean')
    data_no_nan = imputer.fit_transform(data)

    for _ in range(max_iter):
        svd = TruncatedSVD(n_components=k)
        reduced_data = svd.fit_transform(data_no_nan)
        data_reconstructed = svd.inverse_transform(reduced_data)

        diff = np.sqrt(np.nanmean((data_no_nan - data_reconstructed) ** 2))

        data_no_nan = np.where(np.isnan(data), data_reconstructed, data)

        if diff < tol:
            break

    return data_no_nan

def knn_impute(data, k):
    imputer = KNNImputer(n_neighbors=k)
    return imputer.fit_transform(data)

def mice_impute(data):
    imputer = IterativeImputer()
    return imputer.fit_transform(data)

def random_forest_impute(data):
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10), random_state=0)
    return imputer.fit_transform(data)

def bayesian_regression_impute(data):
    imputer = IterativeImputer(estimator=BayesianRidge())
    return imputer.fit_transform(data)

if __name__ == "__main__":
    data = pd.read_csv('D:\\TS\\Group Project\\current\\expanded_data.csv')

    svd_time = 0
    knn_time = 0
    mice_time = 0
    forest_time = 0
    bayesian_time = 0

    # Truncated SVD
    start = time.time()
    imputed_data = truncated_svd_impute(data.drop(columns=['time', 'nodeid']), k=3)
    end = time.time()
    svd_time += end - start
    svd_df = pd.DataFrame(imputed_data, columns=data.columns[2:])
    svd_df.insert(0, 'time', data['time'])
    svd_df.insert(1, 'nodeid', data['nodeid'])

    # KNN
    start = time.time()
    imputed_data = knn_impute(data.drop(columns=['time', 'nodeid']), k=5)
    end = time.time()
    knn_time += end - start
    knn_df = pd.DataFrame(imputed_data, columns=data.columns[2:])
    knn_df.insert(0, 'time', data['time'])
    knn_df.insert(1, 'nodeid', data['nodeid'])

    # MICE
    start = time.time()
    imputed_data = mice_impute(data.drop(columns=['time', 'nodeid']))
    end = time.time()
    mice_time += end - start
    mice_df = pd.DataFrame(imputed_data, columns=data.columns[2:])
    mice_df.insert(0, 'time', data['time'])
    mice_df.insert(1, 'nodeid', data['nodeid'])

    # Random Forest
    start = time.time()
    imputed_data = random_forest_impute(data.drop(columns=['time', 'nodeid']))
    end = time.time()
    forest_time += end - start
    forest_df = pd.DataFrame(imputed_data, columns=data.columns[2:])
    forest_df.insert(0, 'time', data['time'])
    forest_df.insert(1, 'nodeid', data['nodeid'])

    # Bayesian Regression
    start = time.time()
    imputed_data = bayesian_regression_impute(data.drop(columns=['time', 'nodeid']))
    end = time.time()
    bayesian_time += end - start
    bayesian_df = pd.DataFrame(imputed_data, columns=data.columns[2:])
    bayesian_df.insert(0, 'time', data['time'])
    bayesian_df.insert(1, 'nodeid', data['nodeid'])

    # Save combined results to CSV
    svd_df.to_csv("svd.csv", index=False)
    knn_df.to_csv("knn.csv", index=False)
    mice_df.to_csv("mice.csv", index=False)
    forest_df.to_csv("forest.csv", index=False)
    bayesian_df.to_csv("bayesian.csv", index=False)

    print(f"Truncated SVD Time: {svd_time * 10**3} ms")
    print(f"KNN Time: {knn_time * 10**3} ms")
    print(f"MICE Time: {mice_time * 10**3} ms")
    print(f"Random Forest Time: {forest_time * 10**3} ms")
    print(f"Bayesian Regression Time: {bayesian_time * 10**3} ms")
