import pandas as pd
import numpy as np
import joblib
import time

from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,  precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    average_precision_score, mean_squared_error, mean_absolute_error, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score, homogeneity_score, completeness_score, v_measure_score

from kneed import KneeLocator

def other_ens(df, prct):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:, -1], test_size=0.3, shuffle=False)
    # Feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train.values[:200000])
    sc_t = StandardScaler()
    x_test = sc_t.fit_transform(X_test.values)  
    #init
    model1 = OneClassSVM(nu=0.5, kernel='linear', gamma="auto", shrinking=False)
    model2 = IsolationForest(n_estimators=80, max_features=0.2, contamination=0.1, random_state=42)
    model3 = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs = -1)
    #train
    start_time = time.time()
    model3.fit(x_train)
    preds_model3 = model3.predict(x_train).reshape(-1, 1)   
    after_code1_time = time.time()
    print(after_code1_time - start_time)
    preds_model2 = model2.fit_predict(x_train).reshape(-1, 1)    
    after_code2_time = time.time()
    print(after_code2_time - after_code1_time)

    preds_model1 = model1.fit_predict(x_train).reshape(-1, 1)
    after_code3_time = time.time()
    print(after_code3_time - after_code2_time)
    
    # Stack the predictions horizontally
    X_meta = np.hstack((preds_model1, preds_model2, preds_model3))    
    # Assign labels based on the consensus of at least two base models
    x_predictions = (np.sum(X_meta > 0, axis=1) >= 1).astype(int) * 2 - 1
    
    #test
    ypreds_model1 = model1.predict(x_test).reshape(-1, 1)
    ypreds_model2 = model2.predict(x_test).reshape(-1, 1)
    ypreds_model3 = model2.predict(x_test).reshape(-1, 1)
    
    # Stack the predictions horizontally
    stacked_X_test = np.hstack((ypreds_model1, ypreds_model2, ypreds_model3))
    
    # Assign labels based on the consensus of at least two base models
    y_predictions = (np.sum(stacked_X_test > 0, axis=1) >= 1).astype(int) * 2 - 1
    
    ens_metrics = test_scores(y_test, y_predictions, x_test, 'ensemble', prct)
    
    # all_data = impute_outliers(X_train, X_test, x_predictions, y_predictions)
    return ens_metrics
def ensemble(df, prct):
    #split train test and normalize data
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:, -1], test_size=0.3, shuffle=False)
    sc = MinMaxScaler()
    x_train = sc.fit_transform(X_train.values)
    sc_t = MinMaxScaler()
    x_test = sc_t.fit_transform(X_test.values)    
    
    #init
    model1 = IsolationForest(n_estimators = 700, contamination=prct * 0.01)
    model2 = EllipticEnvelope(store_precision=False, contamination= prct * 0.01, support_fraction = 0.1)
    #train
    start_time = time.time()
    preds_model1 = model1.fit_predict(x_train).reshape(-1, 1)
    after_code1_time = time.time()
    print(after_code1_time - start_time)
    preds_model2 = model2.fit_predict(x_train).reshape(-1, 1)
    after_code2_time = time.time()
    print(after_code2_time - after_code1_time)

    # Stack the predictions horizontally
    X_meta = np.hstack((preds_model1, preds_model2))    
    # Assign labels based on the consensus of at least two base models
    x_predictions = (np.sum(X_meta > 0, axis=1) >= 2).astype(int) * 2 - 1
    
    #test
    ypreds_model1 = model1.predict(x_test).reshape(-1, 1)
    ypreds_model2 = model2.predict(x_test).reshape(-1, 1)
    
    # Stack the predictions horizontally
    stacked_X_test = np.hstack((ypreds_model1, ypreds_model2))
    
    # Assign labels based on the consensus of at least two base models
    y_predictions = (np.sum(stacked_X_test > 0, axis=1) >= 2).astype(int) * 2 - 1
    
    ens_metrics = test_scores(y_test, y_predictions, x_test, 'ensemble', prct)
    
    all_data = impute_outliers(X_train, X_test, x_predictions, y_predictions)
    return all_data, ens_metrics

def kmeans(df, prct):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:, -1], test_size=0.3, shuffle=False)
    sc = MinMaxScaler()
    x_train = sc.fit_transform(X_train)
    sc_t = MinMaxScaler()
    x_test = sc_t.fit_transform(X_test)
    inertia = []
    best_score = -1
    optimal_num_clusters = 0
    K = range(1,11)
    for i in K:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x_train)
        inertia.append(kmeans.inertia_)

    kn = KneeLocator(list(K), inertia, curve='convex', direction='decreasing')
    optimal_num_clusters = kn.knee
    
    #train
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_train)
    # Get cluster centers and labels
    x_cluster_centers = kmeans.cluster_centers_
    x_labels = kmeans.labels_
    x_tr_tmp = X_train.reset_index(drop = True)    
    # Calculate the distance of each point to its cluster center
    distances = np.zeros(x_train.shape[0])
    for i in range(5):
        distances[x_labels == i] = np.linalg.norm(x_tr_tmp[x_labels == i] - x_cluster_centers[i], axis=1)

    k = 1.5
    # Set a threshold for anomaly detection (2 standard deviations from the mean distance)
    threshold = np.mean(distances) + k * np.std(distances)

    # Identify anomalies (points with distances greater than the threshold)
    x_anomalies = x_tr_tmp[distances > threshold]
    x_anomaly_indices = distances > threshold
    X_train['labels'] = np.where(x_anomaly_indices, -1, 1)    

    # Test
    y_labels = kmeans.predict(x_test)
    y_cluster_centers = kmeans.cluster_centers_
    x_t_tmp = X_test.reset_index(drop = True)
    # Calculate the distance of each point to its cluster center
    distances = np.zeros(x_test.shape[0])
    for i in range(5):
        distances[y_labels == i] = np.linalg.norm(x_t_tmp[y_labels == i] - y_cluster_centers[i], axis=1)

    # Set a threshold for anomaly detection (2 standard deviations from the mean distance)
    threshold = np.mean(distances) + k * np.std(distances)

    # Identify anomalies (points with distances greater than the threshold)
    y_anomalies = x_t_tmp[distances > threshold]
    y_anomaly_indices = distances > threshold
    X_test['labels'] = np.where(y_anomaly_indices, -1, 1)
    
    kmeans_metrics = test_scores(y_test, X_test['labels'], x_test, 'kmeans', prct)
    
    # all_data = impute_outliers(X_train, X_test, X_train['labels'], X_test['labels'])
    return kmeans_metrics

def dbscan(df, prct):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:, -1], test_size=0.3, shuffle=False)
    sc = MinMaxScaler()
    x_train = sc.fit_transform(X_train)
    sc_t = MinMaxScaler()
    x_test = sc_t.fit_transform(X_test)
    # min_samples_range = [1, 2]
    # eps_range = [0.01,0.07,0.05]
    # best_score = -1
    # best_eps =  -1
    # best_sample = -1

    # Grid search
#     for min_samples in min_samples_range:
#         for eps in eps_range:
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             lbls = dbscan.fit_predict(x_train[:10000])

#             # Calculate silhouette score
#             silhouette_avg = silhouette_score(x_train[:10000], lbls)
#             # Update best parameters if a higher silhouette score is achieved
#             if silhouette_avg >= best_score:
#                 best_score = silhouette_avg
#                 best_eps =  eps
#                 best_sample = min_samples
    dbscan = DBSCAN(eps=0.04, min_samples=2, leaf_size= 10)
    model = dbscan.fit(x_train)
    x_labels = model.labels_
    
    x_outliers_mask = x_labels == -1
    x_outliers = X_train[x_outliers_mask]
    # Remove outliers from the original dataset
    x_data_no_outliers = X_train[~x_outliers_mask]

    X_train['labels'] = np.where(x_outliers_mask, -1, 1)

    y_labels = dbscan.fit_predict(x_test)
    y_outliers_mask = y_labels == -1
    y_outliers = X_test[y_outliers_mask]
    # Remove outliers from the original dataset
    y_data_no_outliers = X_test[~y_outliers_mask]

    X_test['labels'] = np.where(y_outliers_mask, -1, 1)

    dbscan_metrics = test_scores(y_test, X_test['labels'], x_test, 'dbscan', prct)
    
    # all_data = impute_outliers(X_train, X_test, X_train['labels'], X_test['labels'])
    return dbscan_metrics
    
def test_scores(y_test, y_predictions, x_test, ad_method, prct):
    # test scores
    acc = accuracy_score(y_test, y_predictions)
    conf = confusion_matrix(y_test, y_predictions)
    precision = precision_score(y_test, y_predictions,)
    recall = recall_score(y_test, y_predictions)
    f1 = f1_score(y_test, y_predictions)
    roc_auc = roc_auc_score(y_test, y_predictions)
    avg_precision = average_precision_score(y_test, y_predictions)
    mse = mean_squared_error(y_test, y_predictions)
    mae = mean_absolute_error(y_test, y_predictions)
    ari = adjusted_rand_score(y_test, y_predictions)
    homogeneity = homogeneity_score(y_test, y_predictions)
    completeness = completeness_score(y_test, y_predictions)
    v_measure = v_measure_score(y_test, y_predictions)

    fpr, tpr, _ = roc_curve(y_test, y_predictions)
    
    print(' Anomaly detection method: ', ad_method)
    print("Accuracy:", acc)
    print("Confusion Matrix:", conf)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print("Average Precision:", avg_precision)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Adjusted Rand Index:", ari)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    
    # silhouette = silhouette_score(x_test, y_predictions)
    davies_bouldin = davies_bouldin_score(x_test, y_predictions)
    calinski_harabasz = calinski_harabasz_score(x_test, y_predictions)
    # print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", davies_bouldin)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    
    ad_metrics = {
    'Accuracy': acc,
    'Confusion Matrix': conf,
    'Precision': avg_precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': roc_auc,    
    'FPR': fpr,
    'TPR': tpr,
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Adjusted Rand Index': ari,
    'Homogeneity': homogeneity,
    'Completeness': completeness,
    'V-measure': v_measure,
    # 'Silhouette Score': silhouette,
    'Davies-Bouldin Index': davies_bouldin,
    'Calinski-Harabasz Index': calinski_harabasz,
    }
    return ad_metrics

def impute_outliers(X_train, X_test, x_predictions, y_predictions):
    X_train.reset_index(inplace = True)
    X_test.reset_index(inplace = True)
    # Copy the data into NumPy arrays for faster computation
    X_train_arr = X_train.values
    X_test_arr = X_test.values

    # Impute outliers in the training set
    x_outlier_indices = np.where(x_predictions == -1)[0]
    x_good_indices = np.where(x_predictions == 1)[0]
    for outlier_index in x_outlier_indices:
        try:
            previous_good_index = x_good_indices[x_good_indices < outlier_index][-1]
        except IndexError:
            previous_good_index = x_good_indices[outlier_index]
        # Replace the outlier value with the previous good value
        X_train_arr[outlier_index] = X_train_arr[previous_good_index]

    # Impute outliers in the test set
    y_outlier_indices = np.where(y_predictions == -1)[0]
    y_good_indices = np.where(y_predictions == 1)[0]
    for outlier_index in y_outlier_indices:
        try:
            previous_good_index = y_good_indices[y_good_indices < outlier_index][-1]
        except IndexError:
            previous_good_index = y_good_indices[outlier_index]
        # Replace the outlier value with the previous good value
        X_test_arr[outlier_index] = X_test_arr[previous_good_index]

    # Concatenate the arrays into one NumPy array
    all_data_arr = np.concatenate((X_train_arr, X_test_arr), axis=0)

    # Convert the NumPy array back to a DataFrame
    all_data_df = pd.DataFrame(all_data_arr, columns=X_train.columns)

    return all_data_df