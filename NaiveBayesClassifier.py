import sys
import pandas as pd
import numpy as np

CALCULATE_ACCURACY = True  
SMOOTHING_FACTOR = 1e-5  

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    train_data = preprocess_data(train_data, is_training=True)
    test_data = preprocess_data(test_data, is_training=False)
    
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    
    if CALCULATE_ACCURACY and 'label' in test_data.columns:
        X_test = test_data.drop('label', axis=1)
        y_test = test_data['label']
    else:
        X_test = test_data
        y_test = None  
    
    model = train_naive_bayes(X_train, y_train)
    predictions = predict_naive_bayes(model, X_test)
    
    for pred in predictions:
        print(int(pred))
    
    if CALCULATE_ACCURACY and y_test is not None:
        accuracy = np.mean(predictions == y_test.values)
        print(f"Accuracy: {accuracy:.4f}")  

def preprocess_data(data, is_training=True):
    data['home_wl_pre5'] = data['home_wl_pre5'].apply(lambda x: [int(i) for i in x.replace("W", "1").replace("L", "0")])
    data['away_wl_pre5'] = data['away_wl_pre5'].apply(lambda x: [int(i) for i in x.replace("W", "1").replace("L", "0")])
    
    data['home_win_ratio'] = data['home_wl_pre5'].apply(lambda x: sum(x) / len(x))
    data['away_win_ratio'] = data['away_wl_pre5'].apply(lambda x: sum(x) / len(x))
    
    data['reb_diff'] = data['reb_home_avg5'] - data['reb_away_avg5']
    data['pts_diff'] = data['pts_home_avg5'] - data['pts_away_avg5']
    
    data = data.drop(['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5'], axis=1)
    
    return data

def train_naive_bayes(X, y):
    model = {}
    labels = np.unique(y)
    
    for label in labels:
        model[label] = {}
        label_data = X[y == label]
        
        model[label]['mean'] = label_data.mean()
        model[label]['var'] = label_data.var() + SMOOTHING_FACTOR  
        model[label]['prior'] = len(label_data) / len(X) 
    
    return model

def calculate_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def predict_naive_bayes(model, X):
    predictions = []
    for index, row in X.iterrows():
        probs = {}
        for label, params in model.items():
            probs[label] = np.log(params['prior'])  
            

            for feature in X.columns:
                mean = params['mean'][feature]
                var = params['var'][feature]
                x = row[feature]
                feature_prob = calculate_probability(x, mean, var)
                if feature_prob > 0:
                    probs[label] += np.log(feature_prob)
                else:
                    probs[label] += np.log(SMOOTHING_FACTOR)  


        best_label = max(probs, key=probs.get)
        predictions.append(best_label)
    
    return np.array(predictions)

if __name__ == "__main__":
    main()