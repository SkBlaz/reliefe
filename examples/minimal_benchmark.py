## perform a simple benchmark 
from scipy import sparse
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif  # chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import tqdm
import reliefe

def test_simple_benchmark():
    sns.set_style("whitegrid")
    data_obj = load_breast_cancer()
    x = data_obj['data']
    y = data_obj['target']
    names = data_obj['feature_names']

    # let's overfit, just for demo purposes

    reliefE_instance = reliefe.ReliefE(embedding_based_distances=True,
                                   verbose=True,
                                   use_average_neighbour=False,
                                   determine_k_automatically=False,
                                   mlc_distance="hamming")
    reliefE_instance.fit(x, y)
    reliefe_importances = reliefE_instance.feature_importances_
    rf_model = RandomForestClassifier()
    rf_model.fit(x, y)
    rf = rf_model.feature_importances_
    mutual_information = mutual_info_classif(x,y)
    sorted_rf = np.argsort(rf)
    sorted_mi = np.argsort(mutual_information)
    sorted_re = np.argsort(reliefe_importances)
    
    names = ["RF","MI","ReliefE"]
    indices = [sorted_rf,sorted_mi,sorted_re]

    output_struct = {}
    for name, indice_set in zip(names, indices):
        scores = []
        indice_set = indice_set.tolist()
        print("Computing evaluations for: {}".format(name))
        for j in tqdm.tqdm(range(len(indice_set))):        
            selected_features = indice_set[0:j+1]
            subset = x[:,selected_features]
            clf = LogisticRegression(max_iter = 10000000, solver = "lbfgs")
            score = np.mean(cross_val_score(clf, subset, y, cv  = 10))
            scores.append((score,j+1))
        output_struct[name] = scores
        
    print("Plotting ..")
    for k,v in output_struct.items():
        indices = []
        scores = []
        for x,y in v:
            indices.append(y)
            scores.append(x)
        plt.plot(indices, scores, label = k)
    plt.xlabel("Top features")
    plt.ylabel("Performance  (Accuracy)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_simple_benchmark()
