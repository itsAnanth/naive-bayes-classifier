import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

class NaiveBayes:
    
    def __init__(self) -> None:
        self._prior = []
        self._classes = []
        self.n_classes = None
        self._features = []
        self.n_features = None
        self._data = None
        self._target = None
        
    def _calculatePrior(self, data: pd.DataFrame):
        classes = self._classes
        
        for i in range(len(classes)):
            priorProbability = len(data[data[self._target] == i]) / len(data)
            self._prior.append(priorProbability)
        
        return self._prior
    
    def _calculateClasses(self, data: pd.DataFrame):
        self._classes = sorted(list(data[self._target].unique()))
        self.n_classes = len(self._classes)
        return self._classes
    
    def _calculateFeatures(self, data: pd.DataFrame):
        features = list(data.columns[:-1])
        self._features = features
        self.n_features = len(features)
        return self._features
    
    def _calculateLikelihood(self, featureName, featureValue, label):
        data = self._data[self._data[self._target] == label]
        feature_counts = data[featureName].value_counts()
        total_count = len(data)
        smoothing = 1
    
    
        # Apply Laplace smoothing
        probabilityXgivenY = (feature_counts.get(featureValue, 0) + smoothing) / (total_count + smoothing * len(feature_counts))
        
        return probabilityXgivenY
        
    
    def train(self, xtrain: pd.DataFrame, Y: str):
        self._target = Y
        self._calculateClasses(xtrain)
        self._calculateFeatures(xtrain)
        self._calculatePrior(xtrain)
        self._data = xtrain
        
        
    def predict(self, xtest: pd.DataFrame):
        predictions = []
        
        for i in range(len(xtest)):
            
            posterior = [1] * self.n_classes
            for j in range(self.n_classes):
                
                likelihood = 1
                
                for k in range(self.n_features):
                    
                    
                    probabilityXgivenY = self._calculateLikelihood(
                        featureName=self._features[k],
                        featureValue=xtest.iloc[i, k],
                        label=self._classes[j]
                    )
                    
                    likelihood *= probabilityXgivenY
                
                likelihood *= self._prior[j]
                posterior[j] = likelihood
            
            predictions.append(np.argmax(posterior))
            
        return predictions
            
        
        
    
data = pd.read_csv('Breast_cancer_data.csv')
data = data[['mean_radius', 'mean_texture', 'mean_smoothness', 'diagnosis']]

data["cat_mean_radius"] = pd.cut(data["mean_radius"].values, bins = 3, labels = [0,1,2])
data["cat_mean_texture"] = pd.cut(data["mean_texture"].values, bins = 3, labels = [0,1,2])
data["cat_mean_smoothness"] = pd.cut(data["mean_smoothness"].values, bins = 3, labels = [0,1,2])

data = data.drop(columns=["mean_radius", "mean_texture", "mean_smoothness"])
data = data[["cat_mean_radius",	"cat_mean_texture",	"cat_mean_smoothness", "diagnosis"]]



train, test = train_test_split(data, test_size=.2, random_state=41)


xtrain = pd.DataFrame(train)
xtest = pd.DataFrame(test)
ytrain = train.iloc[:, -1].values
ytest = test.iloc[:, -1].values


nb = NaiveBayes()


nb.train(xtrain, 'diagnosis')

# predictions = nb.predict(xtest)

# print(f"{f1_score(predictions, ytest) * 100}% accuracy")



while (True):
    labels = ['benign', 'malignant']
    data = {}
    for i in range(nb.n_features):
        inp = int(input(f"Input for {nb._features[i]}: "))
        data[nb._features[i]] = [inp]
        
    df = pd.DataFrame(data)
    
    diagnosis = nb.predict(df)
    
    
    print(f"Your diagnosis is: {labels[diagnosis[0]]}, {diagnosis}")
    inp = str(input("Continue [Y/N]: "))

    if (inp.lower() == 'n'):
        break