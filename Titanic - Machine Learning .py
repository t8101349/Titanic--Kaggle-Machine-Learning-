import pandas as pd
import numpy as np
from sklearn import preprocessing
# Step1.資料的預處理(Preprocessing data)
# Step1-1 開啟檔案
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
    
print("\n\n訓練集資料：\n\n",train_data)
print("\n\n測試集資料：\n\n",test_data)

# Step1-2 觀察資料
train_data.info()

train_data.columns

train_data.shape 

train_data.describe() 


# Step1-3 處理資料的NAN
print ("\n\n呈現Embarked行(column)內的資料不重複元素： \n\n",train_data["Embarked"].unique())
print ("\n\n呈現Embarked行(column)內的資料一般描述： \n\n",train_data["Embarked"].describe())
train_data['Embarked'] = train_data['Embarked'].fillna('S')  # or dropna(axis=1)
print ("\n\n呈現Embarked行(column)內的資料不重複元素： \n\n",train_data["Embarked"].unique())
print ("\n\n呈現處理完缺值的Embarked行(column)內的資料一般描述： \n\n",train_data["Embarked"].describe())

print ("Age處理nan前",train_data['Age'])
train_data['Age'] = train_data['Age'].fillna(30)
print("Age處理nan後",train_data['Age'])

print ("Cabin資料集",train_data["Cabin"].unique())
print (train_data["Cabin"].describe())

# 考量有些特徵本身就有大量缺值，故筆者在這設定當unique數量 > 100 判定不重要則捨棄特徵。
# 確認有哪些行(columns)，並創一個list使用迴圈去看unique數量
train_data.columns
features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

for i in range(len(features)):
    print("第",i,"行不重複的元素數量：", len(train_data[features[i]].unique()))
    if (len(train_data[features[i]].unique()) > 100):
        print("捨去第",i,"個特徵：",features[i])

#以下指令，從訓練集與測試集合中移除捨棄的四個特徵
abandon_features = ['PassengerId','Name','Ticket','Cabin']
train_data = train_data.drop(abandon_features,axis=1)


# Step1-4 特徵挑選及資料正規化與編碼
#Encoder
Encoder = preprocessing.LabelEncoder()
train_data['Embarked'] = Encoder.fit_transform(train_data['Embarked'])
train_data['Sex'] = Encoder.fit_transform(train_data['Sex'])

#驗證
print(train_data['Embarked'].unique())
print(train_data['Sex'].unique())


#分成特徵與標籤
train_data_feature = train_data.drop('Survived',axis=1)  #捨去生存特徵
train_data_label = train_data['Survived']
train_data_feature

# Step2.模型選擇與建立(Data choose and build)
##1
from sklearn.model_selection import train_test_split
from sklearn import svm

train_feature, val_feature, train_label, val_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

svm_model = svm.SVM()
svm_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("支持向量機(Support Vector Machines)模型準確度(訓練集):",svm_model.score(train_feature, train_label))
print ("支持向量機(Support Vector Machines)模型準確度(測試集):",svm_model.score(val_feature, val_label))
svm_model_acc = svm_model.score(val_feature, val_label)

##2
from sklearn.neighbors import KNeighborsClassifier

KNeighbors_model = KNeighborsClassifier(n_neighbors=2)
KNeighbors_model.fit(train_feature, train_label)

print ("最近的鄰居(Nearest Neighbors)模型準確度(訓練集)：",KNeighbors_model.score(train_feature, train_label))
print ("最近的鄰居(Nearest Neighbors)模型準確度(測試集)：",KNeighbors_model.score(val_feature, val_label))
KNeighbors_model_acc = KNeighbors_model.score(val_feature, val_label)

##3
from sklearn import tree

DecisionTree_model = tree.DecisionTreeClassifier()
DecisionTree_model.fit(train_feature, train_label)

print ("決策樹(Decision Trees)模型準確度(訓練集)：",DecisionTree_model.score(train_feature, train_label))
print ("決策樹(Decision Trees)模型準確度(測試集)：",DecisionTree_model.score(val_feature, val_label))
DecisionTree_model_acc = DecisionTree_model.score(val_feature, val_label)

##4
from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=10)
RandomForest_model.fit(train_feature, train_label)

print ("隨機森林(Forests of randomized trees)模型準確度(訓練集)：",RandomForest_model.score(train_feature, train_label))
print ("隨機森林(Forests of randomized trees)模型準確度(測試集)：",RandomForest_model.score(val_feature, val_label))
RandomForest_model_model_acc = RandomForest_model.score(val_feature, val_label)

##5
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(solver='lbfgs', 
                                   alpha=1e-4,
                                   hidden_layer_sizes=(6, 2), 
                                   )
MLP_model.fit(train_feature, train_label)

print ("神經網路(Neural Network models)模型準確度(訓練集)：",MLP_model.score(train_feature, train_label))
print ("神經網路(Neural Network models)模型準確度(測試集)：",MLP_model.score(val_feature, val_label))
MLP_model_acc = MLP_model.score(val_feature, val_label)

##6
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

GaussianProcess_model = GaussianProcessClassifier()
GaussianProcess_model.fit(train_feature, train_label)

print ("高斯過程(GaussianProcess)模型準確度(訓練集)：",GaussianProcess_model.score(train_feature, train_label))
print ("高斯過程(GaussianProcess)模型準確度(測試集)：",GaussianProcess_model.score(val_feature, val_label))
GaussianProcess_model_acc = GaussianProcess_model.score(val_feature, val_label)



#model比較
models = pd.DataFrame({
    'Model': ['支持向量機(Support Vector Machines)', 
              '最近的鄰居(Nearest Neighbors)', 
              '決策樹(Decision Trees)',
              '隨機森林(Forests of randomized trees)', 
              '神經網路(Neural Network models)'
             ],
    'Score': [svm_model_acc,
              KNeighbors_model_acc,
              DecisionTree_model_acc,
              RandomForest_model_model_acc,
              MLP_model_acc
              ]
                       })
models.sort_values(by='Score', ascending=False)


# STep3.模型驗證(Model validation)

# test_data也需做預先處理，將輸入處理更改成與train_feature 一樣 
test_data.info()
test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data['Age'] = test_data['Age'].fillna(30)
test_data['Fare'] = test_data['Fare'].fillna(35.6)

abandon_features = ['PassengerId','Name','Ticket','Cabin']
test_data_feature = test_data.drop(abandon_features,axis=1)

test_data_feature.info()

#Encoder
Encoder = preprocessing.LabelEncoder()
test_data_feature['Embarked'] = Encoder.fit_transform(test_data_feature['Embarked'])
test_data_feature['Sex'] = Encoder.fit_transform(test_data_feature['Sex'])

test_data_feature

#Training
from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=10)

RandomForest_model.fit(train_data_feature,train_data_label)

final_predictions = RandomForest_model.predict(test_data_feature)

output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': final_predictions})
output.to_csv('Submission.csv', index=False)

##5
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(solver='lbfgs', 
                                   alpha=1e-4,
                                   hidden_layer_sizes=(6, 2), 
                                   )
MLP_model.fit(train_data_feature,train_data_label)

final_predictions_2 = MLP_model.predict(test_data_feature)

output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': final_predictions_2})
output.to_csv('Submission2.csv', index=False)
