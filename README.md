# Machine Learning
## Jenis Machine Learning

![image](https://user-images.githubusercontent.com/49611937/117688727-4055f600-b1e3-11eb-83e5-80b8df160b1c.png)

1. **Regression**: Linear Regression, Decision Tree
2. **Classification**: Logistic Regression, Decision Tree
3. **Clustering**: K-Means Clustering

## Confusion Matrix

![image](https://user-images.githubusercontent.com/49611937/117679762-c1f55600-b1da-11eb-9a6a-a2caec339a3e.png)

- True Positive (TP) : Prediksi dan kenyataan sama-sama +
- True Negative (TN) : Prediksi dan kenyataan sama-sama -
- False Positive (FP) : Prediksi + dan kenyataan -
- False Negative (FN) : Prediksi - dan kenyataan +


## Memilih Metrics
- Jika jumlah **FP** dan **FN** seimbang (*Symmetric*), gunakan **Accuracy**
- Jika jumlah **FP** dan **FN** tidak seimbang, gunakan **F1-Score**
- Jika FP lebih baik dipilih daripada FN (Misalnya kasus **Fraud/Scam**, sebaiknya yg tidak fraud terdeteksi fraud daripada fraud tidak terdeteksi), gunakan **Recall**
- Lebih baik banyak TN daripada FP, gunakan **Precision**.


    - **Accuracy** = (TP + TN ) / (TP+FP+FN+TN).
    - **Precision** = (TP) / (TP+FP)
    - **Recall** = (TP) / (TP + FN)
    - **F1-Score** = 2 * (Recall*Precission) / (Recall + Precission)


## Continuous Value vs Discrete Value
- Continuous Value : numerik (Contohnya: prediksi harga rumah, harga saham, suhu)
- Discrete Value : kategorikal (Contohnya: prediksi SPAM(1) atau NOT SPAM(0))

## Performance Evaluation
- Regression : mean_squared_error, root_mean_squared_error, mean_absolute_error. Semakin kecil error semakin baik model regresinya. (*from sklearn.metrics import mean_squared_error, mean_absolute_error*)
- Classification : confusion_matrix, classification_report (*from sklearn.metrics import confusion_matrix, classification_report*)

## Contoh Implementasi
### 1. Decision Tree
Bisa untuk Klasifikasi dan Regresi. Strukturnya adalah seperti berikut:

![image](https://user-images.githubusercontent.com/49611937/117696601-af374d00-b1eb-11eb-95ea-0960af09991a.png)

Struktur ini terdiri dari:
1. **Decision Node** yang merupakan feature/input variabel
2. **Branch** yang ditunjukkan oleh garis hitam berpanah, yang adalah rule/aturan
3. **Leaf Node** yang merupakan output/hasil.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

X = dataset.drop(['Revenue'], axis=1)  #Revenue=1, Not Revenue=0
y = dataset['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply algorithm and training
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)

# Testing
pred = model.predict(X_test)

# Evaluation
print('Training Accuracy: ', model.score(X_train, y_train)
print('Testing Accuracy: ', model.score(X_test, y_test)

# Confusion matrix
cm = confusion_matrix(y_test, pred)
print(cm)

# Classification_report
cr = classification_report(y_test, pred)
print(cr)
```
Result:

![image](https://user-images.githubusercontent.com/49611937/117684898-a0e33400-b1df-11eb-87fa-e67777a7852b.png)

![image](https://user-images.githubusercontent.com/49611937/117684935-ac365f80-b1df-11eb-9dc8-b77fa93a6427.png)

![image](https://user-images.githubusercontent.com/49611937/117685101-cd974b80-b1df-11eb-9507-23c38cfc97af.png)

---
### 2. Logistic Regression
Klasifikasi. Hanya mengolah data numerik. Selain untuk *Binary Clasification*(1-0), bisa juga untuk *multiclass classification problem*.
Rumusnya sama dengan Linear Regression:

![image](https://user-images.githubusercontent.com/49611937/117693137-e60b6400-b1e7-11eb-8966-8512598f26cf.png)

Karena **Output** Logistic Regression = **1** atau **0**, maka real value perlu diubah ke 1 atau 0 menggunakan **Fungsi Sigmoid**. >0.5 = 1 dan <0.5 =0. Rumus Fungsi Sigmoid adalah:

![image](https://user-images.githubusercontent.com/49611937/117693168-ee639f00-b1e7-11eb-8c66-d68834f2f28f.png)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logreg = LogisticRegression()

# Training
model = logreg.fit(X_train, y_train)

# Testing
pred = logreg.predict(X_test)

print('Training Accuracy: ', model.score(X_train, y_train))
print('Testing Accuracy: ', model.score(X_test, y_test)

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
print(cm)

# Classification Report
cr = classification_report(y_test, pred)
```
Result:

![image](https://user-images.githubusercontent.com/49611937/117695698-b4e06300-b1ea-11eb-9891-36c589fec5f5.png)

![image](https://user-images.githubusercontent.com/49611937/117695734-c0338e80-b1ea-11eb-85ef-c078904cc212.png)

![image](https://user-images.githubusercontent.com/49611937/117695764-c9246000-b1ea-11eb-80e0-3d1d49dfb863.png)

---
### 3. Linear Regression
Digunakan untuk regresi data numerik/ continuous variable. Model regresi ada 2:
1. **Simple regression model** : model regresi paling sederhana, hanya terdiri dari **1 feature (univariate)** dan **1 target**.

![image](https://user-images.githubusercontent.com/49611937/117701065-fd9b1a80-b1f0-11eb-8c2f-3a7b433b97a9.png)

3. **Multiple regression model** : sesuai namanya, terdiri dari **beberapa feature (multivariate)**.

![image](https://user-images.githubusercontent.com/49611937/117701092-05f35580-b1f1-11eb-9d1a-218c73d66e0c.png)

- **y** : target
- **X** : feature
- **a** : intercept
- **b** : slope

![image](https://user-images.githubusercontent.com/49611937/117701267-39ce7b00-b1f1-11eb-808d-a46fc39abc44.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = pd_read.csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv')

# Data rescaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
dataset[['RM', 'LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(dataset[['RM', 'LSTAT','PTRATIO','MEDV']])

X = dataset.drop(['MEDV'], axis=1)
y = dataset['MEDV']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Modeling and Training
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Testing
pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse= np.sqrt(mse)

print('mse: ', mse)
print('mae: ', mae)
print('rmse: ', rmse)

plt.scatter(y_test, pred, c='blue')
plt.title('True value vs predicted value : Linear Regression')
plt.xlabel('Actual Price')
plt.ylabel('Predicted value')
plt.show()

```
Result:

![image](https://user-images.githubusercontent.com/49611937/117748633-973aea00-b23a-11eb-9c27-c46c1624c18b.png)

![image](https://user-images.githubusercontent.com/49611937/117750233-47115700-b23d-11eb-838a-668b3e1be8cf.png)

---
### 4. K-Means Clustering
*Clustering* memproses data dan mengelompokkannya berdasarkan kesamaan antar objek/sampel dalam satu kluster.
K-Means menghitung kesamaan objek dari seberapa dekat jaraknya dari **centroid (K)**. K-Means menghitung jarak antar 2 data (**Jarak Minkowski**). Clustering disebut bagus jika **semakin dekat/rapat data points dengan centroidsnya** dan **semakin jauh jarak antar data points yang beda clusternya**. Katakanlah data pertama adalah **xi**, dan data kedua adalah **xj**. Rumusnya adalah:
![image](https://user-images.githubusercontent.com/49611937/117751136-c7848780-b23e-11eb-91e4-cf54e1400e76.png)
- **Jarak Manhattan**, g=1
- **Jarak Euclidean**, g=2
- **Jarak Chebychev**, g=∞

**Inertia** dipakai untuk mengukur kualitas clustering, yaitu seberapa besar penyebaran object/data points data dalam satu cluster. Semakin kecil nilai inertia, semakin baik.
Cara pakainya adalah dengan:
```
print(
```
Setelah mendapatkan plottingan Inertia, pilih nilai n_clusters yang ada di **elbow** (cekungan). Itu adalah nilai n_clusters terbaik.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv')
X = dataset[['annual_income', 'spending_score']]
X = X.values  #convert dataframe to Array

# Modeling
cluster_model = KMeans(n_clusters=5, random_state=24)
labels = cluster_model.fit_predict(X)

# Pisahkan X untuk membuat plot
xs = X[:,0]
xy = X[:,1]
plt.scatter(xs, xy, c=labels, alpha=0.5)

# Assign cluster_centers_ dan kolom centroids
centroids   = cluster_model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Plotting
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
```
Result:

![image](https://user-images.githubusercontent.com/49611937/117758418-374d3f00-b24c-11eb-9fe2-34e4c00b9747.png)

---
### 
