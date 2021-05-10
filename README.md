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


## Memilih Matrix
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

lr = LogisticRegression()

# Training
model = lr.fit(X_train, y_train)

# Testing
pred = lr.predict(X_test)

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

