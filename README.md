# Machine Learning
## Jenis Machine Learning

![image](https://user-images.githubusercontent.com/49611937/117688727-4055f600-b1e3-11eb-83e5-80b8df160b1c.png)

1. **Regression**: Linear Regression
2. **Classification**: Decision Tree, Logistic Regression
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


## Contoh Implementasi
### 1. Decision Tree
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
print('Akurasi Training: ', model.score(X_train, y_train)
print('Akurasi Testing: ', model.score(X_test, y_test)

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
