# Support Vector Machine (SVM)
**SVM** is good for classification task, even for the data that does not have a linear correspondence (Better than KNN in accuracy for this case). It works by dividing data into multiple classes using **Hyper-plane**.
Hyper-plane can divide **data points**. In 2D, it is simply a line.

Hyper-plane can be made by **Support vectors**. Support vectors is 2 closest point to hyper-plane and having the same distance to hyper-plane. The hyper-plane we should pick is the one with the widest/greatest margin. **Soft Margin** is a margin when there is point (outlier) inside the margin. **Hard Margin** is the one without data points inside the margin.

<img src="https://user-images.githubusercontent.com/49611937/117558332-c3bdfd00-b0a6-11eb-9c26-b4d98573dc4a.png" width="400" height="300">

**Kernel** is needed to create a hyper-plane when we have a data like this (left). It helps to upgrade the dimension, so it might become like this (right). **Kernel** is a function like **(x1)^2 + (x2)^2 = x3**. The example of kernel is: Linear, Polynomial, Circular, and Hyperbolic Tangent (Sigmoid).

<img src="https://user-images.githubusercontent.com/49611937/117558511-48f5e180-b0a8-11eb-8456-f0464506386f.png" width="300" height="200" align="left">
<img src="https://user-images.githubusercontent.com/49611937/117558527-7f336100-b0a8-11eb-8c5c-360e348685b9.png" width="300" height="200">




