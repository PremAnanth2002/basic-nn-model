# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network.
Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.  

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235488/187118111-672bca5f-1969-49b2-a382-6f5928f9e7d1.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

df=pd.read_csv("data.csv")

df.head()

x=df[["X"]].values
y=df[["Y"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

model=Sequential([Dense(16,activation='relu'),
                  Dense(8,activation='relu'),
                  Dense(1)])

model.compile(loss="mae",optimizer="adam",metrics=["mse"])

history=model.fit(x_train,y_train,epochs=1000)

pred=model.predict(x_test)
tf.round(pred)

tf.round(model.predict([[50]]))

pd.DataFrame(history.history).plot()
plt.title("Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")

r=tf.keras.metrics.RootMeanSquaredError()
r(y_test,pred)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/69795479/187755241-71e07e76-f88c-4c43-9c9e-27db1659e6d9.png)

## OUTPUT

![image](https://user-images.githubusercontent.com/69795479/187729554-84ed6458-d23c-49d8-af6b-91349a386dcb.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/69795479/187754270-d88b16f7-7c04-404e-b755-cf6817e0e550.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/69795479/187754374-985f3e99-737f-4c74-b782-05b852b37b97.png)

## RESULT
A Basic neural network regression model for the given dataset is developed successfully.
