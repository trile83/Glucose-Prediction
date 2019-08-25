# Glucose Prediction

### Using type 1 diabetes patient data for training and testing
### Using recurrent neural network for deep learn model

## Using LSTM
### Requirements: 
* Python 3.6
* Tensorflow 1.12
* Pandas 
* Numpy
* Scikitlearn

### Data Summary

![Data Head](/img/DataHead.PNG)


### Variables Heatmap
![Heatmap](/img/Heatmap.PNG)


### Prepare the LSTM model for training

```python3

# design network
model = Sequential()
model.add(LSTM(units=100, input_shape=(train_X.shape[1], train_X.shape[2]) 
               ,  return_sequences= True))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(1))
model.add(Activation("linear"))

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss = 'mse', 
              optimizer = adam)
model.summary()

```

### Model Summary

![Model](/img/Model.PNG)

### Training Process
 We train the model for 30 epochs, using early stopping with patience of 8

![Training Loss](/img/TrainingLoss.PNG)

### Evaluating Model Performance:
The model yields RMSE of 35.74696945470081

Plotting actual glucose level and predicted glucose level

![Prediction vs Actual](/img/PredictionVsActual.PNG)

## Using Random Forest Regressor

### Evaluating Model performance

RMSE = 38.39601968742631

Plotting actual glucose level and predicted glucose level

![Prediction vs Actual](/img/PredictionVsActualRF.PNG)
