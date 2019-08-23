# Glucose-LSTM__RNN

### Using type 1 diabetes patient data for training and testing
### Using recurrent neural network for deep learn model

## Requirements: 
* Python 3.6
* Tensorflow 1.12
* Pandas 
* Numpy
* Scikitlearn

![Heatmap](/img/Heatmap.PNG)
Format: ![Alt Text](url)

### Prepare the LSTM model for training

```python3

# design network
model = Sequential()
model.add(LSTM(units=100, input_shape=(train_X.shape[1], train_X.shape[2]),  
               return_sequences= True))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=100))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss = 'mean_squared_error', 
              optimizer = "adam")
model.summary()


```
