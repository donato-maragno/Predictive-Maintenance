# Predictive maintenance of a turbofan engine
In this project I've developed three Machine learning solutions to predict the Remaining Useful Life (RUL) of a turbofan engine. This is an hot topic in the field of **Predictive Maintenance**, where one of the most importance goals is to predict **when the next fault will happen**. 

In order to achieve this goal, I've developed three algorithms:
- Long Short-term Memory (LSTM)
- Convolutional Neural Network
- Random forest classifier

## Dataset description

The dataset used in this project is: **Turbofan Engine Degradation Simulation [Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)**, provided by NASA.  Data are available in the form of time series: 3 operational settings, 21 sensor measurements and cycle — i.e. observations in terms of time for working life. 
> The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate.

Eache engine has a different life durations. 

## Key points
### Label
In all the three algorithms I used w0 and w1 (with w0 < w1) as periods of cycles. This means that I want to predict if the engine is close to breaking within the next **w0** (or w1) cycles or not. For this purpose, I've realized a label  whose values are:
- 0: the engine is not going to break in the next w0
- 1: the engine will break within w0 cycles
- 2: the engine will break within w1 cycles

### LSTM
In order to feed the LSTM, I had to change the shape of inputs from 2D to 3D. The third dimension is the **sequence lenght** which corresponds to the time window to consider for each single prediction. 

### CNN
In order to apply a CNN I had to transform the time series into images using **Recurrence Plots**, read this [article](https://towardsdatascience.com/remaining-life-estimation-with-keras-2334514f9c61) for more information.

## Addiotional information
Don't estitate to write me for more information: donato.maragno.da@gmail.com