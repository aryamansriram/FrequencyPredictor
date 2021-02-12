import numpy as np
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error

def seed_everything():
    '''
    Seeds everything for reproducible results
    :return: None
    '''
    np.random.seed(42)
    set_seed(42)

def create_rnn_model(config):
    '''
    Creates an RNN model with parameters from the config file
    :param config: dictionary with model parameters
    :return: keras model
    '''
    rnn_model = Sequential()
    rnn_model.add(LSTM(16,input_shape=(20,1)))
    rnn_model.add(Dense(10,activation="relu"))

    rnn_model.add(Dense(1,activation="sigmoid"))
    rnn_model.compile(optimizer=config["opt"],loss=config["loss"])
    return rnn_model

def preproc_rnn(X,y):
    '''
    Converts training data to RNN input format
    :param X: np.ndarray, training features
    :param y: np.ndarray,training labels
    :return: X_rnn,y_rnn, np.ndarrays, preprocessed np arrays
    '''
    X_rnn = X.reshape(-1,20,1)
    y_rnn = y/100
    return X_rnn,y_rnn


def KFoldCrossValidate(X_train,y_train,n_splits,config,model_args,model_func,callbacks=None):
    '''
    Performs KFold cross validation on the given data with the given model
    :param X_train: np.ndarray: Training features
    :param y_train: np.ndarray: Training Labels
    :param n_splits: int: Number of folds
    :param config: dict: Dictionary with model parameters
    :param model_args: dict: Dictionary with training parameters
    :param callbacks: Callback: callback function to save model checkpoint
    :param model_func: function: model creation function
    :return: None
    '''
    kf = KFold(n_splits=n_splits)
    for ii,(train_index,val_index) in enumerate(kf.split(X_train)):
        X_fold,X_fold_val = X_train[train_index],X_train[val_index]
        y_fold,y_fold_val = y_train[train_index],y_train[val_index]
        model = model_func(config)
        if callbacks!=None:
            history = model.fit(x=X_fold,y=y_fold,batch_size=model_args["batch_size"],epochs=model_args["epochs"],validation_data=(X_fold_val,y_fold_val),callbacks=callbacks)
        else:
            history = model.fit(x=X_fold,y=y_fold,batch_size=model_args["batch_size"],epochs=model_args["epochs"],validation_data=(X_fold_val,y_fold_val))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Fold ",)
        plt.legend(["train_loss","val_loss"],loc="upper right")
        plt.show()


def infer(X_test, y_test, model_func, path_to_model, config, preproc_func=None):
    model = model_func(config)
    model.load_weights(path_to_model)
    if preproc_func != None:
        X_test, y_test = preproc_func(X_test, y_test)

    y_pred = model.predict(X_test)

    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("R2: ", r2_score(y_test, y_pred))