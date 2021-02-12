from main_utils import seed_everything,create_rnn_model,preproc_rnn,KFoldCrossValidate,infer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
import numpy as np


if __name__=="__main__":
    seed_everything()
    n_splits=2
    X = np.load("data/X.npy")
    y = np.load("data/Y.npy")

    config = {
        "opt": Nadam(lr=0.001),
        "loss": MeanSquaredError(),

    }

    model_args = {
        "batch_size": 16,
        "epochs": 30,
    }

    ckpt_path = "rnn_cp/rnn.ckpt"
    rnn_cp = ModelCheckpoint(ckpt_path,monitor="val_loss",save_best_only=True,save_weights_only=True)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_rnn,y_rnn = preproc_rnn(X_train,y_train)
    KFoldCrossValidate(X_rnn,y_rnn,n_splits,config,model_args,create_rnn_model,[rnn_cp])

    print("TEST PERFORMANCE: ")
    infer(X_test,y_test,create_rnn_model,ckpt_path,config,preproc_func = preproc_rnn)




