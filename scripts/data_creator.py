from scripts.data_utils import generate_data,plot_signal
import numpy as np




if __name__ == '__main__':
    np.random.seed(42)
    SR = 8000
    T = 20/SR
    LOW = 50
    HIGH = 100
    RES = 0.1
    NUM_DATA = 10000
    t = np.arange(T*SR)/SR
    X,y = generate_data(SR,T,LOW,HIGH,RES,NUM_DATA)
    np.save("data/X.npy",X)
    np.save("data/Y.npy",y)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
