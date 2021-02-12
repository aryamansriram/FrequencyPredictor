# FrequencyPredictor

This project aims to predict the fundamental frequency of a signal using Neural Networks

## Steps to set up environment:
1. Set up a python env (preferable conda environment)<br>
2. Go to the root of the repository,open the terminal and and run `pip install -r requirements.txt` <br>
3. You should see libraries being installed in your system.<br>

## Directory structure

- Scripts: Contains python scripts<br>
  1. `data_utils.py`: Contains functions to create and load the data<br>
  2. `data_creator.py`: Contains code to use functions from data_utils and create the data.<br> 
  3. `main_utils.py`: Contains functions to train and test the data<br>
  4. `main.py`: Contains code to train and test out the model and print the results.<br>
 
- ipython_nbs: Contains jupyter-notebooks which describe all the models explored
- data: Contains dummy data generated for the experiment

## Steps to run

- Open your terminal in the root of your repository and run the data_creator.py file by running the command  `python scripts/data_creator.py` in your terminal.
- This will create the data and store it in the data folder with the name `X.npy` and `y.npy` respectively.
- Now run the command `python scripts/main.py` from the root of your repository in your terminal. 
- The script will run kfold cross validation for 2 times (which can be modified by changing the n_splits variable in the code) and produce a plot for every fold. After this it will load the model which gave the lowest loss and make predictions on the test data using it. It then calculates the mse and r2 score metrics and prints them out as well.
