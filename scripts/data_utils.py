# This script contains functions for generating data for the regression task


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def generate_sin(t,f,sr):
    N = t*sr
    samples = np.arange(N) / sr

    signal = np.sin(2 * np.pi * f * samples)
    return samples,signal

def plot_signal(samples,signal):
    plt.plot(samples, signal)
    plt.show()


def choose_freq(low,high,res):
    freq = np.random.choice(np.arange(low,high,res))
    return freq

def generate_noisy_sig(sig,snr_db=10):
    power_sig = np.mean(sig**2)
    power_sig_db = 10*np.log10(power_sig)
    noise_db = power_sig_db-snr_db
    noise_watts = 10**(noise_db/10)
    mean_noise = 0
    gaussian_noise = np.random.normal(mean_noise,np.sqrt(noise_watts),len(sig))
    noisy_signal = sig+gaussian_noise
    return noisy_signal



def generate_data(sr,t,low_f,high_f,res,num_data=1000):
    sig_samples =[]
    freq = []

    for i in range(num_data):
        print("Iteration: ",i," of ",num_data)
        f = choose_freq(low_f, high_f,res)

        samp, sig = generate_sin(t, f, sr)


        noisy_signal = generate_noisy_sig(sig)
        sig_samples.append(noisy_signal)
        freq.append(f)
    print("L: ",len(np.array(sig_samples)))
    return np.array(sig_samples),np.array(freq)

