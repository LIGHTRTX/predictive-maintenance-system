import numpy as np

def generate_signal(overload_ratio):
    t = np.linspace(0, 1, 1000)
    base = 0.3*np.sin(2*np.pi*30*t) + 0.05*np.random.randn(1000)
    if overload_ratio > 1.5:
        base += 0.4*np.sin(2*np.pi*200*t)
    return base

def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))