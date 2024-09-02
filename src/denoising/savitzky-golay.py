import numpy as np
import matplotlib.pyplot as plt

def savitzky_golay_matrix(degree, n_left, n_right):
    """Compute the Savitzky-Golay matrix."""
    # Compute the coefficients
    x = np.arange(-n_left, n_right + 1)
    A = np.vander(x, degree + 1)
    ATA = np.dot(A.T, A)
    ATAI = np.linalg.inv(ATA)
    ATAIAT = np.dot(ATAI, A.T)
    return ATAIAT

def savitzky_golay(signal, degree, n_left, n_right) -> np.ndarray:
    clean_signal = np.zeros_like(signal)
    ATAIAT = savitzky_golay_matrix(degree, n_left, n_right)
    for i in range(0, len(signal)):
        window = np.zeros(n_left + n_right + 1)
        for j in range(0, n_left + n_right + 1):
            window[j] = signal[(i - n_left + j) % len(signal)]
        clean_signal[i] = np.sum(np.dot(ATAIAT, window))
    return clean_signal



if __name__ == '__main__':
    
    noise = np.random.normal(0, 0.1, 1000)
    original_signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
    signal = original_signal + noise
    
    clean_signal = savitzky_golay(signal, 2, 20, 20)

    plt.plot(original_signal, label='Original signal')
    plt.plot(signal, label='Noisy signal')
    plt.plot(clean_signal, label='Clean signal')
    plt.legend()
    plt.show()

