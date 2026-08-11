import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def simulate_brownian_motion(T=1.0, N=252, n_paths=10, plot=True, savefig = None):
    #T = 1.0                 # Total time in year
    #N = 252                 # Number of time steps/Number of trading days
    dt = T/N                # Size of each time step
    # n_paths = 10            # Paths to plot
    #n_path_large = 10000    # Paths for verification – the law of large numbers needs many samples

    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=(N, n_paths))
    # np.random.normal -> gener ate a normally distributed number
    # loc = 0 -> mean = 0
    # scale = standard deviation -> for a standard brownian motion, the increment W(t+delta_t) - W(t) has a variance of dealt_t so standard deviation is in sqrt(delta_t)
    # size=(N, n_paths) creates a 2‑D array: rows are time steps (0 to N-1), columns are different paths.
    # So dW[i, j] is the random increment for the i-th time step of the j-th path.

    W = np.vstack([np.zeros((1,n_paths)), dw.cumsum(axis=0)])
    # dW.cumsum(axis=0) computes the cumulative sum along the time axis (rows). For a single path, the cumulative sum at step i is the sum of the first i increments. 
    # This gives us the Brownian motion values at each time step, except it starts with the sum of the first increment, not zero.

    # np.zeros((1, n_paths)) creates a row of zeros – one for each path – representing W(0)=0.

    # np.vstack([...]) stacks the zero row on top of the cumulative sums. Now W[0,:] is zero, and W[i,:] for i>0 is the Brownian motion value after i steps.

    time = np.linspace(0,T,N+1)
    # Creates an array of N+1 equally spaced points from 0 to 1. Each element corresponds to a row in W.

    if plot:
        plt.figure(figsize=(10,6))
        for i in range(n_paths):
            plt.plot(time,W[:,i])
        #plot the graph, as we generate 10 paths, with first column equal time, second column equal path 0, third column equal path 1 and continue, we plot it like this

        plt.axhline(y=0)

        plt.title('Simulated Brownian Motion Paths ('+str(n_paths)+' paths)')
        plt.xlabel('Time (years)')
        plt.ylabel('W(t)')
        plt.grid(True, alpha=0.3)

        if savefig is not None:
            plt.savefig(savefig, dpi=150)
        plt.show()
    
    return time, W, T


    