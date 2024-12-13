import numpy as np
import matplotlib.pyplot as plt
import os
def plot_time_response_multiple(time_response, residue_pairs, t, title,name):
    plt.figure(figsize=(12, 8))
    for i, j in residue_pairs:
        R_ij_t = time_response.time_response(i, j, t)
        plt.plot(t, R_ij_t, label=f'R({i},{j})')
    
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if not os.path.exists(f'images/{name}/Multiple_time_response/'):
        os.makedirs(f'images/{name}/Multiple_time_response/')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}/Multiple_time_resposne/risposte.png')


def plot_time_correlation_multiple(time_response, residue_pairs, t, title,name):
    plt.figure(figsize=(12, 8))
    for i, j in residue_pairs:
        R_ij_t = time_response.time_correlation(i, j, t)
        plt.plot(t, R_ij_t, label=f'C({i},{j})')
    
    plt.xlabel('Time')
    plt.ylabel('correlation')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if not os.path.exists(f'images/{name}/Multiple_time_correlation/'):
        os.makedirs(f'images/{name}/Multiple_time_correlation/')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}/Multiple_time_correlation/correlation.png')
def plot_time_entropy_multiple(time_response, residue_pairs, t, title,name):
    plt.figure(figsize=(12, 8))
    for i, j in residue_pairs:
        R_ij_t = time_response.transfer_entropy(i, j, t)
        plt.plot(t, R_ij_t, label=f'TE({i},{j})')
    
    plt.xlabel('Time')
    plt.ylabel('Transfer entropy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if not os.path.exists(f'images/{name}/Multiple_time_correlation/'):
        os.makedirs(f'images/{name}/Multiple_time_correlation/')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}/Multiple_time_correlation/entropy.png')
