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
    if not os.path.exists('images'):
        os.makedirs('images')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}Multiple_time_resposne.png')
