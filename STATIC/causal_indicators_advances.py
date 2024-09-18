import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import os
class BaseCorrelationAnalysis:
    def __init__(self, u, lambdas, mu, sec_struct_data,stringa):
        self.u = u  # Ensure u is at least 2D
        self.lambdas = np.array(lambdas)
        self.mu = mu
        self.sec_struct_data = sec_struct_data
        self.name=stringa

    def _calculate_correlation_cost(self, i, j, t):
        C_ij_t_cost = np.zeros(len(t))
        for idx, z in enumerate(t):
            C_ij_0 = 0
            for k in range(1, len(self.lambdas)):
                C_ij_t_cost[idx] += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]) * np.exp(-self.mu * self.lambdas[k] * z))
                C_ij_0 += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]))
        return C_ij_t_cost, C_ij_0

    def _calculate_correlation_static(self, i, j):
        return self._calculate_correlation_cost(i, j, [0])[0][0]

class TimeCorrelation(BaseCorrelationAnalysis):
    def time_correlation(self, i, j, t):
        C_ij_t_cost, _ = self._calculate_correlation_cost(i, j, t)
        return C_ij_t_cost
    def normalize_autocorrelations(self, C_ii_t):
        C_ii_0 = C_ii_t[0]  # Primo valore di C_ii_t
        return C_ii_t / C_ii_0
    def estimate_tau_2(self, t, normalized_autocorrelations):
        def find_1_e_crossing(t, y):
            # Find the index where y crosses 1/e
            idx = np.argmin(np.abs(y - np.exp(-1)))
            return t[idx]

        tau_values = []
        for i in range(normalized_autocorrelations.shape[0]):
            tau = find_1_e_crossing(t, normalized_autocorrelations[i,:])
            tau_values.append(tau)
        
        
        tau_mean = np.mean(tau_values)
        return tau_mean, tau_values

    def plot_tau_histogram(self, t, normalized_autocorrelations, bins='auto'):
        tau_values = self.estimate_tau_2(t, normalized_autocorrelations)
        
        plt.figure(figsize=(10, 6))
        plt.hist(tau_values, bins=bins, edgecolor='black')
        plt.xlabel('Tau (time to reach 1/e)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Tau Values')
        plt.grid(True, alpha=0.3)
    
        # Check if the 'images' directory exists, if not, create it
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}tau_histogram.png')

    def estimate_tau(self, t, normalized_autocorrelations):
        def exp_fit(t, tau):
            return np.exp(-t / tau)

        taus = []
        for i in range(normalized_autocorrelations.shape[0]):
            popt, _ = curve_fit(exp_fit, t, normalized_autocorrelations[i,:], p0=[1])
            tau_estimated = popt[0]
            taus.append(tau_estimated)
        
        tau_mean = np.mean(taus)
        return tau_mean, taus

    def plot_autocorrelation_fits(self, t, normalized_autocorrelations):
        plt.figure(figsize=(12, 6))
        for i in range(normalized_autocorrelations.shape[0]):
            plt.plot(t, normalized_autocorrelations[i, :], label=f'Residuo {i}')
            tau_estimated = self.estimate_tau(t, normalized_autocorrelations)[0]
            plt.plot(t, np.exp(-t / tau_estimated), '--', label=f'Fit Residuo {i}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Autocorrelation')
        plt.title('Autocorrelation and Fits')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}autocorrelation_fits.png')
        

    def plot_time_correlation(self, i, j, t):
        C_ij_t = self.time_correlation(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, C_ij_t, label=f'Time Correlation C({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.title(f'Time Correlation between {i} and {j}')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}Time Correlation between {i} and {j}.png')

class TransferEntropy(BaseCorrelationAnalysis):
    def __init__(self, u, lambdas, mu, sec_struct_data,stringa):
        super().__init__(u, lambdas, mu, sec_struct_data,stringa)
        self.time_correlation_instance = TimeCorrelation(u, lambdas, mu, sec_struct_data,stringa)

    def transfer_entropy(self, i, j, t):
        C_ii_0 = self._calculate_correlation_static(i, i)
        C_jj_0 = self._calculate_correlation_static(j, j)
        C_ii_t = self.time_correlation_instance.time_correlation(i, i, t)
        C_jj_t = self.time_correlation_instance.time_correlation(j, j, t)
        C_ij_0 = self._calculate_correlation_static(i, j)
        C_ij_t = self.time_correlation_instance.time_correlation(i, j, t)

        alpha_ij_t = (C_ii_0 * C_ij_t - C_ij_0 * C_ii_t) ** 2
        beta_ij_t = (C_ii_0 * C_jj_0-(C_ij_0**2)) * (C_ii_0**2- C_ii_t ** 2)

        ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
        return -0.5 * np.log(1 - ratio)

    def plot_transfer_entropy(self, i, j, t):
        TE_ij = self.transfer_entropy(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, TE_ij, label=f'Transfer Entropy TE({i}->{j})')
        plt.xlabel('Time')
        plt.ylabel('Transfer Entropy')
        plt.title(f'Transfer Entropy from {i} to {j}')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}Transfer Entropy from {i} to {j}.png')

class TimeResponse(BaseCorrelationAnalysis):
    def time_response(self, i, j, t):
        C_ij_t_cost, C_ij_0 = self._calculate_correlation_cost(i, j, t)
        return C_ij_t_cost / C_ij_0

    def plot_time_response(self, i, j, t):
        R_ij_t = self.time_response(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, R_ij_t, label=f'Time Response R({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title(f'Time Response between {i} and {j}')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}Time Response between {i} and {j}.png')

class CorrelationMatrixOperations(BaseCorrelationAnalysis):
    def compute_static_correlation_matrix(self):
        n = self.u.shape[0]
        correlation_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                correlation_matrix[i, j] = self._calculate_correlation_static(i, j)
        return correlation_matrix

    def split_correlation_matrix(self, correlation_matrix):
        positive_matrix = np.where(correlation_matrix > 0, correlation_matrix, 0)
        negative_matrix = np.where(correlation_matrix < 0, correlation_matrix, 0)
        return positive_matrix, negative_matrix

    def plot_correlation_matrix(self, correlation_matrix, title='Correlation Matrix'):
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}Correlation Matrix.png')

    def plot_correlation_matrix_nan(self, correlation_matrix, title='Correlation Matrix', positive_only=False):
        plt.figure(figsize=(8, 6))
        masked_matrix = np.where(correlation_matrix > 0, correlation_matrix, np.nan) if positive_only else np.where(correlation_matrix < 0, correlation_matrix, np.nan)
        plt.imshow(masked_matrix, cmap='coolwarm', interpolation='none',origin='lower')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        #plt.invert_yaxis()
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure
        plt.savefig(f'images/{self.name}{title}.png')

class ResidualAnalysis(TimeCorrelation, TransferEntropy, TimeResponse, CorrelationMatrixOperations):
    def compute_mean_first_passage_time_matrix(self,adjacency_matrix):
        n = self.u.shape[0]
        T = np.zeros((n, n))
        
        # Calculate degrees (d_z)
        degrees = np.sum(adjacency_matrix, axis=1)
        print(degrees)
        # Calculate R matrix
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                R[i, j] = np.sum( 1/ self.lambdas[1:] * (self.u[i, 1:] - self.u[j, 1:])**2)
        
        # Calculate T matrix (MFPT)
        for i in range(n):
            for j in range(n):
                T[i, j] = 0.5 * np.sum(degrees * (R[i, j] + R[:, j] - R[i, :]))
        
        return T

    def plot_mfpt_matrix(self,adjacency_matrix):
        T = self.compute_mean_first_passage_time_matrix(adjacency_matrix)
        '''plt.figure(figsize=(10, 8))
        plt.imshow(T, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean First Passage Time')
        plt.title('Mean First Passage Time Matrix')
        plt.xlabel('Target Node j')
        plt.ylabel('Starting Node i')
        plt.show()'''

    def analyze_mfpt(self, adjacency_matrix):
        T = self.compute_mean_first_passage_time_matrix(adjacency_matrix)
        
        # Focus on residues 5 to 90
        #start_residue, end_residue = 0, 9
        T_subset = T
        
        # Flatten the matrix and remove zeros and diagonal elements
        flat_T = T_subset[~np.eye(T_subset.shape[0], dtype=bool)].flatten()
        non_zero_T = flat_T[flat_T != 0]
        
        # Sort the non-zero values
        sorted_T = np.sort(non_zero_T)
        print(sorted_T)
        # Calculate the 10th percentile as a threshold for "short" passage times
        threshold = np.percentile(sorted_T, 10)
        
        # Plot the distribution of lower MFPT values
        plt.figure(figsize=(10, 6))
        plt.hist(non_zero_T[non_zero_T <= threshold], bins=50, edgecolor='black')
        plt.title('Distribution of Lower Mean First Passage Time Values (Residues)')
        plt.xlabel('Mean First Passage Time')
        plt.ylabel('Frequency')
        plt.axvline(threshold, color='r', linestyle='--', label='10th percentile')
        plt.legend()
        if not os.path.exists(f'images/{self.name}/first_time/'):
            os.makedirs(f'images/{self.name}/first_time/')

        # Save the figure
        plt.savefig(f'images/{self.name}/first_time/Distribution of Lower Mean First Passage Time Values (Residues).png')
        
        # Highlight zones with shorter passage times in the MFPT matrix
        plt.figure(figsize=(10, 8))
        masked_T = np.ma.masked_where(T_subset > threshold, T_subset)
        plt.imshow(masked_T, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='Mean First Passage Time')
        plt.title('Mean First Passage Time Matrix (Residues, Highlighting Shorter Times)')
        plt.xlabel('Target Residue')
        plt.ylabel('Starting Residue')
        
        # Adjust tick labels to show actual residue numbers
        #plt.xticks(range(0, end_residue-start_residue+1, 10), range(start_residue, end_residue+1, 10))
        #plt.yticks(range(0, end_residue-start_residue+1, 10), range(start_residue, end_residue+1, 10))
        
        if not os.path.exists(f'images/{self.name}/first_time/'):
            os.makedirs(f'images/{self.name}/first_time/')

        # Save the figure
        plt.savefig(f'images/{self.name}/first_time/Mean First Passage Time Matrix.png')
        
        # Print some statistics
        print(f"10th percentile (threshold) of MFPT: {threshold:.4f}")
        print(f"Minimum non-zero MFPT: {np.min(non_zero_T):.4f}")
        print(f"Maximum MFPT: {np.max(non_zero_T):.4f}")
        print(f"Average MFPT: {np.mean(non_zero_T):.4f}")

    def compute_mean_correlation_over_segment(self, lista, t, time_idx):
        n = self.u.shape[0]
        residual_correlation_matrix = np.zeros((n, 1))
        print("tempo in cui sto stimando al correlazione:",t[time_idx:time_idx+1])
        for i in lista:
            for j in range(n):
                # Assumiamo che t sia un array di tempi e time_idx sia l'indice del tempo
                if time_idx < len(t):
                    residual_correlation_matrix[j] += self.time_correlation(i, j, t[time_idx:time_idx+1])[0]
                else:
                    raise IndexError(f"Time index {time_idx} out of range for time array.")
        residual_correlation_matrix /= len(lista)
        return residual_correlation_matrix
    
    def compute_residual_transfer_entropy_matrix(self, t, i, time_idx):
        n = self.u.shape[0]
        print(n)
        transfer_entropy_matrix = np.zeros(n)
        print(t[time_idx:time_idx + 1])
        for j in range(n):
            transfer_entropy_matrix[j] = self.transfer_entropy(i, j, t[time_idx:time_idx + 1])
        return transfer_entropy_matrix
    

    def compute_residual_time_response_matrix(self, t, i, time_idx):
        n = self.u.shape[0]
        time_response_matrix = np.zeros(n)
        for j in range(n):
            time_response_matrix[j] = self.time_response(i, j, t[time_idx:time_idx+1])
        print(time_response_matrix)
        return time_response_matrix
    

    def compute_mean_quantity_over_segment(self, lista, t, time_idx, quantity):
        n = self.u.shape[0]
        if quantity == 'correlation':
            mean_quantity_matrix = np.mean([
                self.compute_residual_correlation_matrix(t, i, time_idx)
                for i in lista
            ], axis=0)
        elif quantity == 'linear_response':
            mean_quantity_matrix = np.mean([
                self.compute_residual_response_matrix(t, i, time_idx)
                for i in lista
            ], axis=0)
        elif quantity == 'entropy':
            mean_quantity_matrix = np.mean([
                self.compute_residual_transfer_entropy_matrix(t, i, time_idx)
                for i in lista
            ], axis=0)
        else:
            raise ValueError("Unknown quantity type. Use 'correlation', 'linear_response', or 'entropy'.")
        return mean_quantity_matrix

    def plot_mean_quantity_over_segment(self, lista, t, time_idx, quantity):
        mean_quantity_matrix = self.compute_mean_quantity_over_segment(lista, t, time_idx, quantity)
        ylabel = {
            'correlation': 'Mean Correlation',
            'linear_response': 'Mean Time Response',
            'entropy': 'Mean Transfer Entropy'
        }[quantity]
        
        if quantity in ['correlation', 'linear_response']:
            self._plot_with_secondary_structure_and_time(mean_quantity_matrix, t, ylabel, f'Mean {ylabel} for selected residues')
        else:  # entropy
            self._plot_with_secondary_structure(mean_quantity_matrix, ylabel, f'Mean {ylabel} for selected residues')

    def _plot_with_secondary_structure_and_time(self, matrix, t, ylabel, title):
        sec_struct_info = self.sec_struct_data['Secondary Structure']
        residue_ids = self.sec_struct_data['Residue ID'].astype(int)

        colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
        sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

        plt.figure(figsize=(12, 8))
        for i in range(len(t)):
            plt.plot(range(len(matrix)), matrix[:, i], label=f't={t[i]:.2f}', alpha=0.7)

        # Plot the secondary structure bands
        current_color = 'black'
        start_idx = 0
        for idx, resid in enumerate(residue_ids):
            if sec_struct_colors[idx] != current_color:
                if idx > 0:
                    plt.axvspan(start_idx, idx, color=current_color, alpha=0.2)
                current_color = sec_struct_colors[idx]
                start_idx = idx #- 0.49
        
        # Plot the last segment
        plt.axvspan(start_idx, len(residue_ids) , color=current_color, alpha=0.2)

        # Create custom legend handles for secondary structure
        handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct) for struct, color in colors.items()]
        
        # Add legend for time points
        plt.legend(title='Time', loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add secondary structure legend
        plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.title(title)
        plt.xlabel('Residue Index')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure
        plt.savefig(f'images/{self.name}{title}.png')

    def time_correlation_2(self, i, j, t):
        # No changes made here
        C_ij_t = np.zeros(len(t))
        for idx, z in enumerate(t):
            C_ij_0 = 0
            C_ij_t_cost = 0
            for k in range(1, len(self.lambdas)):
                C_ij_t_cost += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]) * np.exp(-self.mu * self.lambdas[k] * z))
                C_ij_0 += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]))
                #print(k)
            C_ij_t[idx] = C_ij_t_cost
        return C_ij_t
    def time_correlation_3(self, i, j, t):
        # No changes made here
        Rijt_vector = np.zeros(len(t))
        for idx, z in enumerate(t):
            Rijt = 0
            for k in range(0, len(self.lambdas)):
                Rijt += ((self.u[i, k] * self.u[j, k]) * np.exp(-self.mu * self.lambdas[k] * z))
                #C_ij_0 += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]))
            Rijt_vector[idx] = Rijt#C_ij_t_cost#/C_ij_0 
        return Rijt
    
    def compute_residual_correlation_matrix(self, t,i, time_idx):
        """Calcola la matrice di correlazione dei residui per tutti i j e per un intervallo di tempo specificato."""
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]
        t_subset=t
        # Inizializza una matrice per le correlazioni temporali
        residual_correlation_matrix = np.zeros((n, n, len(t_subset)))
        print( t_subset)
        for j in range(n):
            residual_correlation_matrix[i, j, :] = self.time_correlation_2(i, j, t_subset)
        return residual_correlation_matrix[i,:,:]
    def compute_residual_response_matrix(self, t,i, time_idx):
        """Calcola la matrice di correlazione dei residui per tutti i j e per un intervallo di tempo specificato."""
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.3)
        end_idx = int(len(t) * 0.7)
        t_subset = t[start_idx:end_idx]
        t_subset=t
        # Inizializza una matrice per le correlazioni temporali
        residual_correlation_matrix = np.zeros((n, n, len(t_subset)))
        for j in range(n):
            residual_correlation_matrix[i, j, :] = self.time_correlation_3(i, j, t_subset)
        return residual_correlation_matrix[i,:,:]
    
    
    def plot_residual_correlation_vs_j(self, i, t, time_idx):
        residual_correlation_matrix = self.compute_residual_correlation_matrix(t, i, time_idx)
        self._plot_with_secondary_structure(residual_correlation_matrix, f'Correlation with i={i}', f'Residual Correlation C_ij for i={i} as a function of j at time index {time_idx}')

    def plot_residual_time_response_vs_j(self, i, t, time_idx):
        time_response_matrix = self.compute_residual_response_matrix(t, i, time_idx)
        self._plot_with_secondary_structure(time_response_matrix, f'Response with i={i}', f'Time Response R_ij for i={i} as a function of j at time index {time_idx}')

    def plot_residual_transfer_entropy_vs_j(self, i, t, time_idx):
        transfer_entropy_matrix = self.compute_residual_transfer_entropy_matrix(t, i, time_idx)
        self._plot_with_secondary_structure(transfer_entropy_matrix, f'Transfer Entropy with i={i}', f'Transfer Entropy TE_ij for i={i} as a function of j at time index {time_idx}')

    def plot_multiple_time_correlations(self, pairs, t):
        plt.figure(figsize=(12, 8))
        
        for i, j in pairs:
            C_ij_t = self.time_correlation(i, j, t)
            plt.plot(t, C_ij_t, label=f'C({i},{j})')
        
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.title('Time Correlation for Multiple Pairs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'images/{self.name}TimeCorrelationforMultiplePairs.png')
    def _plot_with_secondary_structure(self, matrix, ylabel, title):
        sec_struct_info = self.sec_struct_data['Secondary Structure']
        residue_ids = self.sec_struct_data['Residue ID'].astype(int)

        colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
        sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

        plt.figure(figsize=(12, 8))
        plt.plot(range(len(matrix)), matrix, marker='o', linestyle='-', alpha=0.7)

        # Plot the secondary structure bands
        current_color = 'black'
        start_idx = 0
        for idx, resid in enumerate(residue_ids):
            if sec_struct_colors[idx] != current_color:
                if idx > 0:
                    plt.axvspan(start_idx, idx, color=current_color, alpha=0.2)
                current_color = sec_struct_colors[idx]
                start_idx = idx 
        
        # Plot the last segment
        plt.axvspan(start_idx, len(residue_ids), color=current_color, alpha=0.2)

        # Create custom legend handles
        handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct) for struct, color in colors.items()]
        plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.title(title)
        plt.xlabel('Residue Index')
        plt.ylabel(ylabel)
        #plt.ylim(-0.02,0.02)
        plt.grid(True)
        if not os.path.exists('images'):
            os.makedirs('images')

        # Save the figure
        plt.savefig(f'images/{self.name}{title}.png')
        
