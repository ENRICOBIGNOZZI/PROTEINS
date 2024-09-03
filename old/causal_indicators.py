import numpy as np
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, u, lambdas, mu, sec_struct_data):
        self.u = u # Ensure u is at least 2D
        self.lambdas = np.array(lambdas)
        self.mu = mu
        self.sec_struct_data=sec_struct_data

    def time_correlation(self, i, j, t):
        # No changes made here
        C_ij_t = np.zeros(len(t))
        for idx, z in enumerate(t):
            C_ij_0 = 0
            C_ij_t_cost = 0
            for k in range(3, len(self.lambdas)):
                C_ij_t_cost += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]) * np.exp(-self.mu * self.lambdas[k] * z))
                C_ij_0 += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]))
            C_ij_t[idx] = C_ij_t_cost
        return C_ij_t
    def plot_multiple_time_correlations(self, pairs, t):
        plt.figure(figsize=(10, 8))
        for (i, j) in pairs:
            C_ij_t = self.time_correlation(i, j, t)
            plt.plot(t, C_ij_t, label=f'C({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.title('Time Correlation for Multiple Pairs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def correlation_static_0(self, i, j):
        C_jj_0 = self.time_correlation(j, j, np.array([0.0]))[0]
        C_ii_0 = self.time_correlation(i, i, np.array([0.0]))[0]
        C_ij_0 = self.time_correlation(i, j, np.array([0.0]))[0]
        return C_ij_0 / np.sqrt(C_ii_0 * C_jj_0)


    def transfer_entropy(self, i, j, t):
        C_ii_0 = self.time_correlation(i, i, np.array([0.0]))[0]
        C_jj_0 = self.time_correlation(j, j, np.array([0.0]))[0]
        C_ii_t = self.time_correlation(i, i, t)
        C_jj_t = self.time_correlation(j, j, t)
        C_ij_0 = self.time_correlation(i, j, np.array([0.0]))[0]
        C_ij_t = self.time_correlation(i, j, t)

        alpha_ij_t = (C_ii_0 * C_jj_t - C_ij_0 * C_ii_t) ** 2
        beta_ij_t = (C_ii_0 * C_jj_0) * (C_ii_t - C_ij_t ** 2)

        ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
        return -0.5 * np.log(1 - ratio)
    def static_transfer_entropy(self, i, j):
        C_ii_0 = self.time_correlation(i, i, np.array([0.0]))[0]
        C_jj_0 = self.time_correlation(j, j, np.array([0.0]))[0]
        C_ii_t = self.time_correlation(i, i, np.array([0.3]))
        #C_jj_t = self.time_correlation(j, j, np.array([0.0]))
        C_ij_0 = self.time_correlation(i, j, np.array([0.0]))[0]
        C_ij_t = self.time_correlation(i, j, np.array([0.3]))

        alpha_ij_t = (C_ii_0 * C_ij_t - C_ij_0 * C_ii_t) ** 2
        beta_ij_t = (C_ii_0 * C_jj_0-(C_ij_0**2)) * (C_ii_0**2- C_ii_t ** 2)

        ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
        return -0.5 * np.log(1 - ratio)

    def time_response(self, i, j, t):
        C_ij_t = np.zeros(len(t))
        for idx, z in enumerate(t):
            C_ij_0 = 0
            C_ij_t_cost = 0
            for k in range(1, len(self.lambdas)):
                C_ij_t_cost += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]) * np.exp(-self.mu * self.lambdas[k] * z))
                C_ij_0 += ((self.u[i, k] * self.u[j, k] / self.lambdas[k]))
            C_ij_t[idx] = C_ij_t_cost / C_ij_0
        return C_ij_t
    
    def compute_static_response(self, i, j):
        return self.correlation_static_0( i, j)

    # Plotting time correlation
    def plot_time_correlation(self, i, j, t):
        C_ij_t = self.time_correlation(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, C_ij_t, label=f'Time Correlation C({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.title(f'Time Correlation between {i} and {j}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plotting transfer entropy
    def plot_transfer_entropy(self, i, j, t):
        TE_ij = self.transfer_entropy(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, TE_ij, label=f'Transfer Entropy TE({i}->{j})')
        plt.xlabel('Time')
        plt.ylabel('Transfer Entropy')
        plt.title(f'Transfer Entropy from {i} to {j}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plotting time response
    def plot_time_response(self, i, j, t):
        R_ij_t = self.time_response(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, R_ij_t, label=f'Time Response R({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title(f'Time Response between {i} and {j}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def compute_correlation_matrix(self):
        """Compute the correlation matrix for all pairs (i, j)."""
        n = self.u.shape[0]
        correlation_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                correlation_matrix[i, j] = self.correlation_static_0(i, j)

        return correlation_matrix

    def split_correlation_matrix(self, correlation_matrix):
        """Split the correlation matrix into positive and negative correlation matrices."""
        positive_matrix = np.where(correlation_matrix > 0, correlation_matrix, 0)
        negative_matrix = np.where(correlation_matrix < 0, correlation_matrix, 0)
        return positive_matrix, negative_matrix

    def plot_correlation_matrix(self, correlation_matrix, title='Correlation Matrix'):
        """Plot a correlation matrix."""
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        plt.show()
    def plot_correlation_matrix_2(self, correlation_matrix, title='Correlation Matrix', positive_only=False):
        """Plot a correlation matrix with specific color coding."""
        plt.figure(figsize=(8, 6))

        if positive_only:
            # Set negative correlations to white
            masked_matrix = np.where(correlation_matrix > 0, correlation_matrix, np.nan)
            cmap = 'coolwarm'
        else:
            # Set positive correlations to white
            masked_matrix = np.where(correlation_matrix < 0, correlation_matrix, np.nan)
            cmap = 'coolwarm'

        plt.imshow(masked_matrix, cmap=cmap, interpolation='none')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        plt.show()
    def plot_fixed_i_correlations(self, i):
        """Plot the correlation C_ij as a function of j for a fixed i."""
        n = self.u.shape[0]
        correlations = np.zeros(n)

        for j in range(n):
            correlations[j] = self.correlation_static_0(i, j)

        plt.figure(figsize=(10, 6))
        plt.plot(range(n), correlations, marker='o', linestyle='-', color='blue')
        plt.title(f'Correlation C_ij for i={i} as a function of j')
        plt.xlabel('Index j')
        plt.ylabel(f'C_ij with i={i}')
        plt.grid(True)
        plt.show()
    def compute_transfer_entropy_matrix(self, t):
        n = self.u.shape[0]
        transfer_entropy_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                transfer_entropy_matrix[i, j] = self.transfer_entropy(i, j, t)
        return transfer_entropy_matrix

    def plot_fixed_i_transfer_entropy(self, i):
        n = self.u.shape[0]
        transfer_entropy = np.zeros(n)
        for j in range(n):
            transfer_entropy[j] = self.static_transfer_entropy(i, j)
        plt.figure(figsize=(10, 6))
        plt.plot(range(n), transfer_entropy, marker='o', linestyle='-', color='green')
        plt.title(f'Transfer Entropy T_ij for i={i} as a function of j')
        plt.xlabel('Index j')
        plt.ylabel(f'Transfer Entropy T_ij with i={i}')
        plt.grid(True)
        plt.show()



    def plot_fixed_i_time_response(self, i):
        n = self.u.shape[0]
        time_responses = np.zeros(n)
        for j in range(n):
            time_responses[j] = self.compute_static_response(i, j)
        plt.figure(figsize=(10, 6))
        plt.plot(range(n), time_responses, marker='o', linestyle='-', color='red')
        plt.title(f'Time Response R_ij for i={i} as a function of j')
        plt.xlabel('Index j')
        plt.ylabel(f'Time Response R_ij with i={i}')
        plt.grid(True)
        plt.show()

    def plot_time_response_matrix(self, time_response_matrix, title='Time Response Matrix', positive_only=False):
        plt.figure(figsize=(8, 6))
        if positive_only:
            masked_matrix = np.where(time_response_matrix > 0, time_response_matrix, np.nan)
            cmap = 'coolwarm'
        else:
            masked_matrix = np.where(time_response_matrix < 0, time_response_matrix, np.nan)
            cmap = 'coolwarm'
        plt.imshow(masked_matrix, cmap=cmap, interpolation='none')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        plt.show()
    def compute_residual_time_response_matrix(self, t, i, time_idx):
        """Calcola la matrice di risposta al tempo per tutti i j e per un indice temporale specificato."""
        n = self.u.shape[0]
        # Inizializza una matrice per le risposte temporali
        time_response_matrix = np.zeros((n, n))
        for j in range(n):
            time_response_matrix[i, j] = self.time_response(i, j, t[time_idx:time_idx+1])
        return time_response_matrix

    def compute_mean_correlation_over_segment(self, lista, t, time_idx):
        n = self.u.shape[0]
        residual_correlation_matrix = np.zeros((n, 1))
        for i in lista:
            for j in range(n):
                residual_correlation_matrix[j] += self.time_correlation(i, j, t[time_idx:time_idx+1])[0]
        residual_correlation_matrix /= len(lista)
        return residual_correlation_matrix

    def plot_mean_correlation_over_segment(self, lista, t, time_idx):
        """Visualizza la correlazione media sui segmenti in funzione di j per una lista di i, a un tempo specificato."""
        mean_correlation_matrix = self.compute_mean_correlation_over_segment(lista, t, time_idx)

        def plot_correlation_bands(self, mean_correlation_matrix, ylabel, title):
            sec_struct_info = self.get_sec_struct_info()
            residue_ids = self.sec_struct_data['Residue ID'].astype(int)

            colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
            sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

            plt.figure(figsize=(12, 8))
            plt.plot(range(self.u.shape[0]), mean_correlation_matrix, marker='o', linestyle='-', alpha=0.7)

            current_color = 'black'
            for idx, resid in enumerate(residue_ids):
                if sec_struct_colors[idx] != current_color:
                    if idx > 0:
                        plt.axvspan(idx - 0.5, idx + 0.5, color=current_color, alpha=0.2)
                    current_color = sec_struct_colors[idx]
                    start_idx = idx - 0.5
            plt.axvspan(start_idx, len(residue_ids) - 0.5, color=current_color, alpha=0.2)

            handles = [plt.Line2D([0], [0], color=color, lw=4, label=struct)
                    for struct, color in colors.items()]
            plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            plt.title(title)
            plt.xlabel('Index j')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()

        plot_correlation_bands(self, mean_correlation_matrix, 
                            'Mean Correlation', 
                            f'Mean Correlation C_ij for selected i as a function of j at time index {time_idx}')

    def plot_residual_correlation_vs_j(self, i, t, time_idx):
        """Visualizza le correlazioni dei residui in funzione di j per un dato i, a un tempo specificato."""
        residual_correlation_matrix = self.compute_residual_time_response_matrix(t, i, time_idx)
        print(t[time_idx])
        self.plot_with_secondary_structure_bands(
            i, residual_correlation_matrix, [t[time_idx]], 
            f'Correlation with i={i}', 
            f'Residual Correlation C_ij for i={i} as a function of j at time index {time_idx}',
            self.time_correlation
        )

    def plot_residual_time_response_vs_j(self, i, t, time_idx):
        """Visualizza le risposte al tempo in funzione di j per un dato i, a un tempo specificato."""
        time_response_matrix = self.compute_residual_time_response_matrix(t, i, time_idx)
        self.plot_with_secondary_structure_bands(
            i, time_response_matrix, [t[time_idx]], 
            f'Response with i={i}', 
            f'Time Response R_ij for i={i} as a function of j at time index {time_idx}',
            self.time_response
        )

    def plot_residual_transfer_entropy_vs_j(self, i, t, time_idx):
        """Visualizza la transfer entropy in funzione di j per un dato i, a un tempo specificato."""
        transfer_entropy_matrix = self.compute_residual_transfer_entropy_matrix(t, i, time_idx)
        self.plot_with_secondary_structure_bands(
            i, transfer_entropy_matrix, [t[time_idx]], 
            f'Transfer Entropy with i={i}', 
            f'Transfer Entropy TE_ij for i={i} as a function of j at time index {time_idx}',
            self.transfer_entropy
        )

    def compute_mean_linear_response_over_segment(self, lista, t, time_idx):
        n = self.u.shape[0]
        residual_linear_response_matrix = np.zeros((n, 1))
        for i in lista:
            for j in range(n):
                residual_linear_response_matrix[j] += self.time_response(i, j, t[time_idx:time_idx+1])[0]
        residual_linear_response_matrix /= len(lista)
        return residual_linear_response_matrix

    def compute_mean_entropy_over_segment(self, lista, t, time_idx):
        n = self.u.shape[0]
        residual_entropy_matrix = np.zeros((n, 1))
        for i in lista:
            for j in range(n):
                residual_entropy_matrix[j] += self.transfer_entropy(i, j, t[time_idx:time_idx+1])[0]
        residual_entropy_matrix /= len(lista)
        return residual_entropy_matrix

    def plot_mean_quantity_over_segment(self, lista, t, time_idx, quantity='correlation'):
        if quantity == 'correlation':
            mean_quantity_matrix = self.compute_mean_correlation_over_segment(lista, t, time_idx)
            ylabel = 'Mean Correlation'
            title = 'Mean Correlation C_ij for selected i as a function of j'
        elif quantity == 'linear_response':
            mean_quantity_matrix = self.compute_mean_linear_response_over_segment(lista, t, time_idx)
            ylabel = 'Mean Linear Response'
            title = 'Mean Linear Response for selected i as a function of j'
        elif quantity == 'entropy':
            mean_quantity_matrix = self.compute_mean_entropy_over_segment(lista, t, time_idx)
            ylabel = 'Mean Entropy'
            title = 'Mean Entropy for selected i as a function of j'
        else:
            raise ValueError("Invalid quantity specified. Choose 'correlation', 'linear_response', or 'entropy'.")

        def plot_quantity_bands(self, mean_quantity_matrix, ylabel, title):
            sec_struct_info = self.get_sec_struct_info()
            residue_ids = self.sec_struct_data['Residue ID'].astype(int)

            colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
            sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

            plt.figure(figsize=(12, 8))
            plt.plot(range(self.u.shape[0]), mean_quantity_matrix, marker='o', linestyle='-', alpha=0.7)

            current_color = 'black'
            for idx, resid in enumerate(residue_ids):
                if sec_struct_colors[idx] != current_color:
                    if idx > 0:
                        plt.axvspan(idx - 0.5, idx + 0.5, color=current_color, alpha=0.2)
                    current_color = sec_struct_colors[idx]
                    start_idx = idx - 0.5
            plt.axvspan(start_idx, len(residue_ids) - 0.5, color=current_color, alpha=0.2)

            handles = [plt.Line2D([0], [0], color=color, lw=4, label=struct)
                    for struct, color in colors.items()]
            plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            plt.title(title)
            plt.xlabel('Index j')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()

        plot_quantity_bands(self, mean_quantity_matrix, ylabel, title)


    def compute_autocorrelations(self, t):
        """Calcola le autocorrelazioni C_ii(t) per tutti i residui i."""
        n = self.u.shape[0]
        autocorrelations = np.zeros((n, len(t)))
        
        for i in range(n):
            autocorrelations[i, :] = self.time_correlation(i, i, t)
        
        return autocorrelations

    def normalize_autocorrelations(self,autocorrelations):
        """Normalizza le autocorrelazioni a 1 al tempo t=0."""
        return autocorrelations / autocorrelations[:, 0][:, np.newaxis]

    def compute_tau_values(self, t, normalized_autocorrelations):
        """Calcola i valori tau_ij per ogni residuo i, dove C_ii(tau_ij)/C_ii(0) = exp(-1)."""
        n = self.u.shape[0]
        exp_threshold = np.exp(-1)
        tau_values = np.zeros(n)
        
        for i in range(n):
            # Trova il primo tempo t per cui C_ii(t)/C_ii(0) < exp(-1)
            tau_index = np.where(normalized_autocorrelations[i, :] < exp_threshold)[0][0]
            tau_values[i] = t[tau_index]
        
        return tau_values

    def plot_normalized_autocorrelations(self,t, normalized_autocorrelations, tau_values):
        """Plot delle autocorrelazioni normalizzate con indicazione dei tempi caratteristici tau_ij."""
        n = len(t)
        plt.figure(figsize=(12, 8))
        
        for i in range(normalized_autocorrelations.shape[0]):
            plt.plot(t, normalized_autocorrelations[i, :], label=f'Residue {i+1}')
            plt.axvline(x=tau_values[i], color='r', linestyle='--', alpha=0.5)
        
        plt.axhline(y=np.exp(-1), color='k', linestyle='--', label='exp(-1)')
        plt.title('Normalized Autocorrelations C_ii(t)/C_ii(0) with Characteristic Times')
        plt.xlabel('Time t')
        plt.ylabel('Normalized Autocorrelation')
        plt.legend()
        plt.grid(True)
        plt.show()



    def compute_time_correlation_residuals(self, t):
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]
        
        residual_correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                C_ij_t = self.time_correlation(i, j, t_subset)
                residual_correlation_matrix[i, j] = np.mean(C_ij_t)  # Correlation mean over the subset
        
        return residual_correlation_matrix

    def plot_time_correlation_residuals(self, t):
        residual_correlation_matrix = self.compute_time_correlation_residuals(t)
        plt.figure(figsize=(8, 6))
        plt.imshow(residual_correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title('Residual Time Correlation Matrix')
        plt.xlabel('Index j')
        plt.ylabel('Index i')
        plt.show()
    def compute_residual_correlation_matrix(self, t,i):
        """Calcola la matrice di correlazione dei residui per tutti i j e per un intervallo di tempo specificato."""
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        # Inizializza una matrice per le correlazioni temporali
        residual_correlation_matrix = np.zeros((n, n, len(t_subset)))

        
        for j in range(n):
            residual_correlation_matrix[i, j, :] = self.time_correlation(i, j, t_subset)
        return residual_correlation_matrix

    def compute_residual_time_response_matrix(self, t,i):
        """Calcola la matrice di risposta al tempo per tutti i j e per un intervallo di tempo specificato."""
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        # Inizializza una matrice per le risposte temporali
        time_response_matrix = np.zeros((n, n, len(t_subset)))

        
        for j in range(n):
            time_response_matrix[i, j, :] = self.time_response(i, j, t_subset)

        return time_response_matrix



    def compute_mean_correlation_over_segment(self, lista, t):
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]
        residual_correlation_matrix = np.zeros((n, 1))
        idx = int(len(t_subset) * 0.5)
        for i in lista:
            for j in range(n):
                residual_correlation_matrix[j] += self.time_correlation(i, j, t_subset)[idx]
        residual_correlation_matrix /= len(lista)
        return residual_correlation_matrix

    def plot_mean_correlation_over_segment(self, lista, t):
        """Visualizza la correlazione media sui segmenti in funzione di j per una lista di i, in un intervallo di tempo specificato."""
        mean_correlation_matrix = self.compute_mean_correlation_over_segment(lista, t)
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        # Modifichiamo leggermente la funzione plot_with_secondary_structure_bands per accettare una matrice 2D
        def plot_correlation_bands(self, mean_correlation_matrix, ylabel, title):
            sec_struct_info = self.get_sec_struct_info()
            residue_ids = self.sec_struct_data['Residue ID'].astype(int)

            # Mappa colori per le strutture secondarie
            colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
            sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

            plt.figure(figsize=(12, 8))

            # Plotting the data matrix
            plt.plot(range(self.u.shape[0]), mean_correlation_matrix, marker='o', linestyle='-', alpha=0.7)
            
            # Aggiungi fasce colorate per le strutture secondarie
            current_color = 'black'
            for idx, resid in enumerate(residue_ids):
                if sec_struct_colors[idx] != current_color:
                    if idx > 0:
                        plt.axvspan(start_idx, idx - 0.5, color=current_color, alpha=0.2)
                    current_color = sec_struct_colors[idx]
                    start_idx = idx - 0.5
            plt.axvspan(start_idx, len(residue_ids) - 0.5, color=current_color, alpha=0.2)
            
            # Add legend for secondary structure types
            handles = [plt.Line2D([0], [0], color=color, lw=4, label=struct)
                    for struct, color in colors.items()]
            plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            plt.title(title)
            plt.xlabel('Index j')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()

        # Usa la funzione di plot modificata per la matrice 2D
        plot_correlation_bands(self, mean_correlation_matrix, 
                            'Mean Correlation', 
                            'Mean Correlation C_ij for selected i as a function of j')


    def compute_residual_transfer_entropy_matrix(self, t,i):
        """Calcola la matrice di transfer entropy per tutti i j e per un intervallo di tempo specificato."""
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        # Inizializza una matrice per le transfer entropy temporali
        transfer_entropy_matrix = np.zeros((n, n, len(t_subset)))

       
        for j in range(n):
            transfer_entropy_matrix[i, j, :] = self.transfer_entropy(i, j, t_subset)

        return transfer_entropy_matrix

    def get_sec_struct_info(self):
        """Returns a dictionary mapping residue IDs to secondary structure types."""
        sec_struct_info = {}
        for _, row in self.sec_struct_data.iterrows():
            sec_struct_info[row['Residue ID']] = row['Secondary Structure']
        return sec_struct_info

    def plot_with_secondary_structure_bands(self, i, data_matrix, t, ylabel, title, plot_func):
        """Generates plots with bands indicating secondary structure types along the x-axis."""
        sec_struct_info = self.get_sec_struct_info()
        residue_ids = self.sec_struct_data['Residue ID'].astype(int)

        # Mappa colori per le strutture secondarie
        colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
        sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

        plt.figure(figsize=(12, 8))
        
        # Plotting the data matrix
        for idx, time_point in enumerate(t):
            plt.plot(range(self.u.shape[0]), data_matrix[i, :, idx], marker='o', linestyle='-', alpha=0.7)
        
        # Aggiungi fasce colorate per le strutture secondarie
        current_color = 'black'
        for idx, resid in enumerate(residue_ids):
            if sec_struct_colors[idx] != current_color:
                if idx > 0:
                    plt.axvspan(start_idx, idx - 0.5, color=current_color, alpha=0.2)
                current_color = sec_struct_colors[idx]
                start_idx = idx - 0.5
        plt.axvspan(start_idx, len(residue_ids) - 0.5, color=current_color, alpha=0.2)
        
        # Add legend for secondary structure types
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=struct)
                   for struct, color in colors.items()]
        plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.title(title)
        plt.xlabel('Index j')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def plot_residual_correlation_vs_j(self, i, t):
        """Visualizza le correlazioni dei residui in funzione di j per un dato i, in un intervallo di tempo specificato."""
        residual_correlation_matrix = self.compute_residual_correlation_matrix(t,i)
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        self.plot_with_secondary_structure_bands(
            i, residual_correlation_matrix, t_subset, 
            f'Correlation with i={i}', 
            f'Residual Correlation C_ij for i={i} as a function of j over time',
            self.time_correlation
        )

    def plot_residual_time_response_vs_j(self, i, t):
        """Visualizza le risposte al tempo in funzione di j per un dato i, in un intervallo di tempo specificato."""
        time_response_matrix = self.compute_residual_time_response_matrix(t,i)
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        self.plot_with_secondary_structure_bands(
            i, time_response_matrix, t_subset, 
            f'Response with i={i}', 
            f'Time Response R_ij for i={i} as a function of j over time',
            self.time_response
        )

    def plot_residual_transfer_entropy_vs_j(self, i, t):
        """Visualizza la transfer entropy in funzione di j per un dato i, in un intervallo di tempo specificato."""
        transfer_entropy_matrix = self.compute_residual_transfer_entropy_matrix(t,i)
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]

        self.plot_with_secondary_structure_bands(
            i, transfer_entropy_matrix, t_subset, 
            f'Transfer Entropy with i={i}', 
            f'Transfer Entropy TE_ij for i={i} as a function of j over time',
            self.transfer_entropy
        )
    # Funzione per calcolare la risposta lineare media su un segmento temporale
    def compute_mean_linear_response_over_segment(self, lista, t):
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]
        residual_linear_response_matrix = np.zeros((n, 1))
        idx = int(len(t_subset ) * 0.5)
        for i in lista:
            for j in range(n):
                residual_linear_response_matrix[j] += self.time_response(i, j, t_subset)[idx]
        residual_linear_response_matrix /= len(lista)
        return residual_linear_response_matrix

    # Funzione per calcolare l'entropia media su un segmento temporale
    def compute_mean_entropy_over_segment(self, lista, t):
        n = self.u.shape[0]
        start_idx = int(len(t) * 0.4)
        end_idx = int(len(t) * 0.6)
        t_subset = t[start_idx:end_idx]
        residual_entropy_matrix = np.zeros((n, 1))
        idx = int(len(t_subset ) * 0.5)
        for i in lista:
            for j in range(n):
                residual_entropy_matrix[j] += self.transfer_entropy(i, j, t_subset)[idx]
        residual_entropy_matrix /= len(lista)
        return residual_entropy_matrix

    # Funzione di plot generica per visualizzare la correlazione, risposta lineare o entropia
    def plot_mean_quantity_over_segment(self, lista, t, quantity='correlation'):
        if quantity == 'correlation':
            mean_quantity_matrix = self.compute_mean_correlation_over_segment(lista, t)
            ylabel = 'Mean Correlation'
            title = 'Mean Correlation C_ij for selected i as a function of j'
        elif quantity == 'linear_response':
            mean_quantity_matrix = self.compute_mean_linear_response_over_segment(lista, t)
            ylabel = 'Mean Linear Response'
            title = 'Mean Linear Response for selected i as a function of j'
        elif quantity == 'entropy':
            mean_quantity_matrix = self.compute_mean_entropy_over_segment(lista, t)
            ylabel = 'Mean Entropy'
            title = 'Mean Entropy for selected i as a function of j'
        else:
            raise ValueError("Invalid quantity specified. Choose 'correlation', 'linear_response', or 'entropy'.")

        def plot_quantity_bands(self, mean_quantity_matrix, ylabel, title):
            sec_struct_info = self.get_sec_struct_info()
            residue_ids = self.sec_struct_data['Residue ID'].astype(int)

            colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
            sec_struct_colors = [colors.get(sec_struct_info.get(rid, 'Unknown'), 'black') for rid in residue_ids]

            plt.figure(figsize=(12, 8))
            plt.plot(range(self.u.shape[0]), mean_quantity_matrix, marker='o', linestyle='-', alpha=0.7)

            current_color = 'black'
            for idx, resid in enumerate(residue_ids):
                if sec_struct_colors[idx] != current_color:
                    if idx > 0:
                        plt.axvspan(start_idx, idx - 0.5, color=current_color, alpha=0.2)
                    current_color = sec_struct_colors[idx]
                    start_idx = idx - 0.5
            plt.axvspan(start_idx, len(residue_ids) - 0.5, color=current_color, alpha=0.2)

            handles = [plt.Line2D([0], [0], color=color, lw=4, label=struct)
                    for struct, color in colors.items()]
            plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            plt.title(title)
            plt.xlabel('Index j')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()

        plot_quantity_bands(self, mean_quantity_matrix, ylabel, title)
