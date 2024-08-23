import numpy as np
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, u, lambdas, mu):
        self.u = u # Ensure u is at least 2D
        self.lambdas = np.array(lambdas)
        self.mu = mu

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
        C_ii_t = self.time_correlation(i, i, np.array([0.0]))
        C_jj_t = self.time_correlation(j, j, np.array([0.0]))
        C_ij_0 = self.time_correlation(i, j, np.array([0.0]))[0]
        C_ij_t = self.time_correlation(i, j, np.array([0.0]))

        alpha_ij_t = (C_ii_0 * C_jj_t - C_ij_0 * C_ii_t) ** 2
        beta_ij_t = (C_ii_0 * C_jj_0) * (C_ii_t - C_ij_t ** 2)

        ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
        return -0.5 * np.log(1 - ratio)

    def time_response(self, i, j, t):
        C_ij_t = np.zeros(len(t))
        for idx, z in enumerate(t):
            C_ij_0 = 0
            C_ij_t_cost = 0
            for k in range(3, len(self.lambdas)):
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
