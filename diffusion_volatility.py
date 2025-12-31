"""
Random Walk: Physics vs Finance
Compares physical diffusion to financial volatility using Random Walk models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def random_walk_simulation(n_particles=1000, n_steps=100):
    """
    Simulate random walk for multiple particles.
    
    Parameters:
    -----------
    n_particles : int
        Number of particles to simulate
    n_steps : int
        Number of steps for each particle
    
    Returns:
    --------
    positions : ndarray
        Array of shape (n_particles, n_steps+1) containing positions at each step
    """
    # Initialize all particles at x=0
    positions = np.zeros((n_particles, n_steps + 1))
    
    # Generate random steps: +1 or -1 with 50/50 probability
    # Using vectorized operations for efficiency
    steps = np.random.choice([-1, 1], size=(n_particles, n_steps))
    
    # Cumulative sum gives positions at each step
    positions[:, 1:] = np.cumsum(steps, axis=1)
    
    return positions


def geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2, n_paths=100, n_steps=100, T=1.0):
    """
    Simulate Geometric Brownian Motion for stock prices.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    mu : float
        Drift (expected return)
    sigma : float
        Volatility
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    T : float
        Time horizon (in years)
    
    Returns:
    --------
    stock_paths : ndarray
        Array of shape (n_paths, n_steps+1) containing stock prices at each step
    """
    dt = T / n_steps
    
    # Initialize all paths at S0
    stock_paths = np.zeros((n_paths, n_steps + 1))
    stock_paths[:, 0] = S0
    
    # Generate random normal variables (epsilon)
    epsilon = np.random.normal(0, 1, size=(n_paths, n_steps))
    
    # GBM formula: S_next = S_now * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*epsilon)
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * epsilon
    
    # Calculate log returns
    log_returns = drift_term + diffusion_term
    
    # get cumulative log returns, then exponentiate
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    stock_paths[:, 1:] = S0 * np.exp(cumulative_log_returns)
    
    return stock_paths


def compare_std_and_volatility(particle_positions, stock_paths, sigma):
    """
    Compare the standard deviation of particle positions to stock volatility.
    
    Parameters:
    -----------
    particle_positions : ndarray
        Final positions of particles (or all positions)
    stock_paths : ndarray
        Stock price paths
    sigma : float
        Volatility parameter used in GBM
    
    Returns:
    --------
    dict : Dictionary containing comparison metrics
    """
    # Standard deviation of final particle positions
    final_particle_positions = particle_positions[:, -1]
    particle_std = np.std(final_particle_positions)
    
    # For stock paths, calculate the standard deviation of log returns
    # This is related to volatility
    log_returns = np.diff(np.log(stock_paths), axis=1)
    stock_volatility_empirical = np.std(log_returns) * np.sqrt(len(log_returns[0]))
    
    # Theoretical standard deviation for random walk after n steps
    n_steps = particle_positions.shape[1] - 1
    theoretical_std = np.sqrt(n_steps)
    
    return {
        'particle_std': particle_std,
        'theoretical_std': theoretical_std,
        'stock_volatility_empirical': stock_volatility_empirical,
        'stock_volatility_theoretical': sigma,
        'comparison': {
            'particle_std_vs_theoretical': particle_std / theoretical_std,
            'stock_vol_empirical_vs_theoretical': stock_volatility_empirical / sigma
        }
    }


def plot_comparison(particle_positions, stock_paths, mu=0.05, sigma=0.2):
    """
    Create side-by-side plots comparing particle diffusion and stock price paths.
    
    Parameters:
    -----------
    particle_positions : ndarray
        Particle positions from random walk
    stock_paths : ndarray
        Stock price paths from GBM
    mu : float
        Drift parameter (for display)
    sigma : float
        Volatility parameter (for display)
    """
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Increase default font sizes for better readability
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    
    # Professional color palette: dark blues, grays, and maroons
    colors = {
        'primary_blue': '#1f4e79',      # Dark blue
        'secondary_blue': '#4472c4',    # Medium blue
        'light_blue': '#8faadc',        # Light blue
        'maroon': '#8b1538',            # Maroon
        'dark_gray': '#404040',         # Dark gray
        'medium_gray': '#737373',       # Medium gray
        'light_gray': '#bfbfbf'         # Light gray
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Left column: Particle Diffusion
    # Plot 1: Paths of 5 particles
    ax1 = fig.add_subplot(gs[0, 0])
    n_particles_to_plot = min(5, particle_positions.shape[0])
    for i in range(n_particles_to_plot):
        ax1.plot(particle_positions[i, :], alpha=0.7, linewidth=1.5, color=colors['primary_blue'])
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Random Walk: Paths of 5 Particles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color=colors['dark_gray'], linestyle='--', linewidth=0.5)
    ax1.tick_params(labelsize=11)
    
    # Plot 2: Histogram of final positions
    ax2 = fig.add_subplot(gs[1, 0])
    final_positions = particle_positions[:, -1]
    n_steps = particle_positions.shape[1] - 1
    theoretical_std = np.sqrt(n_steps)
    
    ax2.hist(final_positions, bins=30, density=True, alpha=0.7, 
             color=colors['light_blue'], edgecolor=colors['primary_blue'], linewidth=0.5)
    
    # Overlay theoretical normal distribution
    x_theory = np.linspace(final_positions.min(), final_positions.max(), 100)
    y_theory = (1 / (theoretical_std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * (x_theory / theoretical_std)**2)
    ax2.plot(x_theory, y_theory, color=colors['maroon'], linewidth=2.5, 
             label='Theoretical Normal', linestyle='-')
    
    ax2.set_xlabel('Final Position', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'Final Positions Distribution (n={len(final_positions)} particles)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Plot 3: All particle paths (overview)
    ax3 = fig.add_subplot(gs[2, 0])
    # Plot a sample of paths for visualization
    sample_size = min(100, particle_positions.shape[0])
    sample_indices = np.random.choice(particle_positions.shape[0], sample_size, replace=False)
    for idx in sample_indices:
        ax3.plot(particle_positions[idx, :], alpha=0.1, linewidth=0.5, color=colors['secondary_blue'])
    # Plot mean path
    mean_path = np.mean(particle_positions, axis=0)
    ax3.plot(mean_path, color=colors['maroon'], linewidth=2.5, label='Mean Path')
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Position', fontsize=12)
    ax3.set_title('All Particle Paths (Sample)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color=colors['dark_gray'], linestyle='--', linewidth=0.5)
    ax3.tick_params(labelsize=11)
    
    # Right column: Stock Price Paths
    # Plot 1: Sample of stock paths
    ax4 = fig.add_subplot(gs[0, 1])
    n_paths_to_plot = min(10, stock_paths.shape[0])
    for i in range(n_paths_to_plot):
        ax4.plot(stock_paths[i, :], alpha=0.6, linewidth=1.5, color=colors['primary_blue'])
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Stock Price ($)', fontsize=12)
    ax4.set_title(f'Geometric Brownian Motion (Stock Price Simulation)\n(μ={mu:.2%}, σ={sigma:.2%})', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=stock_paths[0, 0], color=colors['dark_gray'], linestyle='--', linewidth=1.5, 
                label=f'Initial Price: ${stock_paths[0, 0]:.2f}')
    ax4.legend(loc='best', fontsize=11)
    ax4.tick_params(labelsize=11)
    
    # Plot 2: Distribution of final stock prices
    ax5 = fig.add_subplot(gs[1, 1])
    final_prices = stock_paths[:, -1]
    ax5.hist(final_prices, bins=30, density=True, alpha=0.7, 
             color=colors['light_blue'], edgecolor=colors['primary_blue'], linewidth=0.5)
    ax5.axvline(x=stock_paths[0, 0], color=colors['maroon'], linestyle='--', linewidth=2.5, 
                label=f'Initial: ${stock_paths[0, 0]:.2f}')
    ax5.axvline(x=np.mean(final_prices), color=colors['dark_gray'], linestyle='--', linewidth=2.5, 
                label=f'Mean Final: ${np.mean(final_prices):.2f}')
    ax5.set_xlabel('Final Stock Price ($)', fontsize=12)
    ax5.set_ylabel('Probability Density', fontsize=12)
    ax5.set_title(f'Final Price Distribution (n={len(final_prices)} paths)', 
                  fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=11, framealpha=0.95)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=11)
    
    # Plot 3: All stock paths with statistics
    ax6 = fig.add_subplot(gs[2, 1])
    # Plot all paths with transparency
    for i in range(stock_paths.shape[0]):
        ax6.plot(stock_paths[i, :], alpha=0.1, linewidth=0.5, color=colors['secondary_blue'])
    # Plot mean path
    mean_stock_path = np.mean(stock_paths, axis=0)
    ax6.plot(mean_stock_path, color=colors['maroon'], linewidth=2.5, label='Mean Path')
    # Plot confidence intervals
    std_stock_path = np.std(stock_paths, axis=0)
    ax6.fill_between(range(len(mean_stock_path)), 
                     mean_stock_path - std_stock_path,
                     mean_stock_path + std_stock_path,
                     alpha=0.2, color=colors['maroon'], label='±1 Std Dev')
    ax6.set_xlabel('Time Step', fontsize=12)
    ax6.set_ylabel('Stock Price ($)', fontsize=12)
    ax6.set_title('All Stock Price Paths with Statistics', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=11)
    
    plt.suptitle('Random Walk: Physics vs Finance', fontsize=18, fontweight='bold', y=0.995)
    return fig


def main():
    """
    Main function to run the simulation and create graphs.
    """
    # Parameters
    n_particles = 1000
    n_steps = 100
    
    # Part 1: Random Walk (Physics)
    print("Simulating Random Walk for 1,000 particles...")
    particle_positions = random_walk_simulation(n_particles=n_particles, n_steps=n_steps)
    
    # Part 2: Geometric Brownian Motion (Finance)
    # User-configurable parameters
    S0 = 100  # Initial stock price
    mu = 0.05  # Drift (5% annual return)
    sigma = 0.2  # Volatility (20% annual volatility)
    n_paths = 100
    
    print(f"Simulating Geometric Brownian Motion with μ={mu:.2%}, σ={sigma:.2%}...")
    stock_paths = geometric_brownian_motion(S0=S0, mu=mu, sigma=sigma, 
                                           n_paths=n_paths, n_steps=n_steps)
    
    # Compare standard deviation and volatility
    print("\nComparing Standard Deviation and Volatility:")
    comparison = compare_std_and_volatility(particle_positions, stock_paths, sigma)
    print(f"  Particle Standard Deviation: {comparison['particle_std']:.2f}")
    print(f"  Theoretical Std (√n): {comparison['theoretical_std']:.2f}")
    print(f"  Empirical Stock Volatility: {comparison['stock_volatility_empirical']:.4f}")
    print(f"  Theoretical Stock Volatility: {comparison['stock_volatility_theoretical']:.4f}")
    print(f"  Particle Std Ratio: {comparison['comparison']['particle_std_vs_theoretical']:.3f}")
    print(f"  Stock Vol Ratio: {comparison['comparison']['stock_vol_empirical_vs_theoretical']:.3f}")
    
    # Create, save, and display plots
    print("\nGenerating plots...")
    fig = plot_comparison(particle_positions, stock_paths, mu=mu, sigma=sigma)
    
    plt.savefig('diffusion_volatility_comparison_v2.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'diffusion_volatility_comparison_v2.png'")
    
    plt.show()


if __name__ == "__main__":
    main()


