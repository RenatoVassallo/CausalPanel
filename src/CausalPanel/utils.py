import numpy as np
import pandas as pd

def sim_two_periods_panel(n=500, true_att=5, seed=1234):
    """
    Generates artificial panel data with a known true ATT.
    
    Parameters:
    n (int): Number of observations
    true_att (float): True Average Treatment Effect on the Treated (ATT)
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: (y1, y0, D, covariates, true_att)
    """
    np.random.seed(seed)

    # Generate covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)

    z1 = np.exp(x1 / 2)
    z2 = x2 / (1 + np.exp(x1)) + 10
    z3 = (x1 * x3 / 25 + 0.6) ** 3
    z4 = (x1 + x4 + 20) ** 2

    # Standardize covariates
    z1 = (z1 - np.mean(z1)) / np.std(z1)
    z2 = (z2 - np.mean(z2)) / np.std(z2)
    z3 = (z3 - np.mean(z3)) / np.std(z3)
    z4 = (z4 - np.mean(z4)) / np.std(z4)

    covariates = np.column_stack((z1, z2, z3, z4))

    # Propensity score and treatment assignment
    pi = 1 / (1 + np.exp(-(-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4)))
    D = (np.random.rand(n) <= pi).astype(int)

    # Outcome indices
    index_lin = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)
    v = np.random.normal(index_lin * D, 1, n)

    # Generate outcomes
    y0 = index_lin + v + np.random.normal(0, 1, n)  # Pre-treatment outcome
    y10 = index_lin + v + np.random.normal(0, 1, n) + index_lin  # Potential untreated outcome
    y11 = y10 + true_att  # Potential treated outcome (true ATT added)

    y1 = D * y11 + (1 - D) * y10  # Realized post-treatment outcome

    return y1, y0, D, covariates, true_att



def genr_naive_sim_data(seed=20250214):
    """
    Generate a naive version of simulated panel data with treatment effects.
    
    Parameters:
    seed (int): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: Simulated dataset
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Define parameters
    N = 50
    T = 50  # Number of units and time periods
    
    # Create DataFrame
    df = pd.DataFrame({
        'i': np.repeat(np.arange(1, N + 1), T),
        't': np.tile(np.arange(1, T + 1), N)
    })
    
    # Initialize columns
    df['y'] = 0  # Outcome variable
    df['treat_status'] = 0  # Treatment indicator
    df['cohort'] = np.nan  # Treatment cohort
    df['beta'] = np.nan  # Treatment effect size
    df['treat_date'] = np.nan  # Treatment timing
    df['rel_time'] = np.nan  # Relative time to treatment
    
    # Assign cohorts randomly, ensuring consistency within each unit
    unique_ids = df['i'].unique()
    cohort_mapping = {uid: np.random.randint(0, 6) for uid in unique_ids}
    df['cohort'] = df['i'].map(cohort_mapping)
    
    # Assign treatment effects and treatment dates for each cohort
    for cohort in df['cohort'].unique():
        effect_size = np.random.randint(5, 11)  # Effect size between 5 and 10
        timing = np.random.randint(15, 45)   # Treatment timing
        
        df.loc[df['cohort'] == cohort, 'beta'] = effect_size
        df.loc[df['cohort'] == cohort, 'treat_date'] = timing
        df.loc[(df['cohort'] == cohort) & (df['t'] >= timing), 'treat_status'] = 1
    
    # Adjust treatment dates exceeding the end period
    df.loc[df['treat_date'] > T, 'treat_date'] = np.nan
    
    # Compute relative time to treatment
    df['rel_time'] = df['t'] - df['treat_date']
    
    # Generate outcome variable y
    df['y'] = (
        df['i'] + df['t'] +
        np.where(df['treat_status'] == 1, df['beta'] * df['rel_time'], 0) +
        np.random.normal(0, 1, len(df))
    )
    
    return df