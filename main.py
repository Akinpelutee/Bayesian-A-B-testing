import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    return pd.read_csv(file_path)

def wrangle_data(df):
    #dropping irrelevant column 'Unnamed: 0'
    df.drop(columns='Unnamed: 0', inplace=True)
    #Let's check if we have duplicated values in the user id
    dup_val = df.duplicated(subset='user id').sum()
    print(dup_val)
    #drop null values if there're any
    df = df.dropna()
    return df

def explore_data(df):
    #lets know the number of users that are exposed to ads and public announcement
    df['test group'].value_counts(normalize=True).plot(
        kind='bar', xlabel='Test Groups', ylabel='Relative frequency', 
        title='Proportions of the test groups'
    );
    #Percentage of people that converted
    counts = df['converted'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%0.2f%%')
    plt.title('Percentages of Conversions');
    #Let's plot the most ads day and hour
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    
    df['most ads day'].value_counts().sort_values(ascending=True).plot(kind='barh', ax=ax[0])
    ax[0].set_title('Most ads day')
    
    df['most ads hour'].value_counts().sort_values(ascending=False).plot(kind='bar', ax=ax[1])
    ax[1].set_title('Most ads Hours');
    
    plt.tight_layout()
    plt.show()

    #Percentage of conversions per day
    day_converted = pd.crosstab(df['most ads day'], df['converted'], normalize='index')
    print(day_converted.sort_values(by =True, ascending=False))
    day_converted.plot.bar(stacked=True);

    #Bar plot of control group (psa) and treatment group (ad)
    conversion_rates = df.groupby("test group")["converted"].mean().reset_index()
    plt.figure(figsize=(8,6))
    plt.bar(conversion_rates['test group'], conversion_rates['converted'])
    plt.xlabel('Test group')
    plt.ylabel('Conversion rates')
    plt.title('Conversion rates by test groups');
    
    for i, row in conversion_rates.iterrows():
        plt.text(i, row['converted']+0.01, f"{row['converted']*100:.2f}%", ha='center')
    
    plt.show()

def get_posterior(df, prior_alpha_ad, prior_beta_ad, prior_alpha_psa, prior_beta_psa, num_samples):
    #Get the conversions and total users of each groups
    ad_conversions = df[(df['test group'] == 'ad') & (df['converted'] == True)].shape[0]
    ad_total = df[(df['test group'] == 'ad')].shape[0]
    psa_conversions = df[(df['test group'] == 'psa') & (df['converted'] == True)].shape[0]
    psa_total = df[(df['test group'] == 'psa')].shape[0]

    #Define the posterior distributions
    ad_posterior = beta(prior_alpha_ad + ad_conversions, prior_beta_ad + (ad_total - ad_conversions))
    psa_posterior = beta(prior_alpha_psa + psa_conversions, prior_beta_psa + (psa_total - psa_conversions))
    return ad_posterior, psa_posterior

def get_better_group(df):
    #Sample from the posteriors
    sample_posterior_ad = ad_posterior.rvs(num_samples)
    sample_posterior_psa = psa_posterior.rvs(num_samples)
    
    #Calculate the probability of being better
    prob_ad_better = np.mean(sample_posterior_ad > sample_posterior_psa)
    return prob_ad_better
    print(f"{prob_ad_better}")


def ad_effect_on_conversion(df):
    """
    Perform Bayesian linear regression on ad exposure vs. conversion.

    Parameters:
        df (pd.DataFrame): Dataset with 'total_ads' and 'converted'.

    Returns:
        dict: Regression slope and R-squared value.
    """
    x = df[(df['test group'] == 'ad')]['total ads']
    y = df[(df['test group'] == 'ad')]['converted'].astype(int)

    slope, intercept, r_value, _, _ = linregress(x, y)

    return {"slope": slope, "r_squared": r_value**2}

def plot_posterior(df):
    # Plot posterior distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(sample_posterior_ad, label="Ad Group", shade=True, color='blue')
    sns.kdeplot(sample_posterior_psa, label="PSA Group", shade=True, color='red')
    
    # Add vertical lines for means
    plt.axvline(sample_posterior_ad.mean(), color='blue', linestyle='dashed', label=f"Mean Ad: {sample_posterior_ad.mean():.5f}")
    plt.axvline(sample_posterior_psa.mean(), color='red', linestyle='dashed', label=f"Mean PSA: {sample_posterior_psa.mean():.5f}")
    plt.xlabel("Conversion Rate")
    plt.ylabel("Density")
    plt.title("Posterior Distributions of Ad and PSA Groups")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    #reads in the data
    file_path = 'marketing_AB.csv'
    prior_alpha_ad = 1
    prior_beta_ad = 1
    prior_alpha_psa = 1
    prior_beta_psa = 1 
    num_samples = 10000

    #pipeline
    df = load_data(file_path)
    
    df = wrangle_data(df)
    
    ad_posterior, psa_posterior = get_posterior(df,prior_alpha_ad, prior_beta_ad, prior_alpha_psa, prior_beta_psa, num_samples)
    print(f"Posterior : Beta_ad({ad_posterior.args[0]:.2f}, {ad_posterior.args[1]:.2f})")
    print(f"Posterior : Beta_psa({psa_posterior.args[0]:.2f}, {psa_posterior.args[1]:.2f})")

    prob_ad_better = get_better_group(df)
    print(f"{prob_ad_better}")

    result = ad_effect_on_conversion(df)
    print(f"{result}")