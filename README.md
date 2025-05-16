# Bayesian-A-B-testing
This Project aims to use Bayesian methods to draw conclusions whether the marketing ad will yield more returns than psa (Public service announcement) with probabilities and not using the old frequentist approach method.
Bayesian approach to A/B testing gives a probability insight in the sense that, it measures how well a product or feature is better than the other and not just relying on your intuition.
I also wrote an article explaining the methodology in detail [here](https://medium.com/@temitopeakinpelu98/bayesian-a-b-testing-for-website-conversion-rate-optimization-e4aeb5dbb1f9)

### Aims and objectives
The aim and objective of this project is to improve conversion rates using [marketing_AB.csv](https://www.kaggle.com/datasets/rahelederakhshande/marketing-ab) data and to optimize user experience and identifying whether the campaign will be successful or not by investigating which variant ads (treatment group) or psa (control group) is best.

### Methodology
This project follows a function driven approach using libraries like pandas, numpy, matplotlib, seaborn and scipy. 

#### Functions walkthrough
The load_data function reads in the csv file using the pandas library having the file path as it's argument.

The wrangle_data function cleans the data, handles any missing value, drops irrelevant columns like the (unnamed: 0) column and finally checks if we have any duplicates in the data.

The explore_data function explores the data by creating visualizations using the pandas and the matplotlib library in python. The first task of the function was to know the number of users that are exposed to ads and public announcement by plotting a barplot in which about 96% was exposed to ad and about 4% was exposed to the psa(public service announcement).
The fuction also plots the percentage of people that converted, the most ad days and hours, percentage of conversions per day and finally the bar plot of the treatment and control group.

The third function get_posterior starts the deep dive into bayesian inference. The posterior probabilities was calculated in which the priors were set to be (1, 1) since we don't have any prior knowlegde about the data.

The fourth function get_better_group samples from the posterior using a beta distribution and the fifth function plots the density plot.

### Interpretation of results
From the plot, we can infer that the ad group had a higher mean conversion rate indicating that the ad group(treatment) is better.

### Conclusion
The Bayesian framework enables more informed decision making, improved modelling and better uncertainty quatifications across many disciplines. I will love to engage in more data driven projects using Bayesian approach.
