"""
Script to plot age-dependent HIV prevalence without interactions
"""

# Imports
import starsim as ss
import mighti as mi
import pylab as pl
import pandas as pd
import numpy as np

# Create the networks - sexual and maternal
mf = ss.MFNet(
    duration=1/24,  # Mean duration of relationships
    acts=80,
)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(mi.root / 'tests/test_data/nigeria_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(mi.root / 'tests/test_data/nigeria_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)

# Define age-dependent initial prevalence function
def age_dependent_prevalence(sim):
    prevalence = np.zeros(sim.n_agents)
    ages = sim.people.age.raw  # Initial ages of agents

    # Example age-dependent prevalence rates
    prevalence[(ages >= 15) & (ages < 20)] = 0.056
    prevalence[(ages >= 20) & (ages < 25)] = 0.172
    prevalence[(ages >= 25) & (ages < 30)] = 0.303
    prevalence[(ages >= 30) & (ages < 35)] = 0.425
    prevalence[(ages >= 35) & (ages < 40)] = 0.525
    prevalence[(ages >= 40) & (ages < 45)] = 0.572
    prevalence[(ages >= 45) & (ages < 50)] = 0.501
    prevalence[(ages >= 50) & (ages < 55)] = 0.435
    prevalence[(ages >= 55) & (ages < 60)] = 0.338
    prevalence[(ages >= 60) & (ages < 65)] = 0.21
    prevalence[ages >= 65] = 0.147

    return ss.bernoulli(prevalence)

# Initialize HIV with age-dependent initial prevalence
hiv = ss.HIV(init_prev=age_dependent_prevalence)

# Run baseline HIV simulation without interactions
print('Running baseline HIV simulation without interactions')
baseline_sim = ss.Sim(
    n_agents=5000,
    networks=networks,
    diseases=[hiv],
)
baseline_sim.run()

# Calculate and plot HIV prevalence by age group
age_groups = [(15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 100)]
age_group_labels = [f'{age[0]}-{age[1]}' for age in age_groups]
age_results = {label: np.zeros(len(baseline_sim.yearvec)) for label in age_group_labels}
population_by_age_group = {label: np.zeros(len(baseline_sim.yearvec)) for label in age_group_labels}

# Calculate HIV prevalence by age group at each time step
for t in range(len(baseline_sim.yearvec)):
    ages = baseline_sim.people.age.raw  # Get ages of agents
    prevalence = baseline_sim.results.hiv.prevalence[t]  # Get the prevalence at time t
    for (start, end), label in zip(age_groups, age_group_labels):
        age_mask = (ages >= start) & (ages <= end)
        if np.sum(age_mask) > 0:  # To avoid division by zero
            age_results[label][t] = np.sum(prevalence * age_mask) / np.sum(age_mask) * 100  # Calculate prevalence as percentage
            population_by_age_group[label][t] = np.sum(age_mask)  # Population in this age group at time t

# Plot HIV prevalence by age group
fig_age, ax_age = pl.subplots(figsize=(12, 8))
for label in age_group_labels:
    ax_age.plot(baseline_sim.yearvec, age_results[label], label=label)

ax_age.set_title('HIV Prevalence by Age Group')
ax_age.set_xlabel('Year')
ax_age.set_ylabel('Prevalence (%)')
ax_age.legend(title='Age Groups')
ax_age.grid(True)

pl.tight_layout()
pl.show()

# Plot population by age group
fig_pop, ax_pop = pl.subplots(figsize=(12, 8))
for label in age_group_labels:
    ax_pop.plot(baseline_sim.yearvec, population_by_age_group[label], label=label)

ax_pop.set_title('Population by Age Group')
ax_pop.set_xlabel('Year')
ax_pop.set_ylabel('Population Count')
ax_pop.legend(title='Age Groups')
ax_pop.grid(True)

pl.tight_layout()
pl.show()
