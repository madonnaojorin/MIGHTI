"""
Cost-Effectiveness Analysis for Co-delivery of Interventions with HIV Care over a Time Horizon
"""

# Imports
import starsim as ss
import mighti as mi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define baseline cost and utility for HIV care alone
base_cost = 1000  # Annual cost for HIV care alone
base_utility = 0.6  # Annual utility for HIV care alone

# Define additional annual costs and utilities for co-delivery with each intervention
intervention_costs = {
    'Hypertension': 554659.52,
}

# Utilities for the co-delivery scenario (HIV care + intervention)
intervention_utilities = {
    'Hypertension': 656.87,
}

# Define cost-effectiveness threshold (e.g., $3,000 per QALY)
threshold = 3000

# Define the time horizon and discount rate
time_horizon = 20  # years
discount_rate = 0.03  # 3% annual discount rate

# Function to calculate present value
def present_value(annual_value, time_horizon, discount_rate):
    return sum(annual_value / (1 + discount_rate) ** t for t in range(1, time_horizon + 1))

# Function to calculate ICER for each disease over a time horizon
def calculate_icer(base_cost, intervention_cost, base_utility, intervention_utility, time_horizon, discount_rate, num_hiv_patients, num_disease_patients):
    pv_base_cost = present_value(base_cost * num_hiv_patients, time_horizon, discount_rate)
    pv_intervention_cost = present_value(base_cost * num_hiv_patients + intervention_cost * num_disease_patients, time_horizon, discount_rate)
    pv_base_utility = present_value(base_utility * num_hiv_patients, time_horizon, discount_rate)
    pv_intervention_utility = present_value(base_utility * num_hiv_patients + intervention_utility * num_disease_patients, time_horizon, discount_rate)
    
    delta_cost = pv_intervention_cost - pv_base_cost
    delta_utility = pv_intervention_utility - pv_base_utility

    if delta_utility == 0:
        return float('inf')  # To avoid division by zero, return infinity if delta utility is zero

    icer = delta_cost / delta_utility
    return icer

# Function to run the simulation and gather results
def run_simulation(disease_name, connector):
    hiv = ss.HIV(
        beta={
            'mf': [0.0008, 0.0004],  # Per-act transmission probability from sexual contacts
            'maternal': [0.2, 0]},   # MTCT probability
    )
    disease_instance = getattr(mi, disease_name)()
    diseases = [hiv, disease_instance]

    # Create the networks - sexual and maternal
    mf = ss.MFNet(
        duration=1/24,  # Mean duration of relationships
        acts=80,
    )
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    sim = ss.Sim(
        n_agents=5000,
        networks=networks,
        diseases=diseases,
        connectors=connector()
    )
    sim.run()
    return sim

# Store results for plotting
icers = {}
simulation_results = {}
num_disease_patients = {}
disease_names = list(intervention_costs.keys())

# Run simulations for each disease interaction with HIV
for disease_name in disease_names:
    print(f'Running simulation for {disease_name}')
    sim = run_simulation(disease_name, getattr(mi, f'hiv_{disease_name.lower()}'))
    n_infected = sim.results.hiv.n_infected[-1]
    simulation_results[disease_name] = n_infected

    # Extract number of disease patients from simulation results
    num_disease_patients[disease_name] = np.count_nonzero(getattr(sim.results, disease_name.lower()).prevalence[-1])
    
    # Calculate ICERs for each co-delivery intervention over the time horizon
    icer = calculate_icer(
        base_cost, 
        intervention_costs[disease_name], 
        base_utility, 
        intervention_utilities[disease_name],
        time_horizon,
        discount_rate,
        n_infected,  # Using the number of HIV patients from the simulation
        num_disease_patients[disease_name]
    )
    icers[disease_name] = icer

# Print ICERs for each co-delivery intervention
for disease_name, icer in icers.items():
    print(f'Incremental Cost-Effectiveness Ratio (ICER) for {disease_name} co-delivery: {icer}')

# Get the number of HIV patients from one of the simulations
num_hiv_patients = simulation_results[disease_names[0]]

# Plot cost-effectiveness results
cost_values = [present_value(base_cost * num_hiv_patients + intervention_costs[disease] * num_disease_patients[disease], time_horizon, discount_rate) for disease in disease_names]
utility_values = [present_value(base_utility * num_hiv_patients + intervention_utilities[disease] * num_disease_patients[disease], time_horizon, discount_rate) for disease in disease_names]

# Print the calculated values for debugging
print("Cost values:", cost_values)
print("Utility values:", utility_values)

# Define jitter amount
jitter_amount = 0.01

# Apply jitter to utility and cost values
utility_values_jittered = utility_values + np.random.uniform(-jitter_amount * max(utility_values), jitter_amount * max(utility_values), len(utility_values))
cost_values_jittered = cost_values + np.random.uniform(-jitter_amount * max(cost_values), jitter_amount * max(cost_values), len(cost_values))

fig, ax = plt.subplots()

# Plot data points with jitter
ax.scatter(utility_values_jittered, cost_values_jittered, color='blue')

# Annotate points with jitter
for i, txt in enumerate(disease_names):
    ax.annotate(txt, (utility_values_jittered[i], cost_values_jittered[i]), fontsize=8)

# Plot cost-effectiveness threshold line
x_vals = np.linspace(0, max(utility_values_jittered), 100)
y_vals = threshold * (x_vals - present_value(base_utility * num_hiv_patients, time_horizon, discount_rate)) + present_value(base_cost * num_hiv_patients, time_horizon, discount_rate)
ax.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Threshold: ${threshold} per QALY')

ax.set_xlabel('Utility (QALYs)')
ax.set_ylabel('Cost ($)')
ax.set_title('Cost-Effectiveness Analysis of Co-delivery of HIV Interventions')
#ax.set_yscale('log')  # Set y-axis to logarithmic scale
ax.set_ylim(1e6, 0.5*1e11)  # Limit y-axis from 10^9 to 10^11
#ax.set_xlim(0, 0.2*1e9)  # Limit y-axis from 10^9 to 10^11
ax.legend()

plt.grid(True)
plt.show()
