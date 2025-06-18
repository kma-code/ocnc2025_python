import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import yaml
from pathlib import Path
import logging              # handy logging tool for debugging and info
import time                 # to measure time of execution
import multiprocess as mp   # for multiprocessing

logging.basicConfig(format='Runner -- %(levelname)s: %(message)s',
                    level=logging.INFO)


class ExpLIF_population:
    def __init__(self, params):
        # attach parameters to object
        self.V_th, self.V_reset = params["V_th"], params["V_reset"]   
        self.tau_m, self.g_L = params["tau_m"], params["g_L"]     
        self.V_init, self.V_L = params["V_init"], params["V_L"]      
        self.dt = params["dt"]
        self.tau_ref = params["tau_ref"]
        self.DeltaT = params["DeltaT"]
        self.V_exp_trigger = params["V_exp_trigger"]
        
        # number of neurons
        self.n_neurons = params["n_neurons"]

        # initialize voltages
        self.v = np.zeros(self.n_neurons)
        # time steps since last spike
        self.refractory_counter = np.zeros(self.n_neurons)
            
    def LIF_step(self, I):
        """
            Perform one step of the LIF dynamics
        """
        
        currently_spiking = np.array([False for _ in range(self.n_neurons)])
        
        # This is where the magic happens: numpy indexing.
        # first, we need to get indices of neurons which
        # are refractory, above threshold or neither:
        idx_ref = np.where(self.refractory_counter > 0)[0]
        idx_spk = np.where(self.v > self.V_th)[0]
        idx_else = np.where((self.refractory_counter <= 0) & (self.v <= self.V_th))[0]
        
        # if the neuron is still refractory
        self.v[idx_ref] = self.V_reset
        self.refractory_counter[idx_ref] -= 1
        
        # if v is above threshold,
        # reset voltage and record spike event
        currently_spiking[idx_spk] = True
        self.v[idx_spk] = self.V_reset
        self.refractory_counter[idx_spk] = self.tau_ref/self.dt
        
        # calculate the increment of the membrane potential
        dv = self.voltage_dynamics(I)
        # update the membrane potential only for non-spiking neurons
        self.v[idx_else] += dv[idx_else]

        return self.v, currently_spiking
        
    def voltage_dynamics(self, I):
        """
            Calulcates one step of the exp-LI dynamics
        """
        # Fortunately, this code already enabled vectors, due to numpy magic.
        dv = (-(self.v-self.V_L) + I/self.g_L + self.DeltaT * np.exp((self.v-self.V_exp_trigger)/self.DeltaT)) * (self.dt/self.tau_m)
        return dv


def simulate_network(population, params):
    """
        Simulates a population of neurons
        for n_steps defined in params
    """

    voltages_arr = np.zeros((params["n_steps"], params["n_neurons"]))
    spikes_arr = np.zeros((params["n_steps"], params["n_neurons"]))

    for i in range(params["n_steps"]):
        I = np.random.normal(params["mean_I"], params["std_I"], size=params["n_neurons"])
        voltages_arr[i], spikes_arr[i] = population.LIF_step(I=I)

    return voltages_arr, spikes_arr


if __name__ == '__main__':

    # keep a time stamp to see how long the sims take
    t0 = time.time()

    params = yaml.safe_load(Path('params.yaml').read_text())
    logging.info(f"Parameters loaded:")
    for p in params:
        logging.info(f"{p}: {params[p]}")

    n_neurons = params["n_neurons"]

    # define the population
    logging.info(f"Defining population of {n_neurons} neurons")
    population = ExpLIF_population(params)

    # run simulation
    logging.info(f"Starting simulation")

    voltages_arr, spikes_arr = simulate_network(population, params)

    logging.info(f"Finished simulation in {time.time() - t0} seconds.")

    # show only first 1000 nerons and last 1000 steps
    x_range = (-1000,-1)
    y_range = (0,1000)
    spikes_arr = spikes_arr[x_range[0]:x_range[1], y_range[0]:y_range[1]]

    fig = plt.figure()
    for i in range(y_range[1]-y_range[0]):
        spike_times = spikes_arr[x_range[0]:x_range[1],i].nonzero()[0]
        plt.scatter(spike_times, i*np.ones_like(spike_times), marker='.', c='black')
    plt.xlabel('Time step')
    plt.ylabel('# Neuron')
    plt.savefig("spike_raster.png", bbox_inches='tight')
    plt.close(fig)      # good practice to close figs to free up memory

    logging.info(f"Done. Total time: {time.time() - t0} seconds.")
