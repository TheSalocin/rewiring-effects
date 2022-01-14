#!/usr/bin/env python
# coding: utf-8

# In[2]:


#necessary for basic network
import torch
from brian2 import *

#necessary for summary statistics
import numpy as np
import scipy.special as scsp

#For getting connection weights
def get_con_matrix(N_pre, N_post, p, mean, var, connectivity_type = "lognormal"):
    """
    Input:
    
    N_pre: int
        number of presynaptic neurons 
    N_post: int
        number of postsynaptic neurons
    p: float (0,1]
        connection probability
    mean: float
        mean connection weight
    var: float
        variance of connection weights
    connectivity_type: str
        type of connection weight distribution; one of "lognormal" or "normal"
    
    Returns:
    
    connection matrix: np.array()
        N_pre x N_post array of connection weights
    """
    
    assert (connectivity_type in ("lognormal","normal")), "connectivity_type has to be one of 'lognormal' or 'normal'"
    
    if connectivity_type == "lognormal":
        #transform true mean and var into mean and var of underlying normal
        m = 2*log(mean) - 0.5*log(var + mean**2)
        s =  sqrt(-2*log(mean) + log(var + mean**2))
        #get weight matrix
        W = np.random.lognormal(mean=m, sigma=s, size=N_pre*N_post)
        
    elif connectivity_type == "normal":
        W = np.random.normal(loc = mean, scale = np.sqrt(var), size=N_pre*N_post)
    
    #set weights to 0 with probability 1-p
    return np.where(np.random.random(W.shape) > p, 0, W)

def analytic_correlation(W_aa_pre, fr_a_pre, W_aa_post, fr_a_post, W_ba_pre, fr_b_pre, W_ba_post, fr_b_post):
    """
    Calculates the expected correlation according to Mongillo's analytics
    
    Input:
    
    W_aa_pre: np.array
        Weight matrix from a->a pre rewiring
    fr_a_pre: list
        firing rates of population a pre rewiring
    W_aa_post: np.array
        Weight matrix from a->a post rewiring
    fr_a_post: list
        firing rates of population a post rewiring
    W_ba_pre: np.array
        Weight matrix from b->a post rewiring
    fr_b_pre: list
        Firing rates of population b pre rewiring
    W_ba_post: np.array
        Weight matrix from b->a post rewiring
    fr_b_post: list
        Firing rates of population b post rewiring
    
    Returns:
    
    r_a: float
        analytical prediction for firing rate correlation
        
    """
    
    fr_a_pre = np.asarray(fr_a_pre)
    fr_a_post = np.asarray(fr_a_post)
    fr_b_pre = np.asarray(fr_b_pre)
    fr_b_post = np.asarray(fr_b_post)

    
    s_a_sq = (np.mean(W_aa_pre**2)*np.mean(fr_a_pre**2) - (np.mean(W_aa_pre)**2)*(np.mean(fr_a_pre)**2) + 
              np.mean(W_ba_pre**2)*np.mean(fr_b_pre**2) - (np.mean(W_ba_pre)**2)*(np.mean(fr_b_pre)**2))
    
    r_a = (-(np.mean(W_aa_pre)**2)*(np.mean(fr_a_pre)**2) + np.mean(W_aa_pre*W_aa_post)*np.mean(fr_a_pre*fr_a_post)
        -(np.mean(W_ba_pre)**2)*(np.mean(fr_b_pre)**2) + np.mean(W_ba_pre*W_ba_post)*np.mean(fr_b_pre*fr_b_post))/s_a_sq
    
    return r_a

#Calculating summary statistics

#necessary for calculating correlations and summary stats
def get_fr(spikeindices, spiketimes, it, simtime):
    """
    Input:
    
    spikeindices: list
        list of neuron indices corresponding to spikes in spiketimes
    spiketimes: list
        corresponding list of spike times
    it: int
        simulation iteration
    simtime: brian2.units.second
        length of each simulation
    
    Returns:
    
    fr: list
        list of firing rates of each neuron
    """
    
    fr = []
    N_neurons = int(np.max(spikeindices))
    for i in range(N_neurons):
        fr.append(sum(spiketimes[spikeindices == i] > 1000 + it*simtime/ms) /(simtime/second - 1))
        #TODO: also add check that doesn't go above this simulation (< (it+1)*simtime)
        #TODO: make independent of given simulation time
    return fr

def calc_CVs(spikeindices, spiketimes):
    """
    Calculate mean of coefficiens of variation of inter-spike-intervals 
    
    Input:
    
    spikeindices: list
        list of neuron indices corresponding to spikes in spiketimes
    spiketimes: list
        corresponding list of spike times
        
    Returns:
        
    CV: float
        Mean CV of population (or 0 if no spikes recorded)
    """
    CV_list = []
    N_neurons = int(np.max(spikeindices))
    #for each neuron
    for i in range(N_neurons):
        #get spiketimes
        relevant_times = spiketimes[spikeindices == i]
        relevant_times = relevant_times[relevant_times > 1000]
        ISIs = diff(relevant_times)
        #if neuron spiked twice
        if ISIs.size != 0:
            CV_ISI = np.std(ISIs)/np.mean(ISIs)
            CV_list.append(CV_ISI)
    
    #if at least one neuron spiked        
    if len(CV_list) != 0:
        return np.mean(CV_list)
    else: return 0
    
#Takes a discretized raster plot and transforms it to phases
def discretise(spikeindices, spiketimes, binsize):
    """
    Discretise spikes into raster
    
    Input:
    
    spikeindices: list
        list of neuron indices corresponding to spikes in spiketimes
    spiketimes: list [ms]
        corresponding list of spike times
    binsize: float [ms]
        size of bins for spike detection
        
    Returns:
    
    rb: np.array
        N_neurons x N_bins array of spikes that is 1 where a spike was recorded and 0 everywhere else
    N_bins: int
        Number of bins
    N_neurons: int
        Number of neurons
    t_max: float
        Time of last spike
    """
    
    N_neurons = int(np.max(spikeindices)+1)
    t_max = np.max(spiketimes)
    N_bins = int(t_max/binsize)+1
    
    #Create raster
    rb = np.zeros((N_neurons, N_bins))
    #Set times with a spike to 1
    rb[spikeindices.astype(int), (spiketimes/binsize).astype(int)] = 1
    
    return (rb, (N_bins, N_neurons, t_max))

#Result converged as discretization is finer, i.e. nbins increases
#@jit
def interpolate_phase_neuron(n_raster):
    """
    Takes a discretized raster plot and transform it to angular variables. 

    Parameters
    ----------

    n_raster: Numpy array
        A raster discretized using the discretize() function 

    Returns
    -------

    phase: Numpy array
        It has the same shape as n_raster, but now instead of 0 or 1 gives the value of phase at that time.
    """
    nbins = np.size(n_raster)
    
    indices = np.where(n_raster == 1)[0] #Get spiking indices

    #If at least one neuron spiked    
    if np.size(indices > 0):
        phase = np.zeros(nbins)

        #Now for all the indices, fill the phase
        for j in range(np.size(indices) - 1):
            t0,tf =int(indices[j]), int(indices[j+1])
            dphase = 2.0 * np.pi / (1.0 * (tf - t0))
                
            for i in range(tf-t0):
                phase[t0+i] = np.pi - dphase * i

        #Now add periodic boundary conditions
        t0,tf = int(indices[-1]), int(indices[0])
        phase[t0:nbins] = np.random.rand(nbins-t0) * 2.0 * np.pi
        phase[0:tf] = np.random.rand(tf) * 2.0 * np.pi
        return phase

    else: return np.random.rand(nbins) * 2.0 * np.pi

#Compute Kuramoto-Daido parameters up to order n
#@jit
def kd(phases, n=1):
    """
    Compute the Kuramoto-Daido parameters from the discretization 

    Parameters
    ----------

    phases: Numpy array
        An array with the phase evolution of each oscillator. Can be obtained with interpolate_phase_neuron 
    n: int 
        How many Kuramoto-Daido parameters we want to get. By default n=1, the usual complex Kuramoto parameter

    Returns
    -------

    zk: Numpy array
        Each row contains the value of the kth-Kuramoto-Daido order parameter at each time, as zk[k, bin]
    """
    nbins = np.size(phases[0,:])
    zk = np.zeros((n, nbins), dtype=complex)
    for k in range(1,n+1):
        if (k%10==0): print(k)
        zk[k-1] = np.mean(np.exp(1.0j * k * phases), axis=0)
    return zk

def calc_stats(x, simtime=5*second):
    """
    Puts calculation of stats together
    
    Input:
    
    x: list of lists
        list of lists of lists of spike times and indices as returned by the simulator() function
    simtime: brian2.second
        simulation length
        
    Returns:
    
    stats: torch.tensor
        n_sims x 8 tensor consisting of summary statistics for each simulation
    """

    #TODO: verify that x is in correct shape
    
    #number of simulations
    n_sims = np.asarray(x, dtype=object).shape[0]
    
    stats = torch.zeros(n_sims, 8)
    
    #for each simulation
    for sim_idx in range(n_sims):
        
        #relevant simulation
        sim_to_consider = x[sim_idx]
        
        #FR stats
        #get firing rates of each neuron population-wise
        ex_fr = get_fr(sim_to_consider[0], sim_to_consider[1], 0, simtime)
        in_fr = get_fr(sim_to_consider[2], sim_to_consider[3], 0, simtime)
        
        #excitatory firing rate mean and variance
        ex_fr_mean = np.mean(ex_fr)
        ex_fr_var = np.var(ex_fr)
        
        #inhibitory firing rate mean and variance
        in_fr_mean = np.mean(in_fr)
        in_fr_var = np.var(in_fr)
        
        #CVs
        #get mean CV of ISIs population-wise
        ex_CV = calc_CVs(sim_to_consider[0], sim_to_consider[1])
        in_CV = calc_CVs(sim_to_consider[2], sim_to_consider[3])
        
        #Kuramoto Param
        #get raster
        raster_ex, _ = discretise(sim_to_consider[0], sim_to_consider[1], 0.1)
        #find phases for each neuron
        phases_ex = np.zeros(shape(raster_ex))
        for n_idx in range(shape(phases_ex)[0]):
            phases_ex[n_idx, :] = interpolate_phase_neuron(raster_ex[n_idx,:])
        #of interest: mean of absolute kuramoto param
        ex_kd = np.mean(np.abs(kd(phases_ex)))
        
        #same for inhibitory: get raster
        raster_in, _ = discretise(sim_to_consider[2], sim_to_consider[3], 0.1)
        #find phases for each neuron
        phases_in = np.zeros(shape(raster_in))
        for n_idx in range(shape(phases_in)[0]):
            phases_in[n_idx, :] = interpolate_phase_neuron(raster_in[n_idx,:])
        #of interest: mean of absolute kuramoto param
        in_kd = np.mean(np.abs(kd(phases_in)))
        
        #append to stats array
        stats[sim_idx, :] = torch.tensor([ex_fr_mean, ex_fr_var, in_fr_mean, in_fr_var, ex_CV, in_CV, ex_kd, in_kd])
    return stats

#To run the basic network
def simulator(N=1000, EI_ratio=0.2, 
              ee_mi=0.1, ei_mi=0.1, ie_mi=0.4, ii_mi=0.4, e_var=1.0, i_var=1.0, 
              p=0.1, connectivity_type="lognormal",
              theta=-50*mV, vr=-70*mV, tau_m=20*ms, tau_refr=1*ms, 
              E_H = 20*mV, I_H = 21*mV,
              simtime=5*second, report=None):
    """
    runs the network; originally intended for use with sbi
        
    Input:
    
    N: int
        N neurons total
    EI_ratio: float [0,1]
        fraction of inhibitory neurons
    ee_mi: float
        mean E->E connection strength
    ei_mi: float
        mean E->I connection strength
    ie_mi: float
        mean I->E connection strength
    ii_mi: float
        mean I->I connection strength
    var_e: float
        excitatory connection weight variance
    var_i: float
        inhibitory connection weight variance
    p: float (0,1]
        connection probability
    connectivity_type: str
        type of connection weight distribution; one of "lognormal" or "normal"
    theta: brian2.units.mV
        spike emission threshold
    vr: brian2.units.mV
        reset potential
    tau_m: brian2.units.ms
        membrane time constant
    tau_refr: brian2.units.ms
        refractory time
    E_H: brian2.units.mV
        strength of external input to excitatory population
    I_H: brian2.units.mV
        strength of external input to inhibitory population
    simtime: brian2.units.second
        length of simulation
    report: str or None
        passed to brian2.run() function
    
    
    Returns: 
    
    i_e: list
        excitatory neuron indices
    t_e: list
        corresponding excitatory spike times
    i_i: list 
        inhibitory spike indices
    t_i: 
        corresponding inhibitory spike times
    """
    #Setup to let brian2 run multiple times
    device.reinit()
    device.activate()
    start_scope()
    
    #neuron equations
    eqs_neurons='''
    dv/dt = (-(v - vr) + H + 6.5*randn()*mV)/tau_m: volt (unless refractory)
    H: volt'''
    
    #define population sizes
    NI = int(EI_ratio*N)
    NE = int(N-NI)
    
    #set connection params
    ee_var = ei_var = e_var
    ii_var = ie_var = i_var
    ee_p = ei_p = ii_p = ie_p = p
    
    ##########################
    # Initialize neuron groups
    ##########################

    #Excitatoy Neurons
    E = NeuronGroup(NE, model=eqs_neurons, threshold='v > theta',
                          reset='v=vr', refractory=tau_refr, method='euler')
    E.H = E_H


    #Inhibitory Neurons
    I = NeuronGroup(NI, model=eqs_neurons, threshold='v > theta',
                          reset='v=vr', refractory=tau_refr, method='euler')
    I.H = I_H
    
    #################
    # Synapse Groups
    #################

    #E -> E Connections
    S_ee = Synapses(E, E, 'w: 1', on_pre='v += w*mV')
    S_ee.connect()
    m1 = get_con_matrix(NE, NE, ee_p, ee_mi, ee_var, connectivity_type=connectivity_type)
    S_ee.w = m1


    #I -> E Connections
    S_ie = Synapses(I, E, 'w: 1', on_pre='v -= w*mV')
    S_ie.connect()
    m2 = get_con_matrix(NI, NE, ie_p, ie_mi, ie_var, connectivity_type=connectivity_type)
    S_ie.w = m2


    #E -> I Connections
    S_ei = Synapses(E, I, 'w: 1', on_pre='v += w*mV')
    S_ei.connect()
    m3 = get_con_matrix(NE, NI, ei_p, ei_mi, ei_var, connectivity_type=connectivity_type)
    S_ei.w = m3

    #I -> I Connections
    S_ii = Synapses(I, I, 'w: 1', on_pre='v -= w*mV')
    S_ii.connect()
    m4 = get_con_matrix(NI, NI, ii_p, ii_mi, ii_var, connectivity_type=connectivity_type)
    S_ii.w = m4
    
    #################
    # Set up monitors
    #################

    S_E = SpikeMonitor(E)
    S_I = SpikeMonitor(I)
    
    run(simtime, report=report)
    
    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms
    
    return [i_e, t_e, i_i, t_i]

def sbi_simulator(params):
    """
    Wrapper function to run run simulator() in sbi-compatible manner
    
    Input:
    
    params: torch.tensor
        parameters to be fitted with sbi; [EI_ratio, e_mi, i_mi, E_H, I_H]
        
    Returns:
    
    i_e: list
        excitatory neuron indices
    t_e: list
        corresponding excitatory spike times
    i_i: list 
        inhibitory spike indices
    t_i: 
        corresponding inhibitory spike times
    """
    
    EI_rat = params[0]
    e_mi = params[1]
    i_mi = params[2]
    EH = params[3]*mV
    IH = params[4]*mV
    
    [i_e, t_e, i_i, t_i] = simulator(N=500, EI_ratio=EI_rat, 
                                     ee_mi=e_mi, ei_mi=e_mi, ie_mi=i_mi, ii_mi=i_mi, 
                                     E_H=EH, I_H=IH)
    
    return [i_e, t_e, i_i, t_i]


def rewiring_dynamics(N=1000, EI_ratio=0.2, 
                      ee_mi=0.1, ei_mi=0.1, ie_mi=0.4, ii_mi=0.4, e_var=1.0, i_var=1.0, 
                      p=0.1, connectivity_type="lognormal",
                      theta=-50*mV, vr=-70*mV, tau_m=20*ms, tau_refr=1*ms, 
                      E_H = 20*mV, I_H = 21*mV,
                      simtime=5*second, report=None):
    """
    Runs network 5 times with rewiring
    
    Input:
    
    N: int
        N neurons total
    EI_ratio: float [0,1]
        fraction of inhibitory neurons
    ee_mi: float
        mean E->E connection strength
    ei_mi: float
        mean E->I connection strength
    ie_mi: float
        mean I->E connection strength
    ii_mi: float
        mean I->I connection strength
    e_var: float
        excitatory connection weight variance
    i_var: float
        inhibitory connection weight variance
    p: float (0,1]
        connection probability
    connectivity_type: str
        type of connection weight distribution; one of "lognormal" or "normal"
    theta: brian2.units.mV
        spike emission threshold
    vr: brian2.units.mV
        reset potential
    tau_m: brian2.units.ms
        membrane time constant
    tau_refr: brian2.units.ms
        refractory time
    E_H: brian2.units.mV
        strength of external input to excitatory population
    I_H: brian2.units.mV
        strength of external input to inhibitory population
    simtime: brian2.units.second
        length of simulation
    report: str or None
        passed to brian2.run() function
    
    
    Returns:
    
    ex_fr: list of lists
        firing rates of excitatory neurons in the shape [[baseline], [EE], [EI], [IE], [II]]
    in_fr: list of lists
        firing rates of inhibitory neurons in the shape [[baseline], [EE], [EI], [IE], [II]]
    weights: list of lists
        connection weight matrices
    stats: torch.tensor
        tensor of calculated stats in the shape [[ex_fr_mean, ex_fr_var, in_fr_mean, in_fr_var, ex_CV, in_CV, ex_kd, in_kd]]
    
    """
    
    #allow brian2 to run multiple times
    device.reinit()
    device.activate()
    start_scope()

    #Neuronal population sizes
    NI = int(EI_ratio*N)
    NE = int(N-NI)
    
    #set connection params
    ee_var = ei_var = e_var
    ie_var = ii_var = i_var

    ee_p = ei_p = ii_p = ie_p = p

    #neuron equations
    eqs_neurons='''
    dv/dt = (-(v - vr) + H + 6.5*randn()*mV)/tau_m: volt (unless refractory)
    H: volt'''
    
    #Initialise weight list for later use
    weights = []
    
    ##########################
    # Initialize neuron groups
    ##########################

    #Excitatoy Neurons
    E = NeuronGroup(NE, model=eqs_neurons, threshold='v > theta',
                          reset='v=vr', refractory=tau_refr, method='euler')
    E.H = E_H


    #Inhibitory Neurons
    I = NeuronGroup(NI, model=eqs_neurons, threshold='v > theta',
                          reset='v=vr', refractory=tau_refr, method='euler')
    I.H = I_H
    
    #################
    # Synapse Groups
    #################

    #E -> E Connections
    S_ee = Synapses(E, E, 'w: 1', on_pre='v_post += w*mV')
    S_ee.connect()
    m1 = get_con_matrix(NE, NE, ee_p, ee_mi, ee_var, connectivity_type=connectivity_type)
    S_ee.w = m1

    #E -> I Connections
    S_ei = Synapses(E, I, 'w: 1', on_pre='v_post += w*mV')
    S_ei.connect()
    m2 = get_con_matrix(NE, NI, ei_p, ei_mi, ei_var, connectivity_type=connectivity_type)
    S_ei.w = m2
    
    #I -> E Connections
    S_ie = Synapses(I, E, 'w: 1', on_pre='v_post -= w*mV')
    S_ie.connect()
    m3 = get_con_matrix(NI, NE, ie_p, ie_mi, ie_var, connectivity_type=connectivity_type)
    S_ie.w = m3

    #I -> I Connections
    S_ii = Synapses(I, I, 'w: 1', on_pre='v_post -= w*mV')
    S_ii.connect()
    m4 = get_con_matrix(NI, NI, ii_p, ii_mi, ii_var, connectivity_type=connectivity_type)
    S_ii.w = m4

    #################
    # Set up monitors
    #################

    S_E = SpikeMonitor(E)
    S_I = SpikeMonitor(I)

    #run network and extract spike times/indices
    run(simtime, report=report)

    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms

    #initialise list of firing rates
    ex_fr = []
    in_fr = []
    
    #calculate FRs and append to list of FRs
    ex_fr.append(get_fr(i_e, t_e, 0, simtime))
    in_fr.append(get_fr(i_i, t_i, 0, simtime))
    
    #calculate stats; only done once bc. runtime
    stats = calc_stats([[i_e, t_e, i_i, t_i]], simtime)

    #append weight lists
    weights.append([m1, m2, m3, m4])
    
    #Change E -> E connections, n = 1
    m1 = get_con_matrix(NE, NE, ee_p, ee_mi, ee_var, connectivity_type=connectivity_type)
    S_ee.w = m1

    run(simtime, report=report)
    
    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms

    ex_fr.append(get_fr(i_e, t_e, 1, simtime))
    in_fr.append(get_fr(i_i, t_i, 1, simtime))
    
    weights.append([m1, m2, m3, m4])
    
    #Change E -> I connections, n = 2
    m2 = get_con_matrix(NE, NI, ei_p, ei_mi, ei_var, connectivity_type=connectivity_type)
    S_ei.w = m2

    run(simtime, report=report)
    
    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms

    ex_fr.append(get_fr(i_e, t_e, 2, simtime))
    in_fr.append(get_fr(i_i, t_i, 2, simtime))

    weights.append([m1, m2, m3, m4])
    
    #Change I -> E connections, n = 3
    m3 = get_con_matrix(NI, NE, ie_p, ie_mi, ie_var, connectivity_type=connectivity_type)
    S_ie.w = m3

    run(simtime, report=report)
    
    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms

    ex_fr.append(get_fr(i_e, t_e, 3, simtime))
    in_fr.append(get_fr(i_i, t_i, 3, simtime))
    
    weights.append([m1, m2, m3, m4])

    #Change I -> I connections, n = 4
    m4 = get_con_matrix(NI, NI, ii_p, ii_mi, ii_var, connectivity_type=connectivity_type)
    S_ii.w = m4

    run(simtime, report=report)
    
    i_e = S_E.i[:]
    t_e = S_E.t/ms
    i_i = S_I.i[:]
    t_i = S_I.t/ms

    ex_fr.append(get_fr(i_e, t_e, 4, simtime))
    in_fr.append(get_fr(i_i, t_i, 4, simtime))
    
    weights.append([m1, m2, m3, m4])
    
    return ex_fr, in_fr, weights, stats
