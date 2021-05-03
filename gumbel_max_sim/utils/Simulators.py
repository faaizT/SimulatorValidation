from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from gumbel_max_sim.utils.MDP import MdpPyro
from gumbel_max_sim.utils.ObservationalDataset import cols
import torch
import pyro
import pyro.distributions as dist
from pyroapi import infer
from pyro.ops.indexing import Vindex
import logging


def get_vaso_simulator(name):
    if name == "real":
        return MDPVasoReal
    if name == "vaso1":
        return MDPVasoIntensity1
    if name == "vaso2":
        return MDPVasoIntensity2
    if name == "vaso_minus1":
        return MDPVasoIntensity_minus1
    if name == "vaso_minus2":
        return MDPVasoIntensity_minus2

def get_vent_simulator(name):
    if name == "real":
        return MDPVentReal
    if name == "vent1":
        return MDPVentIntensity1
    if name == "vent2":
        return MDPVentIntensity2
    if name == "vent_minus1":
        return MDPVentIntensity_minus1
    if name == "vent_minus2":
        return MDPVentIntensity_minus2

def get_antibiotic_simulator(name):
    if name == "real":
        return MDPAnitibioticReal
    if name == "antibiotic1":
        return MDPAnitibioticIntensity1
    if name == "antibiotic2":
        return MDPAnitibioticIntensity2
    if name == "antibiotic_minus1":
        return MDPAnitibioticIntensity_minus1
    if name == "antibiotic_minus2":
        return MDPAnitibioticIntensity_minus2

def get_combined_simulator(vaso_sim_name, vent_sim_name, antib_sim_name, device):
    vaso_sim = get_vaso_simulator(vaso_sim_name)
    vent_sim = get_vent_simulator(vent_sim_name)
    antib_sim = get_antibiotic_simulator(antib_sim_name)
    class Simulator(vaso_sim, vent_sim, antib_sim, MdpPyro):
        def __init__(self, init_state, device):
            vaso_sim.__init__(self, device)
            vent_sim.__init__(self, device)
            antib_sim.__init__(self, device)
            MdpPyro.__init__(self, init_state, device)
    return Simulator


class MDPVentReal():
    def __init__(self, device):
        self.device = device
    
    def transition_vent_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        """
        percoxyg_probs = torch.FloatTensor([
            [0.3, 0.7],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        """
        vent_state = torch.column_stack([state.vent_state]*4).reshape(batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.1, 0.9]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity2():
    def __init__(self, device):
        self.device = device
    
    def transition_vent_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state on
        percent oxygen: low -> normal w.p. 1.
        """
        percoxyg_probs = torch.FloatTensor([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .7
        """
        vent_state = torch.column_stack([state.vent_state]*4).reshape(batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.7, 0.3]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity1():
    def __init__(self, device):
        self.device = device
    
    def transition_vent_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .8
        """
        percoxyg_probs = torch.FloatTensor([
            [0.2, 0.8],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .4
        """
        vent_state = torch.column_stack([state.vent_state]*4).reshape(batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.4, 0.6]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity_minus1():
    def __init__(self, device):
        self.device = device

    def transition_vent_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .2
        """
        percoxyg_probs = torch.FloatTensor([
            [0.8, 0.2],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*batch_size)])
        return percoxyg_probs.to(self.device)
    
    def transition_vent_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        """
        vent_state = torch.column_stack([state.vent_state]*4).reshape(batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.1, 0.9]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity_minus2():
    def __init__(self, device):
        self.device = device

    def transition_vent_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state on
        percent oxygen: low -> normal w.p. 0.0
        """
        percoxyg_probs = torch.FloatTensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        """
        vent_state = torch.column_stack([state.vent_state]*4).reshape(batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.1, 0.9]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPAnitibioticReal():
    def __init__(self, device):
        self.device = device

    def transition_antibiotics_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.5, 0.5]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.5, 0.5]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        """
        antibiotic_state = torch.column_stack([state.antibiotic_state]*9).reshape(batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity2():
    def __init__(self, device):
        self.device = device

    def transition_antibiotics_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .9
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .6
        """
        antibiotic_state = torch.column_stack([state.antibiotic_state]*9).reshape(batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity1():
    def __init__(self, device):
        self.device = device

    def transition_antibiotics_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .7
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7, 0.3]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7, 0.3]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .4
        """
        antibiotic_state = torch.column_stack([state.antibiotic_state]*9).reshape(batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity_minus1():
    def __init__(self, device):
        self.device = device

    def transition_antibiotics_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .3
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 0.7]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        """
        antibiotic_state = torch.column_stack([state.antibiotic_state]*9).reshape(batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity_minus2():
    def __init__(self, device):
        self.device = device

    def transition_antibiotics_on(self):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .1
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self, state):
        batch_size = state.vent_state.size(0)
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        """
        antibiotic_state = torch.column_stack([state.antibiotic_state]*9).reshape(batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPVasoReal():
    def __init__(self, device):
        self.device = device

    def transition_vaso_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.5, 0.4],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.5, 0.5, 0.0, 0.0, 0.0], 
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0], 
            [0.0, 0.0, 0.0, 0.5, 0.5], 
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*batch_size), torch.stack([glucose_probs_diab]*batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self):
        batch_size = state.vent_state.size(0)
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        vaso_state = torch.column_stack([state.vaso_state]*2*9).reshape(2,batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity2():
    def __init__(self, device):
        self.device = device
    
    def transition_vaso_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .2, low -> high w.p. .5, 
                    normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .1, lo -> hi w.p. .8
            raise blood glucose by 3 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.1, 0.8],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.2, 0.5],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.3, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.3, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.3, 0.7],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*batch_size), torch.stack([glucose_probs_diab]*batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self, state):
        batch_size = state.vent_state.size(0)
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .5
            for diabetics, blood pressure falls by 1 w.p. .3
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        vaso_state = torch.column_stack([state.vaso_state]*2*9).reshape(2,batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity1():
    def __init__(self, device):
        self.device = device

    def transition_vaso_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .3, low -> high w.p. .4, 
                    normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .3, lo -> hi w.p. .6
            raise blood glucose by 2 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.3, 0.6],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.3, 0.4],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.2, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*batch_size), torch.stack([glucose_probs_diab]*batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self, state):
        batch_size = state.vent_state.size(0)
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .3
            for diabetics, blood pressure falls by 1 w.p. .2
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.2, 0.8]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        vaso_state = torch.column_stack([state.vaso_state]*2*9).reshape(2,batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity_minus1():
    def __init__(self, device):
        self.device = device

    def transition_vaso_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .3
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.5, 0.4],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*batch_size), torch.stack([glucose_probs_diab]*batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self, state):
        batch_size = state.vent_state.size(0)
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        vaso_state = torch.column_stack([state.vaso_state]*2*9).reshape(2,batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity_minus2():
    def __init__(self, device):
        self.device = device

    def transition_vaso_on(self, state):
        batch_size = state.vent_state.size(0)
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .2
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.5, 0.4],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*batch_size), torch.stack([glucose_probs_diab]*batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self, state):
        batch_size = state.vent_state.size(0)
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*batch_size), torch.stack([sysbp_probs_diab]*batch_size)))
        vaso_state = torch.column_stack([state.vaso_state]*2*9).reshape(2,batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)
