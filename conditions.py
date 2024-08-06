import numpy as np
import starsim as ss
import mighti as mi

__all__ = [
    'Hypertension',
]

class Hypertension(ss.Disease):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            prevalence=ss.bernoulli(0.092999814),  # Initial prevalence of hypertension
            incidence=ss.bernoulli(0.005832899),  # Incidence at each point in time
            p_death=ss.bernoulli(0.01),  # Risk of death from hypertension
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('susceptible'),  # Boolean array to track susceptibility
            ss.BoolArr('affected'),  # Boolean array to track who is affected
            ss.FloatArr('ti_affected'),  # Float array to track time of being affected
            ss.FloatArr('ti_dead'),  # Float array to track time of death
        )

        return

    def init_vals(self):
        """ Populate initial values in states """
        initial_cases = self.pars['prevalence'].filter()
        self.affected[initial_cases] = True
        self.susceptible[~initial_cases] = True
        return initial_cases

    def update_pre(self):
        sim = self.sim
        deaths = (self.ti_dead == sim.ti).uids
        sim.people.request_death(deaths)
        self.results.new_deaths[sim.ti] = len(deaths)  # Log deaths attributable to this module
        return

    def make_new_cases(self):
        new_cases = self.pars['incidence'].filter(self.susceptible.uids)
        self.set_prognoses(new_cases)
        return

    def set_prognoses(self, uids):
        sim = self.sim
        p = self.pars
        self.affected[uids] = True

        # Determine who dies and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        self.ti_dead[dead_uids] = sim.ti + np.inf  # Chronic condition, death time far in the future

        return

    def init_results(self):
        sim = self.sim
        super().init_results()
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
        ]
        return

    def update_results(self):
        super().update_results()
        sim = self.sim
        ti = sim.ti
        self.results.prevalence[ti] = np.count_nonzero(self.affected)/len(sim.people)
        self.results.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        return
