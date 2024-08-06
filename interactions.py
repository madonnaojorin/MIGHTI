import pandas as pd
import starsim as ss
import mighti as mi
from collections import defaultdict

__all__ = [
    'hiv_hypertension'
]

class hiv_hypertension(ss.Connector):
    """ Simple connector to make people with hypertension more likely to contract HIV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.Hypertension])
        self.default_pars(
            rel_sus_hiv_hypertension=1.3,  # People with hypertension are 1.3x more likely to acquire HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        """ Specify HIV-hypertension interactions """
        sim = self.sim
        sim.diseases.hiv.rel_sus[sim.people.hypertension.affected] = self.pars.rel_sus_hiv_hypertension
        return

# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions
    Note, this is not yet used, but at some point this could be adjusted so that it automatically
    created the Connectors
    """
    if datafile is None:
        datafile = '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile)

    rel_sus = defaultdict(dict)

    for cond in df.has_condition.unique():
        conddf = df.loc[df.has_condition == cond]
        conddf.reset_index(inplace=True, drop=True)
        interacting_conds = conddf.columns[~conddf.isna().any()].tolist()
        interacting_conds.remove('has_condition')
        for interacting_cond in interacting_conds:
            rel_sus[interacting_cond][cond] = conddf.loc[0, interacting_cond]

    return rel_sus
