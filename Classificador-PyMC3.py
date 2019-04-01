#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:32:03 2019

@author: wesley
"""

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

plot_options = dict(linewidth=3, alpha=0.6)


from collections import Counter

class Pmf(Counter):
    
    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = sum(self.values())
        for key in self:
            self[key] /= total
            
    def sorted_items(self):
        """Returns the outcomes and their probabilities."""
        return zip(*sorted(self.items()))



def underride(options):
    """Add key-value pairs to d only if key is not in d.

    options: dictionary
    """

    for key, val in plot_options.items():
        options.setdefault(key, val)
    return options

def plot(xs, ys, **options):
    """Line plot with plot_options."""
    plt.plot(xs, ys, **underride(options))

def bar(xs, ys, **options):
    """Bar plot with plot_options."""
    plt.bar(xs, ys, **underride(options))

def plot_pmf(sample, **options):
    """Compute and plot a PMF."""
    pmf = Pmf(sample)
    pmf.normalize()
    xs, ps = pmf.sorted_items()
    bar(xs, ps, **options)
    
def pmf_goals():
    """Decorate the axes."""
    plt.xlabel('Number of goals')
    plt.ylabel('PMF')
    plt.title('Distribution of goals scored')
    legend()
    
def legend(**options):
    """Draw a legend only if there are labeled items.
    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if len(labels):
        plt.legend(**options)

df = pd.read_csv('./datasets/assis-estud-agregado-media-clean.csv', delimiter=';')
dfOrder = df.sort_values(by=['renda_mensal'])
ACEITOS = dfOrder['deferimento']==1
REJEITADOS = dfOrder['deferimento']==0

dfRejeitados = dfOrder[REJEITADOS]
dfAceitos = dfOrder[ACEITOS]







iqr = dfRejeitados['renda_mensal'][dfRejeitados['renda_mensal'].between(dfRejeitados['renda_mensal'].quantile(.0), dfRejeitados['renda_mensal'].quantile(.25), inclusive=True)]

teste = iqr = dfRejeitados['renda_mensal'][dfRejeitados['renda_mensal'].between(dfRejeitados['renda_mensal'].quantile(.26), dfRejeitados['renda_mensal'].quantile(1), inclusive=True)]

intervalos = [(.0, .24), (.25, .50), (.51, .75), (.76, 1)]

for i in intervalos:
    piso = i[0]
    teto = i[1]
    iqr = dfRejeitados['renda_mensal'][dfRejeitados['renda_mensal'].between(dfRejeitados['renda_mensal'].quantile(piso), dfRejeitados['renda_mensal'].quantile(teto), inclusive=True)]
    print('Intervalo entre %.2f e %.2f: %d itens.' % (i[0], i[1], len(iqr)))
    


def estimate_gamma_params(xs):
    """Estimate the parameters of a gamma distribution.
    
    See https://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
    """
    s = np.log(np.mean(xs)) - np.mean(np.log(xs))
    k = (3 - s + np.sqrt((s-3)**2 + 24*s)) / 12 / s
    theta = np.mean(xs) / k
    alpha = k
    beta = 1 / theta
    return alpha, beta  


xs = iqr


alpha, beta = estimate_gamma_params(xs)
print(alpha, beta)

def plot_cdf(sample, **options):
    """Compute and plot the CDF of a sample."""
    pmf = Pmf(sample)
    xs, freqs = pmf.sorted_items()
    ps = np.cumsum(freqs, dtype=np.float)
    ps /= ps[-1]
    plot(xs, ps, **options)
    
def cdf_rates():
    """Decorate the axes."""
    plt.xlabel('Goal scoring rate (mu)')
    plt.ylabel('CDF')
    plt.title('Distribution of goal scoring rate')
    legend()

def cdf_goals():
    """Decorate the axes."""
    plt.xlabel('Number of goals')
    plt.ylabel('CDF')
    plt.title('Distribution of goals scored')
    legend()

def plot_cdfs(*sample_seq, **options):
    """Plot multiple CDFs."""
    for sample in sample_seq:
        plot_cdf(sample, **options)
    cdf_goals()


def make_gamma_dist(alpha, beta):
    """Returns a frozen distribution with given parameters.
    """
    return st.gamma(a=alpha, scale=1/beta)


dist = make_gamma_dist(alpha, beta)
print(dist.mean(), dist.std())


size = len(iqr)
mu = iqr.mean()
sample_poisson = np.random.poisson(mu, size)
np.mean(sample_poisson)


plot_cdf(sample_poisson, label='poisson', linestyle='dashed')
legend()


def poisson_likelihood(goals, mu):
    """Probability of goals given scoring rate.
    
    goals: observed number of goals (scalar or sequence)
    mu: hypothetical goals per game
    
    returns: probability
    """
    return np.prod(st.poisson.pmf(goals, mu))

print(poisson_likelihood(goals=12, mu=500))

#dfRejeitados.renda_mensal.hist()

#dfAceitos.renda_mensal.hist()


class Suite(Pmf):
    """Represents a set of hypotheses and their probabilities."""
    
    def bayes_update(self, data, like_func):
        """Perform a Bayesian update.
        
        data:      some representation of observed data
        like_func: likelihood function that takes (data, hypo), where
                   hypo is the hypothetical value of some parameter,
                   and returns P(data | hypo)
        """
        for hypo in self:
            self[hypo] *= like_func(data, hypo)
        self.normalize()
        
    def plot(self, **options):
        """Plot the hypotheses and their probabilities."""
        xs, ps = self.sorted_items()
        plot(xs, ps, **options)
        

def pdf_rate():
    """Decorate the axes."""
    plt.xlabel('RENDA MÉDIA POR ALUNO (mu)')
    plt.ylabel('PDF')
    #plt.title('Distribution of goal scoring rate')
    plt.title('Distribution of RENDA MÉDIA')
    
    legend()

hypo_mu = np.linspace(500, 1500, num=50)
hypo_mu

suite = Suite(hypo_mu)
suite.normalize()
suite.plot(label='prior')
pdf_rate()



suite.bayes_update(data=477, like_func=poisson_likelihood)
suite.plot(label='posterior')
pdf_rate()


    