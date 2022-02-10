#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pathlib import Path
from cc_model.load_datasets import load_gt_dataset_cached


import graph_tool.all as gt
from cc_model.wl import WL, labelings_are_equivalent
from cc_model.fast_graph import FastGraph
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt


# In[10]:


# Code that should make matplotlib use ACM font (libertine)
import matplotlib

rc_fonts = {
    "font.family": "serif",
    "font.size": 20,
    'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """,
}
matplotlib.rcParams.update(rc_fonts)


# In[35]:


datasets = [#"karate",
            "phonecalls",
            "HepPh",
            "AstroPh",
            "web-Google",
             "soc-Pokec"
#            "deezer_HR", "deezer_HU", "deezer_RO","tw_musae_DE",
#            "tw_musae_ENGB","tw_musae_FR","lastfm_asia","fb_ath",
#            "fb_pol", "facebook_sc"
           ]


# In[12]:


dataset_path = Path("/home/felix/projects/colorful_configuration/datasets")


# In[13]:


epsilon=1e-16
max_iter = 1000
alpha=0.85


# In[14]:


def rewire_run_pagerank_for_round(G, depth, in_base_pagerank, number_of_rewires=10, verbosity=0):
    """ Calculate similarities in pagerank vectors for rewired G
    G :      FastGraph
    depth:    depth of WL iteration to be used
    in_base_pagerank :    in pagerank of G
    Generate synthethic networks which have the same WL colors as G
      at specific depth and return absolute error sum
    """
    pagerank_args = {"mode":"in",
                     "epsilon":epsilon,
                     "max_iter":max_iter,
                     "alpha":alpha,
                     "return_err":True}

    errors = []
    for i in range(number_of_rewires):
        G.rewire(depth)

        in_pagerank, err = G.calc_pagerank(**pagerank_args)

        if verbosity > 0:
            print("the error in pagerank is:\r\n", err)
        if err < 0:
            print("did not converge!")
        pagerank_diff = np.sum(np.abs(in_base_pagerank-in_pagerank))
        errors.append(pagerank_diff)

    return errors


# In[26]:


def compute_pagerank_on_all_datasets(n_rewires, verbosity=0):
    """ computes pagerank and compares it with rewired graphs
    """
    list_values = []
    list_stds = []
    for dataset in datasets:
        if dataset is None:
            list_means.append([])
            list_stds.append([])
            continue
        if verbosity > 0:
            print(dataset)
        G_base = load_gt_dataset_cached(dataset_path,
                                        dataset,
                                        verbosity=verbosity,
                                        force_reload=False)
        edges = np.array(G_base.get_edges(), dtype=np.uint32)



        G = FastGraph(edges, G_base.is_directed())

        values = get_SAE_for_iterations(G,
                                        n_rewires=n_rewires,)
        list_values.append(values)
    return list_values


# In[27]:


def get_SAE_for_iterations(G, n_rewires):
    values = []
    G.ensure_edges_prepared(initial_colors="out_degree")
    print(G.dead_arr.sum(axis=1)/G._edges.shape[0])
    in_base_pagerank, _ = G.calc_pagerank("base in",
                                          epsilon=epsilon,
                                          max_iter=max_iter,
                                          alpha=alpha,
                                          return_err=True)

    for WL_round in range(G.wl_iterations-1,-1,-1):
        if verbosity>0:
            print("WL round", WL_round)

        SAEs = rewire_run_pagerank_for_round(G,
                                             WL_round,
                                             in_base_pagerank,
                                             number_of_rewires=n_rewires,
                                             verbosity=0)
        values.append(SAEs)
    return values


# In[31]:



verbosity=1
number_of_samples = 100
list_values = compute_pagerank_on_all_datasets(number_of_samples, verbosity=1)


# In[ ]:





# In[32]:


print()
now = datetime.now()
if number_of_samples > 10:
    save_prefix = now.strftime("%Y_%m_%d__%H_%M_%S")
    out_name = "./results/"+"_pagerank_"+save_prefix+".pkl"
    print(out_name)
    with open(out_name, "wb") as f:
        pickle.dump((list_values, datasets), f)


# In[ ]:





# In[33]:


def get_mean_std(list_values):
    list_means = []
    list_stds0 = []
    list_stds1 = []

    for values in list_values:
        mean = np.mean(values)+1e-20
        quantiles = np.quantile(values, [0.5-0.68/2, 0.5+0.68/2,])
        list_means.append(mean)
        list_stds0.append(quantiles[0])
        list_stds1.append(quantiles[1])
    list_stds = [list_stds0[::-1], list_stds1[::-1]]
    return list_means[::-1], list_stds

def quickplot(list_values, datasets, show_alpha=False, save_date=None, xlim=None):
    plt.figure(figsize=(10,6))
    if show_alpha:
        x=np.linspace(0,15)
        y=2 *alpha**(x+1)
        plt.plot(x,y)

    markers = [".", "o", "<", "s", "*"]
    markers2 = ["^", "o", "+", "x", "_"]
    marker_sizes = 6*np.array([1.4, 1, 1.5, 1, 1.9,])



    for values, label, marker, ms in zip(list_values, datasets, markers2, marker_sizes):
        means, stds = get_mean_std(values)
        plt.errorbar(x=np.arange(len(means)),y=means, yerr=stds, label=label, fmt=marker+"--", markersize=ms)


    plt.ylabel("SAE of PageRank centrality")
    plt.xlabel("depth $d$")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.yscale("log")
    plt.legend()
    #
    if save_date is None:
        plt.title("Convergence of pagerank for synthetic networks ")
    else:
        time_str = save_date.strftime("%Y_%m_%d__%H_%M_%S")
        plt.savefig(Path(".")/Path('images')/f'pagerank_{time_str}.pdf', bbox_inches = 'tight')


# In[34]:


quickplot(list_values, datasets, show_alpha=False, save_date=datetime.now())


# In[ ]:


load_dataset=True
if load_dataset:
    load_date = datetime(2021,12,11,3,2,49)
    save_prefix = load_date.strftime("%Y_%m_%d__%H_%M_%S")
    with open("./results/"+save_prefix+".pkl", "rb") as f:
        (list_means2, list_stds2, datasets2) = pickle.load(f)


# In[ ]:


quickplot(list_means2, list_stds2, datasets2, save_date=load_date)


# In[ ]:


plt.rcParams['lines.markersize']


# In[ ]:




