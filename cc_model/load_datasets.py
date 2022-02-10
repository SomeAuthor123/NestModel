import numpy as np
import pandas as pd
from cc_model.utils import graph_tool_from_edges

def relabel_edges(edges):
    unique = np.unique(edges.ravel())
    mapping = {key:val for key, val in zip(unique, range(len(unique)))}
    out_edges = np.empty_like(edges)
    for i,(e1,e2) in enumerate(edges):
        out_edges[i,0] = mapping[e1]
        out_edges[i,1] = mapping[e2]
    return out_edges

def check_is_directed(edges):
    d = {(a,b) for a,b in edges}
    for a,b in edges:
        assert (b,a) in d

class Dataset:
    def __init__(self, name, file_name, is_directed=False, delimiter=None):
        self.name=name
        self.file_name = file_name
        self.get_edges = self.get_edges_pandas
        self.skip_rows = 0
        self.is_directed=is_directed
        self.delimiter = delimiter
        self.requires_node_renaming=False



    def get_edges_pandas(self, datasets_dir):
        df = pd.read_csv(datasets_dir/self.file_name, skiprows=self.skip_rows, header=None, sep=self.delimiter)
        edges = np.array([df[0].to_numpy(), df[1].to_numpy()],dtype=np.uint64).T


        if self.requires_node_renaming:
            return relabel_edges(edges)
        else:
            return edges

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Dataset):
            return self.name==other.name
        else:
            raise ValueError()

    def get_edges_karate(self, datasets_dir):
        import networkx as nx
        import numpy as np
        G = nx.karate_club_graph()
        edges = np.array(list(G.edges), dtype=int)
        return edges

Phonecalls = Dataset("phonecalls", "phonecalls.edgelist.txt", delimiter="\t")

AstroPh = Dataset("AstroPh", "ca-AstroPh.txt", delimiter="\t", is_directed=True)
AstroPh.skip_rows=4
AstroPh.requires_node_renaming=True

HepPh = Dataset("HepPh", "cit-HepPh.txt", delimiter="\t", is_directed=True)
HepPh.skip_rows=4
HepPh.requires_node_renaming=True

Karate = Dataset("karate", "karate")
Karate.get_edges = Karate.get_edges_karate

Google= Dataset("web-Google", "web-Google.txt", delimiter="\t", is_directed=True)
Google.skip_rows=4
Google.requires_node_renaming=True

Pokec= Dataset("soc-Pokec", "soc-pokec-relationships.txt", delimiter="\t", is_directed=True)
Pokec.skip_rows=0
Pokec.requires_node_renaming=True

all_datasets = [Karate, Phonecalls, AstroPh, HepPh, Google, Pokec]

def find_dataset(dataset_name):
    dataset = None
    for potential_dataset in all_datasets:
        if potential_dataset == dataset_name:
            dataset = potential_dataset
            break
    assert not dataset is None, f"You have specified an unknown dataset {dataset}"
    return dataset



def load_dataset(datasets_dir, dataset_name):
    #"deezer_HR", "deezer_HU", "deezer_RO","tw_musae_DE",
    #            "tw_musae_ENGB","tw_musae_FR","lastfm_asia","fb_ath",
    #            "fb_pol","phonecalls", "facebook_sc"]

    dataset = find_dataset(dataset_name)
    edges = dataset.get_edges(datasets_dir)

    if dataset.is_directed==False:
        edges = edges[edges[:,0] < edges[:,1],:]
        #[(e1, e2) for e1, e2 in edges if e1 < e2]
    #print("A", dataset.is_directed)
    return edges, dataset.is_directed



def load_gt_dataset_cached(datasets_dir, dataset_name, verbosity=0, force_reload=False):
    dataset = find_dataset(dataset_name)
    cache_file = datasets_dir/(dataset.file_name+".gt")
    if cache_file.is_file() and not force_reload:
        if verbosity>1:
            print("loading cached")
        import graph_tool.all as gt
        return gt.load_graph(str(cache_file.absolute()))
    else:
        if verbosity>1:
            print("loading raw")
        edges, is_directed = load_dataset(datasets_dir, dataset_name)
        g = graph_tool_from_edges(edges, None, is_directed=is_directed)
        g.save(str(cache_file.absolute()))
        return g