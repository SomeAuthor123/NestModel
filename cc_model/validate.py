import numpy as np
from collections import Counter
from wl import WL, labelings_are_equivalent


def check_WL_agree(original_g, synthetic_g, max_rounds, verbosity=0):
    _, labelings = WL(synthetic_g, k_max = max_rounds, verbosity=verbosity)
    originals = [original_g.vp[f"color_{WL_round}"] for WL_round in range(max_rounds)]
    result = []
    for original, synthetic in zip(originals, labelings):
        
        c1 = np.sort(np.fromiter(Counter(original).values(),dtype=int))[::-1]
        c2 = np.sort(np.fromiter(Counter(synthetic).values(),dtype=int))[::-1]
        if verbosity>3:
            print(np.sum(c1), np.sum(c2))
            print(len(c1), len(c2))
            if len(c1)==len(c2):
                print(np.sum(np.abs(c2-c1)))
                print(c1[(c2-c1)!=0])
                print(c2[(c2-c1)!=0])
            else:
                print("number of colors disagree")
        result.append(labelings_are_equivalent(original, synthetic, verbosity=5))
    return result


def check_edges_agree(g1,g2, print_disagree=False):
    a=g1.get_edges()
    a=np.vstack((a, np.vstack((a[:,1], a[:,0])).T))
    ind = np.lexsort((a[:,1], a[:,0]), axis=0)
    b1=a[ind,:]
    
    a=g2.get_edges()
    a=np.vstack((a, np.vstack((a[:,1], a[:,0])).T))
    ind = np.lexsort((a[:,1], a[:,0]), axis=0)
    b2=a[ind,:]
    
    agree = np.all(b1==b2, axis=1)

    #print("disagreements1", b[~agree,:])
    #print("disagreements2",b3[~agree,:])
    if print_disagree:
        
        for i, v in enumerate(g1.get_vertices()):
            deg1 = np.sort(g1.get_all_neighbors(v))
            deg2 = np.sort(g2.get_all_neighbors(v))
            if not np.all(deg1==deg2):
                print(deg1, deg2)
    if not np.all(agree):
        return False
    else:
        return True