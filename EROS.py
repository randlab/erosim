import numpy as np
import matplotlib.pyplot as plt
import geone
import geone.covModel as gcm
import pyvista as pv
import scipy
from skimage.measure import label  # for connectivity
import time

from shapely.geometry import Polygon, LineString, MultiPolygon
import rasterio
import rasterio.features
from rasterio import Affine


class Graph:
    def __init__(self, list_ids, directed=False):
        
        self.list_ids = list_ids.astype(int)
        self.directed = directed
        
        self.list = {}
        for i in self.list_ids:
            self.list[i] = []
        

    def add_edge(self, node1, node2, weight=1):

        self.list[node1].append((node2, weight))

        if not self.directed:
            self.list[node2].append((node1, weight))

    def print_list(self):
        print(self.list)
        
# functions   
def apply_facies(arr, dic_res):
    
    res = arr.copy()
    for k,v in dic_res.items():
        res[arr==k]=v
    return res

def plot_bh(w_logs, width=1):
    
    for s in w_logs:
        
        x,y, log = s
        z_0 = log[0][1] - y
        for unit in log:
            np.random.seed(unit[0]+24)
            plt.bar(x, unit[1] - z_0, bottom=z_0, linewidth=width, edgecolor="black", color=np.random.random(size=3), width=width)

## 3D functions
def compute_domain(z0, z1, nx, ny, nz, sz, s1, s2):

    """
    Return a bool 2D array that define the domain where the units
    exist (between two surfaces, s1 and s2)

    s1, s2: 2D arrays, two given surfaces over simulation domain size: (ny, nx)),
    s1 is top surface, s2 is bot surface
    """

    s1[s1 < z0]=z0
    s1[s1 > z1]=z1
    s2[s2 < z0]=z0
    s2[s2 > z1]=z1

    idx_s1=(np.round((s1-z0)/sz)).astype(int)
    idx_s2=(np.round((s2-z0)/sz)).astype(int)
    
    diff = s1 - s2
    list_iy, list_ix = np.where(diff > 0)
    
    #domain
    a=np.zeros([nz, ny, nx])
    for iy, ix in zip(list_iy, list_ix):
        a[idx_s2[iy, ix]: idx_s1[iy, ix], iy, ix]=1

    return a

class Graph_3D:
    def __init__(self, directed=False):
        
        self.directed = directed
        
        self.list = {}
#         for i in self.list_ids:
#             self.list[i] = []
        

    def add_edge(self, node1, node2, weight=1):
        
        if node1 not in self.list.keys():
            self.list[node1] = []
        if node2 not in self.list.keys():
            self.list[node2] = []
        
        self.list[node1].append((node2, weight))

        if not self.directed:
            self.list[node2].append((node1, weight))

    def print_list(self):
        print(self.list)


def plot_bhs_3D(w_logs, z0, plotter=None, v_ex=1):

    """
    Plot boreholes in w_logs

    #parameters#
    w_logs   : borehole logs
    plotter: pyvista plotter
    v_ex  : float, vertical exaggeration
    """

    def lines_from_points(points):
        """Given an array of points, make a line set"""
        poly=pv.PolyData()
        poly.points=points
        cells=np.full((len(points)-1, 3), 2, dtype=np.int_)
        cells[:, 1]=np.arange(0, len(points)-1, dtype=np.int_)
        cells[:, 2]=np.arange(1, len(points), dtype=np.int_)
        poly.lines=cells
        return poly

    if plotter is None:
        p=pv.Plotter()
    else:
        p=plotter

    for bh in w_logs:
        for i in range(len(bh[3])):

            l=[]
            st=bh[3][i][0]
            l.append(bh[3][i][1])
            if i < len(bh[3])-1:
                l.append(bh[3][i+1][1])

            if i == len(bh[3])-1:
                l.append(bh[3][0][1]-bh[2])
            pts=np.array([np.ones([len(l)])*bh[0], np.ones([len(l)])*bh[1], l]).T
           
            line=lines_from_points(pts)
            line.points[:, -1]=(line.points[:, -1] - z0)*v_ex+z0
            if st is not None:
                np.random.seed(st+24)
                color=np.random.random(size=3)
                opacity=1
            else:
                color="white"
                opacity=0
            p.add_mesh(line, color=color, interpolate_before_map=True, render_lines_as_tubes=True, line_width=15, opacity=opacity)

    if plotter is None:
        p.add_bounding_box()
        p.show_axes()
        p.show()
          

def sim(N, covmodels, means_surf, dimension, spacing, origin, w_logs, nreal=1, bot=None, top=None, xi=0.2, grf_method = "sgs",
        facies_ids = [1, 2, 3, 4], proba_cdf = [0.25, 0.25, 0.25, 0.25], alpha=1, seed = 5):
    

    return


def sim_uncond_2D(N, covmodels, means, dimension, spacing, origin, bot=None, top=None, xi=0.5, grf_method = "sgs",
        facies_ids = [1, 2, 3, 4], proba = [0.25, 0.25, 0.25, 0.25], alpha = 0.5, p_combi="log", seed = 5, verbose=1):
    
    """
    
    
    ##inputs##
    N 
    covmodels
    means
    dimension  : (nx, ny), dimension of the simulation grid
    spacing    : (sx, sy), spacing of the simulation grid
    origin     : origin of the simulation grid
    xi         : float btw 0 and 1, fraction of erosive surface
    facies_ids : seq of int, facies ids to simulate
    proba      : proba target of the different facies
    alpha      : float btw 0 and 1, global proba fraction 
                 (1 mean only global proba is taken into account
                 and 0 only proba of neighbours)

    """

    np.random.seed(seed)

    nx, nz = dimension
    sx, sz = spacing
    ox, oz = origin
    z1 = oz + nz*sz
    x1 = ox + nx*sx
    
    if bot is None:
        bot = oz*np.ones(nx)
    if not isinstance(bot, np.ndarray):
        bot = np.ones(nx)*bot
        
    if top is None:
        top = z1*np.ones(nx)
    if not isinstance(top, np.ndarray):
        top = np.ones(nx)*top
    
    one_cm = False
    if isinstance(covmodels, gcm.CovModel1D):
        one_cm = True
    
    # adjust surfaces
    real_surf = np.ones([N, nx])
    
    i = 0
    while i < N:

        erod_layer = np.random.uniform() < xi

        # simulate surface
        if one_cm:
            cm = covmodels
        else:
            cm = covmodels[i]
        
        if grf_method == "fft":
            s1 = geone.grf.grf1D(cm, nx, sx, ox, mean=means[i])[0]
        else:
            s1 = geone.geosclassicinterface.simulate1D(cm, nx, sx, ox, nreal=1, mean=means[i], verbose=verbose, 
                                                       searchRadiusRelative=1, nneighborMax=12)["image"].val[0, 0, 0, :]
        
        s1[s1 > top] = top[s1 > top]
        s1[s1 < bot] = bot[s1 < bot]
        
        # loop over prexisting surfaces and apply erosion rules
        for o in range(i):
            s2 = real_surf[o]
            if erod_layer:
                s2[s2 > s1] = s1[s2 > s1] 
            else:
                s1[s1 < s2] = s2[s1 < s2]
                
        if i > 0 and erod_layer:
            s1[s1 > real_surf[i-1]] = real_surf[i-1][s1 > real_surf[i-1]]  # erode no deposition
        
        if not erod_layer:
            real_surf[i] = s1
            i += 1

    real_surf = np.concatenate((bot.reshape(1, nx), real_surf, top.reshape(1, nx)), axis=0)  # add top and bot
    real_surf[real_surf>z1]=z1
    real_surf[real_surf<oz]=oz
    # real_surf[real_surf>top]=top[real_surf>top]
    # real_surf[real_surf<bot]=bot[real_surf<bot]

    ## polygons
    list_p = []
    list_ids = []
    xg = np.linspace(ox, ox+sx*nx, nx)
    ID = 0
    for i in range(real_surf.shape[0]-1):

        s1=real_surf[i]
        s2=real_surf[i+1]

        mask_g = s2>s1

        mark = False
        ia = 0
        ib = 0
        g_1 = mask_g[0]
        g_2 = mask_g[-1]

        idx_g = np.where(mask_g[1:] != mask_g[:-1])[0]
        if len(idx_g) > 0:
            if g_1:
                start = 0
            else:
                start = 1

            for o in range(start, len(idx_g)+1, 2):
                if o == 0:
                    ia = 0
                    ib = idx_g[o]+2

                elif o < len(idx_g):
                    ia = idx_g[o-1]
                    ib = idx_g[o]+2
                else:
                    ia = idx_g[o-1]
                    ib = None

                coord_l1 = [(x,y) for x,y in zip(xg[ia:ib], s1[ia:ib])]
                coord_l2 = [(x,y) for x,y in zip(xg[ia:ib], s2[ia:ib])]

                if len(coord_l1) > 1 or len(coord_l2) > 1:
                    l1 = LineString(coord_l1)
                    l2 = LineString(coord_l2)
                    p = Polygon([*list(l2.coords), *list(l1.coords)[::-1]])
                    list_ids.append(ID)
                    # p.ID = ID
                    ID += 1
                    list_p.append(p)
                else:
                    pass

        elif g_1:
            coord_l1 = [(x,y) for x,y in zip(xg, s1)]
            coord_l2 = [(x,y) for x,y in zip(xg, s2)]

            if len(coord_l1) > 1 or len(coord_l2) > 1:
                l1 = LineString(coord_l1)
                l2 = LineString(coord_l2)
                p = Polygon([*list(l2.coords), *list(l1.coords)[::-1]])
                # p.ID = ID
                list_ids.append(ID)
                ID += 1
                list_p.append(p)
 
    # rasteriser les polygones 
    arr = rasterio.features.rasterize(shapes=zip(list_p, np.arange(len(list_p))), out_shape=(nz, nx),
                                transform=Affine(sx, 0.0, ox, 0.0, sz, oz), fill=-99)
    
    list_ids = np.array(list_ids)

    def create_graph():
        # if necessary create graph
        g = Graph(list_ids, False)

        for o in range(len(list_p)):
            p = list_p[o]
            # for i in list_p[o:]:
            for ip, i in enumerate(list_p[o:]):
                if p.intersects(i) and p != i:
                    res = (list_ids[o], list_ids[o+ip], p.intersection(i).length)
                    # res = (p.ID, i.ID, p.intersection(i).length)
                    if res[2] > 0:
                        g.add_edge(res[0], res[1], res[2])
        return g
            
    
    ## simulation of the facies
    facies_ids = np.array(facies_ids)
    proba_cdf = np.array(proba)
    
    if proba_cdf.sum() != 1:
        proba_cdf /= proba_cdf.sum()
        
    if alpha < 1:
        g = create_graph()
        
    # set initial facies to polygon (0 mean unknown)
    dic_res = {}
    for i in list_ids:
        dic_res[i] = 0

    # dictionary of facies area
    dic_area = {}
    for i in facies_ids:
        dic_area[i] = 0

    area_sim = 0  # total area simulated
    total_area = np.sum([p.area for p in list_p])  # total area
    
    ## algo with a graph
    ids_to_sim = [i for i in dic_res.keys() if dic_res[i]==0]  # cell id to simulate
    
    proba = proba_cdf.copy()
    while len(ids_to_sim) > 0:
        
        if area_sim > 0:
            # update proba according to area simulated
            area_ratio = area_sim/total_area
            for i in range(len(proba)):
                p = proba_cdf[i]
                facies_id = facies_ids[i]
                new_p = (p - area_ratio*dic_area[facies_id]/area_sim)/(1- area_ratio)
                if new_p < 0:
                    new_p = 0
                proba[i] = new_p


        proba = proba / proba.sum()
        id_sim = np.random.choice(ids_to_sim)  # select a volume to simulate

        p_neig = proba
        if alpha < 1:
            neigs = g.list[id_sim]  # neighbours 

            if len(neigs) == 1:  # only 1 neighbour

                fac = dic_res[neigs[0][0]]
                if fac == 0:
                    p_neig = proba
                else:
                    p_neig = facies_ids==fac

            elif len(neigs) > 1:

                sum_w = 0
                p_neig = np.zeros(facies_ids.shape)
                for n in neigs:
                    cell_id = n[0]  # cell id
                    w = n[1]  # weight
                    sum_w += w
                    fac = dic_res[cell_id]  # value at the cell
                    if fac == 0:  # no value, take proba
                        p_neig += proba*w
                    else:
                        p_neig += (fac==facies_ids)*w
                p_neig /= sum_w       
        else:
            p_neig=np.zeros(facies_ids.shape)

        # restrict p interval between 0.001 and 0.999
        p_neig[p_neig<0.001] = 0.001
        p_neig[p_neig>0.999] = 0.999
        p_neig[(p_neig > 0.001) & (p_neig < 0.999)].sum()/(1 - p_neig[(p_neig <= 0.001) | (p_neig >= 0.999)].sum())

        proba[proba<0.001] = 0.001
        proba[proba>0.999] = 0.999
        proba[(proba > 0.001) & (proba < 0.999)].sum()/(1 - proba[(proba <= 0.001) | (proba >= 0.999)].sum())

        ## mix p with global p
        if (p_neig * proba).sum() == 0:
            p_combi = "linear"
        if p_combi == "linear":
            p = (1-alpha)*p_neig + alpha*proba
        elif p_combi == "log":
            p = p_neig**(1-alpha) * proba**alpha
        else:
            print("Invalid p_combi, use linear or log")
        p = p / p.sum()
          
        facies_choice = np.random.choice(facies_ids, p=p)
        dic_res[id_sim] = facies_choice
        poly_id = np.where(list_ids==id_sim)[0][0]
        poly = list_p[poly_id]
        # poly = [i for i in list_p if i.ID == id_sim]
        area_sim += poly.area
        dic_area[facies_choice] += poly.area

        ids_to_sim.remove(id_sim)  # remove id from list
    
    arr_res = apply_facies(arr, dic_res)
    
    return real_surf, arr_res, list_p


def sim_cond_2D(N, covmodels, means_surf, dimension, spacing, origin, w_logs, nreal=1, bot=None, top=None, xi=0.5,
        facies_ids = [1, 2, 3, 4], proba_cdf = [0.25, 0.25, 0.25, 0.25], alpha=1, p_combi = "log", seed = 5, plots=False, verbose=1):
    
    np.random.seed(seed)
    
    def add_line():
        
        global N_surf, means, erod_lst, real_surf
        
        N_surf += 1
        
        if len(means.shape) == 1:
            means = np.concatenate((means, np.array(means[-1]).reshape(-1)))
        elif len(means.shape) == 2:
            means = np.concatenate((means, np.array(means[-1]).reshape(-1, nx)))

        erod_lst = np.concatenate((erod_lst, np.array((np.random.random() < xi)).reshape(-1)))
        real_surf = np.concatenate((real_surf, np.ones(nx).reshape(-1, nx)))
         

    def check_dic_c(dic_c, prog_logs=None):
            
        global N_surf, means, erod_lst, real_surf

        if prog_logs is None:  # prog is a dictionnary storing which intervals have been checked in each boreholes
            prog={}  # prog dic
            for k, v in sorted(dic_c.items()):
                for iv in v:
                    if iv not in prog.keys():
                        prog[iv] = -1
        else:
            prog={}  # prog dic
            for k, v in sorted(dic_c.items()):
                for iv in v:
                    if iv not in prog.keys():
                        prog[iv] = prog_logs[iv]
            
        for k, v in sorted(dic_c.items()):
            v = np.copy(v)
            if len(v) > 1:  # if multiple intervals constrained by the same surface
                prev = 0
                for o in range(len(v)):
                    ov = v[o]
                    if o == 0:
                        prev = w_logs[ov][2][prog[ov]][0]  # previous facies
                    else:
                        if prev != w_logs[ov][2][prog[ov]][0]:  # if facies are different
                            dic_c[k].remove(ov)  # remove the interval from the list
                            if not dic_c[k]:  # if the list is empty delete
                                del(dic_c[k])
                            flag = True
                            inc = 1
                            while flag:  # find new surface for removed interval
                                if k+np.abs(inc) >= N_surf:
                                    add_line()
                                    #raise ValueError("Error, difficulties to constrain, increase the number of lines (N)")
                                if k+inc not in dic_c.keys():
                                    dic_c[k+inc] = [ov]
                                    flag = False
                                inc += 1
                                # if k-inc not in dic_c.keys() and k-inc > i:
                                    # dic_c[k-inc] = [ov]
                                    # flag = False
                                # if inc < 0:
                                    # inc -= 1
                            # inc *= -1
            for iv in v:
                if iv not in prog.keys():
                    prog[iv] = -1
                else:
                    prog[iv] += -1
    
    def check_bh_compa(i2_max, bh_id, plot=False):


        """
        Check that a borehole is compatible with actual surfaces or not
        Return correct if no there is no problem. 
        If there is, returns an altitude indicating a maximal bound for the next grf to simulate
        """

        def plot_things():

            plot_bh(w_logs)
            new_shape = MultiPolygon(list_p)

            for geom in new_shape.geoms: 
                xs, ys = geom.exterior.xy    
                plt.fill(xs, ys, alpha=.5, fc=np.random.random(3), ec='none')        

            plt.show()

        list_p = []

        if plot:
            plt.figure(figsize=(12, 3), dpi=200)

        fa_id_to_const = w_logs[bh_id][2][prog_logs[bh_id]][0]  # facies id to constrained
        idx_bh_x = np.round(((w_logs[bh_id][0] - ox - sx/2)/sx)).astype(int)  # x_idx position of the borehole

        # loop over surfaces until below facies to constrained
        # this allows to keep only surfaces that are above what have already been constrained

        # get index of lowest surfaces below facies to constrained
        i2_min = i2_max

        if prog_logs[bh_id]+1 != 0:
            height_interface = w_logs[bh_id][2][prog_logs[bh_id]+1][1]
        else:
            height_interface = w_logs[bh_id][2][0][1] - w_logs[bh_id][1]

        while np.abs(real_surf[i2_min, idx_bh_x] - height_interface) > 0.1 and real_surf[i2_min, idx_bh_x] > height_interface:  
            i2_min -= 1

            if i2_min == 0:
                break
        
        for i2 in range(i2_min, i2_max):  # loop from lowest surfaces to highest

            s1 = real_surf[i2]
            s2 = real_surf[i2+1]

            l1 = s1[idx_bh_x]
            l2 = s2[idx_bh_x]

            if l1 < l2 and l1 < w_logs[bh_id][2][prog_logs[bh_id]][1] :
                if plot:
                    plt.plot(xgc, s1,c = "k", linewidth=0.5)
                    plt.plot(xgc, s2,c = "k", linewidth=0.5)

                #polygons
                mask_g = s2>s1

                mark = False
                ia = 0
                ib = 0
                g_1 = mask_g[0]
                g_2 = mask_g[-1]

                idx_g = np.where(mask_g[1:] != mask_g[:-1])[0]
                
                if len(idx_g) > 0:
                    if g_1:
                        start = 0
                    else:
                        start = 1

                    for o in range(start, len(idx_g)+1, 2):
                        if o == 0:
                            ia = 0
                            ib = idx_g[o]+2

                        elif o < len(idx_g):
                            ia = idx_g[o-1]
                            ib = idx_g[o]+2
                        else:
                            ia = idx_g[o-1]
                            ib = len(s1)

                        if idx_bh_x > ia and idx_bh_x < ib:  # if polygon touch the borehole where we are constraining
            
                            # make a polygon
                            coord_l1 = [(x,y) for x,y in zip(plot_xg[ia:ib], s1[ia:ib])]
                            coord_l2 = [(x,y) for x,y in zip(plot_xg[ia:ib], s2[ia:ib])]
                            
                elif g_1:  # case polygon go through the whole domain
                    coord_l1 = [(x,y) for x,y in zip(plot_xg, s1)]
                    coord_l2 = [(x,y) for x,y in zip(plot_xg, s2)]
                       
                if len(coord_l1) > 1 or len(coord_l2) > 1:
                    l1 = LineString(coord_l1)
                    l2 = LineString(coord_l2)
                    p = Polygon([*list(l2.coords), *list(l1.coords)[::-1]])
                    list_p.append(p)                  
                    # now check if that these polygons intersect the others boreholes
                    
                    for k,v in well_in_lines.items():
                        if k != bh_id:  # check only other boreholes
                            for idx_k in range(-1, prog_logs[k], -1):  # check only already constrained intervals
                                lin = v[0][idx_k]
                                if p.intersects(lin):
                                    if v[1] != fa_id_to_const:
                                        if plot:
                                            plot_things()

                                        #  return elevation up which there is a problem
                                        return real_surf[i2, idx_bh_x]  # incorrect connection
                                    
        if plot:
            plot_things()


        return "correct"
    
    # inputs
    global N_surf
    N_surf = N
    
    # grid
    nx, nz = dimension
    sx, sz = spacing
    ox, oz = origin
    z1 = oz + nz*sz
    x1 = ox + nx*sx
    xgc = np.linspace(ox+sx/2, ox+sx*nx-sx/2, nx)
    xg = np.arange(ox, ox+sx*(nx+1), sx)
    plot_xg = np.linspace(ox, ox+nx*sx, nx)

    if bot is None:
        bot = oz*np.ones(nx)
    if not isinstance(bot, np.ndarray):
        bot = np.ones(nx)*bot

    if top is None:
        top = z1*np.ones(nx)
    if not isinstance(top, np.ndarray):
        top = np.ones(nx)*top

    one_cm = False
    if isinstance(covmodels, gcm.CovModel1D):
        one_cm = True

    elif isinstance(covmodels, list):
        for cm in covmodels:
            assert isinstance(cm, gcm.CovModel1D), "object in covmodels must be geone CovModel1D objects"

    # adjust surfaces
    global erod_lst, real_surf, means # global variables
    
    means = means_surf.copy()
    erod_lst = np.random.uniform(size=N_surf) < xi  # determine which layers will be erode
    real_surf = np.ones([N_surf, nx])

    # correct position of the boreholes
    new_w_logs = []
    for w in w_logs:
        w =( plot_xg[np.round((w[0] - ox - sx/2)/sx).astype(int)], w[1], w[2]) # set x to the nearest center cell
        new_w_logs.append(w)
    w_logs = new_w_logs

    # warning --> does not allow borehole on the same location, to do
    for w in w_logs:
        for w2 in w_logs:
            if w != w2:
                if w[0] == w2[0]:  # same position
                    if w[1] > w2[1]:
                        w_logs.remove(w2)
                    else:
                        w_logs.remove(w)
    
    nwells = len(w_logs)
    
    ## put boreholes into lines
    well_in_lines = {}  # dictionary that contains linestrings of boreholes
    i = 0
    for well in w_logs:
        lines = []
        xbh = well[0]
        depth = well[1]
        z0 = well[2][0][1]

        for index in range(len(well[2])):
            fa = well[2][index]

            if index < len(well[2])-1:
                fa2 = well[2][index+1]

                line = LineString([(xbh, fa[1]-1e-3), (xbh, fa2[1]+1e-3)])
                # line.id= fa[0]
            else:
                line = LineString([(xbh, fa[1]-1e-3), (xbh, z0-depth+1e-3)])
                # line.id= fa[0]

            lines.append((line, fa[0]))
        well_in_lines[i] = lines
        i += 1
    
    # boreholes indexes
    bh_idxs = [np.round(((bh[0] - ox - sx/2)/sx)).astype(int) for bh in w_logs]

    # choose when to respect HD
    # dictionary of constrained from boreholes
    dic_c = {}
    for o in range(nwells):
        l = [i[1] for i in w_logs[o][-1]]  # interfaces in borehole

        for i in l:
            if one_cm:
                dis = scipy.stats.norm(i, np.sqrt(covmodels.sill()))
                probas = dis.pdf(means)

            else:
                probas = np.ones(N_surf, dtype=np.float32)
                for isurf in range(N_surf):
                    cm = covmodels[isurf]
                    dis = scipy.stats.norm(i, np.sqrt(cm.sill()))
                    probas[isurf] = dis.pdf(means[isurf])

            p = np.random.choice(range(N_surf), p=probas/probas.sum())

            if p not in dic_c.keys():
                dic_c[p] = []
            if o not in dic_c[p]:
                dic_c[p].append(o)
            else:
                flag = True
                while flag: 
                    p = np.random.choice(range(N_surf), p=probas/probas.sum())
                    if p not in dic_c.keys():
                        dic_c[p] = [o]
                        flag=False
    
    # some useful arrays
    prog_logs = -1*np.ones([nwells], dtype=int)  # progression of the constrained on the logs
    idx_const = np.sort(list(dic_c.keys()))  # idx of constrained

    dic_ineq = {}

    # simulation of the surfaces
    i = 0
    s1 = bot.copy()
    plot_to_do = False
    while i < N_surf:

        if plot_to_do:
            plt.plot(plot_xg, s1_org, c="r", linewidth=1, alpha=0.9, label="simulated surface")
            plt.legend()
            plt.show()
            plot_to_do=False

        if i in dic_c.keys() and plots:
            print(i, dic_c[i])
            fig, axs = plt.subplots(figsize=(10,5), dpi=200)
            plt.plot(plot_xg, real_surf.T, c="k", linewidth=.5)
            plot_bh(w_logs, 1)
            np.random.seed(seed)
            plt.ylim(0, 35)

            plot_to_do = True
            #return real_surf, prog_logs, dic_c

        x_hd = []  # constraints on grf
        z_hd = []  # constraints on grf
        ineq_x = []
        ineq_v = []
        ineq_max_x = []
        ineq_max_v = []
        others_interfaces = None

        # simulate surface
        if one_cm:
            cm = covmodels
        else:
            cm = covmodels[i]
        
        sigma = np.sqrt(cm.sill())

        # print(i)
        # if i == 96:
        #    return real_surf, prog_logs, dic_c

        check_dic_c(dic_c, prog_logs) 

        idx_const = np.sort(list(dic_c.keys()))
        
        erod_layer = False
        if np.random.random() < xi:  # if erode
            erod_layer = True

        if i in idx_const and not erod_layer:  # if a constraint must be respected

            # check to correct a bug that can appear some times 
            if len(dic_c[i]) > 1:

                l  = []
                for iwell in dic_c[i]:

                    test = check_bh_compa(i-1, iwell) 
                    x, depth, log = w_logs[iwell]
                    s_max = real_surf[i-1, bh_idxs[iwell]]  # maximum height of the surfaces previously simulated
                    height_log = log[prog_logs[iwell]][1]

                    if (s_max > height_log and prog_logs[iwell] != -len(log)) or test != "correct":
                        l.append("erode")
                    else:
                        l.append("onlap")

                    #print(l)
                if "erode" in l and "onlap" in l:  # this is a problem
                    l = np.array(l)
                    others_interfaces = list(np.array(dic_c[i])[l == "onlap"]).copy()
                    dic_c[i] = list(np.array(dic_c[i])[l == "erode"])  # keep interface that will be corrected by the erode surface
                else:
                    pass   

            # determine constraints
            to_remove = []  # list to interface to remove from dic_c
            for iwell in dic_c[i]:
                constraints = True  # flag to apply constraints on the conditional surfaces to prevent some issues
                test = check_bh_compa(i-1, iwell) 
                if test == "correct":  # no problem of connexion with others bh
                    x, depth, log = w_logs[iwell]
                    s_max = real_surf[i-1, bh_idxs[iwell]]  # maximum height of the surfaces previously simulated
                    height_log = log[prog_logs[iwell]][1]

                    if s_max > height_log and prog_logs[iwell] != -len(log):  # if surfaces are above contact --> erosion has to be set
                        erod_layer = True
                        x_hd.append(x+1e-5)
                        z_hd.append(height_log)
                    elif s_max < height_log and prog_logs[iwell] == -len(log):  # if top of the borehole
                        ineq_x = np.insert(ineq_x, 0, x+1e-5)
                        ineq_v = np.insert(ineq_v, 0, height_log)
                        
                    elif s_max < height_log:
                        x_hd.append(x+1e-5)
                        z_hd.append(height_log)

                    elif s_max > height_log and prog_logs[iwell] == -len(log):  # if surface are above contact but topest unit of borehole
                        constraints = False  #disable constraints as intervals already good


                    dic_ineq[x] = height_log  # update ineq
                    
                    if not erod_layer and constraints:
                        ### add more constraints to prevent that the surface cross cut other boreholes
                        s_bef = s1.copy()
                        h_sim = [s_bef[ibh] for ibh in bh_idxs]  # height of simulation at bh positions

                        ## select a bh where to simulate the surface
                        l_int = []
                        for bh_id in range(nwells):
                            bh = w_logs[bh_id]
                            pr = prog_logs[bh_id]

                            if -pr - 1 != len(bh[2]):  # not completed bh
                                el = bh[2][pr]  # (facies, altitude)
                                if h_sim[bh_id] < el[1]:  # if surfaces are below facies to constrained
                                    if pr == -1:
                                        l_int.append((el[0], el[1], max(bh[2][0][1]-bh[1], h_sim[bh_id])))
                                    else:
                                        l_int.append((el[0], el[1], max(bh[2][pr+1][1], h_sim[bh_id])))
                                else:
                                    l_int.append((None, h_sim[bh_id]))


                            else:  # bh completed
                                l_int.append(None)

                        choice = iwell
                        facies = l_int[choice][0]

                        # create ineq
                        for iw in range(nwells):
                            ibh = bh_idxs[iw]
                            t = l_int[iw]
                            if t is not None:
                                if t[0] is None:  # case where surfaces are above interface between two next facies of iw well
                                    ineq_max_x.append(xgc[ibh])
                                    ineq_max_v.append(t[1])
                                else:
                                    if t[0] != facies:
                                        ineq_max_x.append(xgc[ibh])
                                        ineq_max_v.append(max(s_bef[ibh], t[2]))

                    prog_logs[iwell] -= 1  # update prog logs

                    to_remove.append(iwell)

                else:  # we have to fix that
                    print(i, "gne")
                    x, depth, log = w_logs[iwell]
                    height_log = log[prog_logs[iwell]][1]
                    ineq_max_x.append(x)
                    
                    # un bout de scotch
                    good = True
                    for j in range(len(ineq_v)):
                        if ineq_x[j] == x:
                            if ineq_v[j] < test - 2*sz:
                                ineq_max_v.append(test - 2*sz)
                                good = False
                    if good:
                        ineq_max_v.append(test)

                    erod_layer = True

                    # now that we have fix the problem we must simulate correctly the next surface, adapt dic_c
                    dic_c[i].remove(iwell)
                    if not dic_c[i]:  # empty
                        del(dic_c[i])
                    # add a new entry
                    up = 1
                    flag = True
                    while flag:
                        if i+up not in dic_c.keys():
                            dic_c[i+up] = [iwell]
                            break
                        else:
                            if iwell not in dic_c[i+up]:
                                dic_c[i+up].append(iwell) 
                                break

                        up += 1
                        if i + up > N_surf:
                            raise ValueError ("Error")

                    if i+1 == N_surf:  # if no more surfaces availables, add a new one
                        add_line()
                    idx_const = np.sort(list(dic_c.keys()))
            
            # remove in dic_c
            for iwell in to_remove:
                dic_c[i].remove(iwell)
                if not dic_c[i]:  # empty
                    del(dic_c[i])  # remove value from dic c when corrected

            # min inequality constraints
            if erod_layer :
                temp = np.array([(k, v) for k,v in dic_ineq.items() if k not in x_hd])
                if temp.shape[0] > 0:
                    ineq_x = temp[:, 0]
                    ineq_v = temp[:, 1]

            # if i == 95:
            #     print(ineq_x,ineq_v, ineq_max_x, ineq_max_v)
            s1 = geone.geosclassicinterface.simulate1D(cm, nx, sx, ox, nreal=1, mean=means[i], verbose=verbose, 
                                                       searchRadiusRelative=1, nneighborMax=12,
                                                       x=x_hd, v=z_hd,
                                                       xIneqMin=ineq_x,vIneqMin=ineq_v,
                                                       xIneqMax=ineq_max_x, vIneqMax=ineq_max_v)["image"].val[0, 0, 0, :]
            s1_org = s1.copy()

        else:  # no constraints --> just simulate with lower bounds OR apply more rules
            # erode layers
            if erod_layer:
                # ineq inf
                ineq_x = list(dic_ineq.keys())
                ineq_v = list(dic_ineq.values())
                
                if np.random.random() < 0:  # A layer to correct a borehole
                    bhid_to_correct = np.random.choice(list(well_in_lines.keys()))
                    if len(w_logs[bhid_to_correct][2]) == -prog_logs[bhid_to_correct] - 1:  # bh already constrained
                        ineq_max_x = []
                        ineq_max_v = []
                        x_hd = []  # constraints on grf
                        z_hd = []  # constraints on grf 
                        
                    else:
                        res = check_bh_compa(i-1, bhid_to_correct)
                        
                        if res != "correct":
                            print(i, "gne2")
                            x, depth, log = w_logs[bhid_to_correct]
                            height_log = log[prog_logs[bhid_to_correct]][1]
                            ineq_max_x.append(x)

                            # un bout de scotch
                            good = True
                            for j in range(len(ineq_v)):
                                if ineq_x[j] == x:
                                    if ineq_v[j] < res - 2*sz:
                                        ineq_max_v.append(res - 2*sz)
                                        good = False
                            if good:
                                ineq_max_v.append(res)
                            
            elif i > 0 and np.random.random() > 0:  # onlap 

                s_bef = s1.copy()
                h_sim = [s_bef[ibh] for ibh in bh_idxs]  # height of simulation at bh positions

                ## select a bh where to simulate the surface
                l_int = []
                pro = []
                for bh_id in range(nwells):
                    bh = w_logs[bh_id]
                    pr = prog_logs[bh_id]

                    if -pr - 1 != len(bh[2]):  # not completed bh
                        el = bh[2][pr]  # (facies, altitude)
                        if h_sim[bh_id] < el[1]:  # if surfaces are below facies to constrained

                            if pr == -1:
                                l_int.append((el[0], el[1], bh[2][0][1]-bh[1]))
                            else:
                                l_int.append((el[0], el[1], bh[2][pr+1][1]))
                        else:
                            l_int.append((None, h_sim[bh_id]))

                    else:  # bh completed or too many surfaces simulated
                        l_int.append(None)

                for t in l_int:
                    if t is not None:
                        if t[0] is not None:
                            if means[i] + sigma > t[2] and means[i] - sigma < t[1]:

                                v = scipy.stats.norm(means[i], cm.sill()).cdf((t[2], t[1]))
                                proba = v[1] - v[0]
                                #proba = scipy.stats.norm(means[i], cm.sill()).pdf((np.linspace(t[2], t[1], 10))).sum()
                                pro.append(proba)
                            else:
                                pro.append(0)
                        else:
                            pro.append(0)
                    else:
                        pro.append(0)

                pro = np.array(pro)

                if sum((pro > 0))>1:
                    pro = pro/sum(pro)
                    choice = np.random.choice(range(nwells), p=pro)  # choose a bh

                    facies = l_int[choice][0]

                    # create ineq
                    for iw in range(nwells):
                        ibh = bh_idxs[iw]
                        t = l_int[iw]

                        if t is not None:
                            if t[0] is None:
                                ineq_max_x.append(xgc[ibh])
                                ineq_max_v.append(t[1] - 2*sz)
                            else:
                                if t[0] != facies:
                                    ineq_max_x.append(xgc[ibh])
                                    ineq_max_v.append(s_bef[ibh] - 2*sz)

                        if iw == choice:  # choosen one
                            ineq_max_x.append(xgc[ibh])
                            ineq_max_v.append(t[1])

                            if xgc[ibh] not in ineq_x:
                                ineq_x.append(xgc[ibh])
                                ineq_v.append(t[2])
            #if i == 26:
            #    print(x_hd, z_hd, ineq_x, ineq_v, ineq_max_x, ineq_max_v)
            s1 = geone.geosclassicinterface.simulate1D(cm, nx, sx, ox, nreal=1, mean=means[i], verbose=0, 
                                                       searchRadiusRelative=1, nneighborMax=12, x=x_hd, v=z_hd,
                                                       xIneqMin=ineq_x, vIneqMin=ineq_v,
                                                       xIneqMax=ineq_max_x, vIneqMax=ineq_max_v)["image"].val[0, 0, 0, :]

            # limit surface between top and bot
            s1[s1 > top] = top[s1 > top]
            s1[s1 < bot] = bot[s1 < bot]

        # loop over prexisting surfaces and apply erosion rules
        for o in range(i):
            s2 = real_surf[o]
            if erod_layer:
                s2[s2 > s1] = s1[s2 > s1]
            else:
                s1[s1 < s2] = s2[s1 < s2]

        if i > 0 and erod_layer:
            s1[s1 > real_surf[i-1]] = real_surf[i-1][s1 > real_surf[i-1]]  # erode no deposition
        
        if not erod_layer:
            real_surf[i] = s1  # store surfaces
            i += 1
        
        if others_interfaces is not None:
            if i not in dic_c.keys():
                dic_c[i] = []

            for qwer in others_interfaces:
                dic_c[i].append(qwer)

    # return real_surf    
    real_surf = np.concatenate((bot.reshape(1, nx), real_surf, top.reshape(1, nx)), axis=0)  # add top and bot

    real_surf[real_surf>z1]=z1
    real_surf[real_surf<oz]=oz
    # real_surf[real_surf>top]=top[real_surf>top]
    # real_surf[real_surf<bot]=bot[real_surf<bot]

    ## polygons
    list_p = []
    list_ids = []
    ID = 0
    for i in range(real_surf.shape[0]-1):

        s1=real_surf[i]
        s2=real_surf[i+1]

        mask_g = s2>s1

        mark = False
        ia = 0
        ib = 0
        g_1 = mask_g[0]
        g_2 = mask_g[-1]

        idx_g = np.where(mask_g[1:] != mask_g[:-1])[0]
        if len(idx_g) > 0:
            if g_1:
                start = 0
            else:
                start = 1

            for o in range(start, len(idx_g)+1, 2):
                if o == 0:
                    ia = 0
                    ib = idx_g[o]+2

                elif o < len(idx_g):
                    ia = idx_g[o-1]
                    ib = idx_g[o]+2
                else:
                    ia = idx_g[o-1]
                    ib = None

                coord_l1 = [(x,y) for x,y in zip(plot_xg[ia:ib], s1[ia:ib])]
                coord_l2 = [(x,y) for x,y in zip(plot_xg[ia:ib], s2[ia:ib])]

                if len(coord_l1) > 1 or len(coord_l2) > 1:
                    l1 = LineString(coord_l1)
                    l2 = LineString(coord_l2)
                    p = Polygon([*list(l2.coords), *list(l1.coords)[::-1]])
                    # p.ID = ID
                    list_ids.append(ID)
                    ID += 1
                    list_p.append(p)
                else:
                    pass

        elif g_1:  # polygon over the whole domain (touch left and right border)
            coord_l1 = [(x,y) for x,y in zip(plot_xg, s1)]
            coord_l2 = [(x,y) for x,y in zip(plot_xg, s2)]

            if len(coord_l1) > 1 or len(coord_l2) > 1:
                l1 = LineString(coord_l1)
                l2 = LineString(coord_l2)
                p = Polygon([*list(l2.coords), *list(l1.coords)[::-1]])
                # p.ID = ID
                list_ids.append(ID)
                ID += 1
                list_p.append(p)
    
    
    arr = rasterio.features.rasterize(shapes=zip(list_p, np.arange(len(list_p))), out_shape=(nz, nx),
                            transform=Affine(sx, 0.0, ox, 0.0, sz, oz), fill=-99)


    # list_ids = np.array([i.ID for i in list_p])
    list_ids = np.array(list_ids)
    
    # def create_graph():
    #     # if necessary create graph
    #     g = Graph(list_ids, False)

    #     for o in range(len(list_p)):
    #         p = list_p[o]
    #         for i in list_p[o:]:
    #             if p.intersects(i) and p != i:
    #                 res = (p.ID, i.ID, p.intersection(i).length)
    #                 if res[2] > 0:
    #                     g.add_edge(res[0], res[1], res[2])

    def create_graph():
        # if necessary create graph
        g = Graph(list_ids, False)

        for o in range(len(list_p)):
            p = list_p[o]
            for ip, i in enumerate(list_p[o:]):
                if p.intersects(i) and p != i:
                    res = (list_ids[o], list_ids[o+ip], p.intersection(i).length)
                    if res[2] > 0:
                        g.add_edge(res[0], res[1], res[2])
        return g


    ## simulation of the facies
    facies_ids = np.array(facies_ids)
    proba_cdf = np.array(proba_cdf)

    if proba_cdf.sum() != 1:
        proba_cdf /= proba_cdf.sum()

    if alpha < 1:
        g = create_graph()

    # set initial facies to polygon (0 mean unknown)
    dic_res = {}
    for i in list_ids:
        dic_res[i] = 0

    # dictionary of facies area
    dic_area = {}
    for i in facies_ids:
        dic_area[i] = 0

    area_sim = 0  # total area simulated
    total_area = np.sum([p.area for p in list_p])  # total area
    
    # set a hard data 
    for p_id, p in zip(list_ids, list_p):
        b=[]
        b_id = []
        for v in well_in_lines.values():
            for line in v:
                if p.intersects(line[0]):
                    b.append(line[0])
                    b_id.append(line[1])
            
        if len(b) > 0:
            if len(b) > 1:  # if a polygon cross cut more than one borehole with different facies (can happen)
                # fa_ids = np.array([i.id for i in b])
                fa_ids = np.array([b_id])
                if not (fa_ids == fa_ids[0]).all():
                    lengths = np.array([i.intersection(p).length for i in b])
                    idx = np.where(lengths == lengths.max())[0][0]   
                    line = b[idx]  # select line
                    id_line = b_id[idx]
                else:
                    line = b[0]
                    id_line = b_id[0]

            elif len(b) == 1:
                line = b[0]

            # dic_res[p.ID] = line.id
            dic_res[p_id] = id_line
            area_sim += p.area
            dic_area[id_line] += p.area

    ## algo with a graph
    ids_to_sim = [i for i in dic_res.keys() if dic_res[i]==0]  # cell id to simulate
    proba = np.array(proba_cdf).copy()

    while len(ids_to_sim) > 0:

        # update proba according to area simulated
        area_ratio = area_sim/total_area
        for i in range(len(proba)):
            p = proba_cdf[i]
            facies_id = facies_ids[i]
            new_p = (p - area_ratio*dic_area[facies_id]/area_sim)/(1- area_ratio)
            if new_p < 0:
                new_p = 0

            proba[i] = new_p


        proba = proba / proba.sum()
        id_sim = np.random.choice(ids_to_sim)  # select a volume to simulate

        if alpha < 1:
            neigs = g.list[id_sim]  # neighbours 

            if len(neigs) == 1:  # only 1 neighbour

                fac = dic_res[neigs[0][0]]
                if fac == 0:
                    p_neig = proba
                else:
                    p_neig = facies_ids==fac

            elif len(neigs) > 1:

                sum_w = 0
                p_neig = np.zeros(facies_ids.shape)
                for n in neigs:
                    cell_id = n[0]  # cell id
                    w = n[1]  # weight
                    sum_w += w
                    fac = dic_res[cell_id]  # value at the cell
                    if fac == 0:  # no value, take proba
                        p_neig += proba*w
                    else:
                        p_neig += (fac==facies_ids)*w
                p_neig /= sum_w       
        else:
            p_neig=np.zeros(facies_ids.shape)

        # restrict p interval between 0.001 and 0.999
        p_neig[p_neig<0.001] = 0.001
        p_neig[p_neig>0.999] = 0.999
        p_neig[(p_neig > 0.001) & (p_neig < 0.999)].sum()/(1 - p_neig[(p_neig <= 0.001) | (p_neig >= 0.999)].sum())

        proba[proba<0.001] = 0.001
        proba[proba>0.999] = 0.999
        proba[(proba > 0.001) & (proba < 0.999)].sum()/(1 - proba[(proba <= 0.001) | (proba >= 0.999)].sum())

        ## mix p with global p
        if (p_neig * proba).sum() == 0:
            p_combi = "linear"
        if p_combi == "linear":
            p = (1-alpha)*p_neig + alpha*proba
        elif p_combi == "log":
            p = p_neig**(1-alpha) * proba**alpha
        else:
            print("Invalid p_combi, use linear or log")
        p = p / p.sum()

        facies_choice = np.random.choice(facies_ids, p=p)
        dic_res[id_sim] = facies_choice
        poly_id = np.where(list_ids == id_sim)[0][0]
        poly = list_p[poly_id]
        # poly = [i for i in list_p if i.ID == id_sim]
        area_sim += poly.area
        dic_area[facies_choice] += poly.area

        ids_to_sim.remove(id_sim)  # remove id from list

    arr_res = apply_facies(arr, dic_res)

    return real_surf, arr_res, list_p


### 3D
def sim_uncond_3D(N, covmodels, means_surf, dim, spa, ori, covmodels_erod=None, bot=None, top=None, xi=0.5,
        facies_ids = [1, 2, 3, 4], proba_cdf = [0.25, 0.25, 0.25, 0.25], alpha=1, p_combi="log", seed=5, verbose=0):
    
    start = time.time()  # ini time
    
    np.random.seed(seed)

    # inputs
    global N_surf
    N_surf = N

    # grid
    nx, ny, nz = dim
    sx, sy, sz = spa
    ox, oy, oz = ori
    z1 = oz + nz*sz
    y1 = oy + ny*sy
    x1 = ox + nx*sx
    xgc = np.linspace(ox+sx/2, ox+sx*nx-sx/2, nx)
    ygc = np.linspace(oy+sy/2, oy+sy*ny-sy/2, ny)
    zg = np.linspace(oz, oz+nz*sz, nz)

    if bot is None:
        bot = oz*np.ones((ny, nx))
    if not isinstance(bot, np.ndarray):
        bot = np.ones((ny, nx))*bot

    if top is None:
        top = z1*np.ones((ny, nx))
    if not isinstance(top, np.ndarray):
        top = np.ones((ny, nx))*top

    one_cm = False
    if isinstance(covmodels, gcm.CovModel2D):
        one_cm = True

    mean_array = 0
    if len(means_surf.shape) == 1 and means_surf.shape[0] == N:
        mean_array = 1
    elif len(means_surf.shape) > 1 and means_surf.shape == (N, ny, nx):
        mean_array = 2  # sequence of 2D arrays
    else:
        raise ValueError ("Invalid shape {} for means_surf argument".format(means_surf.shape))
        
    # adjust surfaces
    global real_surf, means # global variables

    means= means_surf.copy()
    real_surf = np.ones([N_surf, ny, nx])
    
    t1 = time.time()
    if verbose > 0:
        print("setup phase : time elapsed {} s".format(np.round(t1 - start, 2)))
          
    ## simulations of the surfaces
    i = 0
    while i < N_surf:

        # simulate surface
        if one_cm:
            cm = covmodels
        else:
            cm = covmodels[i]

        sigma = np.sqrt(cm.sill())

        # erode layer ?
        erod_layer = False
        if i > 0:
            if np.random.random() < xi:  # if erode
                erod_layer = True

        if erod_layer and covmodels_erod is not None:
            if isinstance(covmodels_erod, gcm.CovModel2D):
                cm = covmodels_erod

        s1 = geone.geosclassicinterface.simulate2D(cm, (nx, ny), (sx, sy), (ox, oy), nreal=1, mean=means[i], verbose=verbose, 
                                                   searchRadiusRelative=1, nneighborMax=12)["image"].val[0, 0, :]

        # limit surface between top and bot
        s1[s1 > top] = top[s1 > top]
        s1[s1 < bot] = bot[s1 < bot]

        # loop over prexisting surfaces and apply erosion rules
        for o in range(i):
            s2 = real_surf[o]
            if erod_layer:
                s2[s2 > s1] = s1[s2 > s1]
            else:
                s1[s1 < s2] = s2[s1 < s2]

        if i > 0 and erod_layer:
            s1[s1 > real_surf[i-1]] = real_surf[i-1][s1 > real_surf[i-1]]  # erode no deposition

        if not erod_layer:
            real_surf[i] = s1  # store surfaces
            i += 1  

    real_surf = np.concatenate((bot.reshape(1, ny, nx), real_surf, top.reshape(1, ny, nx)), axis=0)  # add top and bot
    real_surf[real_surf>z1]=z1
    real_surf[real_surf<oz]=oz
#    real_surf[real_surf>top]=top[real_surf>top]
#    real_surf[real_surf<bot]=bot[real_surf<bot]
    
    t2 = time.time()
    if verbose > 0:
        print("Compute surfaces : time elapsed {} s".format(np.round(t2 - t1, 2)))
          
    # discretize the domain and build graph
    arr_res = np.zeros((nz, ny, nx), dtype=int)
    ID = 1
    situation = np.zeros((ny, nx), dtype=int)
    g = Graph_3D()

    for i in range(1, int(real_surf.shape[0])):
        s2 = real_surf[i-1]
        s1 = real_surf[i]
        thk = s2 - s1
        if thk.any():
            a = compute_domain(oz, z1, nx, ny, nz, sz, s1, s2).astype(bool)
            mask = (s1 - s2) > 0
            a2d_lab = label(mask)
            l = []
            for iz in range(nz):
                l.append(a2d_lab)
            a2d_lab_3D = np.stack(l)
            for iv in np.unique(a2d_lab):
                if iv > 0:
                    mask = ((a2d_lab_3D==iv) & a==True)
                    if mask.any():
                        arr_res[mask] = ID
                        # connexion of ID volume with previous volumes
                        mask_connections = ((a2d_lab==iv) & ((situation)!=0))  # mask where there is connections
                        connections = situation[mask_connections]  # extract ID values
                        vol_touched = np.unique(connections)  # volumes touched by the new volume
                        surf_contact = [((connections==i).sum()) for i in vol_touched]  # surface of contact for each touched volumes
                        # add each new edge to the graph
                        for ivol in range(len(vol_touched)): 
                            g.add_edge(ID, vol_touched[ivol], surf_contact[ivol])

                        # update situation
                        situation[a2d_lab == iv] = ID

                        ID += 1
     
    lst_ids = np.unique(arr_res)  # ids of volumes

    t3 = time.time()
    if verbose > 0:
        print("Discretization : time elapsed {} s".format(np.round(t3 - t2, 2)))
          
    ## simulation of the facies
    facies_ids = np.array(facies_ids)
    proba_cdf = np.array(proba_cdf)

    if proba_cdf.sum() != 1:
        proba_cdf /= proba_cdf.sum()

    # dictionary of facies area
    dic_volume = {}
    for i in facies_ids:
        dic_volume[i] = 0

    volume_sim = 0  # total area simulated
    total_volume = np.sum(arr_res>0)  # total volume

    # set initial facies to polygon (0 mean unknown)
    dic_res={}
    for i in lst_ids:
        if i not in dic_res.keys():
            dic_res[i] = 0

    ## algo with a graph
    ids_to_sim = [i for i in dic_res.keys() if dic_res[i]==0 and i != 0]  # cell id to simulate
    proba = np.array(proba_cdf).copy()

    while len(ids_to_sim) > 0:

        # update proba according to area simulated
        v_ratio = volume_sim/total_volume
        if v_ratio > 0:
            for i in range(len(proba)):
                p = proba_cdf[i]
                facies_id = facies_ids[i]
                new_p = (p - v_ratio*dic_volume[facies_id]/volume_sim)/(1-v_ratio)
                if new_p < 0:
                    new_p = 0

                proba[i] = new_p


            proba = proba / proba.sum()
        id_sim = np.random.choice(ids_to_sim)  # select a volume to simulate

        if alpha < 1:
            neigs = g.list[id_sim]  # neighbours 

            if len(neigs) == 1:  # only 1 neighbour

                fac = dic_res[neigs[0][0]]
                if fac == 0:
                    p_neig = proba
                else:
                    p_neig = facies_ids==fac

            elif len(neigs) > 1:

                sum_w = 0
                p_neig = np.zeros(facies_ids.shape)
                for n in neigs:
                    cell_id = n[0]  # cell id
                    w = n[1]  # weight
                    sum_w += w
                    fac = dic_res[cell_id]  # value at the cell
                    if fac == 0:  # no value, take proba
                        p_neig += proba*w
                    else:
                        p_neig += (fac==facies_ids)*w
                p_neig /= sum_w       
        else:
            p_neig=np.zeros(facies_ids.shape)

        # restrict p interval between 0.001 and 0.999
        p_neig[p_neig<0.001] = 0.001
        p_neig[p_neig>0.999] = 0.999
        p_neig[(p_neig > 0.001) & (p_neig < 0.999)].sum()/(1 - p_neig[(p_neig <= 0.001) | (p_neig >= 0.999)].sum())

        proba[proba<0.001] = 0.001
        proba[proba>0.999] = 0.999
        proba[(proba > 0.001) & (proba < 0.999)].sum()/(1 - proba[(proba <= 0.001) | (proba >= 0.999)].sum())

        ## mix p with global p
        if (p_neig * proba).sum() == 0:
            p_combi = "linear"
        if p_combi == "linear":
            p = (1-alpha)*p_neig + alpha*proba
        elif p_combi == "log":
            p = p_neig**(1-alpha) * proba**alpha
        else:
            print("Invalid p_combi, use linear or log")
        p = p / p.sum()

        facies_choice = np.random.choice(facies_ids, p=p)
        dic_res[id_sim] = facies_choice
        volume_id_sim = (arr_res==id_sim).sum()  # volume occupied by the volume id_sim
        volume_sim += volume_id_sim 
        dic_volume[facies_choice] += volume_id_sim 

        ids_to_sim.remove(id_sim)  # remove id from list

    arr_final = apply_facies(arr_res, dic_res)
          
    t4 = time.time()
    if verbose > 0:
        print("Assign facies : time elapsed {} s".format(np.round(t4 - t3, 2)))
    
    arr_final[arr_final == 0] = -99       
          
    return arr_final, real_surf


def sim_cond_3D(N, covmodels, means_surf, dim, spa, ori, w_logs, covmodels_erod=None, bot=None, top=None, xi=0.5,
        facies_ids = [1, 2, 3, 4], proba_cdf = [0.25, 0.25, 0.25, 0.25], alpha=1, p_combi="log", seed=5, verbose=0):
    
    start = time.time()  # ini time
    
    np.random.seed(seed)

    # inputs
    global N_surf
    N_surf = N

    # grid
    nx, ny, nz = dim
    sx, sy, sz = spa
    ox, oy, oz = ori
    z1 = oz + nz*sz
    y1 = oy + ny*sy
    x1 = ox + nx*sx
    xgc = np.linspace(ox+sx/2, ox+sx*nx-sx/2, nx)
    ygc = np.linspace(oy+sy/2, oy+sy*ny-sy/2, ny)
    zg = np.linspace(oz, oz+nz*sz, nz)

    if bot is None:
        bot = oz*np.ones((ny, nx))
    if not isinstance(bot, np.ndarray):
        bot = np.ones((ny, nx))*bot

    if top is None:
        top = z1*np.ones((ny, nx))
    if not isinstance(top, np.ndarray):
        top = np.ones((ny, nx))*top

    one_cm = False
    if isinstance(covmodels, gcm.CovModel2D):
        one_cm = True

    mean_array = 0
    if len(means_surf.shape) == 1 and means_surf.shape[0] == N:
        mean_array = 1
    elif len(means_surf.shape) > 1 and means_surf.shape == (N, ny, nx):
        mean_array = 2  # sequence of 2D arrays
    else:
        raise ValueError ("Invalid shape {} for means_surf argument".format(means_surf.shape))
        
    # adjust surfaces
    global erod_lst, real_surf, means # global variables

    means= means_surf.copy()
    erod_lst = np.random.uniform(size=N_surf) < xi  # determine which layers will be erode
    real_surf = np.ones([N_surf, ny, nx])

    nwells = len(w_logs)

    # boreholes indexes
    bh_idxs = [(np.round(((bh[0] - ox - sx/2)/sx)).astype(int), np.round(((bh[1] - oy - sy/2)/sy)).astype(int)) for bh in w_logs]

    for i in range(nwells):
        idx = bh_idxs[i]
        for i2 in range(nwells):
            idx2 = bh_idxs[i2]
            if idx == idx2 and i!=i2:  # two diff boreholes in the same cell
                bh1 = w_logs[i]
                bh2 = w_logs[i2]
                if bh1[2] > bh2[2]:
                    w_logs.remove(bh2)
                else:
                    w_logs.remove(bh1)

    # recompute boreholes indexes
    bh_idxs = [(np.round(((bh[1] - oy - sy/2)/sy)).astype(int), np.round(((bh[0] - ox - sx/2)/sx)).astype(int)) \
               for bh in w_logs]

    # choose when to respect HD
    # dictionary of constrained from boreholes
    dic_c = {}
    for o in range(nwells):
        l = [i[1] for i in w_logs[o][-1]]
        iy, ix = bh_idxs[o]

        if one_cm:
            for i in l:
                if mean_array == 1:
                    dis = scipy.stats.norm(i, np.sqrt(covmodels.sill()))
                    probas = dis.pdf(means)
                elif mean_array == 2:  # non stationarity in mean
                    dis = scipy.stats.norm(i, np.sqrt(covmodels.sill()))
                    probas = dis.pdf([m[iy, ix] for m in means])
                p = np.random.choice(range(N_surf), p=probas/probas.sum())
                
                # store value
                if p not in dic_c.keys():
                    dic_c[p] = []
                if o not in dic_c[p]:
                    dic_c[p].append(o)
                else:
                    flag = True
                    while flag:
                        p = np.random.choice(range(N_surf), p=probas/probas.sum())
                        if p not in dic_c.keys():
                            dic_c[p] = [o]
                            flag=False
                            
        else:  # case with different covmodels
            probas = []
            for i in range(N_surf):

                if mean_array == 1:
                    probas.append(scipy.stats.norm(means[i], covmodels[i].sill()).pdf(l))
                elif mean_array == 2:  # non stationarity in mean
                    probas.append(scipy.stats.norm(means[i][iy, ix], covmodels[i].sill()).pdf(l))
            
            probas = np.array(probas)
            for interf in range(len(l)):            
                p = np.random.choice(range(N_surf), p=probas[:, interf]/probas[:, interf].sum())
                   
                # store value
                if p not in dic_c.keys():
                    dic_c[p] = []
                if o not in dic_c[p]:
                    dic_c[p].append(o)
                else:
                    flag = True
                    while flag:
                        p = np.random.choice(range(N_surf), p=probas/probas.sum())
                        if p not in dic_c.keys():
                            dic_c[p] = [o]
                            flag=False

    def add_line():
        
        global N_surf, means, erod_lst, real_surf
        
        N_surf += 1
        
        if len(means.shape) == 1:
            means = np.concatenate((means, np.array(means[-1]).reshape(-1)))
        elif len(means.shape) == 2:
            means = np.concatenate((means, np.array(means[-1]).reshape(-1, nx, ny)))

        real_surf = np.concatenate((real_surf, np.ones((ny, nx)).reshape(-1, ny, nx)))

    def check_dic_c(dic_c, first=False):

        global N_surf, means, erod_lst, real_surf

        prog={}  # prog dic
        for k, v in sorted(dic_c.items()):
            for iv in v:
                if iv not in prog.keys():
                    prog[iv] = -1

        for k, v in sorted(dic_c.items()):
            v = np.copy(v)
            if len(v) > 1:
                prev = 0
                for o in range(len(v)):
                    ov = v[o]
                    if o == 0:
                        prev = w_logs[ov][3][prog[ov]][0]
                    else:
                        if prev != w_logs[ov][3][prog[ov]][0]:
                            dic_c[k].remove(ov)
                            if not dic_c[k]:
                                del(dic_c[k])
                            flag = True
                            inc = 1

                            while flag:
                                if k+np.abs(inc) >= N_surf:
                                    add_line()
                                    #raise ValueError("Error, difficulties to constrain, increase the number of lines (N)")
                                if k+inc not in dic_c.keys():
                                    dic_c[k+inc] = [ov]
                                    flag = False
                                elif first:
                                    if k-inc not in dic_c.keys():
                                        dic_c[k-inc] = [ov]
                                        flag = False
                                inc += 1


            for iv in v:
                if iv not in prog.keys():
                    prog[iv] = -1
                else:
                    prog[iv] += -1

    def extract(z, bh, vb=0):

        if z < bh[3][0][1] - bh[2]:
            if vb:
                print("extraction below borehole")
            return None
        elif z > bh[3][0][1]:
            if vb:
                print("extraction above borehole")
            return None
        else:
            pos_in_log = np.where(z < np.array([i[1] for i in bh[3]]))[0][-1]
            return bh[3][pos_in_log]

    def check_bh_compa(i2_max, bh_id, bh_idxs):


        """
        Check that a borehole is compatible with actual surfaces or not
        Return correct if no there is no problem. 
        If there is, returns an altitude indicating a minimal bound for the next grf to simulate
        """

        list_p = []

        fa_id_to_const = w_logs[bh_id][3][prog_logs[bh_id]][0]  # facies id to constrained

        # loop over surfaces until below facies to constrained
        # this allows to keep only surfaces that are above what have already been constrained

        # get index of lowest surfaces below facies to constrained
        i2_min = i2_max

        if prog_logs[bh_id]+1 != 0:
            height_interface = w_logs[bh_id][3][prog_logs[bh_id]+1][1]
        else:
            height_interface = w_logs[bh_id][3][0][1] - w_logs[bh_id][2]

        while np.abs(real_surf[i2_min][bh_idxs[bh_id]] - height_interface) > 0.1 and real_surf[i2_min][bh_idxs[bh_id]] > height_interface:  
            i2_min -= 1

            if i2_min == 0:
                break

        for i2 in range(i2_min, i2_max):  # loop from lowest surfaces to highest

            s1 = real_surf[i2]
            s2 = real_surf[i2+1]

            l1 = s1[bh_idxs[bh_id]]
            l2 = s2[bh_idxs[bh_id]]

            mask = (s2 - s1) > 0
            if mask.any() and mask[bh_idxs[bh_id]] > 0:
                a2d_lab = label(mask)  # volumes
                for io in range(len(w_logs)):  # loop over boreholes
                    other_bh_id = bh_idxs[io]  # get idx position of other borehole
                    if other_bh_id != bh_idxs[bh_id]:
                        if a2d_lab[other_bh_id] == a2d_lab[bh_idxs[bh_id]]:  # same volume
                            for idx_k in range(-1, prog_logs[bh_id], -1):  # check only already constrained intervals

                                facies_intersected = extract(s1[other_bh_id], w_logs[io])
                                if facies_intersected is not None:
                                    facies_intersected = facies_intersected[0]
                                    if facies_intersected != fa_id_to_const:  # not same facies inside the same volume

                                        #  return elevation up which there is a problem
                                        return real_surf[i2][bh_idxs[bh_id]]  # incorrect connection

        return "correct"
    
    
    # check dic_c
    check_dic_c(dic_c, True)
    
    # some useful arrays
    prog_logs = -1*np.ones([nwells], dtype=int)  # progression of the constrained on the logs
    idx_const = np.sort(list(dic_c.keys()))  # idx of constrained

    dic_ineq = {}
    
    t1 = time.time()
    if verbose > 0:
        print("setup phase : time elapsed {} s".format(np.round(t1 - start, 2)))
          
    ## simulations of the surfaces
    i = 0
    while i < N_surf:

        x_hd = []  
        z_hd = []  # hard data constraints
        ineq_x = []
        ineq_v = [] # min ineq constraints
        ineq_max_x = []
        ineq_max_v = [] # max ineq constraints

        # simulate surface
        if one_cm:
            cm = covmodels
        else:
            cm = covmodels[i]

        sigma = np.sqrt(cm.sill())

        check_dic_c(dic_c)  
        idx_const = np.sort(list(dic_c.keys()))

        # erode layer ?
        erod_layer = False
        if i > 0:
            if np.random.random() < xi:  # if erode
                erod_layer = True

        if erod_layer and covmodels_erod is not None:
            if isinstance(covmodels_erod, gcm.CovModel2D):
                cm = covmodels_erod
    
        if i in idx_const and not erod_layer:  # if a constraint must be respected
            # determine constraints

            # if i == 25:
            #     return real_surf, w_logs, dic_c, bh_idxs, prog_logs

            for iwell in dic_c[i]:

                test = check_bh_compa(i-1, iwell, bh_idxs) 
                if test == "correct":  # no problem of connexion with others bh
                    x, y, depth, log = w_logs[iwell]
                    s_max = real_surf[i-1, bh_idxs[iwell][0], bh_idxs[iwell][1]]  # maximum height of the surfaces previously simulated
                    height_log = log[prog_logs[iwell]][1]
                    if s_max > height_log and prog_logs[iwell] != -len(log):  # if surfaces are above contact --> erosion has to be set
                        erod_layer = True
                        x_hd.append((x+1e-5, y+1e-5))
                        z_hd.append(height_log)
                    elif s_max < height_log and prog_logs[iwell] == -len(log):  # if top of the borehole
                        ineq_x = np.insert(ineq_x, 0, (x+1e-5, y+1e-5))
                        ineq_v = np.insert(ineq_v, 0, height_log)

                    elif s_max < height_log:
                        x_hd.append((x+1e-5, y+1e-5))
                        z_hd.append(height_log)

                    dic_ineq[(x, y)] = height_log  # update ineq min

                    if not erod_layer:
                        ### add more constraints to prevent that the surface cross cut other boreholes
                        s_bef = s1.copy()
                        h_sim = [s_bef[ibh] for ibh in bh_idxs]  # height of simulation at bh positions

                        ## select a bh where to simulate the surface
                        l_int = []
                        for bh_id in range(nwells):
                            bh = w_logs[bh_id]
                            pr = prog_logs[bh_id]

                            if -pr - 1 != len(bh[3]):  # not completed bh
                                el = bh[3][pr]  # (facies, altitude)
                                if h_sim[bh_id] < el[1]:  # if surfaces are below facies to constrained
                                    if pr == -1:
                                        l_int.append((el[0], el[1], max(bh[3][0][1]-bh[2], h_sim[bh_id])))
                                    else:
                                        l_int.append((el[0], el[1], max(bh[3][pr+1][1], h_sim[bh_id])))
                                else:
                                    l_int.append((None, h_sim[bh_id]))

                            else:  # bh completed
                                l_int.append(None)

                        choice = iwell
                        facies = l_int[choice][0]

                        # create ineq
                        for iw in range(nwells):
                            ibh = bh_idxs[iw]
                            t = l_int[iw]
                            if t is not None:
                                if t[0] is None:  # case where surfaces are above interface between two next facies of iw well
                                    ineq_max_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                                    ineq_max_v.append(t[1])
                                else:
                                    if t[0] != facies:
                                        ineq_max_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                                        ineq_max_v.append(max(s_bef[ibh], t[2]))

                    prog_logs[iwell] -= 1  # update prog logs
                    dic_c[i].remove(iwell)  # remove value from dic c when corrected
                    if not dic_c[i]:  # empty
                        del(dic_c[i])

                else:  # we have to fix that
                    print("gne")
                    x, y, depth, log = w_logs[iwell]
                    height_log = log[prog_logs[iwell]][1]
                    ineq_max_x.append((x, y))

                    # un bout de scotch
                    good = True
                    for j in range(len(ineq_v)):
                        if ineq_x[j][0] == x and ineq_x[j][1] == y:
                            if ineq_v[j] < res - 2*sz:
                                ineq_max_v.append(res - 2*sz)
                                good = False
                    if good:
                        ineq_max_v.append(res)

                    erod_layer = True

                    # now that we have fix the problem we must simulate correctly the next surface, adapt dic_c
                    dic_c[i].remove(iwell)
                    if not dic_c[i]:  # empty
                        del(dic_c[i])
                    # add a new entry
                    up = 1
                    flag = True
                    while flag:
                        if i+up not in dic_c.keys():
                            dic_c[i+up] = [iwell]
                            break
                        else:
                            if iwell not in dic_c[i+up]:
                                dic_c[i+up].append(iwell) 
                                break

                        up += 1
                        if i + up > N_surf:
                            raise ValueError ("Error")

                    if i+1 == N_surf:  # if no more surfaces availables, add a new one
                        add_line()
                        print(2)
                    idx_const = np.sort(list(dic_c.keys()))

            # min inequality constraints
            if erod_layer :
                ineq_x = np.array([k for k,v in dic_ineq.items() if k not in x_hd])
                ineq_v = np.array([v for k,v in dic_ineq.items() if k not in x_hd])
#                 if temp.shape[0] > 0:
#                     ineq_x = temp[:, 0]
#                     ineq_v = temp[:, 1]

            s1 = geone.geosclassicinterface.simulate2D(cm, (nx, ny), (sx, sy), (ox, oy), nreal=1, mean=means[i], verbose=verbose, 
                                                       searchRadiusRelative=1, nneighborMax=12,
                                                       x=x_hd, v=z_hd,
                                                       xIneqMin=list(ineq_x), vIneqMin=list(ineq_v),
                                                       xIneqMax=ineq_max_x, vIneqMax=ineq_max_v)["image"].val[0, 0, :]


        else:  # no constraints --> just simulate with lower bounds OR apply more rules

            # erode layers
            if erod_layer:
                # ineq inf
                ineq_x = list(dic_ineq.keys())
                ineq_v = list(dic_ineq.values())

                if np.random.random() > xi:  # A layer to correct a borehole

                    bhid_to_correct = np.random.choice(range(nwells))
                    if len(w_logs[bhid_to_correct][3]) == -prog_logs[bhid_to_correct] - 1:  # bh already constrained
                        ineq_max_x = []
                        ineq_max_v = []
                        x_hd = []  # constraints on grf
                        z_hd = []  # constraints on grf 

                    else:
                        res = check_bh_compa(i-1, bhid_to_correct, bh_idxs)

                        if res != "correct":
                            print("gne2")
                            x, y, depth, log = w_logs[bhid_to_correct]
                            height_log = log[prog_logs[bhid_to_correct]][1]
                            ineq_max_x.append((x, y))

                            # un bout de scotch
                            good = True
                            for j in range(len(ineq_v)):
                                if ineq_x[j][0] == x and ineq_x[j][1] == y:
                                    if ineq_v[j] < res - 2*sz:
                                        ineq_max_v.append(res - 2*sz)
                                        good = False
                            if good:
                                ineq_max_v.append(res)

            elif i > 0:  # onlap                
                s_bef = s1.copy()
                h_sim = [s_bef[ibh] for ibh in bh_idxs]  # height of simulation at bh positions

                ## select a bh where to simulate the surface
                l_int = []  # list to store intervals of facies, the next to simulate
                pro = []
                for bh_id in range(nwells):
                    bh = w_logs[bh_id]
                    pr = prog_logs[bh_id]

                    if -pr - 1 != len(bh[3]):  # not completed bh
                        el = bh[3][pr]  # (facies, altitude)
                        if h_sim[bh_id] < el[1]:  # if surfaces are below facies to constrained

                            if pr == -1:
                                l_int.append((el[0], el[1], bh[3][0][1]-bh[2]))
                            else:
                                l_int.append((el[0], el[1], bh[3][pr+1][1]))
                        else:
                            l_int.append((None, h_sim[bh_id]))

                    else:  # bh completed or too many surfaces simulated
                        l_int.append(None)

                for t in l_int:
                    if t is not None:
                        if t[0] is not None:
                            if means[i] + sigma > t[2] and means[i] - sigma < t[1]:
                                #pro.append(1)
                                v = scipy.stats.norm(means[i], cm.sill()).cdf((t[2], t[1]))
                                proba = v[1] - v[0]
                                pro.append(proba)
                            else:
                                pro.append(0)
                        else:
                            pro.append(0)
                    else:
                        pro.append(0)

                pro = np.array(pro)
                if sum((pro > 0))>1:
                    pro = pro/sum(pro)
                    choice = np.random.choice(range(nwells), p=pro)  # choose a bh

                    facies = l_int[choice][0]

                    # create ineq
                    for iw in range(nwells):
                        ibh = bh_idxs[iw]
                        t = l_int[iw]

                        if t is not None:
                            if t[0] is None:
                                ineq_max_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                                ineq_max_v.append(t[1])
                            else:
                                if t[0] != facies:
                                    ineq_max_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                                    ineq_max_v.append(s_bef[ibh])

                        elif iw == choice:  # choosen one
                            ineq_max_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                            ineq_max_v.append(t[1])

                            if (xgc[ibh[1]], ygc[ibh[0]]) not in ineq_x:
                                ineq_x.append((xgc[ibh[1]], ygc[ibh[0]]))
                                ineq_v.append(t[2])

            s1 = geone.geosclassicinterface.simulate2D(cm, (nx, ny), (sx, sy), (ox, oy), nreal=1, mean=means[i], verbose=verbose, 
                                                       searchRadiusRelative=1, nneighborMax=12,
                                                       x=x_hd, v=z_hd,
                                                       xIneqMin=ineq_x, vIneqMin=ineq_v,
                                                       xIneqMax=ineq_max_x, vIneqMax=ineq_max_v)["image"].val[0, 0, :]

            # limit surface between top and bot
            s1[s1 > top] = top[s1 > top]
            s1[s1 < bot] = bot[s1 < bot]

        # loop over prexisting surfaces and apply erosion rules
        for o in range(i):
            s2 = real_surf[o]
            if erod_layer:
                s2[s2 > s1] = s1[s2 > s1]
            else:
                s1[s1 < s2] = s2[s1 < s2]

        if i > 0 and erod_layer:
            s1[s1 > real_surf[i-1]] = real_surf[i-1][s1 > real_surf[i-1]]  # erode no deposition

        if not erod_layer:
            real_surf[i] = s1  # store surfaces
            i += 1  


    real_surf = np.concatenate((bot.reshape(1, ny, nx), real_surf, top.reshape(1, ny, nx)), axis=0)  # add top and bot
    real_surf[real_surf>z1]=z1
    real_surf[real_surf<oz]=oz
    # real_surf[real_surf>top]=top[real_surf>top]
    # real_surf[real_surf<bot]=bot[real_surf<bot]
    
    t2 = time.time()
    if verbose > 0:
        print("Compute surfaces : time elapsed {} s".format(np.round(t2 - t1, 2)))
          
    # discretize the domain and build graph
    arr_res = np.zeros((nz, ny, nx), dtype=int)
    ID = 1
    situation = np.zeros((ny, nx), dtype=int)
    g = Graph_3D()

    for i in range(1, int(real_surf.shape[0])):
        s2 = real_surf[i-1]
        s1 = real_surf[i]
        thk = s2 - s1
        if thk.any():
            a = compute_domain(oz, z1, nx, ny, nz, sz, s1, s2).astype(bool)
            mask = (s1 - s2) > 0
            a2d_lab = label(mask)
            l = []
            for iz in range(nz):
                l.append(a2d_lab)
            a2d_lab_3D = np.stack(l)
            for iv in np.unique(a2d_lab):
                if iv > 0:
                    mask = ((a2d_lab_3D==iv) & a==True)
                    if mask.any():
                        arr_res[mask] = ID
                        # connexion of ID volume with previous volumes
                        mask_connections = ((a2d_lab==iv) & ((situation)!=0))  # mask where there is connections
                        connections = situation[mask_connections]  # extract ID values
                        vol_touched = np.unique(connections)  # volumes touched by the new volume
                        surf_contact = [((connections==i).sum()) for i in vol_touched]  # surface of contact for each touched volumes
                        # add each new edge to the graph
                        for ivol in range(len(vol_touched)): 
                            g.add_edge(ID, vol_touched[ivol], surf_contact[ivol])

                        # update situation
                        situation[a2d_lab == iv] = ID

                        ID += 1
     
    lst_ids = np.unique(arr_res)  # ids of volumes

          
    t3 = time.time()
    if verbose > 0:
        print("Discretization : time elapsed {} s".format(np.round(t3 - t2, 2)))
          
    ## simulation of the facies
    facies_ids = np.array(facies_ids)
    proba_cdf = np.array(proba_cdf)

    if proba_cdf.sum() != 1:
        proba_cdf /= proba_cdf.sum()

    # dictionary of facies area
    dic_volume = {}
    for i in facies_ids:
        dic_volume[i] = 0

    volume_sim = 0  # total area simulated
    total_volume = np.sum(arr_res>0)  # total volume

    # set initial facies to polygon (0 mean unknown)
    dic_res={}
    for i in lst_ids:
        if i not in dic_res.keys():
            dic_res[i] = 0

    # set a hard data
    for iwell in range(nwells):

        well = w_logs[iwell]
        bh_idx = bh_idxs[iwell]

        for iz in range(nz):
            z = zg[iz] + sz/2
            idomain = arr_res[iz, bh_idx[0], bh_idx[1]]
            ext = extract(z, well)  # get facies at altitude z
            if ext is not None:
                fa, alt = ext
                if dic_res[idomain] == 0:  # attribute facies to volume encountered if not already defined
                    dic_res[idomain] = fa
                    volume = (arr_res==idomain).sum()
                    dic_volume[fa] += volume
                    volume_sim += volume

    ## algo with a graph
    ids_to_sim = [i for i in dic_res.keys() if dic_res[i]==0 and i != 0]  # cell id to simulate
    proba = np.array(proba_cdf).copy()

    while len(ids_to_sim) > 0:

        # update proba according to area simulated
        v_ratio = volume_sim/total_volume
        if v_ratio > 0:
            for i in range(len(proba)):
                p = proba_cdf[i]
                facies_id = facies_ids[i]
                new_p = (p - v_ratio*dic_volume[facies_id]/volume_sim)/(1-v_ratio)
                if new_p < 0:
                    new_p = 0

                proba[i] = new_p


            proba = proba / proba.sum()
        id_sim = np.random.choice(ids_to_sim)  # select a volume to simulate

        if alpha < 1:
            neigs = g.list[id_sim]  # neighbours 

            if len(neigs) == 1:  # only 1 neighbour

                fac = dic_res[neigs[0][0]]
                if fac == 0:
                    p_neig = proba
                else:
                    p_neig = facies_ids==fac

            elif len(neigs) > 1:

                sum_w = 0
                p_neig = np.zeros(facies_ids.shape)
                for n in neigs:
                    cell_id = n[0]  # cell id
                    w = n[1]  # weight
                    sum_w += w
                    fac = dic_res[cell_id]  # value at the cell
                    if fac == 0:  # no value, take proba
                        p_neig += proba*w
                    else:
                        p_neig += (fac==facies_ids)*w
                p_neig /= sum_w       
        else:
            p_neig=np.zeros(facies_ids.shape)

        # restrict p interval between 0.001 and 0.999
        p_neig[p_neig<0.001] = 0.001
        p_neig[p_neig>0.999] = 0.999
        p_neig[(p_neig > 0.001) & (p_neig < 0.999)].sum()/(1 - p_neig[(p_neig <= 0.001) | (p_neig >= 0.999)].sum())

        proba[proba<0.001] = 0.001
        proba[proba>0.999] = 0.999
        proba[(proba > 0.001) & (proba < 0.999)].sum()/(1 - proba[(proba <= 0.001) | (proba >= 0.999)].sum())

        ## mix p with global p
        if (p_neig * proba).sum() == 0:
            p_combi = "linear"
        if p_combi == "linear":
            p = (1-alpha)*p_neig + alpha*proba
        elif p_combi == "log":
            p = p_neig**(1-alpha) * proba**alpha
        else:
            print("Invalid p_combi, use linear or log")
        p = p / p.sum()

        facies_choice = np.random.choice(facies_ids, p=p)
        dic_res[id_sim] = facies_choice
        volume_id_sim = (arr_res==id_sim).sum()  # volume occupied by the volume id_sim
        volume_sim += volume_id_sim 
        dic_volume[facies_choice] += volume_id_sim 

        ids_to_sim.remove(id_sim)  # remove id from list

    arr_final = apply_facies(arr_res, dic_res)
          
    t4 = time.time()
    if verbose > 0:
        print("Assign facies : time elapsed {} s".format(np.round(t4 - t3, 2)))
          
    arr_final[arr_final == 0] = -99 
          
    return arr_final, real_surf