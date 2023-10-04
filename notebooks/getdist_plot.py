from getdist import loadMCSamples
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from getdist.mcsamples import MCSamplesFromCobaya
from getdist import loadMCSamples
from getdist.mcsamples import MCSamples
import getdist.plots as gdplt
import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 40})
# rc('text', usetex=True)



path_class = '/work/bk935060/Looti/cobaya/class_TTTEEE_2/'
path1 = '/work/bk935060/Looti/cobaya/looti_TTTEEE_8sigma_100/chains/'
# path2 = '/work/bk935060/Looti/cobaya/looti_TTTEEE_5sigma_400/chains/'
# path3 = '/work/bk935060/Looti/cobaya/looti_TTTEEE_5sigma_300/chains/'

# n_train = []

# samples = []
# legends = []
# for nn in n_train:
#     path = '/work/bk935060/Looti/cobaya/looti_TTTEEE_5sigma_%i/chains/' %(nn)
#     sample = loadMCSamples(path, settings={'ignore_rows':0.1})
#     samples.append(sample)
#     legends.append('looti %i' %(nn))

# samples.append(loadMCSamples(path_class, settings={'ignore_rows':0.1}))
# legends.append("class")

samples_class = loadMCSamples(path_class, settings={'ignore_rows':0.1})
samples1 = loadMCSamples(path1, settings={'ignore_rows':0.1})

# samples3 = loadMCSamples(path3, settings={'ignore_rows':0.1})


# Customized triangle plots
def make_triangle_plot(
    samples,
    parameters,
    path='./', 
    name='triangle',
    filled = True):

    # Create path
    if os.path.isdir(path)==False:
        os.mkdir(path)
    
    # Create plotter
    plot = gdplt.get_subplot_plotter()

    plot.triangle_plot(samples, parameters, filled = filled)

    plot.export(name+'.png',adir=path)





# Compare two triangle plots
def compare_triangles(
    samples,
    parameters,
    path='./',
    name='triangle_compare',
    legends=[],
    filled=True
    ):

    # Create path
    if os.path.isdir(path)==False:
        os.mkdir(path)

    # contour_args = {'alpha': 0.3}
    
    # Create plotter
    plot = gdplt.get_subplot_plotter()

    plot.settings.num_plot_contours = 2

    plot.settings.axes_fontsize = 20
    plot.settings.lab_fontsize = 20
    plot.settings.legend_fontsize = 30

    plot.triangle_plot(samples, parameters, legend_labels=legends, filled = filled) # , contour_args=contour_args

    plot.export(name+'.png',adir=path)


parameters = ['H0','A_s','n_s','omega_b','omega_cdm','tau_reio']
# make_triangle_plot(samples=samples1, path=path1, name='triangle', parameters=parameters)
compare_triangles(samples=[samples_class, samples1], parameters=parameters, path='/work/bk935060/Looti/looti/plots/triangle_plots/', name='Looti_class_8sigma_100_trainprior', legends=['class', 'looti 100'], filled=True)