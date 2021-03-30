from __future__ import division,unicode_literals
import scipy.constants as const

import sys, os
import numpy as np 
import pandas as pd
import pylab
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

import scipy
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy import signal

from BaselineRemoval import BaselineRemoval

class rR_Analyzer:
    def __init__(self, path):
        self._path = path
        self._foldername = None
        self._filenames = None
        self._x_raw = None
        self._y_raw = None
        self._x_spline = None
        self._y_spline = None
        self._solvent_index = None
        self._x_frame_data = None
        self._y_frame_data = None
        self._frame_names = None

        self._normalized = False
        self._difference = False
        self._bgr = False

    def copy(self):
        '''Copy an instance

        Returns:
            instance
        '''

        tmp = rR_Analyzer(self._path)
        tmp._filenames = self._filenames.copy()
        tmp._foldername = self._foldername
        if tmp._filenames:
            tmp._x_raw = self._x_raw.copy()
            tmp._y_raw = self._y_raw.copy()
            tmp._x_spline = self._x_spline.copy()
            tmp._y_spline = self._y_spline.copy()
            tmp._solvent_index = self._solvent_index

        tmp._normalized = self._normalized
        tmp._difference = self._difference
        tmp._bgr = self._bgr
        return tmp

    def read_files(self, foldername='data', end='.txt', xmin=700, xmax=2500, steps=None):
        ''' Reads raw data (single column exported WinSpec ascii-files)

        Args:
            foldername: name of the folder including the data (string, default: 'data')
            end: file-ending (string, default: '.txt')
            xmin: minimum rel. waveneumber value (float/integer, default: 700)
            xmax: maximum rel. wavenumber value (float/integer, default: 2500)
            steps: specify number of points in the ranging from xmin to xmax (default:
            None, calculates the number of steps from xmin and xmax)

        Returns:
            The splined raw_data (dictionary)
        '''

        if steps == None:
            steps = (xmax - xmin)*4

        print('Number of x-points = '+str(steps))

        file_path = os.path.join(self._path, foldername)
        filenames = sorted([n for n in os.listdir(file_path) if n.endswith(end)])
        self._filenames = filenames

        y_dict = {}
        x_dict = {}
        y_spline_dict = {}
        for nr in range(len(filenames)):
            df = pd.read_csv(os.path.join(file_path,str(filenames[nr])), sep="\t", header=0)
            data_array = np.array(df)
            x_data = data_array[0:,0]
            y_data = data_array[0:,-1]
            x_dict['x_'+self._get_key(nr)] = x_data
            y_dict[self._get_key(nr)] = y_data
            y_spline = interp1d(x_data, y_data)
            x_spline = np.linspace(xmin, xmax, num=steps, endpoint=True)
            y_spline_dict[self._get_key(nr)] = y_spline(x_spline)

        self._x_raw = x_dict
        self._y_raw = y_dict
        self._x_spline = x_spline
        self._y_spline = y_spline_dict
        self._foldername = foldername

    def _get_key(self, nr, start_cut=3, end_cut=4):
        '''Get dictionary keys from raw-data filenames

        Args:
            nr: Integer in range of filenames in the working directory
            start_cut: number of elements ignored at the filename beginning
            end_index: number of elements ignored at the end of the filename
                       (default: 4, e.g. ignores '.txt'-ending)
        Returns:
            filename (string)
        '''
        return str(self._filenames[nr][start_cut:-end_cut])

    def BGR(self, data, method, polynomial_degree, lambda_, porder, itermax):
        '''Background correction of the data

        Args:
            data: data to background correct
            method:
                ZhangFit [https://doi.org/10.1039/B922045C]:
                    porder: signal-to-noise ratio (typically 3)
                    lambda_ : smoothing-factor for the background (typically in the order of 500)
                    itermax: number of iterations
                ModPoly [https://doi.org/10.1366/000370203322554518]
                ModPoly [https://doi.org/10.1366/000370207782597003]
                    polynomial_degree: degree of the polynomial for background correction (typically 3)

        Returns:
            Background corrected data

        Raises:
            Method not found. Possible options are: ZhangFit, ModPoly and IModPoly. Check spelling.
        '''

        baseObj = BaselineRemoval(data)
        if method == 'ZhangFit':
            BGR_output = baseObj.ZhangFit(lambda_=lambda_, porder=porder, itermax=itermax)
        elif method == 'ModPoly':
            BGR_output = baseObj.ModPoly(polynomial_degree)
        elif method == 'IModPoly':
            BGR_output = baseObj.IModPoly(polynomial_degree)
        else:
            raise Exception('Method not found. Possible options are: ZhangFit, ModPoly and IModPoly. Check spelling.')

        return BGR_output


    def BGR_corr(self, method='ZhangFit', polynomial_degree=2, lambda_=100, porder=1, itermax=15,
                 figurename='BGR', savefig=False, xmin_p=700, xmax_p=2500, steps=None):
        '''Call background correction function and plot background corrected data and shows plots of the raw-
        and background corrected data

        Args:
            method:
                ZhangFit [https://doi.org/10.1039/B922045C]:
                    porder: signal-to-noise ratio (typically 3)
                    lambda_ : smoothing-factor for the background (typically in the order of 500)
                    itermax: number of iterations
                ModPoly [https://doi.org/10.1366/000370203322554518]
                ModPoly [https://doi.org/10.1366/000370207782597003]
                    polynomial_degree: degree of the polynomial for background correction (typically 3)
            xmin_p: lower rel. wavenumber limit for plot
            xmax_p: upper rel. wavenumber limit for plot
            steps: number of points (linear interpolation) between xmin_p and xmax_p
            savefig: if 'True' save figure to output folder (default format: '.pdf')
            figurename: name of the figure saved to the output folder (string, default: 'BGR')

        Returns:
            Instance of the class (background corrected data)
        '''

        BGR_data_dict = {}

        for nr in range(len(self._filenames)):
            datafile = self._y_spline[self._get_key(nr)]
            BGR_data_dict[self._get_key(nr)] = self.BGR(data=datafile, method=method,
                                                        polynomial_degree=polynomial_degree,
                                                        lambda_=lambda_, porder=porder, itermax=itermax)

        tmp = self.copy()
        tmp._bgr = True
        tmp._normalized = False
        tmp.plot_data(data_to_plot=BGR_data_dict, savefig=savefig, bgr_plot=True, figurename=figurename,
                      title=tmp._get_message_string(), xmin_p=xmin_p, xmax_p=xmax_p, steps=None)
        tmp._y_spline = BGR_data_dict

        return tmp

    def roi(self, list_data, roi_list):
        '''Reduce a list (list_data) into a region-of-interest (roi) specified by a range (roi_list)

        Args:
            list_data: List of floats
            roi_list: [lower_limit, upper_limit] with lower_limit and upper_limit in list_data

        Returns:
            cutindex: List of indizes of the lower and upper value specified in roi_list
        '''

        cutindex=[]
        for element in roi_list:
            index_ober = [n for n, i in enumerate(list_data) if i >= element][0]
            index_unter = [n for n, i in enumerate(list_data) if i < element][0]
            if list_data[index_ober] - element >= element - list_data[index_unter]:
                cutindex.append(index_unter)
            else:
                cutindex.append(index_ober)

        return cutindex

    def norm_peak(self, peak=1373, bnd=5, xmin_p=800, xmax_p=2000, steps=None, figurename='norm',
                  savefig=False):

        '''Normalize data on the maximum in the given range peak plus/minus bnd and shows a plot of the
        normalized data in the given plot-range [xmin_p, xmax_p]

        Args:
            peak: estimation of the rel. wavenumber of the peak maximum for normalization (integer/float,
                  default: 1373 (acetonitrile))
            bnd: boundaries added and subtracted from peak for finding a local maximum (interger, default: 5)
            xmin_p: lower rel. wavenumber limit for plot
            xmax_p: upper rel. wavenumber limit for plot
            steps: number of points (linear interpolation) between xmin_p and xmax_p
            savefig: if 'True' save figure to output folder (default format: '.pdf')
            figurename: name of the figure saved to the output folder (string, default: 'norm')

        Returns:
            Instance of the class (normalized spectra)
        '''

        if steps == None:
            steps = (xmax_p - xmin_p)*2

        roi_list = [peak-bnd, peak+bnd]
        roi_bnd = self.roi(list_data=self._x_spline, roi_list=roi_list)

        norm_dict = {}
        for nr in range(len(self._filenames)):
            data = self._y_spline[self._get_key(nr)]

            maximum = np.max(data[roi_bnd[0]:roi_bnd[1]+1])
            max_index = [i for i,j in enumerate(data[roi_bnd[0]:roi_bnd[1]+1]) if j==maximum][0]+roi_bnd[0]
            norm = data / data[max_index]
            norm_dict[self._get_key(nr)] = norm

        tmp = self.copy()
        tmp._normalized = True
        tmp.plot_data(data_to_plot=norm_dict, xmin_p=xmin_p, xmax_p=xmax_p, steps=steps, savefig=savefig,
                      title=tmp._get_message_string(), figurename=figurename)
        tmp._y_spline = norm_dict
        return tmp


    def _get_message_string(self):
        '''Generate message string dependent on the status of self._normalized, self._bgr,
        and self._difference

        Returns:
            Message string
        '''

        message_string = "Normalized " if self._normalized else ""
        message_string += "BGR " if self._bgr else ""
        message_string += "Difference " if self._difference else ""
        if not self._normalized and not self._bgr and not self._difference:
            message_string = "Raw data"
        else:
            message_string += "Spectra"
        return message_string

    def subtract_solvent(self, solvent_index=0, xmin_p=800, xmax_p=2000, steps=None, savefig=False,
                         figurename='difference'):
        '''Subtract a (solvent) spectrum from the set of spectra and shows a plot of the subtracted
        data in the given plot-range [xmin_p, xmax_p]

        Args:
            solvent_index: Index of the spectrum that should be subtracted (integer, default: 0)
            xmin_p: lower rel. wavenumber limit for plot
            xmax_p: upper rel. wavenumber limit for plot
            steps: number of points (linear interpolation) between xmin_p and xmax_p
            savefig: if 'True' save figure to output folder (default format: '.pdf')
            figurename: name of the figure saved to the output folder (string, default: 'difference')

        Returns:
            Instance of the class (subtracted spectra)
        '''

        if steps == None:
            steps = (xmax_p - xmin_p)*2

        subtract_dict = {}

        for nr in range(len(self._filenames)):

            data = self._y_spline[self._get_key(nr)]
            data_to_subtract = self._y_spline[self._get_key(nr=solvent_index)]

            subtract_dict[self._get_key(nr)] = data - data_to_subtract

        tmp = self.copy()
        tmp._difference = True
        tmp._normalized = False
        tmp.plot_data(data_to_plot=subtract_dict, xmin_p=xmin_p, xmax_p=xmax_p, steps=steps,
                      savefig=savefig, title=tmp._get_message_string(), figurename=figurename)
        tmp._solvent_index = solvent_index
        tmp._filenames.remove(self._filenames[solvent_index])
        tmp._y_spline = subtract_dict

        return tmp

    def plot_data(self, data_to_plot, xmin_p, xmax_p, steps, savefig, figurename, title, bgr_plot=False):
        '''Plot manipulated and raw data

        Args:
            data_to_plot: manipulated data which will be plotted
            xmin_p: lower rel. wavenumber limit for plot
            xmax_p: upper rel. wavenumber limit for plot
            steps: number of points (linear interpolation) between xmin_p and xmax_p
            savefig: if 'True' save figure to output folder (default format: '.pdf')
            figurename: name of the figure saved to the output folder (string)
            title: title that will be shown in the plot
            bgr_plot: if True plot raw and background corrected data in one plot for each spectrum
        '''
        if steps == None:
                steps = (xmax_p - xmin_p)*4

        if savefig:
            if not os.path.exists(os.path.join(self._path, self._foldername, 'output')):
                os.makedirs(os.path.join(self._path, self._foldername, 'output'))

        if bgr_plot:
            for nr in range(len(self._filenames)):
                fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
                plt.rcParams.update({'font.size': 16})
                ax.axhline(y=0.0, color='k', linestyle='--')
                ax.plot(self._x_spline, self._y_spline[self._get_key(nr)], color='g', label='raw-data')
                ax.plot(self._x_spline, data_to_plot[self._get_key(nr)], color='k', label='BGR-corr')
                ax.set_ylabel('Raman intensity')
                ax.set_xlabel('Raman shift / cm$^{-1}$')
                ax.legend(bbox_to_anchor=(1.0, 1.2), loc='upper right')
                fig.tight_layout()
                ax.set_title(str(self._get_message_string())+' '+self._get_key(nr))

                if savefig:
                    figurepath = os.path.join(self._path, self._foldername, 'output',
                                              self._get_key(nr)+'_'+str(figurename)+'.pdf')
                    plt.savefig(figurepath)

        else:
            y_plot_manipulate = {}
            for nr in range(len(data_to_plot.keys())):
                x_plot = np.linspace(xmin_p, xmax_p, num=steps, endpoint=True)
                y_plot = interp1d(self._x_spline, data_to_plot[self._get_key(nr)])
                y_plot_manipulate[self._get_key(nr)] = y_plot(x_plot)

            plt.rcParams["figure.figsize"] = (10,7)
            fig = plt.figure()
            gs = gridspec.GridSpec(2,1,height_ratios=[4,4])
            plt.rcParams.update({'font.size': 16})

            ax1 = fig.add_subplot(gs[0])
            ax1.axhline(y=0.0, color='k', linestyle='--')

            for nr in range(len(data_to_plot.keys())):
                ax1.plot(x_plot, y_plot_manipulate[self._get_key(nr)], label=self._get_key(nr), linewidth=3)

            ax1.set_ylabel('rel. Raman intensity')
            ax1.set_xlabel('Raman shift / cm$^{-1}$')
            ax1.legend(bbox_to_anchor=(1.0, 1.5), loc='upper right')
            ax1.set_title(str(self._get_message_string()))

            if savefig:
                figurepath = os.path.join(self._path, self._foldername, 'output', str(figurename)+'.pdf')
                plt.savefig(figurepath)


    def write_data(self, name='manipulated_data'):
        '''Write manipulated data to output folder

        Args:
            name: filename (string, default: manipulated_data)
        '''

        if not os.path.exists(os.path.join(self._path, self._foldername, 'output')):
                os.makedirs(os.path.join(self._path, self._foldername, 'output'))

        file_name = str(name)+'.dat'
        with open(os.path.join(self._path, self._foldername, 'output', file_name), 'w') as f:
            f.write('wn')
            for nr in range(len(self._filenames)):
                f.write('\t'+str(self._filenames[nr][3:-4]))
            f.write('\n')
            for i in range(len(self._x_spline)):
                f.write(str(self._x_spline[i]))
                for nr in range(len(self._filenames)):
                    f.write('\t'+str(self._y_spline[self._get_key(nr)][i]))
                f.write('\n')

    def write_peaks(self, widths=np.arange(1,50,0.005), minimum=0.01, name='peak_data'):
        '''Write peak positions and intensity to output folder

        Args:
            widths: a range speciefied for the find_peaks_cwt (wavelet transformation)
            minimum: threshold of the peak intensity (values >= minimum are printed to the peak list)
            name: filename (string, default: peak_data)
        '''

        if not os.path.exists(os.path.join(self._path, self._foldername, 'output')):
            os.makedirs(os.path.join(self._path, self._foldername, 'output'))

        peakind_dict = {}
        for nr in range(len(self._filenames)):
            peakind = signal.find_peaks_cwt(self._y_spline[self._get_key(nr)], widths=widths, wavelet=None,
                                            max_distances=widths/2, gap_thresh=None, min_length=None, min_snr=1,
                                            noise_perc=10)
            peaks_n = []
            for index in peakind:
                if self._y_spline[self._get_key(nr)][index] > minimum:
                    peaks_n.append(index)
            peakind_dict['peaks_'+self._get_key(nr)] = peaks_n

            file_name = str(name)+'_'+self._get_key(nr)+'.dat'
            with open(os.path.join(self._path, self._foldername, 'output', file_name), 'w') as f:
                f.write('wn \t rel_Int \r\n')
                for i in peakind_dict['peaks_'+self._get_key(nr)]:
                    f.write(str(int(self._x_spline[i]))+'\t'+str(round(self._y_spline[self._get_key(nr)][i],3)))
                    f.write('\r\n')


    def plot_frames(self, nr_of_frames, filenumber, xmin_p, xmax_p, steps, data_to_plot):
        '''Plot the single frames in multiple subplots (depending on the number of frames)

        Args:
            nr_of_frames: number of frames (integer)
            filenumber: loop index (file number) specifying the working-file
            xmin_p: lower rel. wavenumber limit for plot
            xmax_p: upper rel. wavenumber limit for plot
            steps: number of points (linear interpolation) between xmin_p and xmax_p
            data_to_plot: manipulated data which will be plotted
        '''

        y_plot = {}
        nr = filenumber

        for i in range(nr_of_frames):
            name = 'y'+str(i)+'_'+str(self._frame_names[nr][3:-4])
            x_plot = np.linspace(xmin_p, xmax_p, num=steps, endpoint=True)
            y_plot_n = interp1d(self._x_frame_data['x_'+str(nr)], data_to_plot[name])
            y_plot[name] = y_plot_n(x_plot)

        if nr_of_frames > 30:
            nr_of_plots = int(nr_of_frames / 30)+1
            plt.rcParams.update({'font.size': 16})
            f, ax = plt.subplots(nr_of_plots, sharex=True, figsize=(10, 5*nr_of_plots))
            for plot_nr in range(nr_of_plots):
                for i in range(nr_of_frames):
                    name = 'y'+str(i)+'_'+str(self._frame_names[nr][3:-4])
                    color = next(ax[plot_nr]._get_lines.prop_cycler)['color']
                    if i <= 10+(plot_nr*30) and i >= 0+(plot_nr*30):
                        ax[plot_nr].plot(x_plot, y_plot[name], '-', color=color, linewidth=3, label=str(i))
                    elif i <= 20+(plot_nr*30) and i >= 0+(plot_nr*30):
                        ax[plot_nr].plot(x_plot, y_plot[name], '-.', color=color, linewidth=3, label=str(i))
                    elif i <= 30+(plot_nr*30) and i >= 0+(plot_nr*30):
                        ax[plot_nr].plot(x_plot, y_plot[name], '--', color=color, linewidth=3, label=str(i))
                ax[plot_nr].set_ylabel('Raman intensity')
                ax[plot_nr].set_xlabel('rel. wavenumber / cm$^{-1}$')
                ax[plot_nr].set_title('Plot '+str(nr))
                ax[plot_nr].legend(ncol=3, bbox_to_anchor=(1.5, 1.0), loc='upper right')

        else:
            plt.rcParams["figure.figsize"] = (10,5)
            fig = plt.figure()
            gs = gridspec.GridSpec(1,1, height_ratios=[1])
            plt.rcParams.update({'font.size': 16})
            ax1 = fig.add_subplot(gs[0])
            ax1.axhline(y=0.0, color='k', linestyle='--')
            for i in range(nr_of_frames):
                name = 'y'+str(i)+'_'+str(self._frame_names[nr][3:-4])
                color = next(ax1._get_lines.prop_cycler)['color']
                if i <= 10:
                    ax1.plot(x_plot, y_plot[name], '-', color=color, linewidth=3, label=str(i))
                elif i <= 20:
                    ax1.plot(x_plot, y_plot[name], '-.', color=color, linewidth=3, label=str(i))
                else:
                    ax1.plot(x_plot, y_plot[name], '--', color=color, linewidth=3, label=str(i))
            ax1.set_ylabel('Raman intensity')
            ax1.set_xlabel('rel. wavenumber / cm$^{-1}$')
            ax1.set_title('Plot '+str(nr))
            ax1.legend(ncol=3, bbox_to_anchor=(1.5, 1.0), loc='upper right')


    def read_frames(self, foldername='data', subfoldername='SEC', file_ending='.txt', xmin=700,
                    xmax=2500, steps=None, columntype='single'):
        ''' Reads raw data (multiple frames exported WinSpec ascii-files)

        Args:
            foldername: name of the folder including the data (string, default: 'data')
            subfoldername: name of the folder including data with multiple frames (string, default: 'SEC')
            file_ending: file-ending (string, default: '.txt')
            xmin: minimum rel. waveneumber value (float/integer, default: 700)
            xmax: maximum rel. wavenumber value (float/integer, default: 2500)
            steps: specify number of points in the ranging from xmin to xmax (default:
            None, calculates the number of steps from xmin and xmax)
            columntype: single or multiple (depends on exporpt function in WinSpec, default:'single')

        Raises:
            columntype must be either single or multiple
        Returns:
            The splined raw_data of each frame
        '''

        if columntype is not 'single' and columntype is not 'multiple':
            raise Exception('Columntype must be single (one column including several frames) or multiple (each frame one row).')

        if steps == None:
            steps = (xmax - xmin)*4

        print('Number of x-points = '+str(steps))

        self._foldername = foldername

        file_frame_path = os.path.join(self._path, foldername, subfoldername)
        frame_names = sorted([n for n in os.listdir(file_frame_path) if n.endswith(file_ending)])
        self._frame_names = frame_names

        x_dict = {}
        y_dict = {}
        for nr in range(len(frame_names)):
            path = os.path.join(file_frame_path,str(frame_names[nr]))
            df = pd.read_csv(path, sep="\t", header=None)
            data_array = np.array(df)
            
            if columntype == 'single':
                frames_n, indices = np.unique(data_array[0:,-2], return_index=True)
                frames = [int(i) for i in frames_n]
                nr_of_frames=frames[-1]
                print('File ('+str(frame_names[nr])+') No.'+str(nr)+' -> '+str(nr_of_frames)+' frames')

                x_dict['x_'+str(nr)] = data_array[indices[0]:indices[1],0]
                for i in range(len(frames)):
                    if i < int(len(frames)-1):
                        y_dict['y'+str(i)+'_'+str(frame_names[nr][3:-4])] = data_array[indices[i]:indices[i+1],-1]
                    else:
                        y_dict['y'+str(i)+'_'+str(frame_names[nr][3:-4])] = data_array[indices[i]:,-1]
            
            elif columntype == 'multiple':
                frames_n = data_array[3:,0]
                frames = [int(i) for i in frames_n]
                nr_of_frames=frames[-1]
                print('File ('+str(frame_names[nr])+') No.'+str(nr)+' -> '+str(nr_of_frames)+' frames')
                x_dict['x_'+str(nr)] = data_array[1,2:-1]
                for i in range(len(frames)):
                    y_dict['y'+str(i)+'_'+str(frame_names[nr][3:-4])] = data_array[i+3,2:-1]

            self._x_frame_data = x_dict
            self._y_frame_data = y_dict
            self.plot_frames(nr_of_frames=nr_of_frames, xmin_p=xmin, xmax_p=xmax, steps=steps,
                            data_to_plot=y_dict, filenumber=nr)
        
        self._x_frame_data = x_dict
        self._y_frame_data = y_dict

    def average_frames(self, number=[0,1], average_start=[0,0], average_end=[3,3], specification=['ocp','red']):
        '''Average frames in a given range and write the data to the folder with the single column data

        Args:
            number: List of the spectral number (e.g. 2 files in a folder requires the list [0,1] to work wih
            both files)
            average_start: List of integers specifying the start-index of the average list
            average_end: List of integers specifying the last-index of the average list
            specification: List of strings specifying the averaged data (string is added in the file, which is written
            upon making the average)

        Raises:
            Entry numbers in average lists do not equal the number of files. Numbers must be equal.
            Average lists are of unequal length. Use equal numbers of list entries in average_end and average_start.
        '''

        if len(average_start) == len(average_end) == len(number):
            None
        else:
            if len(average_start) == len(average_end):
                raise Exception('Entry numbers in average lists do not equal the number of files. Numbers must be equal.')
            else:
                raise Exception('Average lists are of unequal length. Use equal numbers of list entries in average_end and \
                average_start.')

        y_mean = {}
        for index, nr in enumerate(number):
            frame_list = [average_start[index], average_end[index]]
            y_mean_data = list(np.mean([self._y_frame_data['y'+str(frame_idx)+'_'+str(self._frame_names[nr][3:-4])] for frame_idx in frame_list], axis=0))

            name = str(self._frame_names[nr][0:-4])+'_'+str(specification[index])+'_mean.txt'
            with open(os.path.join(self._path, self._foldername, name), 'w') as f:
                for i in range(len(self._x_frame_data['x_'+str(nr)])):
                    f.write(str(self._x_frame_data['x_'+str(nr)][i]))
                    f.write('\t'+str(y_mean_data[i]))
                    f.write('\n')

        print('Averaged files are written to the folder.')

