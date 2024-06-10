import sys
sys.path.append("C:\\Users\\User\\Vrije Universiteit Brussel\\Mehdi Feizpour - Halewijn's Thesis Project\\master thesis\\other")
sys.path.append("C:\\Users\\mfeizpou\\OneDrive - Vrije Universiteit Brussel\\Halewijn's Thesis Project\\master thesis\other")

import os
import uuid
import tkinter as tk
from tkinter import *
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets import interact, widgets
from sklearn.preprocessing import MinMaxScaler
from renishawWiRE import WDFReader
import matplotlib.animation as animation
from scipy.signal import find_peaks
from adjustText import adjust_text  # Import ad
from PIL import Image
import ramanspy as rp
import subprocess
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




# from renishawWiRE import WDFReader
# from renishawWiRE import export
import numpy as np
import importlib
import my_functions as mf
import filepaths as fp
import figure_label_names as fln
import color_maps as cm
import variables
import classes
mf = importlib.reload(mf)
fp = importlib.reload(fp)
cm = importlib.reload(cm)
variables = importlib.reload(variables)
classes = importlib.reload(classes)

class RamanMeasurement(): 
    def __init__(self, filepath, preprocessing = False, preprocessing_params = None):
        self.filepath = filepath
        self.reader = WDFReader(filepath)
        self.num_spectra = self.reader.count
        self.SpectralContainer = rp.load.renishaw(self.filepath)
        self.measurement_type = self.reader.measurement_type #this is basically a number
        self.preprocessing_params = preprocessing_params
        #number = 1, it's a spectral measurement
        #number = 2, it's a depth series, or timeseries
        #number = 3, it's a mapping spectral measurement
        self.num_timepoints = self.reader.count

        if preprocessing: 
            self.cosmic_ray_removal = self.preprocessing_params['cosmic_ray_removal']
            self.window_length = self.preprocessing_params['window_length']
            self.polyorder = preprocessing_params['polyorder']
            self.normalization = preprocessing_params['normalization']
            if self.cosmic_ray_removal: 
                self.cosmic_ray_z_score = self.preprocessing_params['cosmic_ray_z_score']
            if self.window_length is None or self.polyorder is None: 
                raise ValueError('window length and polyorder need to be provided')
            self.preprocess_data()

        else: 
            self.window_length = None
            self.polyorder = None
            self.spectra = self.reader.spectra
            self.xdata = self.reader.xdata
            self.num_spectra = self.reader.count
        #run some functions
        self.convert_wdf_to_csv()
        
    def preprocess_data(self): 
        preprocessing_pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.denoise.SavGol(window_length=self.window_length, polyorder=self.polyorder),
            rp.preprocessing.baseline.ASLS(), 
]) 
        if self.cosmic_ray_removal: 
            preprocessing_pipeline.insert(0, rp.preprocessing.despike.WhitakerHayes(threshold = self.cosmic_ray_z_score))
        if self.normalization:
            preprocessing_pipeline.append(rp.preprocessing.normalise.MinMax())
            
        preprocessed_spectrum = preprocessing_pipeline.apply(self.SpectralContainer)
        self.xdata = preprocessed_spectrum.spectral_axis
        self.spectra = preprocessed_spectrum.spectral_data
        
    def convert_wdf_to_csv(self):
        if self.measurement_type == 1: #if it's just a spectral measurement
            if self.num_timepoints == 1: 
                x_axis = self.xdata
                y_axis = self.spectra
                self.csv_df = pd.DataFrame({'wavenumber': x_axis, 
                             'intensity': y_axis})
        elif self.measurement_type == 2: #spectral series
            df_array = []
            spectra = self.spectra
            for spectrum_nr in range(self.num_spectra): 
                spectrum = spectra[spectrum_nr]
                df = pd.DataFrame({'wavenumber': self.xdata, 
                                  'intensity': spectrum})
                df.loc[:, 'spectrum_nr'] = [spectrum_nr]*df.shape[0]
                df_array += [df]
            concat_df = pd.concat(df_array)
            
            self.csv_df = concat_df[['spectrum_nr', 'wavenumber', 'intensity']] #change the order
        elif self.measurement_type == 3: #a map
            unique_xpositions = mf.get_unique_elements_in_order(self.reader.xpos)
            unique_ypositions = mf.get_unique_elements_in_order(self.reader.ypos)
            df_array = []
            unique_position_counter = 0
            for x_index, unique_xposition in enumerate(unique_xpositions):
                for y_index, unique_yposition in enumerate(unique_ypositions): 
                    intensity_array = self.spectra[y_index, x_index, :]
                    df = pd.DataFrame({'wavenumber': self.xdata, 
                                     'intensity': intensity_array})
                    df.loc[:, 'x'] = [unique_xposition]*df.shape[0]
                    df.loc[:, 'y'] = [unique_yposition]*df.shape[0]
                    df.loc[:, 'measurement_index'] = [unique_position_counter]*df.shape[0]
                    unique_position_counter += 1
                    df_array += [df]
            concat_df = pd.concat(df_array)
            column_order = ['x', 'y','measurement_index', 'wavenumber', 'intensity']
            concat_df = concat_df.reindex(columns = column_order)
            self.csv_df = concat_df
        else: 
            self.csv_df = None
            print("measurement type not identified")
        return self.csv_df  
    def get_image_data(self): 
        if self.measurement_type == 1: 
            print('write this code')
        elif self.measurement_type == 2: 
            print('code stil has to be written')
        elif self.measurement_type == 3: 
            bytes_io = self.reader.img
            bytes_io.seek(0)
            image = Image.open(bytes_io)
            self.image = np.array(image)
            img_x0, img_y0 = self.reader.img_origins
            img_w, img_h = self.reader.img_dimensions
            self.extent = (img_x0, img_x0 + img_w,
                img_y0 + img_h, img_y0)
            return self.image, self.extent
        else: 
            print('code not yet written for this measurement type')
    def show_image(self): 
        self.convert_wdf_to_csv()
        self.get_image_data()
        scatter_coordinates = self.csv_df.groupby(['x', 'y'], as_index = False).first()[['x','y']] #get the unique cooridnates of where the spectra are captured
        plt.imshow(self.image, 
        extent=self.extent)
        sns.scatterplot(data = scatter_coordinates, x = 'x', y = 'y', s = 5, color = 'black')
        plt.show()

    def get_raman_peaks(self, prominence = 1,distance = 1,width = 1, threshold = 1, height = 0): 
        self.convert_wdf_to_csv()
        
        peaks, _ = find_peaks(self.csv_df['intensity'].values, prominence=prominence,distance =  distance, width = width, threshold = threshold,height = height)
        print(height)
        idx = self.SpectralContainer.peaks()
        
        x_positions = []
        y_positions = []
        texts = []
        for peak_idx in peaks:
            peak_wavenumber = self.csv_df['wavenumber'].values[peak_idx]
            x_positions.append(peak_wavenumber)
            peak_height = self.csv_df['intensity'].values[peak_idx]
            y_positions.append(peak_height)
            texts.append(str(int(peak_wavenumber)))
            #plt.scatter(peak_wavenumber, peak_height + 2, color = 'black', alpha = 0.5, s = 2)
            #texts.append(plt.text(peak_wavenumber-10, peak_height + 5, str(int(np.round(peak_wavenumber, 0))),fontsize=variables.raman_wavenumber_fontsize))
        #return x_positions, y_positions, texts
        self.peaks_df = pd.DataFrame({
            'x_positions':x_positions,
            'y_positions':y_positions,
            'texts' :texts
            
        })
        print(x_positions)
        return self.peaks_df
    def plot_raman_spectrum(self, plot_peaks = False, threshold = 1, height = 0):
        sns.lineplot(data = self.csv_df, x = 'wavenumber', y= 'intensity')
        if plot_peaks: 
            self.get_raman_peaks(threshold = threshold, height = height)
            sns.scatterplot(data = self.peaks_df, x = 'x_positions', y = 'y_positions')
            for ii in range(self.peaks_df.shape[0]):
                text = self.peaks_df['texts'].values[ii]
                x_pos = self.peaks_df['x_positions'].values[ii]
                y_pos = self.peaks_df['y_positions'].values[ii]
                plt.text(x_pos, y_pos, text)
        plt.xlabel(fln.spectrum_x, fontsize = variables.x_y_label_fontsize)
        plt.ylabel(fln.spectrum_y, fontsize = variables.x_y_label_fontsize)
        plt.xticks(fontsize=variables.tick_fontsize)
        plt.yticks(fontsize=variables.tick_fontsize)
        plt.show()

    def create_map_gui(self):
        df = self.convert_wdf_to_csv()
        self.show_image()
        x_points = df['x'].unique()
        y_points = df['y'].unique()
        num_x_points = len(x_points)
        num_y_points = len(y_points)
        # Create the main Tkinter window
        root = tk.Tk()
        root.title("Grid of Buttons Example")
    
        # Create a frame for the canvas
        canvas_frame = Frame(root)
        canvas_frame.grid(row=0, column=0, columnspan=13, rowspan=13)
    
        fig = Figure(figsize=(5, 5), dpi=100) 
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw() 
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=13, rowspan=13)
    
        # Create a frame for the buttons
        buttons_frame = Frame(root)
        buttons_frame.grid(row=0, column=13, columnspan=12, rowspan=13)
    
        for row in range(num_y_points):
            for col in range(num_x_points):
                button = tk.Button(buttons_frame, width=2, height=1)
                button.grid(row=row, column=col, padx=1, pady=1)
    
                # Bind hover events
                button.bind("<Enter>", lambda event, r=row, c=col: mf.on_hover(event, r, c, fig, canvas, df,x_points,y_points))
    
        # Start the Tkinter event loop
        root.mainloop()
        
class RamanStudies(): 
    def __init__(self, raman_spectra_path, gui_dataframe_path): 
        self.raman_spectra_path = raman_spectra_path
        self.gui_dataframe_path = gui_dataframe_path
        self.get_concatenated_gui_dataframes()
        self.get_study_overview_list()

    def get_concatenated_gui_dataframes(self): 
        filenames = mf.get_files_with_extension(self.gui_dataframe_path, '.csv')
        df_array = []
        for filename in filenames: 
            df = pd.read_csv(self.gui_dataframe_path + '/' + filename)
            df_array += [df]
        self.concat_df = pd.concat(df_array)
        return self.concat_df

    def get_study_overview_list(self): 
        self.study_overview_list = self.concat_df['study_name'].unique()
        return self.study_overview_list

    def get_study_dataframe(self, study_name): 
        filt = self.concat_df['study_name'] == study_name
        self.study_dataframe = self.concat_df[filt]
        return self.study_dataframe

    def get_raman_measurement(self, measurement_id, preprocessing = False, preprocessing_params = None): 
        if preprocessing: 
            self.RamanMeasurement = RamanMeasurement(self.raman_spectra_path + '/' +  measurement_id + '.wdf', preprocessing, preprocessing_params=preprocessing_params)
        else:
            self.RamanMeasurement = RamanMeasurement(self.raman_spectra_path + '/' +  measurement_id + '.wdf', preprocessing)

        return self.RamanMeasurement

    def open_study_dataframe(self, study_name):
        self.get_study_dataframe(study_name)
        
        self.study_dataframe.to_csv(fp.raman_path + '/' + 'delete.csv')
        subprocess.Popen([r'C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE', fp.raman_path + '/' + 'delete.csv'], shell=True)
        



class Raman_database(): 
    def __init__(self, measurement_dict, preprocessing_params = variables.preprocessing_params, order = None):
        self.measurement_dict = measurement_dict
        self.ramanstudies = classes.RamanStudies(fp.raman_spectra_path, fp.raman_GUI_dataframe_path)
        self.order = order
        self.preprocessing_params = preprocessing_params
        
        
    def concatenate_raman_dataframes(self): 
        measurement_index_count = 0
        df_array = []
        for Class in self.measurement_dict: 
            for id in self.measurement_dict[Class]: 
                raman_measurement = self.ramanstudies.get_raman_measurement(id, preprocessing = True, preprocessing_params=self.preprocessing_params)
                num_spectra = raman_measurement.num_spectra #get the number of spectra
                
                df = raman_measurement.convert_wdf_to_csv() #read in the dataframe that also contains a column of the measurement index that starts at 0
                last_measurement_index = df['measurement_index'].values[-1]
                df.loc[:, 'measurement_index'] = df.loc[:, 'measurement_index'] + measurement_index_count
                measurement_index_count = measurement_index_count + last_measurement_index + 1
                df.loc[:, 'Class'] = [Class]*df.shape[0]
                # Define the desired order
                if self.order is not None:
                    df['Class'] = pd.Categorical(df['Class'], categories=self.order, ordered=True)
                df_array += [df]
            self.concat_df = pd.concat(df_array)
        return self.concat_df
        

    def get_stacked_plot_raman_dataframe(self, distance): 
        self.concatenate_raman_dataframes()
        mean_concat_df = self.concat_df.groupby(['wavenumber','Class'], as_index = False)['intensity'].mean()
        gr_df = mean_concat_df.groupby(['Class'],as_index = False)['intensity'].max() #group and get the max value
        cummulative_sum_df = pd.DataFrame(gr_df['intensity'].cumsum()) #make the cummulative sum
        cummulative_sum_df['add_distance'] = cummulative_sum_df.index*distance
        cummulative_sum_df['intensity'] = (cummulative_sum_df['intensity']).shift(1).replace({np.NaN: 0}) + cummulative_sum_df['add_distance'] #add a shift, and also add the offset
        cummulative_sum_df['Class'] = gr_df['Class'] #put the labels back for merging later
        cummulative_sum_df = cummulative_sum_df.rename({'intensity': 'offset'}, axis = 1) #rename the column to offset
        merged_df = pd.merge(self.concat_df, cummulative_sum_df, how = 'left', on ='Class') #merge the dataframes
        merged_df['intensity'] = merged_df['intensity'] + merged_df['offset']
        self.stacked_conated_df = merged_df[['x', 'y', 'measurement_index','wavenumber', 'intensity', 'Class']]
        return self.stacked_conated_df