import importlib
import pandas as pd
import numpy as np
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import my_functions as mf
import variables as variables
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
import filepaths as fp
import pickle
from sklearn.model_selection import train_test_split
import uuid
from sklearn.metrics import accuracy_score, f1_score
import subprocess
import tkinter as tk
from tkinter import Frame, Button, Label
import re
from PIL import Image
import io
from renishawWiRE import WDFReader
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from ipywidgets import interact, widgets
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture









variables = importlib.reload(variables)
# fontsize = variables.fontsize


def get_melt_df(vector_df): 
    
    melted_df = vector_df.melt(id_vars = ['measurement_index', 'Class'], var_name = 'wavenumber', value_name = 'intensity').sort_values(['measurement_index', 'wavenumber'])
    melted_df.loc[:, 'wavenumber'] = melted_df.loc[:, 'wavenumber'].astype('float64')
    return melted_df

def get_vector_df(melted_df, intensity_column_name):
    """
    Transforms a melted DataFrame containing spectral data into a pivoted DataFrame, 
    where each row represents a measurement with intensity values at different wavelengths.
    """ 
    melted_df = melted_df[['measurement_index', 'Class', 'wavenumber', intensity_column_name]]
    pivoted_df = melted_df.pivot_table(index=['measurement_index', 'Class'], columns='wavenumber', values=intensity_column_name).reset_index()
    pivoted_df.columns.name = None  # Remove the column name after pivot
    return pivoted_df

def get_mean_spectrum(melt_df): 
    mean_spectrum = melt_df.groupby(['wavenumber'], as_index = False)['intensity'].mean()
    return mean_spectrum

def get_measurement_spectrum(measurment_index, melt_df): 
    return melt_df[melt_df.loc[:, 'measurement_index'] == measurment_index]

def describe_class_distribution(vector_df): 
    df = pd.DataFrame(vector_df.groupby('Class').size()).reset_index().rename({0: 'num_samples'}, axis = 1)
    tot_samples = df.loc[:, 'num_samples'].sum()
    df.loc[:, 'percentage'] = np.round(df.loc[:, 'num_samples']/tot_samples*100, 1)
    display(df)
    print('in total, there are ' + str(tot_samples) + ' samples')

def get_random_sample_from_class(melt_df, Class, num_samples): 
    cropped_df = melt_df.groupby(['measurement_index', 'Class'],as_index = False)['Class'].first()
    #filter on class
    filt_cropped_df = cropped_df[cropped_df.loc[:, 'Class'] == Class]
    #get the measurment indices that all belong to that class
    measurements_indices = list(filt_cropped_df.loc[:, 'measurement_index'])
    #sample without replacement
    selected_measurement_indices = random.sample(list(measurements_indices),num_samples)
    #np.random.sample(measurements_indices)
    filtered_df = melt_df[melt_df.loc[:, 'measurement_index'].isin(selected_measurement_indices)]
    return filtered_df

#replaces the names of the bacterial strain so that it's in italics
def replace_bacterial_names_for_plotting(df): 
    new_df = df.replace({'E coli': '$\it{E}$' + ' ' + '$\it{coli}$',
                     'H influenza': '$\it{H}$' + ' ' + '$\it{influenza}$',
                    'N meningitidis': '$\it{N}$' + ' ' + '$\it{meningitidis}$',
                   'S pneumoniae': '$\it{S}$' + ' ' + '$\it{pneumoniae}$',
                    'S epidermis': '$\it{S}$' + ' ' + '$\it{epidermis}$', 
                    'P vulneris': '$\it{P}$' + ' ' + '$\it{vulneris}$'})
    return new_df


def stitch_labels(df, labels_series): 
    df['Class'] = labels_series.values
    return df

def get_most_common_element(labels, predicted_labels, label):
    # Count occurrences of each string element
    element_counts = Counter(np.array(labels[predicted_labels == label]))
    most_common_element = element_counts.most_common(1)[0][0]
    return most_common_element

def get_label_map(test_df):
    labels = test_df.loc[:, 'Class']
    predicted_labels_gmm = test_df.loc[:, 'predicted_Class numerical']
    # Map predicted GMM labels to original class labels
    label_map = {0: get_most_common_element(labels, predicted_labels_gmm, 0),
                1: get_most_common_element(labels, predicted_labels_gmm, 1), 
                2: get_most_common_element(labels, predicted_labels_gmm, 2), 
                3: get_most_common_element(labels, predicted_labels_gmm, 3)}

    # Reverse the mapping dictionary
    reverse_label_map = {v: k for k, v in label_map.items()}

    return label_map, reverse_label_map

def transform_categorical_to_numerical(df, label_map, reverse_label_map): 
    df.loc[:, 'Class numerical'] = df.loc[:, 'Class'].map(reverse_label_map)
    df.loc[:, 'predicted Class'] = df.loc[:, 'predicted_Class numerical'].map(label_map)
    return df

def plot_confusion_matrix(df, map, filename, fontsize): 
    conf_matrix = confusion_matrix(df.loc[:, 'true_gmm_class'], df.loc[:, 'gmm_predicted_class'])
    #convert into percentages
    percentages_conf_matrix = conf_matrix.astype('float')/conf_matrix.sum(axis = 1)[:, np.newaxis]*100
    percentages_conf_matrix = np.round(percentages_conf_matrix, 2)
    #plot the heatmap
    
    labels = list(replace_bacterial_names_for_plotting(map.gmm_class))
    sns.heatmap(percentages_conf_matrix, annot=True, fmt = '', cmap='Blues', linewidths=0.5, xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.ylabel('True class', fontsize = fontsize)
    plt.xlabel('Predicted class', fontsize = fontsize)
    plt.setp(plt.gca().get_xticklabels(), fontsize=variables.tick_fontsize)
    
    plt.setp(plt.gca().get_yticklabels(), fontsize=variables.tick_fontsize)
    plt.savefig(fp.figure_filepath + '/'  + 'results_figures' + '/'+ filename, bbox_inches = 'tight', dpi = 1200, transparent = True)
    


    plt.show()
    
    

def save_pickle(filepath, pipeline):
    with open(filepath, 'wb') as file:
        pickle.dump(pipeline, file)

def map_color(df, column_name, mapping): 
    df.loc[:, 'color'] = df.loc[:, column_name].map(mapping)
    return df

def get_X_y_from_vector_df(vector_df): 
    X = vector_df.iloc[:, 2:]
    y = vector_df.loc[:, 'Class']
    return X, y

def majority_vote(df):
    majority_label = df['gmm_predicted_class'].mode()[0]  # Get the mode (most frequent label)
    return majority_label
    
def convert_GMM_to_class_labels(df, map): 
    class_train_df = pd.merge(df, map, left_on='gmm_predicted_class', right_on='gmm_class_numerical', how = 'left')
    class_train_df = class_train_df.drop('gmm_class_numerical', axis = 1).rename({'gmm_class':'predicted_class'}, axis = 1)
    class_train_df = pd.merge(class_train_df, map, left_on='true_class', right_on='gmm_class').drop('gmm_class', axis =1).rename({'gmm_class_numerical': 'true_gmm_class'}, axis =1)
    class_train_df
    return class_train_df

def raman_shift_to_wavelength(raman_shift, laser_wavelength_nm): 
    """
    Converts Raman shift to scattered wavelength using the laser wavelength.

    Parameters:
    - raman_shift (float): Raman shift value in reciprocal centimeters (cm⁻¹).
    - laser_wavelength_nm (float): Wavelength of the incident laser in nanometers (nm).

    Returns:
    float: Wavelength of the Raman-scattered light in nanometers (nm).
    """
    
    laser_wavelength_cm = laser_wavelength_nm/1e7
    wavelength_cm = 1/((1/laser_wavelength_cm)-raman_shift)
    wavelength_nm = wavelength_cm*1e7
    return wavelength_nm
    
def get_dummy_accuracy(vector_df): 
    """
    Calculates the accuracy of a random untrained model using the DummyClassifier.
    """
    
    # get the x, y vector
    X, y = get_X_y_from_vector_df(vector_df)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a random untrained model using DummyClassifier
    random_model = DummyClassifier(strategy="uniform", random_state=42)
    # Fit the model (even though it's random, scikit-learn requires a fit before predictions)
    random_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = random_model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_dataframe_without_overwrite(dataframe, folder_path, filename, index):
    """
    Save a DataFrame to a CSV file without overwriting existing files.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be saved.
    - original_filename (str): The original filename without the extension.
    - folder_path (str): The folder path where the file will be saved. Default is the current directory.
    """
    # Create the full file path
    full_file_path = folder_path + '/' + filename + '.csv'

    # Check if the file already exists
    counter = 1
    while os.path.exists(full_file_path):
        # Append an increment to the filename
        full_file_path = folder_path + '/' + filename + '_' + str(counter) + '.csv'
        counter += 1

    # Save the DataFrame to a CSV file with the new filename
    dataframe.to_csv(full_file_path, index=index)
    return full_file_path


def get_gaussian_noise_intensity(melt_df, mean, std_dev): 
    """
    Add Gaussian noise to the 'intensity' column in a DataFrame.
    """
    melt_df_copy = melt_df.copy()
    # Calculate the number of datapoints
    num_datapoints = melt_df_copy.shape[0]
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, num_datapoints)
    melt_df_copy.loc[:, 'intensity_noise'] = melt_df_copy.loc[:, 'intensity'] + noise
    return melt_df_copy

def get_files_with_extension(path, extension): 
    files = [file for file in os.listdir(path) if file.endswith(extension)]
    return files

# Function to convert wavelength to Raman shift
def wavelength_to_raman_shift(wavelength_nm, laser_wavelength_nm): 
    """
    Calculate the Raman shift based on the wavelength of scattered light
    and the wavelength of the incident laser light.
    Parameters:
    - wavelength_nm: The wavelength of the scattered light in nanometers.
    - laser_wavelength_nm: The wavelength of the incident laser light in nanometers
    Returns:
    - raman_shift: The calculated Raman shift in inverse centimeters.
    """
    # Calculate inverse wavelength in centimeters
    inverse_cm = 10*1e6/wavelength_nm
    # Calculate Raman shift using the given formula
    raman_shift = 10*1e6*(1/wavelength_nm - 1/laser_wavelength_nm)
    # Return the calculated Raman shift
    return raman_shift

# Function to convert Raman shift to wavelength
def raman_shift_to_wavelength(raman_shift, laser_wavelength_nm):
    """
    Convert Raman shift (in inverse centimeters) back to the wavelength of scattered light
    Parameters:
    - raman_shift: The Raman shift in inverse centimeters.
    - laser_wavelength_nm: The wavelength of the incident laser light in nanometers
    Returns:
    - wavelength: The calculated wavelength of the scattered light in nanometers.
    """
    # Calculate inverse wavelength in centimeters
    inverse_wavelength_cm = (raman_shift + 10*1e6/laser_wavelength_nm)
    # Convert inverse wavelength to nanometers
    inverse_wavelength_nm = inverse_wavelength_cm * 1e-7
    # Calculate the wavelength of the scattered light
    wavelength = 1 / inverse_wavelength_nm
    # Return the calculated wavelength
    return wavelength

def generate_experiment_id():
    unique_id = str(uuid.uuid4())
    # Take a portion of the UUID (e.g., first 8 characters)
    shortened_uuid = unique_id.split('-')[0]
    return shortened_uuid

def get_experiment_name_suggestions():
    od_GUI_dataframe_path = fp.od_GUI_dataframe_path
    raman_GUI_dataframe_path = fp.raman_GUI_dataframe_path
    microfluidic_GUI_dataframe_path = fp.microfluidic_GUI_dataframe_path
    filepaths_to_check = [od_GUI_dataframe_path, raman_GUI_dataframe_path, microfluidic_GUI_dataframe_path]
    
    suggestions_array = []
    for path in filepaths_to_check: 
        for file_name in os.listdir(path): 
            filepath = path + '/' + file_name
            try: 
                suggestions = list(pd.read_csv(filepath).loc[:, 'experiment_name'].unique())
                for suggestion in suggestions: 
                    if suggestion not in suggestions_array:
                        suggestions_array += [suggestion]
            except: 
                pass
    return np.array(suggestions_array).astype('str')

def get_study_name_suggestions():
    od_GUI_dataframe_path = fp.od_GUI_dataframe_path
    raman_GUI_dataframe_path = fp.raman_GUI_dataframe_path
    microfluidic_GUI_dataframe_path = fp.microfluidic_GUI_dataframe_path
    filepaths_to_check = [od_GUI_dataframe_path, raman_GUI_dataframe_path, microfluidic_GUI_dataframe_path]
    
    suggestions_array = []
    for path in filepaths_to_check: 
        for file_name in os.listdir(path): 
            filepath = path + '/' + file_name
            try: 
                suggestions = list(pd.read_csv(filepath).loc[:, 'study_name'].unique())
                for suggestion in suggestions: 
                    if suggestion not in suggestions_array:
                        suggestions_array += [suggestion]
            except: 
                pass
    return np.array(suggestions_array).astype('str')

def open_csv_in_excel(csv_file_path):
    try:
        subprocess.run(['start', 'excel', csv_file_path], shell=True)
    except Exception as e:
        print(f"Error opening Excel: {e}")


#for the study
def study_on_key_release(event,study_name_var, suggestion_listbox, study_suggestion_items):
    search_term = study_name_var.get().lower()
    # Clear previous suggestions
    suggestion_listbox.delete(0, tk.END)
    if search_term:
        # Filter suggestions based on the entered text
        suggestions = [item for item in study_suggestion_items if search_term in item.lower()]
        # Display filtered suggestions
        for suggestion in suggestions:
            suggestion_listbox.insert(tk.END, suggestion)

def on_study_suggestion_selected(event,study_name_var, study_suggestion_listbox):
    # Get the selected suggestion and insert it into the entry
    selected_index = study_suggestion_listbox.curselection()
    if selected_index:
        selected_suggestion = study_suggestion_listbox.get(selected_index)
        study_name_var.set(selected_suggestion)
        study_suggestion_listbox.delete(0, tk.END)  # Clear suggestions after selecting one

def on_experiment_key_release(event,experiment_name_var, suggestion_listbox, experiment_suggestion_items):
    search_term = experiment_name_var.get().lower()
    # Clear previous suggestions
    suggestion_listbox.delete(0, tk.END)
    if search_term:
        # Filter suggestions based on the entered text
        suggestions = [item for item in experiment_suggestion_items if search_term in item.lower()]
        # Display filtered suggestions
        for suggestion in suggestions:
            suggestion_listbox.insert(tk.END, suggestion)

def on_experiment_suggestion_selected(event,experiment_name_var, experiment_suggestion_listbox, experiment_suggestion_items):
    # Get the selected suggestion and insert it into the entry
    selected_index = experiment_suggestion_listbox.curselection()
    if selected_index:
        selected_suggestion = experiment_suggestion_listbox.get(selected_index)
        experiment_name_var.set(selected_suggestion)
        experiment_suggestion_listbox.delete(0, tk.END)  # Clear suggestions after selecting one

def read_od600_spectrum_df(filepath): 
    df = pd.read_csv(filepath, sep=";", skiprows=5, decimal=',', engine='python')
    df = df.drop(0).reset_index(drop = True)
    df.columns = ['wavelength', 'sample', 'dark', 'reference', 'transmittance']
    df['wavelength'] = df['wavelength'].astype('float64')
    df['sample'] = df['sample'].astype('float64')
    df['dark'] = df['dark'].astype('float64')
    df['reference'] = df['reference'].astype('float64')
    df['transmittance'] = df['transmittance'].astype('float64')
    df.loc[:, 'od'] = -np.log(df['transmittance']/100)
    return df

def get_od_spectra_from_study(study_name): 
    #concatenate all the GUI dataframes
    file_names = get_files_with_extension(fp.od_GUI_dataframe_path, 'csv')
    df_array = []
    for file_name in file_names: 
        df = pd.read_csv(fp.od_GUI_dataframe_path + '/' + file_name)
        df_array += [df]
    
    concat_df = pd.concat(df_array)
    
    #now filter on the study
    study_filt = concat_df.loc[:, 'study_name'] == study_name
    study_filt_concat_df = concat_df[study_filt]
    
    #now concatenate the sprectra, but first look if the spectra id occurs in the study
    pattern = re.compile(r'^(.*?)_\w{9}\.TXT')
    
    spectra_df_array = []
    txt_file_names =get_files_with_extension(fp.od_spectra_path, 'TXT')
    study_spectra_df_array = []
    for txt_file_name in txt_file_names: 
        ID = re.search(pattern, txt_file_name).group(1)
        if ID in list(study_filt_concat_df.loc[:, 'ID']): 
            df = read_od600_spectrum_df(fp.od_spectra_path + '/' + txt_file_name)
            df.loc[:, 'ID'] = [ID]*df.shape[0]
            study_spectra_df_array += [df]
            
    study_spectra_df = pd.concat(study_spectra_df_array)
    study_df = pd.merge(study_filt_concat_df, study_spectra_df, how = 'left', on = 'ID')

    
    return study_df




def get_image_and_extent_from_wdf(filepath): 
    reader = WDFReader(filepath)
    bytes_io = reader.img
    # Rewind the BytesIO object to the beginning
    bytes_io.seek(0)
    
    # Open the image using Pillow
    image = Image.open(bytes_io)
    image = np.array(image)
    # Now you can work with the image, for example, display it or save it to a file
    img_x0, img_y0 = reader.img_origins
    img_w, img_h = reader.img_dimensions
    extent=(img_x0, img_x0 + img_w,
                       img_y0 + img_h, img_y0)
    return image, extent

def get_concatenated_GUI_dataframes(measurement_type): 
    #concatenate all the GUI dataframes
    if measurement_type == 'raman': 
        path = fp.raman_GUI_dataframe_path
    if measurement_type == 'od':
        path = fp.od_GUI_dataframe_path
        
    file_names = mf.get_files_with_extension(path, 'csv')
    df_array = []
    for file_name in file_names: 
        df = pd.read_csv(path + '/' + file_name)
        df_array += [df]
    concat_df = pd.concat(df_array)
    return concat_df

# def get_study_GUI_dataframe(measurement_type, study_name): 
#     concat_df = get_concatenated_GUI_dataframes('raman')
#     study_filt = concat_df.loc[:, 'study_name'] == study_name
#     study_filt_concat_df = concat_df[study_filt]
#     return study_filt_concat_df

def get_raman_spectra_from_study(study_name): 
    study_filt_concat_df = get_study_GUI_dataframe('raman', study_name)
    txt_file_names = mf.get_files_with_extension(fp.raman_spectra_path, 'csv')
    study_spectra_df_array = []
    for txt_file_name in txt_file_names: 
        ID = txt_file_name[:-4]
        if ID in list(study_filt_concat_df.loc[:, 'ID']): 
            df = mf.read_raman_spectrum_df(fp.raman_spectra_path + '/' + txt_file_name)
            df.loc[:, 'ID'] = [ID]*df.shape[0]
            study_spectra_df_array += [df]
            
    study_spectra_df = pd.concat(study_spectra_df_array)
    study_df = pd.merge(study_filt_concat_df, study_spectra_df, how = 'left', on = 'ID')
    return study_df

# def convert_wdf_to_txt_files(path): 
#     wdf_files = mf.get_files_with_extension(path, '.wdf')
#     wdf_without_extension = [x[:-4] for x in wdf_files]
#     txt_files = mf.get_files_with_extension(path, '.txt')
#     txt_without_extension = [x[:-4] for x in txt_files]
    
#     for wdf_file in wdf_without_extension: 
#         if wdf_file not in txt_without_extension: 
#             print(wdf_file)
#             input_filepath = path + '/' + wdf_file + '.wdf'
#             output_filepath = path + '/' + wdf_file + '.csv'
#             subprocess.run(['wdf-export', input_filepath, '-o', output_filepath], check=True)


def get_unique_elements_in_order(arr):
    """
    Returns a list containing unique elements from the input array while maintaining the original order.
    Parameters:
        arr (list): The input list.
    Returns:
        list: A list containing unique elements from the input list while maintaining the original order.
    """
    # Initialize an empty list to store unique elements while maintaining order
    unique_elements_list = []
    # Iterate through the original list
    for element in arr:
        # Check if the element is not already in the unique elements list
        if element not in unique_elements_list:
            # Add the element to the unique elements list
            unique_elements_list.append(element)
    return unique_elements_list

def convert_wdf_to_txt_files(path): 
    wdf_files = mf.get_files_with_extension(path, '.wdf')
    wdf_without_extension = [x[:-4] for x in wdf_files]
    csv_files = mf.get_files_with_extension(path, '.csv')
    csv_without_extension = [x[:-4] for x in csv_files]
    
    for wdf_file in wdf_without_extension: 
        if wdf_file not in csv_without_extension: 
            print(wdf_file)
            input_filepath = path + '/' + wdf_file + '.wdf'
            output_filepath = path + '/' + wdf_file + '.csv'
            mf.convert_wdfmap_file_to_csv(input_filepath, output_filepath)


def on_hover(event, row, col, fig,canvas, df,x_points, y_points):
    # Clear the existing figure
    fig.clear()
    max_intensity = df.intensity.values.max()
    ax = fig.add_subplot(111)
    x_point = x_points[col]
    y_point = y_points[row]
    location_filter = (df['x'] == x_point) & (df['y'] == y_point)
    location_filt_df = df[location_filter]

    sns.lineplot(data = location_filt_df, x= 'wavenumber', y = 'intensity', ax = ax)
    ax.set_ylim((0, max_intensity))
    canvas.draw()
    
def create_map_gui(df):
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
            button.bind("<Enter>", lambda event, r=row, c=col: on_hover(event, r, c, fig, canvas, df,x_points,y_points))

    # Start the Tkinter event loop
    root.mainloop()


def find_closest_value(df, column_name, target_value):
    # Extract the specified column
    column_values = df[column_name]
    # Calculate the absolute differences between each element and the target value
    differences = (column_values - target_value).abs()
    # Find the value corresponding to the minimum difference
    closest_value = column_values.loc[differences.idxmin()]
    return closest_value

def draw_plot(x, df, mean_df, min_intensity, max_intensity): 
    plt.subplot(1,2,1)
    sns.lineplot(mean_df, x ='wavenumber', y = 'intensity')
    plt.vlines(x, min_intensity, max_intensity, colors='tab:red')

    plt.subplot(1,2,2)
    closest_wavenumber = find_closest_value(df, 'wavenumber', x)
    image = df[df['wavenumber'] == closest_wavenumber]
    image = image.drop('wavenumber', axis = 1)
    image = image.pivot('y', 'x', 'intensity')
    plt.imshow(image)
    plt.show()
    
def create_intensity_map(df): 
    mean_df = df.groupby('wavenumber', as_index = False)['intensity'].mean()
    min_wavenumber = mean_df['wavenumber'].min()
    max_wavenumber = mean_df['wavenumber'].max()
    min_intensity = mean_df['intensity'].min()
    max_intensity = mean_df['intensity'].max()
    widgets.interact(draw_plot, 
                         x=widgets.FloatSlider(min=min_wavenumber, max=max_wavenumber, step=50, value=min_wavenumber, continuous_update=False),
                     df = widgets.fixed(df),
                     mean_df = widgets.fixed(mean_df), 
                    min_intensity = widgets.fixed(min_intensity),
                    max_intensity = widgets.fixed(max_intensity))

def multivariate_gaussian_pdf(x, mean, covariance):
    n = len(mean)
    coefficient = 1
    exponent = -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean)
    return np.exp(exponent)

def calculate_gmm_probability(gmm_pipeline, coordinate): 
    gmm = gmm_pipeline['gmm']
    covariance_matrices = gmm.covariances_
    means = gmm.means_
    probability_array = []
    #for each class, calculate the probability
    for ii in range(4): 
        covariance_matrix = covariance_matrices[ii]
        mean = means[ii]
        probability = multivariate_gaussian_pdf(coordinate[0], mean, covariance_matrix)
        probability_array += [probability]
    sum = np.sum(np.array(probability_array))
    return np.array(probability_array)/sum

def predict_class_probability(vector_df):
    with open(fp.model_filepath + '/' + 'lda_pipeline.pkl', 'rb') as lda_file:
        lda_pipeline = pickle.load(lda_file)
    
    with open(fp.model_filepath + '/' + 'gmm_pipeline.pkl', 'rb') as gmm_file:
        gmm_pipeline = pickle.load(gmm_file)
    map = pd.read_csv(fp.intermediate_data_filepath + '/' + 'gmm_map.csv')
    X, y = mf.get_X_y_from_vector_df(vector_df)
    # Perform LDA transformation
    lda_transformed_data = lda_pipeline.transform(X.values)
    gmm_predictions = gmm_pipeline.predict_proba(pd.DataFrame(lda_transformed_data))
    probability_df = map.copy()
    probability_df = probability_df.sort_values('gmm_class_numerical')
    probability_df.loc[:, 'probability'] = calculate_gmm_probability(gmm_pipeline, lda_transformed_data)
    probability_df.loc[:, 'color'] =  probability_df['gmm_class'].map(variables.bacterial_colormap)
    return probability_df


def mahalanobis_distance(x, mean, cov):
    """
    Compute the Mahalanobis distance between a data point x and the mean of a class with covariance matrix cov.
    """
    x_minus_mean = x - mean
    cov_inv = np.linalg.inv(cov)
    distance = np.sqrt(np.dot(np.dot(x_minus_mean, cov_inv), x_minus_mean.T))
    return distance

def get_cluster_mean_and_covariance(lda_pipeline, vector_df, bacterial_class):
    """
    Determine the threshold for Mahalanobis distance-based outlier detection using the Interquartile Range (IQR) method.

    Parameters:
    - bacterial_class: The class label for which the threshold is being determined.
    - iqr_multiplier: A multiplier to adjust the threshold relative to the IQR.

    Returns:
    - threshold: The threshold for outlier detection based on Mahalanobis distances.
    """
    # Extract features (X) and labels (y) from the vector data
    X, y = mf.get_X_y_from_vector_df(vector_df)
    # Filter the data for the specified bacterial class
    class_filt = (y == bacterial_class)
    X_class_vector_df = X[class_filt]
    # Transform the data using the LDA pipeline
    class_lda_transformation = lda_pipeline.transform(X_class_vector_df.values)
    # Compute mean vector and covariance matrix of the transformed data
    mean = np.mean(class_lda_transformation, axis=0)
    covariance_matrix = np.cov(class_lda_transformation, rowvar=False)
    return mean, covariance_matrix, class_lda_transformation

def get_mahalanobis_distances(lda_pipeline, vector_df, bacterial_class): 
    mean, covariance_matrix, class_vector_df = get_cluster_mean_and_covariance(lda_pipeline, vector_df,bacterial_class)
    # Compute Mahalanobis distances for each data point
    mahalanobis_distances = []
    for point in class_vector_df:
        mahalanobis_distances.append(mahalanobis_distance(point, mean, covariance_matrix))
    return mahalanobis_distances

def get_mahalanobis_distance_threshold(lda_pipeline, vector_df, bacterial_class, iqr_multiplier): 
    # Calculate the first quartile (Q1) and third quartile (Q3) of the Mahalanobis distances
    mahalanobis_distances = get_mahalanobis_distances(lda_pipeline,vector_df, bacterial_class)
    Q1 = np.percentile(mahalanobis_distances, 25)
    Q3 = np.percentile(mahalanobis_distances, 75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Define the threshold as a multiple of the IQR
    threshold = Q3 + iqr_multiplier * IQR
    return threshold

def outlier_detection(vector_df,vector):
    with open(fp.model_filepath + '/' + 'lda_pipeline.pkl', 'rb') as lda_file:
        lda_pipeline = pickle.load(lda_file)
    point = lda_pipeline.transform(vector.values)
    classes = vector_df['Class'].unique()
    mahalanobis_distance_array = []
    for class_ in classes:
        mean, cov, a = get_cluster_mean_and_covariance(lda_pipeline, vector_df, class_)
        distance = mahalanobis_distance(point, mean, cov)
        mahalanobis_distance_array +=  [distance]
    
    distances_df = pd.DataFrame({'Class': classes, 
                 'calculated_distance': mahalanobis_distance_array})
    
    threshold_df = pd.read_csv(fp.intermediate_data_filepath +'/' + 'threshold_distance.csv')
    merged_df = pd.merge(distances_df, threshold_df, how = 'left', on = 'Class')
    if (merged_df['calculated_distance'] < merged_df['threshold']).sum() >= 1: 
        anamoly = False
    else: 
        anamoly = True
    return anamoly

def get_idx_peaks(df, averaging_window_size, height,prominence): 
    df = df.copy()
    # Smooth the intensity data
    df.loc[:,'smoothed_intensity'] = df.loc[:,'intensity'].rolling(window=averaging_window_size, center=True).mean()
    
    # Find peaks in the smoothed intensity data
    peaks, _ = find_peaks(df['smoothed_intensity'].values, height=height, prominence=prominence)
    return peaks


def train_model_and_get_results(vector_df, n_components):
    X, y = mf.get_X_y_from_vector_df(vector_df)
    # Splitting the data into training and testing sets with a 70-30 split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    
    #make the pipeline
    lda_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    #GMM pipeline
    gmm_pipeline = Pipeline([
        ('gmm', GaussianMixture(n_components=n_components))  # Adjust n_components for GMM
    ])
    
    #fit_transform the lda_df
    train_lda = pd.DataFrame(lda_pipeline.fit_transform(X_train, y_train))
    #fit_transform the lda_df
    test_lda = pd.DataFrame(lda_pipeline.transform(X_test))
    
    #make the train_lda_df
    train_lda_df = pd.DataFrame(lda_pipeline.fit_transform(X_train, y_train))
    #predict the train labels
    train_predictions = gmm_pipeline.fit_predict(train_lda_df)
    #make the test_lda_df
    test_lda_df = pd.DataFrame(lda_pipeline.transform(X_test))
    #make the test predictions
    test_predictions = gmm_pipeline.predict(test_lda_df)
    # Create a DataFrame to associate original labels with cluster assignments
    train_df = pd.DataFrame({'true_class': y_train, 'gmm_predicted_class': train_predictions})
    #create the test df
    test_df = pd.DataFrame({'true_class': y_test, 'gmm_predicted_class': test_predictions})

    #get the map based on majority voting
    map = train_df.groupby('true_class').apply(mf.majority_vote).reset_index().rename({
        'true_class': 'gmm_class',
        0: 'gmm_class_numerical'}, axis = 1)
    
    #now apply this majority map
    class_train_df = mf.convert_GMM_to_class_labels(train_df, map)
    class_test_df = mf.convert_GMM_to_class_labels(test_df, map)

    return class_train_df, class_test_df

def train_model_and_get_results_for_learning_curve(X_train, X_test, y_train, y_test, n_components):    
    #make the pipeline
    lda_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    #GMM pipeline
    gmm_pipeline = Pipeline([
        ('gmm', GaussianMixture(n_components=n_components))  # Adjust n_components for GMM
    ])
    
    #fit_transform the lda_df
    train_lda = pd.DataFrame(lda_pipeline.fit_transform(X_train, y_train))
    #fit_transform the lda_df
    test_lda = pd.DataFrame(lda_pipeline.transform(X_test))
    
    #make the train_lda_df
    train_lda_df = pd.DataFrame(lda_pipeline.fit_transform(X_train, y_train))
    #predict the train labels
    train_predictions = gmm_pipeline.fit_predict(train_lda_df)
    #make the test_lda_df
    test_lda_df = pd.DataFrame(lda_pipeline.transform(X_test))
    #make the test predictions
    test_predictions = gmm_pipeline.predict(test_lda_df)
    # Create a DataFrame to associate original labels with cluster assignments
    train_df = pd.DataFrame({'true_class': y_train, 'gmm_predicted_class': train_predictions})
    #create the test df
    test_df = pd.DataFrame({'true_class': y_test, 'gmm_predicted_class': test_predictions})

    #get the map based on majority voting
    map = train_df.groupby('true_class').apply(mf.majority_vote).reset_index().rename({
        'true_class': 'gmm_class',
        0: 'gmm_class_numerical'}, axis = 1)
    
    #now apply this majority map
    class_train_df = mf.convert_GMM_to_class_labels(train_df, map)
    class_test_df = mf.convert_GMM_to_class_labels(test_df, map)

    return class_train_df, class_test_df

def get_accuracy_and_f1(true_labels, predictions): 

    #calculate the accuracy
    accuracy = accuracy_score(true_labels, predictions)
    #calculate the f1 score
    f1 = f1_score(true_labels, predictions, average = 'macro')
    return accuracy, f1