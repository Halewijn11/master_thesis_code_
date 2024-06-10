import os
current_working_directory = os.getcwd()

if "\\Users\\mfeizpou\\OneDrive" in current_working_directory:
    working_directory =  "C:\\Users\\mfeizpou\\OneDrive - Vrije Universiteit Brussel\\Halewijn's Thesis Project\\master thesis"
else: 
    working_directory =  "C:\\Users\\User\\Vrije Universiteit Brussel\\Mehdi Feizpour - Halewijn's Thesis Project\\master thesis"
    
    
first_semester_directory = working_directory + '/' + 'first_semester'
second_semester_directory = working_directory + '/' + 'second_semester'

first_semester_data_directory = first_semester_directory + '/' + 'data'

other_directory = working_directory + '/' + 'other'
figure_filepath = other_directory + '/' + 'figures'
video_filepath = figure_filepath + '/' + 'videos'

first_semester_data_filepath = first_semester_directory + '/' + 'data'
model_filepath = first_semester_directory + '/' + 'models'
intermediate_data_filepath = first_semester_directory + '/' + 'intermediate_data'
simulation_data_filepath = first_semester_data_filepath + '/' + 'simulation_data'

od_path = second_semester_directory + '/' + 'od600_measurements'
od_spectra_path = od_path + '/' + 'spectra'
od_GUI_dataframe_path = od_path + '/' + 'GUI_dataframes'

raman_path = second_semester_directory + '/' + 'raman_measurements'
raman_spectra_path = raman_path + '/' + 'spectra'
raman_GUI_dataframe_path = raman_path + '/' + 'GUI_dataframes'
raman_study_dataframe_path = raman_path + '/' + 'study_dataframes'

second_semester_data_processing_path = second_semester_directory +'/' + 'data_processing'
second_semester_data_path = second_semester_data_processing_path + '/'  'data'
second_semester_model_path = second_semester_data_processing_path + '/' + 'models'

second_semester_simulation_data_path = second_semester_data_path + '/' + 'simulation_data'


microfluidic_GUI_dataframe_path = second_semester_directory + '/' + 'microfluidic_control' + '/' + 'microfluidic_GUI_dataframe'



