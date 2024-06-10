random_state = 42
x_y_label_fontsize = 16
tick_fontsize = 16
raman_wavenumber_fontsize = 12
legend_fontsize = 16


scatterplot_size = 15
bacterial_colormap = {
    'E coli': 'tab:blue', 
    'H influenza': 'tab:orange',
    'N meningitidis': 'tab:green',
    'S pneumoniae': 'tab:red'
}

new_bacterial_colormap = {
    'E coli': 'tab:red', 
    'P vulneris': 'tab:orange',
    'S epidermis': 'tab:blue'
}

italics_bacterial_colormp = {'$\it{E}$' + ' ' + '$\it{coli}$': 'tab:red', 
                            '$\it{S}$' + ' ' + '$\it{epidermis}$':'tab:orange', 
                            '$\it{P}$' + ' ' + '$\it{vulneris}$':'tab:blue' }


preprocessing_params = {'window_length': 15, 
                            'polyorder':3, 
                       'normalization': False, 
                       'cosmic_ray_removal':True,
                       'cosmic_ray_z_score': 8}