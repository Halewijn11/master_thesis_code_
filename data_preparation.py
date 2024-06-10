# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:18:22 2020

@author: Arash
"""

#%%imports
"""IMPORTS"""
#import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from pandas import DataFrame, read_csv, concat
from scipy.io import savemat











#%%initiation
"""Parameters"""
fileTypeRaman = -1
rangeSpec = range(0, 1015) #0-644 = full range

mainDirectory = 'C:\\Users\\mfeizpou\\OneDrive - Vrije Universiteit Brussel\\Desktop\\31-07-2023 CSF Bacteria Salma' #remember the two slashes!
os.chdir(mainDirectory)
if not os.path.exists('Figures'):
    os.makedirs('Figures')
    
#os.chdir(mainDirectory)
if not os.path.exists('Data'):
    os.makedirs('Data')
dataDir = mainDirectory + '\\Data'
os.chdir(dataDir)

#finding the data folder addresses
deviceFolders = [name for name in os.listdir(dataDir)] #if os.path.isfile(os.path.join(os.curdir, name))] 
# if deviceFolders[0].lower() == 'ftir':
#     FTIRFolder = dataDir + '\\' + deviceFolders[0]
#     RamanFolder = dataDir + '\\' + deviceFolders[1]
if deviceFolders[0].lower() == 'raman':
    #FTIRFolder = dataDir + '\\' + deviceFolders[1]
    RamanFolder = dataDir + '\\' + deviceFolders[0]
# dataFoldersAdsFTIR = [FTIRFolder + '\\' + name\
#                       for name in sorted(os.listdir(FTIRFolder))] #addresses sorted alphabetically based on plastic names
# dataLabelsFTIR = [name for name in sorted(os.listdir(FTIRFolder))] #addresses sorted alphabetically based on plastic names

dataFoldersAdsRaman = [RamanFolder + '\\' + name\
                      for name in sorted(os.listdir(RamanFolder))] #addresses sorted alphabetically based on plastic names
dataLabelsRaman = [name for name in sorted(os.listdir(RamanFolder))] #addresses sorted alphabetically based on plastic names


# FTIR_Coords = {  'PVC' : [ [[367,953] ,[387,963]] ,  [[418,828] ,[438,838]]   ,
#                             [[300,424] ,[320,434]] ,  [[163,648] ,[183,658]]   ,
#                             [[326,783] ,[346,793]] ,
#                             [[166,780] ,[176,800]] ,  [[573,520] ,[593,530]]   ,
#                             [[155,176] ,[175,186]] ,  [[280,1  ] ,[300,11 ]]   ,
#                             [[267,159] ,[287,169]] ,  [[395,1112] ,[415,1122]] ,
#                             [[307,1039] ,[317,1059]]                           ]        }
                

# FTIR_Coords = {   'PP'  : [ [[475,64]  ,[495,74]]  ,  [[570 ,591],[590 ,601]]  ,
#                             [[886,60]  ,[906,70]]  ,  [[698 ,122],[718 ,132]]  ,
#                             [[486,252] ,[496,272]] ,  [[400,139] , [420,149]]  ,
#                             [[565,135] ,[585,145]] ,  [[703,211] ,[723,221]]   ,
#                             [[912 ,298],[922,318]] ,  [[639,288] ,[659,298]]   ,
#                             [[805,170] ,[825,180]] ,  [[824,559] ,[844,569]]   ,
#                             [[865,467] ,[885,477]]                            ],
               
#                   'PET' : [ [[372,480] ,[382,500]] ,  [[344 ,27] ,[364 ,37]]   ,
#                             [[625,495] ,[645,505]] ,  [[572,206] ,[582,226]]   , 
#                             [[533,179] ,[543,199]] ,  [[113,263] ,[123,283]]   ,
#                             [[635,280] ,[645,300]] ,  [[652,314] ,[660,339]]   ,
#                             [[665,369] ,[675,389]] ,  [[221,157] ,[231,177]]   ,
#                             [[458,132] ,[478,142]] ,  [[454,145] ,[464,165]]   ,
#                             [[429,184] ,[439,204]] ,  [[70,412]  , [80,432]]   ,
#                             [[210,625] ,[218,650]] ,  [[295,680] ,[305,700]]   ,
#                             [[414,701] ,[424,721]] ,  [[531,662] ,[539,687]]   ,
#                             [[506,662] ,[526,672]] ,  [[217,656] ,[225,681]]   ,
#                             [[332,583] ,[357,591]]                             ],
                  
#                   'HDPE': [ [[842,643] ,[852,663]] ,  [[1041,422],[1051,442]]  ,
#                             [[917,439] ,[927,459]] ,  [[977,117] ,[997,127]]   ,
#                             [[272,581] ,[292,591]] ,  [[384,225] ,[404,235]]   ,
#                             [[285,477] ,[295,497]] ,  [[448,522] ,[468,532]]   ,
#                             [[854,287] ,[864,307]] ,  [[446,347] ,[456,367]]   ,
#                             [[353,481] ,[373,491]]                             ],
                  
#                   'PS'  : [ [[726,81]  ,[736,101]] ,  [[522,417] ,[542,427]]   ,
#                             [[320,380] ,[340,390]] ,  [[557,515] ,[577,525]]   ,
#                             [[434,580] ,[444,600]] ,  [[582,690] ,[602,700]]   ,
#                             [[605,590] ,[615,610]] ,  [[16 ,82 ] ,[36 , 92]]   ,
#                             [[361,61 ] ,[371,81 ]] ,  [[293,165] ,[313,175]]   ,
#                             [[29 ,635] ,[49 ,645]] ,  [[439,101] ,[459,111]]   ],
                  
#                   'PVC' : [ [[367,953] ,[387,963]] ,  [[418,828] ,[438,838]]   ,
#                             [[300,424] ,[320,434]] ,  [[163,648] ,[183,658]]   ,
#                             [[326,783] ,[346,793]] ,
#                             [[166,780] ,[176,800]] ,  [[573,520] ,[593,530]]   ,
#                             [[155,176] ,[175,186]] ,  [[280,1  ] ,[300,11 ]]   ,
#                             [[267,159] ,[287,169]] ,  [[395,1112] ,[415,1122]] ,
#                             [[307,1039] ,[317,1059]]                           ]} 

# FTIR_Coords = dict(sorted(FTIR_Coords.items())) # sorting the dictionary element alphabetically

data_name = 'Data-unprocessed' #the name with which to output the cleaned data



#%% Functions

"""Definitions"""

def DataFileCleaner(file, filType=4, decimal=','):
    
    if filType == 0:
        #Single Files in one file same spot
        data = read_csv(file, sep="\t", header = None, decimal=decimal, engine='python')
        data = data.drop([0], axis = 0)
        data = data.drop([3,4], axis = 1)
        data.columns = ['T','RS', 'I'] 
    
    elif filType == 1:
        #Single Files in one file
        data = read_csv(file, sep="\t", header = None, decimal=decimal, engine='python')
        data = data.drop([0], axis = 0)
        data = data.drop([3,4], axis = 1)
        data.columns = ['Z','RS', 'I']    
    elif filType == 2:
        #Single Files
        data = read_csv(file, sep="\t", header = None, decimal=decimal, engine='python')
        data = data.drop([0], axis = 0)
        data = data.drop([2], axis = 1)
        data.columns = ['RS', 'I']    
        
    elif filType == 3:
        #depth
        data = read_csv(file, sep="\t", header = None, decimal=decimal, engine='python')
        data = data.drop([0], axis = 0)
        data = data.drop([3,4], axis = 1)
        data.columns = ['Z', 'RS', 'I']  
    elif filType == 4:
        #MAP
        data = read_csv(file, sep="\t", header = None, decimal=decimal, engine='python')
        data = data.drop([0], axis = 0)
        data = data.drop([4,5,6], axis = 1)
        data.columns = ['X', 'Y', 'RS', 'I']
    elif filType == 5:
        #FTIR Maps
        data = read_csv(file, sep=";", header = None, decimal=decimal, engine='python')
        # data = data.drop([0], axis = 0)
        # data = data.drop([4,5,6], axis = 1)
        # data.columns = ['X', 'Y', 'RS', 'I']
    
    return(data) #lineListCleaned

def drop_rows_with_zeros(df):
    # Filter rows where all values are not equal to zero
    df = df[(df != 0).all(axis=1)].reset_index(drop=True)
    return df














#%% Body     
#Raman Data INPUT     
print('\n\nRaman data-cleaning is beginning..\n')
for htype in dataLabelsRaman:
    address = [name for name in dataFoldersAdsRaman if htype in name][0] #going over the plastic folders for Raman
    print('Current Raman folder: {}'.format(address))
    os.chdir(address)

    print(htype)
    
    print("Data being processed (Raman)):" +
          dataLabelsRaman[dataFoldersAdsRaman.index(address)] + '\n') # what plastic is being processed
    for file in os.listdir(address):
        # Finding file type
        firstElement = read_csv(file, sep="\t", header = None, decimal='.', engine='python').iloc[0,0]
        if firstElement == '#Time':
            fileTypeRaman = 0
        elif firstElement == '#Wave':
            fileTypeRaman = 2
        elif firstElement == '#X':
            fileTypeRaman = 4
            
        cleanedFileData = DataFileCleaner(file, fileTypeRaman, decimal='.') #Function Call
        if fileTypeRaman == 4:
            numSpectra = len((cleanedFileData['X']).astype(float).unique())*len(cleanedFileData['Y'].astype(float).unique()) # number of spectra 
        elif fileTypeRaman == 0:
            numSpectra = len(cleanedFileData['T'].unique()) # number of spectra in the file
        elif fileTypeRaman == 2:
            numSpectra = 1
        
        # numSpectra fail safe       
        if numSpectra != cleanedFileData.shape[0]//len(rangeSpec):
            print('Fail safe activated for number of spectra! fileName:')
            print(file)
            numSpectra = cleanedFileData.shape[0]//len(rangeSpec)
        
        dataDim = (numSpectra, len(rangeSpec))   
        data = cleanedFileData['I'].values.reshape(dataDim).astype(float)#.transpose()
        lambs = cleanedFileData['RS'].values.reshape(dataDim).astype(float) #all lambs
        if os.listdir(address).index(file) == 0 and\
            dataLabelsRaman.index(htype) == 0: #for the slight changes in Raman shift
            lambsVector = lambs[0,:] #as all lambs should be the same, 1 is enough
        # if fileTypeRaman == 4:
        #     Xs = cleanedFileData['X'].astype(float).unique()
        #     Ys = cleanedFileData['Y'].astype(float).unique()
        #     locLabels = np.array(np.meshgrid(Ys , Xs)).T.reshape(-1,2)
        # elif fileTypeRaman == 0:
        #     locLabels = np.zeros(numSpectra)
        # elif fileTypeRaman == 2:
        #     locLabels = [1]
            
        data = DataFrame(data, columns = lambsVector.round(1).astype(str))
        """
        The maps must be averaged for SERS
        """
        data = drop_rows_with_zeros(data) # dropping the saturated spectra before averaging the map
        data = data.mean(axis=0)
        data = DataFrame(data).T
        """"""
        #data['Raman Coordinate'] = [(locLabel[1], locLabel[0]) for locLabel in locLabels[:]]
        data['Class'] = [dataLabelsRaman[dataFoldersAdsRaman.index(address)]] * data.shape[0] #### numSpectra changed to data.shape[0] for SERS
        if os.listdir(address).index(file) == 0:
            data_address = data
        else:
            data_address = concat([data_address, data], ignore_index=True)
    if dataLabelsRaman.index(htype) == 0:
        all_data_Raman = data_address
    else:
        all_data_Raman = concat([all_data_Raman, data_address], ignore_index=True)

all_data_Raman = all_data_Raman[all_data_Raman.columns[::-1]]   #### changed for SERS

print('Raman files reading completed.\n\n\n')



# #%% FTIR INPUT
# print("\n\nFTIR data-cleaning began..\n")
# for ptype in FTIR_Coords.keys(): #going over the plastic types we have coordinates for
#     address = [name for name in dataFoldersAdsFTIR if ptype in name][0] #going over the plastic folders for Raman
#     print('\nCurrent FTIR folder: {}\n'.format(address))
#     os.chdir(address)
    
#     print(ptype)
    
#     #fileNamesFTIR = [name for name in os.listdir(os.curdir)]
#     for coord in FTIR_Coords[ptype]: #going over the plastic coordinates -> files: each coordinate gives a tileData
#         crossTileSwitch = 0
#         print(coord)
#         # map beginning coordinates
#         XtileBeg = (coord[0][0]+1)//128
#         if (coord[0][0]+1)%128 == 0:
#             XpixelBeg = 127
#             XtileBeg = (coord[0][0]+1)//128-1
#         else:
#             XpixelBeg = (coord[0][0]+1)%128 -1 # -1 because the #px starts from 0, 127: inversing the axes
#         YtileBeg = (coord[0][1]+1)//128
#         if (coord[0][1]+1)%128 == 0:
#             YpixelBeg = 127
#             YtileBeg = (coord[0][1]+1)//128-1
#         else:
#             YpixelBeg = (coord[0][1]+1)%128 -1 # -1 because the #px starts from 0, 127: inversing the axes
        
#         # beginning coordinates
#         XtileEnd = (coord[1][0]+1)//128
#         if (coord[1][0]+1)%128 == 0:
#             XpixelEnd = 127
#             XtileEnd = (coord[1][0]+1)//128-1
#         else:
#             XpixelEnd = (coord[1][0]+1)%128 -1 # -1 because the #px starts from 0, 127: inversing the axes
#         YtileEnd = (coord[1][1]+1)//128
#         if (coord[1][1]+1)%128 == 0:
#             YpixelEnd = 127
#             YtileEnd = (coord[1][1]+1)//128-1
#         else:
#             YpixelEnd = (coord[1][1]+1)%128 -1 # -1 because the #px starts from 0, 127: inversing the axes
        
#         #printing what pixels to look for. 
#         print('# # # Beginning pixel along X: {}\n Ending pixel along X: {} \n Beginning pixel along Y: {}\n Ending pixel along Y: {} # # #\n'.format(XpixelBeg, XpixelEnd, YpixelBeg, YpixelEnd))
#         #Checking if the map is over several tiles and the data should be extracted from multiple files
#         if XtileBeg != XtileEnd or YtileBeg != YtileEnd:
#             crossTileSwitch = 1 # the data goes over multiple tiles (files)
        
#         #if you do not have to go over tiles (files)    
#         if crossTileSwitch == 0: 
#             tileCoordName = 'xy_' + '0' * (4-len(str(XtileBeg))) + str(XtileBeg) +\
#                 '_' + '0' * (4-len(str(YtileBeg))) + str(YtileBeg) 
            
#             targetTileFileName = [name for name in os.listdir(os.curdir) if tileCoordName in name][0]
#             print('Target FTIR file name is: {}\n'.format(targetTileFileName))
            
#             targetTileData = DataFileCleaner('.\\' + targetTileFileName, 5, decimal=',') #Function Call
#             if ptype == list(FTIR_Coords.keys())[0] and coord == FTIR_Coords[ptype][0]:
#                 FTIRshifts = (targetTileData.iloc[0][1:])[::-1].reset_index(drop=True) # [1:] to remove the title, [::-1] to reverse
#                 FTIRshifts = [round(float( bb ), 1)\
#                               for bb in FTIRshifts]# changing , to . for decimals
#                 FTIRshifts_columns = [str(bb) + "f"\
#                               for bb in FTIRshifts]# changing , to . for decimals
#             targetTileData = targetTileData.drop([0], axis = 0)
#             #targetTileData = targetTileData.drop([0], axis = 1)
#             #targetTileData.columns =  np.linspace(0, targetTileData.shape[1]-1,\
#             #                                      targetTileData.shape[1]).astype(int)
#             targetTileData = targetTileData.reset_index(drop=True)
            
#             # FTIR tiles start from top right corner and move to left
#             for y in np.arange(YpixelBeg, YpixelEnd, dtype = int): #note that YpaxielBeg starts from 0, YpixelEnd will not be included
#                 rowBeg = int(targetTileData.shape[0] - (y * 128 + XpixelBeg)) 
#                 rowEnd = int(targetTileData.shape[0] - (y * 128 + XpixelEnd)) 
#                 data_line = targetTileData[rowEnd:rowBeg] # rowBeg is not included in this range, note that rowEnd<rowBeg
#                 data_line = data_line.iloc[::-1] # reversing the rows order
#                 #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                 #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                 if y == YpixelBeg:
#                    data_tile = data_line
#                 else:
#                    data_tile = concat([data_tile, data_line], ignore_index=True)
               
#         else: #in case you have to go over tiles: if crossTileSwitch == 1
#             """in case you have to go over tiles""" 
#             tileCoordNameBEG = 'xy_' + '0' * (4-len(str(XtileBeg))) + str(XtileBeg) +\
#                 '_' + '0' * (4-len(str(YtileBeg))) + str(YtileBeg)    
#             tileCoordNameEND = 'xy_' + '0' * (4-len(str(XtileEnd))) + str(XtileEnd) +\
#                 '_' + '0' * (4-len(str(YtileEnd))) + str(YtileEnd)
                
#             # inputting the beginning and ending tiles
#             #beginning
#             targetTileFileNameBEG = [name for name in os.listdir(os.curdir) if tileCoordNameBEG in name][0]
#             print('Target FTIR file name (BEGINNING) is: {}\n'.format(targetTileFileNameBEG))
            
#             targetTileDataBEG = DataFileCleaner('.\\' + targetTileFileNameBEG, 5, decimal=',') #Function Call
            
#             if ptype == list(FTIR_Coords.keys())[0] and coord == FTIR_Coords[ptype][0]:
#                 FTIRshifts = (targetTileDataBEG.iloc[0][1:])[::-1].reset_index(drop=True) # [1:] to remove the title, [::-1] to reverse
#                 FTIRshifts = [round(float( bb ), 1)\
#                               for bb in FTIRshifts]# changing , to . for decimals
#                 FTIRshifts_columns = [str(bb) + "f"\
#                               for bb in FTIRshifts]# changing , to . for decimals
            
#             targetTileDataBEG = targetTileDataBEG.drop([0], axis = 0)
#             #targetTileData = targetTileData.drop([0], axis = 1)
#             #targetTileData.columns =  np.linspace(0, targetTileData.shape[1]-1,\
#             #                                      targetTileData.shape[1]).astype(int)
#             targetTileDataBEG = targetTileDataBEG.reset_index(drop=True)
            
#             #Ending
#             targetTileFileNameEND = [name for name in os.listdir(os.curdir) if tileCoordNameEND in name][0]
#             print('Target FTIR file name (ENDING) is: {}\n'.format(targetTileFileNameEND))
#             targetTileDataEND = DataFileCleaner('.\\' + targetTileFileNameEND, 5, decimal=',') #Function Call
#             """The next 5 lines are commented bc we need the FTIR shifts only once (they are the same for all spectra)"""
#             # FTIRshiftsEND = targetTileDataEND.iloc[0][1:] # [1:] to remove the title, [::-1] to reverse
#             # FTIRshiftsEND = [round(float( bb ), 1)\
#             #               for bb in FTIRshiftsEND]# changing , to . for decimals
#             # FTIRshifts_columnsEND = [str(bb) + "f"\
#             #               for bb in FTIRshiftsEND]# changing , to . for decimals
#             targetTileDataEND = targetTileDataEND.drop([0], axis = 0)
#             #targetTileData = targetTileData.drop([0], axis = 1)
#             #targetTileData.columns =  np.linspace(0, targetTileData.shape[1]-1,\
#             #                                      targetTileData.shape[1]).astype(int)
#             targetTileDataEND = targetTileDataEND.reset_index(drop=True)         
            
            
#             if XtileBeg != XtileEnd and not YtileBeg != YtileEnd: #if the map extension is only horizontal
#                 print('\n <<Going over two tiles horizontally>> \n')
#                 # FTIR tiles start from top right corner and move to left
#                 for y in np.arange(YpixelBeg, YpixelEnd, dtype = int): #note that YpaxielBeg starts from 0
#                     rowBegBEG = int(targetTileDataBEG.shape[0] - (y * 128 + XpixelBeg)) # targetTileDataBEG.shape[0] = 128*128
#                     rowEndBEG = int(targetTileDataBEG.shape[0] - (y * 128 + 127+1)) # +1 because the 127th element is in the range and should be included
#                     rowBegEND = int(targetTileDataEND.shape[0] - (y * 128 + 0)) # targetTileDataEND.shape[0] = 128*128
#                     rowEndEND = int(targetTileDataEND.shape[0] - (y * 128 + XpixelEnd))
#                     data_line = concat([targetTileDataEND[rowEndEND:rowBegEND].reset_index(drop = True), # reset index to fix the indices in place
#                                        targetTileDataBEG[rowEndBEG:rowBegBEG].reset_index(drop = True)], # reset index to fix the indices in place
#                                        ignore_index=True) # rowBeg is not included in this range, note that rowEnd<rowBeg
#                     data_line = data_line.iloc[::-1] # reversing the rows order
#                     #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                     #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                     if y == YpixelBeg:
#                        data_tile = data_line
#                     else:
#                        data_tile = concat([data_tile, data_line], ignore_index=True)

#             elif YtileBeg != YtileEnd and not XtileBeg != XtileEnd: # YtileBeg != YtileEnd, i.e. if the map extension is only normal
#                 print('\n <<Going over two tiles vertically>> \n')
#                 # FTIR tiles start from top right corner and move to left
#                 for y in np.arange(YpixelBeg, 127+1, dtype = int): #note that YpaxielBeg starts from 0, +1 to include 127 as well
#                     rowBeg = int(targetTileDataBEG.shape[0] - (y * 128 + XpixelBeg)) 
#                     rowEnd = int(targetTileDataBEG.shape[0] - (y * 128 + XpixelEnd)) # targetTileDataEND.shape[0] = 128*128
#                     data_line = targetTileDataBEG[rowEnd:rowBeg]
#                     data_line = data_line.iloc[::-1] # reversing the rows order
#                     #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                     #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                     if y == YpixelBeg:
#                        data_tile = data_line
#                     else:
#                        data_tile = concat([data_tile, data_line], ignore_index=True)
#                 # for the y values in the second tile file
#                 for y in np.arange(0, YpixelEnd, dtype = int): #note that YpaxielBeg starts from 0
#                     rowBeg = int(targetTileDataEND.shape[0] - (y * 128 + XpixelBeg))  
#                     rowEnd = int(targetTileDataEND.shape[0] - (y * 128 + XpixelEnd))  # targetTileDataEND.shape[0] = 128*128
#                     data_line = targetTileDataBEG[rowEnd:rowBeg]
#                     data_line = data_line.iloc[::-1] # reversing the rows order
#                     #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                     #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                     data_tile = concat([data_tile, data_line], ignore_index=True)
                       
#             else: #in case the tile is over 4 tiles, raise an error
#                 print("\n\n<<<<<<<Warning: Four tiles are involved in this map.>>>>>>>\n\n")
#                 """in case you have to go over 4 tiles, you will need 2 more tiles in addition to the ones defined earlier""" 
#                 tileCoordNameBEG1 = 'xy_' + '0' * (4-len(str(XtileEnd))) + str(XtileEnd) +\
#                     '_' + '0' * (4-len(str(YtileBeg))) + str(YtileBeg)    
#                 tileCoordNameEND0 = 'xy_' + '0' * (4-len(str(XtileBeg))) + str(XtileBeg) +\
#                     '_' + '0' * (4-len(str(YtileEnd))) + str(YtileEnd)
                    
#                 # inputting the beginning and ending tiles
#                 #beginning
#                 targetTileFileNameBEG1 = [name for name in os.listdir(os.curdir) if tileCoordNameBEG1 in name][0]
#                 print('Target FTIR file name (BEGINNING) is: {}\n'.format(targetTileFileNameBEG1))
                
#                 targetTileDataBEG1 = DataFileCleaner('.\\' + targetTileFileNameBEG1, 5, decimal=',') #Function Call
#                 """The next 5 lines are commented bc we need the FTIR shifts only once (they are the same for all spectra)"""
#                 # FTIRshiftsBEG1 = targetTileDataBEG1.iloc[0][1:] # [1:] to remove the title, [::-1] to reverse
#                 # FTIRshiftsBEG1 = [round(float( bb ), 1)\
#                 #               for bb in FTIRshiftsBEG1]# changing , to . for decimals
#                 # FTIRshifts_columnsBEG = [str(bb) + "f"\
#                 #               for bb in FTIRshiftsBEG1]# changing , to . for decimals
#                 targetTileDataBEG1 = targetTileDataBEG1.drop([0], axis = 0)
#                 #targetTileData = targetTileData.drop([0], axis = 1)
#                 #targetTileData.columns =  np.linspace(0, targetTileData.shape[1]-1,\
#                 #                                      targetTileData.shape[1]).astype(int)
#                 targetTileDataBEG1 = targetTileDataBEG1.reset_index(drop=True)
                
#                 #Ending
#                 targetTileFileNameEND0 = [name for name in os.listdir(os.curdir) if tileCoordNameEND0 in name][0]
#                 print('Target FTIR file name (ENDING) is: {}\n'.format(targetTileFileNameEND0))
#                 targetTileDataEND0 = DataFileCleaner('.\\' + targetTileFileNameEND0, 5, decimal=',') #Function Call
#                 """The next 5 lines are commented bc we need the FTIR shifts only once (they are the same for all spectra)"""
#                 # FTIRshiftsEND0 = targetTileDataEND0.iloc[0][1:] # [1:] to remove the title, [::-1] to reverse
#                 # FTIRshiftsEND0 = [round(float( bb ), 1)\
#                 #               for bb in FTIRshiftsEND0]# changing , to . for decimals
#                 # FTIRshifts_columnsEND0 = [str(bb) + "f"\
#                 #               for bb in FTIRshiftsEND0]# changing , to . for decimals
#                 targetTileDataEND0 = targetTileDataEND0.drop([0], axis = 0)
#                 #targetTileData = targetTileData.drop([0], axis = 1)
#                 #targetTileData.columns =  np.linspace(0, targetTileData.shape[1]-1,\
#                 #                                      targetTileData.shape[1]).astype(int)
#                 targetTileDataEND0 = targetTileDataEND0.reset_index(drop=True)  
#                 # FTIR tiles start from top right corner and move to left
#                 for y in np.arange(YpixelBeg, 127+1, dtype = int): #note that YpaxielBeg starts from 0
#                     rowBegBEG =  int(targetTileDataBEG.shape[0] - (y * 128 + XpixelBeg)) # targetTileDataBEG.shape[0] = 128*128
#                     rowEndBEG =  int(targetTileDataBEG.shape[0] - (y * 128 + 127+1)) # +1 because the 127th element is in the range and should be included
#                     rowBegBEG1 = int(targetTileDataBEG1.shape[0] - (y * 128 + 0)) # targetTileDataEND.shape[0] = 128*128
#                     rowEndBEG1 = int(targetTileDataBEG1.shape[0] - (y * 128 + XpixelEnd))
#                     data_line = concat([targetTileDataBEG1[rowEndBEG1:rowBegBEG1].reset_index(drop = True), # reset index to fix the indices in place
#                                        targetTileDataBEG[rowEndBEG:rowBegBEG].reset_index(drop = True)], # reset index to fix the indices in place
#                                        ignore_index=True) # rowBeg is not included in this range, note that rowEnd<rowBeg
#                     data_line = data_line.iloc[::-1] # reversing the rows order
#                     #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                     #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                     if y == YpixelBeg:
#                        data_tile = data_line
#                     else:
#                        data_tile = concat([data_tile, data_line], ignore_index=True)
#                 # for the y values in the second tile file
#                 for y in np.arange(0, YpixelEnd, dtype = int): #note that YpaxielBeg starts from 0
#                     rowBegEND0 = int(targetTileDataEND0.shape[0] - (y * 128 + XpixelBeg)) # targetTileDataBEG.shape[0] = 128*128
#                     rowEndEND0 = int(targetTileDataEND0.shape[0] - (y * 128 + 127+1)) # +1 because the 127th element is in the range and should be included
#                     rowBegEND =  int(targetTileDataEND.shape[0] - (y * 128 + 0)) # targetTileDataEND.shape[0] = 128*128
#                     rowEndEND =  int(targetTileDataEND.shape[0] - (y * 128 + XpixelEnd)) 
#                     data_line = concat([targetTileDataEND[rowEndEND:rowBegEND].reset_index(drop = True), # reset index to fix the indices in place
#                                            targetTileDataEND0[rowEndEND0:rowBegEND0].reset_index(drop = True)], # reset index to fix the indices in place
#                                           ignore_index=True) # rowBeg is not included in this range, note that rowEnd<rowBeg
#                     data_line = data_line.iloc[::-1] # reversing the rows order
#                     #data_line = data_line[data_line.columns[::-1]] # reversing the columns order to have low to high
#                     #data_line.columns = data_line.columns[::-1] # to reset the column numbers
#                     data_tile = concat([data_tile, data_line], ignore_index=True)
#         # after if crossTileSwitch == 0 or else
#         data_tile = data_tile.loc[::-1].reset_index(drop = True) #reverse, reset index to fix the indices in place
#         FTIRmapPixels = data_tile.pop(0)#data_tile.shape[1]-1) # the coordinates in respective tiles
#         data_tile.columns = FTIRshifts_columns #data_tile.reset_index(drop=True)  
#         data_tile['FTIR Map Pixels'] = FTIRmapPixels     
#         if coord == FTIR_Coords[ptype][0]:
#             print('current coord: {}\n <<0>>\n'.format(coord))
#             data_material = data_tile
#         else:
#             print('current coord: {}\n <<1>>'.format(coord))
#             print('data_tile: ', data_tile.shape)
#             print('data_material: ', data_material.shape)
#             data_material = concat([data_material, data_tile], ignore_index=True)
#             print(data_material.shape,'\n\n')
        
#     data_material = data_material.reset_index(drop = True)
#     if list(FTIR_Coords.keys()).index(ptype) == 0:
#         all_data_FTIR = data_material
#     else:
#         all_data_FTIR = concat([all_data_FTIR, data_material], ignore_index=True)
#     print("{} MPs are read in. \n\n\n".format(ptype))

    
# #%% Fusion
# print('\n\nData fusion has began.')
# all_data = concat([all_data_Raman, all_data_FTIR], axis = 1)
# all_data.insert(0, 'FTIR Map Pixels', all_data.pop('FTIR Map Pixels'))
fusedShifts = DataFrame(lambsVector)[::-1].transpose()
fusedShifts.columns = [str(int(pq)) for pq in np.arange(len(lambsVector), dtype=int)]

os.chdir(mainDirectory)
#all_data.to_excel("{}.xlsx".format(data_name)) 
all_data_Raman.to_feather("{}".format(data_name)) 
#savemat('Raman_Data_Matlab_input', all_data_Raman) # Mat file for inputing into MAtlab
fusedShifts.to_feather("{}".format(data_name+'_shifts')) 



