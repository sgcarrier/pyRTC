# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""

import numpy as np
from os import path
from OOPAO.tools.tools  import createFolder


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #everything needs to be in units of D/x
    param['r0'                   ] = 0.045                                       # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [1.0]                                            # Cn2 profile
    param['windSpeed'            ] = [7]                                          # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [90]                                            # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0]                                          # altitude of the different layers in [m]
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 1.2                                             # diameter in [m]
    param['nSubaperture'         ] = 17                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 2                                          # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0                                             # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                             # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 1                                   # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 4.0                                             # magnitude of the guide star
    param['opticalBand'          ] = 'R'                                            # optical band of the guide star
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1                                             # number of actuators
    param['mechanicalCoupling'   ] = 0.3
    param['isM4'                 ] = False                                           # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    param['influence_fnt_filename']= '/Users/wasi/Desktop/ML_PyRTC/pyRTC/trwfs_tb/res/influence_fnts_mask_june19_2024.npy'
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    param['dmPupil'            ] = 1                                    # physical size of DM pupil in units of m. So ALPAO dm97 should be 13.5mm


    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 3                                            # modulation radius in ratio of wavelength over telescope diameter
    param['pupilSeparationRatio'  ] = 1.2                                           # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['calibrationModulation' ] = 20                                            # modulation radius used to select the valid pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['edgePixel'             ] = 4                                             # number of pixel on the external edge of the PWFS pupils
    param['extraModulationFactor' ] = 4                                             # factor to add/remove 4 modulation points (one for each PWFS face)
    param['nTheta_user_defined'   ] = 48                                            # Number for frames taken during the modulation. Must be a multiple of 4 for the pyramid
    param['postProcessing'        ] = 'fullFrame'                                   # post-processing of the PWFS signals
    param['unitCalibration'       ] = False                                         # calibration of the PWFS units using a ramp of Tip/Tilt    

    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 1000                                           # number of iteration
    param['photonNoise'           ] = True                                         # Photon Noise enable
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.4                                          # integrator gain
    param['nModes'                ] = 1200                                          # number of KL modes controlled
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'AO_lab_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param
