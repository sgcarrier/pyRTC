
import time
import numpy as np

def modWeightsFromIMCube(im_cube):

    numFrames =im_cube.shape[1]
    maxNumModes = im_cube.shape[2]
    weighting_cube = np.zeros((numFrames, maxNumModes))
    for i in range(maxNumModes):
        avg_val = np.mean(im_cube[:, :, i], axis=0)
        weighting_cube[:,i] = np.sqrt(((np.mean((im_cube[:, :, i]-avg_val[np.newaxis,:])**2, axis=0))))
        weighting_cube[:,i] = (weighting_cube[:,i]  / np.sum(np.abs(weighting_cube[:,i])))*numFrames

    return weighting_cube


def calcIM_cube(wfc, fsm, loop, maxNumModes=None):

    #slope.takeRefSlopes()
    wfc.flatten()

    fsm.stop()
    fsm.currentPos = None
    pos = {"A": 5.0, "B": 5.0}
    fsm.goTo(pos)
    time.sleep(1)
    IM_cube = pushPullIM_cube_2(loop, fsm, maxNumModes=maxNumModes)


    return IM_cube


def pushPullIM_cube(loop, fsm):
    IM_cube = np.zeros((loop.signalSize, 48, loop.numModes),dtype=loop.signalDType)

    ref_slopes = np.zeros((loop.signalSize, 48))
    fsm.currentPos = None

    # for s in range(len(fsm.points)):
    #     fsm.step()
    #     time.sleep(0.1)
    #     #Average out N new WFS frames
    #     tmp_plus =  np.zeros((loop.signalSize))
    #     for n in range(loop.numItersIM):
    #         tmp_plus += loop.wfsShm.read()
    #     tmp_plus /= loop.numItersIM

    #     #Compute the normalized difference
    #     ref_slopes[:,s] =tmp_plus


    #For each mode
    for i in range(10):
        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.1)
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(loop.IM[:,i])
            for n in range(loop.numItersIM):
                tmp_plus += loop.wfsShm.read()
            tmp_plus /= loop.numItersIM


            #plus_signal = tmp_plus/np.sum(tmp_plus) #- ref_slopes[:,s]/np.sum(ref_slopes[:,s])


        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.1)
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(loop.IM[:,i])
            for n in range(loop.numItersIM):
                tmp_minus += loop.wfsShm.read()
            tmp_minus /= loop.numItersIM


            # #Minus amplitude
            # correction[i] = -loop.pokeAmp
            # #Post a new shape to be made
            # loop.wfcShm.write(correction)
            # #Add some delay to ensure one-to-one
            # time.sleep(loop.hardwareDelay)
            # #Burn the first new image since we were moving the DM during the exposure
            # loop.wfsShm.read()
            # #Average out N new WFS frames
            # tmp_minus = np.zeros_like(loop.IM[:,i])
            # for n in range(loop.numItersIM):
            #     tmp_minus += loop.wfsShm.read()
            # tmp_minus /= loop.numItersIM

        #Compute the normalized difference
        IM_cube[:,s,i] = (tmp_plus-tmp_minus)/(2*loop.pokeAmp)

    return IM_cube

def pushPullIM_cube_2(loop, fsm, maxNumModes=None):

    numModFrames = len(fsm.points)

    if maxNumModes is None:
        maxNumModes = loop.numModes
    if maxNumModes > loop.numModes:
        maxNumModes = loop.numModes


    IM_cube = np.zeros((loop.signalSize, numModFrames, maxNumModes),dtype=loop.signalDType)


    ref_slopes = np.zeros((loop.signalSize, numModFrames))
    fsm.currentPos = None

    for s in range(len(fsm.points)):
        fsm.step()
        time.sleep(0.1)
        #Average out N new WFS frames
        ref_slopes[:,s] =  np.zeros((loop.signalSize))
        for n in range(loop.numItersIM):
            ref_slopes[:,s] += loop.wfsShm.read()
        ref_slopes[:,s] /= loop.numItersIM



    #For each mode
    for i in range(maxNumModes):
        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        tmp_plus =  np.zeros((loop.signalSize, 48))
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.01)
            #Average out N new WFS frames
            for n in range(loop.numItersIM):
                tmp_plus[:,s] += loop.wfsShm.read()
            tmp_plus[:,s] /= loop.numItersIM


            tmp_plus[:,s] = tmp_plus[:,s] - ref_slopes[:,s]



        #minus amplitude
        correction[i] = -loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        tmp_minus =  np.zeros((loop.signalSize, numModFrames))
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.01)
            #Average out N new WFS frames
            for n in range(loop.numItersIM):
                tmp_minus[:,s] += loop.wfsShm.read()
            tmp_minus[:,s] /= loop.numItersIM

            tmp_minus[:,s] = tmp_minus[:,s] - ref_slopes[:,s]


            # #Minus amplitude
            # correction[i] = -loop.pokeAmp
            # #Post a new shape to be made
            # loop.wfcShm.write(correction)
            # #Add some delay to ensure one-to-one
            # time.sleep(loop.hardwareDelay)
            # #Burn the first new image since we were moving the DM during the exposure
            # loop.wfsShm.read()
            # #Average out N new WFS frames
            # tmp_minus = np.zeros_like(loop.IM[:,i])
            # for n in range(loop.numItersIM):
            #     tmp_minus += loop.wfsShm.read()
            # tmp_minus /= loop.numItersIM

        #Compute the normalized difference
        IM_cube[:,:,i] = (tmp_plus-tmp_minus)/(2*loop.pokeAmp)

    return IM_cube

