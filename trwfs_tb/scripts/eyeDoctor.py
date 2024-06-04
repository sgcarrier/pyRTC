import numpy as np
import time

def eyeDoctorPSF(wfc, psf, maxModeToSharpen=25, stepsPerMode=11, maxStepVal=0.01, numImagesToStack=5, numPixelsToSharpen=8, saveFilename=None):

    offset = np.zeros(wfc.numActuators)

    modeSteps = np.linspace(-1, 1, stepsPerMode) * maxStepVal
    imStack = np.zeros((psf.imageShape[0], psf.imageShape[1], numImagesToStack))
    maxInt = np.zeros(stepsPerMode)


    for mode in range(2, maxModeToSharpen):
        for s in range(stepsPerMode):
            wfc.flatten()
            #wfc.write(offset)
            # Apply to DM
            to_apply = offset
            to_apply[mode] =  modeSteps[s]
            print(f"to apply = {to_apply}")
            #wfc.push(mode, modeSteps[s])
            wfc.write(to_apply)

            print(f"Max of current shape:{np.max(np.abs(wfc.currentShape))}")
            time.sleep(0.1)
            # Grab PSF frames 
            for i in range(numImagesToStack):
                psf.expose()
                im = psf.read()
                imStack[:,:, i] = im
            ima = np.mean(imStack, axis=2)
            ima_flat = ima.flatten()
            idx = np.argsort(-ima_flat)
            sorted_values = ima_flat[idx]
            maxInt[s] = np.sum(sorted_values[:numPixelsToSharpen])

            # Average frame stack
        time.sleep(0.5)

        # determine best coeff
        idx = np.argmax(maxInt)
        # Add to the offset 
        offset[mode] = modeSteps[idx]

    print(offset)
    wfc.write(offset)
    time.sleep(1)

    psf.expose()
    im = psf.read()

    print(f"Final Shape: {wfc.currentShape}")
    if saveFilename is not None:
        np.save(saveFilename, wfc.currentShape)