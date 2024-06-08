import aotools
import aotools.functions.zernike
import numpy as np
# First we generate the modal basis we will use
numOfModes = 97 #Number of modes
res = 11 # Resolution of the pupil. Here I use the number of actuators accross the diameter
bases,_,_,_ = aotools.functions.karhunenLoeve.make_kl(numOfModes, res, stf="kolmogorov")

max_basis_value = 1
bases = bases / np.max(np.abs(bases)) * max_basis_value
#for i in range(bases.shape[0]):
#    bases[i,:,:] = bases[i,:,:] / np.max(np.abs(bases[i,:,:])) * max_basis_value


mask = np.abs(bases[0,:,:]) > 0
print(mask)

M2C = np.zeros((np.sum(mask), numOfModes))
for m in range(numOfModes):
    M2C[:,m] = bases[m,mask]


#np.save("M2C_KL_97.npy", M2C)

def findEdgePixelIdx(dmRes):
    xx,yy = np.meshgrid(np.arange(dmRes),np.arange(dmRes))
    zz = np.sqrt((xx-(dmRes//2))**2 + (yy-(dmRes//2))**2)
    actMap = np.ones((dmRes,dmRes))*-1
    actMap[zz<=(dmRes/2)] = list(range(np.sum(zz<=(dmRes/2))))
    edgesID_all = actMap[zz>=(dmRes//2)]
    edgesIdx = (edgesID_all[edgesID_all != -1]).astype("int")
    return actMap, edgesIdx

print(findEdgePixelIdx(11))
_, edgeAct = findEdgePixelIdx(11)

np.save("floating_actuactors.npy", edgeAct)

M2C_reduced = M2C.copy()
for idx in edgeAct:
    M2C_reduced[idx,:] = 0

M2C_reduced_less_modes = M2C_reduced[:,:-len(edgeAct)]

np.save(f"M2C_KL_{97-len(edgeAct)}_edgeless.npy", M2C_reduced_less_modes)