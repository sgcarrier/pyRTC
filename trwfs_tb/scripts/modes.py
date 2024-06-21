#%%
import aotools
import aotools.functions.zernike
import numpy as np
# First we generate the modal basis we will use
numOfModes = 60 #Number of modes
actuators= 97
res = 9 # Resolution of the pupil. Here I use the number of actuators accross the diameter
bases,_,_,_ = aotools.functions.karhunenLoeve.make_kl(numOfModes, res, stf="kolmogorov")

#max_basis_value = 1
#bases = bases / np.max(np.abs(bases)) * max_basis_value
#for i in range(bases.shape[0]):
#    print( np.max(np.abs(bases[i,:,:])))
#    bases[i,:,:] = bases[i,:,:] / np.max(np.abs(bases[i,:,:])) * max_basis_value




base_mask = np.abs(bases[0,:,:]) > 0

#%%
def genMask(res, radius):
    xx,yy = np.meshgrid(np.arange(res),np.arange(res))
    zz = np.sqrt((xx-(res//2))**2 + (yy-(res//2))**2)
    actMap = np.zeros((res,res)).astype('bool')
    actMap[zz<=(radius)] = True
    return actMap
#%%
for i in range(97):
    a[i, ~mask] = 0
#%%
mask = small_mask[big_mask]
a_masked = np.zeros((97, 97))
for i in range(69):
    a_masked[i, :] = a[i,mask]
#%%

from astropy.io import fits as pyfits
hdul = pyfits.open('mat69_from_poke_influence_97_19juin.fits')
a = hdul[0].data

#%%
disp_a = np.zeros((11,11))
disp_a[small_mask] = a[5,:]
plt.imshow(disp_a)
plt.colorbar()
#%%

a_no_flat = a[1:,:]
for i in range(68):
    a_no_flat[i,:] -= a[0,:]
#%%
factor = 1/np.max(np.abs(a_no_flat)) * 1
a_norm = a_no_flat * factor
a_norm_no_piston = a_norm
#%%
a_final = np.zeros((68, 97))
flat_mask = small_mask[big_mask]
for i in range(68):
    a_final[i,flat_mask] = a_norm_no_piston[i,:]
#%%
disp_a = np.zeros((11,11))
disp_a[big_mask] = a_final[4,:]
plt.imshow(disp_a)
plt.colorbar()

#%%
np.save("new_M2C_68_JP_max1_19june_2.npy", a_final.T)

#%%
small_mask = genMask(11, 9/2)
big_mask = genMask(11, 11/2)
mask = small_mask[big_mask]

poke_mat = np.zeros((97, 69))

for i in range(69):
    tmp = np.zeros(69)
    tmp[i] = 1
    poke_mat[mask, i] =  tmp

#%%
np.save("M2C_POKE_69.npy", poke_mat)
#%%
small_mask = genMask(11, 9/2)
big_mask = genMask(11, 11/2)

M2C = np.zeros((actuators, numOfModes))
for m in range(numOfModes):
    tmp = np.zeros((11,11))
    tmp[small_mask] = bases[m,base_mask]
    M2C[:,m] = tmp[big_mask]


np.save("M2C_KL_60_interior.npy", M2C)
#%%



#%%

phase = np.zeros((11,11))
phase[big_mask] =  M2C[:,-1]
plt.figure()
plt.imshow(phase)
plt.colorbar()
plt.show()


#%%
def findEdgePixelIdx(dmRes):
    xx,yy = np.meshgrid(np.arange(dmRes),np.arange(dmRes))
    zz = np.sqrt((xx-(dmRes//2))**2 + (yy-(dmRes//2))**2)
    actMap = np.ones((dmRes,dmRes))*-1
    actMap[zz<=(dmRes/2)] = list(range(np.sum(zz<=(dmRes/2))))
    edgesID_all = actMap[zz>=(dmRes//2)]
    edgesIdx = (edgesID_all[edgesID_all != -1]).astype("int")
    return actMap, edgesIdx

print(findEdgePixelIdx(11))
actMap, edgeAct = findEdgePixelIdx(11)

asdads = actMap 


final_map1 = np.zeros(actMap.shape)
final_map2 = np.zeros(actMap.shape)
final_map3 = np.zeros(actMap.shape)

final_map1[actMap in edgeAct ] = 1
final_map2[actMap not in edgeAct ] = 2
final_map3[actMap==-1 ] = 0

sum_map = final_map1+final_map2+final_map3

#%%
#np.save("floating_actuactors.npy", edgeAct)

M2C_reduced = M2C.copy()
for idx in edgeAct:
    M2C_reduced[idx,:] = 0

M2C_reduced_less_modes = M2C_reduced #[:,:-len(edgeAct)]

np.save(f"M2C_KL_{97-len(edgeAct)}_edgeless_3.npy", M2C_reduced_less_modes)
# %%
