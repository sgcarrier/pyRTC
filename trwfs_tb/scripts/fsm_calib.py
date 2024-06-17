
# %%
import numpy as np
import matplotlib.pyplot as plt

t = modpsf.read()
ts = np.argsort(t)
t_flat= t.flatten()
ts = np.argsort(t_flat)[::-1]

percent = 0.5
indCutOff = int(ts.shape[0]*percent/100)

top_ts = ts[0:indCutOff]
top_t_flat = t_flat
top_t_flat[ts[indCutOff:]] = 0
t_new= top_t_flat.reshape(t.shape)

plt.figure()
plt.imshow(t_new)
plt.show()
# %%
# find middle point

X_center = 0
Y_center = 0

t_new_ones = t_new
t_new_ones[t_new_ones>0] = 1 

for x in range(t.shape[0]):
    X_center += np.sum(t_new_ones[x,:]*x)
X_center /= np.sum(t_new_ones)
X_center = int( X_center)

for y in range(t.shape[1]):
    Y_center += np.sum(t_new_ones[:,y]*y)
Y_center /= np.sum(t_new_ones)
Y_center = int( Y_center)

print(f"centerpoint is {X_center}, {Y_center}")


# %%
# find ratio in elipse lengths
low_x =0
high_x = t_new.shape[0]

low_y =0
high_y = t_new.shape[1]

slice_x = t[X_center, :]
slice_y = t[:, Y_center]

slice_x_sorted_1 = np.argsort(slice_x[:Y_center])
slice_x_sorted_2 = np.argsort(slice_x[Y_center:])
slice_y_sorted_1 = np.argsort(slice_y[:X_center])
slice_y_sorted_2 = np.argsort(slice_y[X_center:])

low_x = slice_x_sorted_1[-1]
high_x = slice_x_sorted_2[-1]+Y_center

low_y = slice_y_sorted_1[-1]
high_y = slice_y_sorted_2[-1]+X_center

print(f"axes points are ({low_x}, {Y_center}),({high_x}, {Y_center}),  ({X_center}, {low_y}),({X_center}, {high_y})")

points_x = [low_x, high_x, Y_center, Y_center]
points_y = [X_center, X_center, low_y, high_y]


plt.figure()
plt.imshow(t)
plt.scatter(points_x, points_y, c='r')
plt.show()


# %%

relative_factor = (high_x-low_x) / (high_y-low_y)

print(f"Relative factor between X and Y is {relative_factor}")

# %%


#%%
t_im = modpsf.read()
for i in range(9):
    t_im += modpsf.read()
t_im = t_im / 10.0
t_im_slice = t_im[257,:]

FWHM_lvl = np.max(t_im_slice)/2

FWHM = t_im_slice < FWHM_lvl



plt.figure()
plt.plot(t_im_slice[250:450])
plt.axhline(FWHM_lvl)
plt.show()
# %%
