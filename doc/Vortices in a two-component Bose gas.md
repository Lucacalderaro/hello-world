
### Set-up


```python
%matplotlib inline
import numpy as np
import trottersuzuki as ts
from matplotlib import pyplot as plt
from __future__ import division # to carry out normal division with just one '/'.

from os.path import expanduser; home = expanduser("~")  # returns the home directory
directory = home + '/Test/vortices/'

dim = 150      # square lattice
length = 40.
delta_x = delta_y = length / dim
periods = [0, 0]      # open boundary conditions
kernel_type = "cpu"      #the evolution is performed by the CPU kernel

# use harmonic oscillator units
particle_mass_A = particle_mass_B = 1.
w_x = w_y = 1.   # frequencies of the harmonic confinement
rot_coord_x = rot_coord_y = dim / 2   # the axis of rotation will be located at the lattice center

TF_radius = length/4.   # wished TF radius, in harmonic oscillator units
coupling_const = np.pi * (TF_radius**4) / 4.   # this will be multiplying a wavefunction normalized to unity

# vectors containing the physical coordinates (in h.o. units, and centered in the lattice)
x_vec = y_vec = (np.arange(dim) - dim*0.5) * delta_x

# define the external potential
external_pot_A = np.zeros((dim, dim))
for i in range(0, dim):
    y = y_vec[i]
    for j in range(0, dim):
        x = x_vec[j]
        external_pot_A[i,j] = 0.5 * particle_mass_A * (w_x**2 * x**2 + w_y**2 * y**2)
external_pot_B = external_pot_A

# exports or displays a figure with the present density and phase
def heatmaps_of_density_and_phase(time, export, show, onlyDensity=False):
    density_A = ts.get_wave_function_density(p_real_A, p_imag_A)
    density_B = ts.get_wave_function_density(p_real_B, p_imag_B)
    if onlyDensity==True:
        fig = plt.figure(figsize=(9,4));
        ax1 = fig.add_subplot(121); 
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(density_A)
        ax2 = fig.add_subplot(122); 
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(density_B)
    else:
        phase_A = ts.get_wave_function_phase(p_real_A, p_imag_A)
        phase_B = ts.get_wave_function_phase(p_real_B, p_imag_B)
        fig = plt.figure(figsize=(19,4));
        ax1 = fig.add_subplot(141); 
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(density_A)
        plt.set_cmap('afmhot')
        ax2 = fig.add_subplot(142); 
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(density_B)
        ax3 = fig.add_subplot(143);
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(phase_A)
        plt.set_cmap('hsv')
        ax3 = fig.add_subplot(144);
        plt.xticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.yticks([0, dim/4, dim/2, 3*dim/4, dim],[str(-length/2), str(-length/4), '0', str(length/4), str(length/2)])
        plt.pcolor(phase_B)
        
    if export==True:
        fig.savefig(directory + ('imagTime' if imag_time==True else 'realTime') + '_plots-' + str(time) +'.png')
    if show==False:
        plt.close(fig)
    return

# exports to file the energies at a given time
def export_info(time):
    tot_energy = ts.calculate_total_energy_2GPE(p_real_A, p_imag_A, p_real_B, p_imag_B, particle_mass_A, particle_mass_B,
                                                [g_A, g_B, g_AB, Omega_Rabi, Omega_Rabi_imag], external_pot_A, external_pot_B,
                                                omega, rot_coord_x, rot_coord_y, delta_x, delta_y)
    file_info = open(directory + ('imagTime' if imag_time==True else 'realTime') + '_info.txt',"a")
    file_info.write(str(time) + #"\t" + str(rot_energy) +\
                                #"\t" + str(kin_energy) + 
                                "\t" + str(tot_energy) + "\n")
    file_info.close()
    
# this is the basic evolution step
def evolution_step():
    ts.evolve_2GPE(p_real_A, p_imag_A, p_real_B, p_imag_B, particle_mass_A, particle_mass_B, external_pot_A, external_pot_B,
                   omega, rot_coord_x, rot_coord_y, delta_x, delta_y, delta_t, iterations,
                   [g_A, g_B, g_AB, Omega_Rabi, Omega_Rabi_imag], kernel_type,
                   periods, imag_time)
```

# Dynamics in presence of Rabi coupling
 (starting with a vortex in BEC1, and no vortex in BEC2)

### Initial condition


```python
# width of the gaussian envelope
width = 4

p_real_A = np.zeros((dim,dim))
p_imag_A = np.zeros((dim,dim))
p_real_B = np.zeros((dim,dim))
p_imag_B = np.ones((dim,dim))

for i in range(0, dim):
    y = y_vec[i]
    for j in range(0, dim):
        x = x_vec[j]
        z = x + 1j*y
        val = np.exp(-(x**2 + y**2)/(2. * width**2))
        p_real_A[i, j] = np.real(val)
        p_imag_A[i, j] = np.imag(val)
        val = np.exp(-(x**2 + y**2)/(2. * width**2))
        p_real_B[i, j] = np.real(val)
        p_imag_B[i, j] = np.imag(val)

# normalize the initial condition
norm=np.sqrt(delta_x * delta_y * np.sum(np.abs(p_real_A)**2 + np.abs(p_imag_A)**2))
p_real_A=p_real_A/norm
p_imag_A=p_imag_A/norm
norm=np.sqrt(delta_x * delta_y * np.sum(np.abs(p_real_B)**2 + np.abs(p_imag_B)**2))
p_real_B=p_real_B/norm
p_imag_B=p_imag_B/norm

# plot the initial condition
imag_time = True
heatmaps_of_density_and_phase(time=0, export=False, show=True)
```

### Find the ground state (with no rotation, no Rabi, and with $g_{AB}\leq\sqrt{g_A g_B}$ so the BECs remain mixed)


```python
omega = 0.   # no rotation
imag_time = True    # True: imaginary time evolution; False: real time evolution
delta_t = 2.5e-4     #evolution time of a single iteration step
iterations = 1000   #number of iterations
start_it=1
max_it = 4
g_A = coupling_const
g_B = coupling_const
g_AB = g_A

Rabi_period = 20
Omega_Rabi = 0   # 2 * np.pi/Rabi_period
Omega_Rabi_imag = 0.


file_info = open(directory + ('imagTime' if imag_time==True else 'realTime') + '_info.txt',"w")
file_info.write("time\ttot_energy\n")#\tkin_energy\ttot_energy\n")
file_info.close()
export_info(0)

for cont in range(start_it-1, max_it):
    evolution_step()
    time=(cont + 1) * iterations * delta_t
    heatmaps_of_density_and_phase(time,export=False, show=True)
    #export_info(time)
    

p_real_A_0 = p_real_A.copy()
p_imag_A_0 = p_imag_A.copy()
p_real_B_0 = p_real_B.copy()
p_imag_B_0 = p_imag_B.copy()
```

With these initial conditions (no Rabi during imaginary time evolution, no vortices imprinted), a real Rabi frequency turned on at t=0 generates no dynamics.

A purely imaginary Rabi frequency (or, equivalently, a relative phase of $\pi$/2 between BEC-1 and BEC-2) instead generates periodic oscillations of the relative densities

# Imprint a vortex in BEC1


```python
p_real_A = p_real_A_0.copy()
p_imag_A = p_imag_A_0.copy()
p_real_B = p_real_B_0.copy()
p_imag_B = p_imag_B_0.copy()

p_real_A_before = p_real_A_0.copy()

for i in range(0, dim):
    y = y_vec[i]
    for j in range(0, dim):
        x = x_vec[j]
        z = x + 1j*y
        p_real_A[i, j] = p_real_A_before[i, j] * np.cos(np.angle(z))
        p_imag_A[i, j] = p_real_A_before[i, j] * np.sin(np.angle(z))

heatmaps_of_density_and_phase(time,export=False, show=True)
```

## Real time evolution


```python
imag_time = False    # True: imaginary time evolution; False: real time evolution
iterations = 400   #number of iterations
start_it=1
max_it = 25

file_info = open(directory + ('imagTime' if imag_time==True else 'realTime') + '_info.txt',"w")
file_info.write("time\ttot_energy\n")
file_info.close()
export_info(0)

Rabi_period = 20
Omega_Rabi = 2 * np.pi/Rabi_period
Omega_Rabi_imag = 0

time=0
for cont in range(start_it-1, max_it):
    heatmaps_of_density_and_phase(time,export=True, show=True)
    #export_info(time)
    evolution_step()
    print(ts.calculate_norm2(p_real_A,p_imag_A,delta_x,delta_y), ts.calculate_norm2(p_real_B,p_imag_B,delta_x,delta_y))
    time=(cont + 1) * iterations * delta_t

    
heatmaps_of_density_and_phase(time,export=True, show=False)
```

Nothing changes employing either a real or imaginary Rabi frequency. (apart maybe from a motion of the vortices, which results rotated by 90 degrees in the plane?)

This may be so, because it is not possible to define a "relative phase" between BEC-A and BEC-B, given that at t=0 BEC-A includes a vortex. 


```python

```
