#setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.units import eV, Ang, GPa
import sys
sys.path.append(r'files')
import Morse
calc = Morse.MorsePotential()

"""DELIVERABLE 1"""

#a function to compute the morse energy at a distance d
def morse_energy(d):
    """Function which returns the morse energy of a 
    placement of copper atoms given a distance d 
    between them"""
    a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., d)])
    a.set_calculator(calc)
    energy = a.get_potential_energy()
    return energy

#producing x and y values of energy vs distance plot
x = np.linspace(2.3, 5, 1000)
y = np.zeros(1000)
for i in range(len(x)):
    y[i] = morse_energy(x[i] * Ang)

#printing the distance at which the energy is lowest
min_energy_position = np.argmin(y)
print("Separation of lowest energy in Ang: ")
print(x[min_energy_position])

#plotting the energy vs distance
plt.plot(x, y)
plt.xlabel("Distance/Ang")
plt.ylabel("Energy.eV")
plt.title("Plot of the energy at different separations of nuclei")
plt.show()

"""DELIVERABLE 2"""

#function which calculates the magnitude of the force at a given distance d
def morse_force(d):
    a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., d)])
    a.set_calculator(calc)
    force = a.get_forces()
    #mag_force = np.sqrt(force[0][0]**2 + force[0][1]**2 + force[0][2]**2)
    z_force = force[1][2]
    return z_force

#producting an array of forces
f = np.zeros(1000)
for i in range(len(x)):
    f[i] = morse_force(x[i] * Ang)

#plotting force vs distance
plt.plot(x, f)
plt.xlabel("Distance/Ang")
plt.ylabel("Force/(eV/Ang)")
plt.title("Force between nuclei at different separations")
plt.show()

#producing the the gradient of the energy at 4Ang
def gradient_of_energy(epsilon):
    energy_gradient = -1 * (morse_energy((4 + epsilon) *Ang) - morse_energy(4 * Ang)) / epsilon
    return energy_gradient

#producing the calculated force at 4Ang
force = morse_force(4)

#producing an array of the error at different values of epsilon
epsilon_values = np.logspace(-5, -10, 50)
error_array = []
for i in epsilon_values:
    error_value = abs(force - gradient_of_energy(i))
    error_array.append(error_value)

#producing a logarithmic plot of the error against epsilon
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xscale("log")
plt.plot(epsilon_values, error_array)
plt.xlabel("epsilon")
plt.ylabel("Error")
plt.title("Logarithmic plot of the error for different sizes of epsilon")
plt.show()

#If epsilon becomes too small, the calculation no longer works as epsilon is so small that information is lost and the computer is effectively dividing by 0.
#There are rounding errors


"""Deliverable 3"""

#setup for lattice
from ase.build import bulk
cu = bulk("Cu", "fcc", a=3.6, cubic=True)
cu.cell
np.array(cu.cell)
cu.set_calculator(calc)

#function to find potential energy at given strain factor
def potential_energy(strain_factor):
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    cu.set_calculator(calc)
    cell = cu.get_cell()
    cell *= strain_factor
    cu.set_cell(cell, scale_atoms=True) # To apply strain, the atomic positions need to be scaled together with the unit cell 
    cu.get_cell()
    energy = cu.get_potential_energy() / cu.get_number_of_atoms()
    return energy

#plotting the potential energies at various different strain factors
strain_factors = np.linspace(0.95, 1.1, 50)
volume_axis = (3.6 * strain_factors)**3
energies = []
for i in strain_factors:
    energies.append(potential_energy(i))
plt.plot(volume_axis, energies)
plt.xlabel("Volume/Ang^3")
plt.ylabel("Energy/eV")
plt.title("Plot of the energy at different volumes due to various levels of lattice straining")
plt.show()

#function to find the pressure at a given strain factor
def pressure(strain_factor):
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    cu.set_calculator(calc)
    cell = cu.get_cell()
    cell *= strain_factor
    cu.set_cell(cell, scale_atoms=True) # To apply strain, the atomic positions need to be scaled together with the unit cell 
    cu.get_cell()
    pressure = -1 / 3 * (cu.get_stress(voigt=False)[0][0] + cu.get_stress(voigt=False)[1][1] + cu.get_stress(voigt=False)[2][2]) / (GPa)
    return pressure

#plotting the pressures at various different strain factos
pressures = []
for j in strain_factors:
    pressures.append(pressure(j))
plt.plot(volume_axis, pressures)
plt.xlabel("Volume/Ang^3")
plt.ylabel("Pressure/(GPa)")
plt.title("Pressures at different volumes due to varied straining")
plt.show()

#finding dPdV at minimum of energy plot and therefore Bulk Modulus
minimum_potential = min(energies)
minimum_volume_index = energies.index(minimum_potential)
minimum_volume = volume_axis[minimum_volume_index]
strain_at_minimum_volume = minimum_volume**(1/3) / 3.6
dPdV = (pressures[minimum_volume_index + 1] - pressures[minimum_volume_index]) / (volume_axis[minimum_volume_index + 1] - volume_axis[minimum_volume_index])
K = -1 * volume_axis[minimum_volume_index] * dPdV
print("Bulk modulus in GPa: ")
print(K)

#This is 25% away from the experimental value for the bulk modulus. 

"""Deliverable 4"""

#setup and calculation for shear modulus
applied_shear_strain = 0.01
L = 3.6
cu = bulk("Cu", "fcc", a=3.6, cubic=True)
cu.set_calculator(calc)
cell = cu.get_cell()
cell_matrix = np.array(cell)
cell_matrix[0,1] = applied_shear_strain * L
cu.set_cell(cell_matrix, scale_atoms=True)
cu.get_stress(voigt=False)
shear_modulus = cu.get_stress(voigt=False)[0, 1] /(2 * applied_shear_strain * GPa)
print("Shear Modulus in GPa: ")
print(shear_modulus)

#setup and calculation for poisson's ratio

#creating a range of strains in the yz direction and setup
strains_yz = np.linspace(-0.01, 0.01, 100)
stresses_yz = []
cu = bulk("Cu", "fcc", a=3.6, cubic=True)
cu.set_calculator(calc)
cell = cu.get_cell()
cell_matrix = np.array(cell)

#trying to find zero stress points
for strain in strains_yz:
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    #use equilibrium value of a
    cu.set_calculator(calc)
    cell = cu.get_cell()
    cell_matrix = np.array(cell)
    cell_matrix[0, 0] *= (1.01)
    cell_matrix[1, 1] *= (strain + 1)
    cell_matrix[2, 2] *= (strain + 1)

    cu.set_cell(cell_matrix, scale_atoms=True)
    stresses_yz.append(cu.get_stress(voigt=False)[1,1])

#using the fact that the zero points are where there are minimums in the absolute values
index_of_min = np.abs(stresses_yz).argmin()
poisson = -strains_yz[index_of_min]/0.01
print("Poisson Ratio: ")
print(poisson)

#The simple relationships are G = 3/8 * K and v = 1/3
G_simple = 3/8 * K
v_simple = 1/3

#Finding percentage uncertainties from experimental and simple values
G_dev_from_exp = (abs(46.5 - shear_modulus)) * 100 / 46.5
v_dev_from_exp = (abs(0.35 - poisson)) * 100 / 0.35

G_dev_from_simple = (abs(G_simple - shear_modulus)) * 100 / shear_modulus
v_dev_from_simple = (abs(v_simple - poisson)) * 100 / poisson

print("Percentage Deviation of shear modulus from experimental value: ")
print(G_dev_from_exp)
print("Percentage Deviation of poisson ratio from experimental value: ")
print(v_dev_from_exp)
print("Percentage Deviation of shear modulus from simple calculation: ")
print(G_dev_from_simple)
print("Percentage Deviation of poisson ratio from simple calculation: ")
print(v_dev_from_simple)

"""Deliverable 5"""

#The dislocations move from the notches across the lattice structure
#When the dislocations reach the edges, they cause plastic deformation of the whole structure.
#Over time, the notch becomes much thinner













