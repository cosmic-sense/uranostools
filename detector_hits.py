# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis of detector hits from URANOS output
#
# The is an example script to read and plot URANOS output from a physical detector of a single scenario to examine each individual neutron.
#
# *-- last modified: June 29, 2022*

# %%
# import the URANOS class which can deal with all the details on its own
from lib.uranos import URANOS

# %% [markdown]
# ## 0. TL;DR
# This tutorial uses only a few short-hand functions to read in URANOS output. Here is the compacted list without explanations. Jump to the next section go through the more elaborate tutorial.

# %%
# Read
U = URANOS(folder='example_uranos_output/complex_fields/')
U = U.read_hits().only_soil_contact().drop_multicounts()
U = U.weight_by_detector_response(method='1/sqrt(E)')
# Plot
U.plot_xy_hits()
U.plot_z_hits()
U.footprint_by_hits('r')
U.depth_distribution(var='z_max')
U.plot_angle_of_origin(polar=True)

# %% [markdown]
# ## 1. Setup the URANOS object
# Assuming that you have all the URANOS output files in one folder, the URANOS class can automatically recognize the detector hits file based on its name. Hence, to initialize the object, it is sufficient to provide only the folder name. Other options for initialization are:
# - `folder` - folder which contains the density maps, origin files, png image, etc.
# - `scaling=2` - one pixel of the matrix corresponds to how many meters? Defaults to 2, i.e., a 500x500 matrix represents a 1000x1000 m² area
# - `default_material` - dominant material that is used as region number 0. If not provided, tries to guess the dominant material of the environment by looking at the central pixel (0,0).
# - `hum=5, press=1013` - air humidity (g/m³) and air pressure (hPa) used in the simulation, will be used for further internal calculations.

# %%
# Create URANOS object based on the given folder
U = URANOS(folder='example_uranos_output/complex_fields/')

# %% [markdown]
# Now read the file (defaults to 'detectorNeutronHitData.dat') containing all the neutrons which got detected by the central detector. Since we are only interested in neutrons with soil contact and we want to count each wandering neutron only once, we also filter the data right away.

# %%
# Filters only neutrons with soil contact = 1 and z > 0 
# Drop duplicate counts
U = U.read_hits().only_soil_contact().drop_multicounts()

# %%
# This is how the neutron hits DataFrame looks like:
U.Hits

# %% [markdown]
# As seen above, the initialization already prepares useful short-hand columns for certain variables:
# - `x` - equals 'x_at_interface_[m]' (first soil contact)
# - `y` - equals 'y_at_Interface_[m]' (first soil contact)
# - `z` - equals 'z_at_Interface_[m]' (first soil contact)
# - `z_max` - equals 'maximum_Depth_[m]'
# - `w` - angle of origin: `np.arctan2(y, x) + 3.141`
# - `w_deg` - w in degrees
# - `r` -  Distance to origin: `np.sqrt(x**2 + y**2)`
# - `ex` - equals 'previous_x_[m]' (last position before detection)
# - `ey` - equals 'previous_y_[m]' (last position before detection)
# - `we` - angle of entry
# - `we_deg` - we in degrees
# - `E` - equals 'Energy_[MeV]'
# - `thermal` - thermal neutron flag, `E < 0.5e-6`

# %% [markdown]
# ## 2. Energy weighting
# Depending on the detector and the URANOS configuration used, the detected neutrons might need to be weighted by their probability to get detected by a real detector. This can be done by a realistic detector response function, or a assuming the standaed $1/\sqrt{E}$ weighting for bare counters.
#
# Function `U.weight_by_detector_response(method, file)` with the options:
# - `method='1/sqrt(E)'` - standard 1/sqrt(E) weighting for bare counters
# - `method='drf'` - use a detector response function, specify a file
# - `file` - csv file that contains two columns: Energy and Weight.

# %%
U = U.weight_by_detector_response(method='1/sqrt(E)')

# %% [markdown]
# How relevant is the weighting for the data analysis? Well, since detectors have a certain probability to detect neutrons of a certain energy, not every neutron count has an equal value in the final results. 
# Here is an example: *How many neutrons are detected within 10 m distance?*

# %%
# How many neutrons are detected within 10 m distance?
r10m = U.Hits[U.Hits.r < 10]
len(r10m)/len(U.Hits) # => 39 %

# %%
# More realistic: weight by the probability to be counted
# (i.e., weighted by the detector energy response)
r10m.Weight.sum() # => 29 %

# %% [markdown]
# ## 3. Spatial Plots
# Let's look at the neutron origins from the birds-eye view, where do the neutrons come from? The following functions have the same argument list with the following options:
# - `ax=None` - uses a given axis to plot on a subpanel of an overarching figure, or create an own figure
# - `thermal=False` - consider thermal or epithermal neutrons, defaults to epithermal
# - `footprint=True` - indicate the footprint ellipse
# - `quantile=0.865` - quantile definition for the footprint
# - `weighted=True` - use energy-weighted neutrons
#
# Notes:
#
# - All plotting routines return an `ax` object which can be used to make further changes to the plot. 
# - If you need to save the plot, use: `ax.figure.savefig('my_plot.pdf', bbox_inches="tight")`

# %%
# XY Plot with origins and footprint
U.plot_xy_hits()

# %%
# Z Plot with origins and footprint
# Suggestion: use ax = ... above and ax.set_xlim(-10,10) to zoom in.
U.plot_z_hits()

# %% [markdown]
# ## 4. Distance and Depth distribution
# Let's look at the distance and depth distribution

# %%
# Distance distribtion
U.distance_distribution(var='r')

# %%
# Distribution of depth of first contact in the soil
U.depth_distribution(var='z')

# %%
# Distribution of maximum depth in the soil
U.depth_distribution(var='z_max')

# %% [markdown]
# ## 5. Footprint
# The footprint can be calculated by counting the number of (weighted) neutrons per distance travelled, the footprint radius then is the distance within which 86.5% of all detected neutrons have originated.

# %%
U.footprint_by_hits('r')

# %%
# Can also be calculated for thermal neutrons,
# while there is a debate on how this footprint should be calculated
# (see Jakobi et al. 2021 and Rasche et al. 2021).
U.footprint_by_hits('r', thermal=True)

# %%
# The peneration depth calculation is not trivial.
# The following calculations only indicate the possible range of penetration.
# Refer to Markus Köhli for details.
print('z_86 = {z:.3f} m, zmax_86 = {zmax:.3f} m' 
      .format(z   =U.footprint_by_hits('z'),
              zmax=U.footprint_by_hits('z_max')))

# %% [markdown]
# ## 6. Angular distribution
# When looking at every single neutron that hit the detector, we can also calculate the angle from which the neutrons originated. The following plot shows the angles for four different distances of origin to answer the question whether long-range and short-range neutrons have the same angular distribution. For an application, see e.g., Francke et al. 2021.

# %%
# Angular distribution
U.plot_angle_of_origin()

# %%
# Angular distribution in polar coordinates
U.plot_angle_of_origin(polar=True)
