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
# # Analysis of complex spatial pattern from URANOS output
#
# The is an example script to read and plot URANOS output (neutron density maps) of a single scenario to explore the spatial patterns.
# The scenarios are based on work by [Schrön et al. (2023)](http://dx.doi.org/10.5194/hess-27-723-2023).
#
# *-- last modified: Mar 10, 2023*

# %%
import sys
# download corny (https://git.ufz.de/CRNS/cornish_pasdy) to local directory
sys.path.append('../../../../cornish_pasdy/')  #set path to local directory of corny
#import corny.grains #this should work, if the path is set correctly.

# import the URANOS class which can deal with all the details on its own
from lib.uranos import URANOS
import corny.grains.Schroen2017hess


# %% [markdown]
# ## 0. TL;DR
# This tutorial uses only a few short-hand functions to read in URANOS output. Here is the compacted list without explanations. Jump to the next section go through the more elaborate tutorial.

# %%
# Read
U = URANOS(folder='example_uranos_output/complex_fields/', scaling=2, hum=5)
U = U.read_materials('6.png').material2sm(SM_gui=0.2)
U = U.generate_distance().find_regions()
U = U.read_density('densityMapSelected*', pad=True)
U = U.read_origins('detectorOrigins*', pad=True)
# Plot
U.region_data['freetext'] = [r'$\theta=20\%$', r'wet', '     sand',
                             'also wet', 'grass','dry', 'meadow1', 'meadow2',
                             'pasture', 'parking', 'peatland','desert']
U.histogram(var=['area', 'Materials', 'SM', 'Density', 'Origins'])
U.plot(image='Regions', annotate='freetext', contour=True)
U.plot(image='Density', annotate='Density', contour=True,
       cmap='Spectral_r', cmap_scale=1, colorbar=True, interpolation='bicubic')
ax = U.plot(image='SM', cmap='Blues', annotate='Origins', overlay='Origins',
            contour=True, colorbar=True)
ax.figure.savefig('my_plot.pdf', bbox_inches="tight")

# %% [markdown]
# ## 1. Setup the URANOS object
# Assuming that you have all the URANOS output files in one folder, the URANOS class can automatically recognize the different files based on its name. Hence, to initialize the object, it is sufficient to provide only the folder name. Other options for initialization are:
# - `folder` - folder which contains the density maps, origin files, png image, etc.
# - `scaling=2` - one pixel of the matrix corresponds to how many meters? Defaults to 2, i.e., a 500x500 matrix represents a 1000x1000 m² area
# - `default_material` - dominant material that is used as region number 0. If not provided, tries to guess the dominant material of the environment by looking at the central pixel (0,0).
# - `hum=5, press=1013` - air humidity (g/m³) and air pressure (hPa) used in the simulation, will be used for further internal calculations.

# %%
# Create URANOS object based on the given folder
U = URANOS(folder='example_uranos_output/complex_fields/', scaling=2, hum=5)

# %% [markdown]
# ### Read input spatial patterns
# Will use the input layer definition to learn about the used materials.

# %%
U = U.read_materials('6.png')

# %%
# This created a material matrix as a 2D numpy array which looks like this:
U.Materials

# %% [markdown]
# Now that we have an idea about the used materials, let's try to convert it to soil moisture values automatically. Of course this does not work for values that represent non-soil material.

# %%
# Convert Material to Soil Moisture,
# where material 100 corresponds to the slider value set in the URANOS GUI.
U = U.material2sm(SM_gui=0.2)

# %%
# This created a soil moisture matrix as a 2D numpy array which looks like this:
U.SM

# %% [markdown]
# Let's also create a matrix of distances, this will be very handy if the distance of a certain pixel to the center (or central detector) is requested.

# %%
U = U.generate_distance()

# %% [markdown]
# It is now easy to sample a certain pixel and ask for its attributes:

# %%
i, j = 42, 23
print('Pixel ({i},{j}) is {r:.1f} m from center and '
      +'consists of material {material} with {sm:.0f} % soil moisture.' \
      .format(i=i, j=j, r=U.Distance[i,j], material=U.Materials[i,j], sm=(U.SM[i,j]*100)))

# %% [markdown]
# ## 2. Identify sub-areas (=Regions)
# Normally spatial patterns are devided into certain sub-areas. The URANOS object can automatically identify *connected* areas with the *same material code* as one region.

# %%
# Identify regions
U = U.find_regions()

# %%
# Plot regions
U.plot(image='Regions', annotate='Regions', contour=True)

# %% [markdown]
# `U.find_regions()` has generated a DataFrame containing all the relevant information for each region, e.g. its area, distance, or soil moisture value. Single values can be accessed by `U.region_data.loc[8,'SM']`, for instance. Since it is a normal DataFrame, it can be saved with `U.region_data.to_csv('region_data.csv')`.

# %%
U.region_data

# %% [markdown]
# It is also possible to give each region a name that can be used for annotating plots, for instance:

# %%
U.region_data['freetext'] = [r'$\theta=20\%$', r'wet', '     sand',
                             'also wet','grass','dry', 'meadow1', 'meadow2',
                             'pasture', 'parking', 'peatland','desert']

# %%
# Plot regions
U.plot(image='Regions', annotate='freetext', contour=True)

# %% [markdown]
# ## 3. Basic plotting
# The URANOS-specifc plotting routine is very handy to quickly make plots of the generated matrices. They can be used as soon as regions have been identified, as the arguments for annotation and contours refer to the corresponding entries in the `region_data` DataFrame.
#
# The function `U.plot(ax, image, annotate, overlay, ...) -> ax` has the following options:
# - `ax=None` - uses a given axes from an overarching figure panel, or creates a new axis and figure
# - `image='SM'` - name of the matrix used for colored plotting
# - `annotate=None` - attempts to annotate regions with a label given by a column name in U.region_data
# - `overlay=None` - overlay 'Origins' to draw crosses of neutron origins onto the map
# - `fontsize=10, title=None` - additional formatting
# - `contour=False` - if True, draw the region borders on the map
# - `regions=None` - list of regions to plot, e.g., `[0,4,5]`, defaults to all regions.
# - `extent=500` - zoom in to a certain width/height square (500 is the whole matrix)
# - `cmap='Greys'` - colorscale used for the image plot
# - `cmap_scale=2` - scales the colorscale by factor 2, which is often lighter and increases readability for annotation labels.
# - `x_marker=None` - x-distance in meters to draw a marker on top of the axis, e.g., to highlight a distance of a patch from the center
# - `cross_alpha=0.5` - transparency of the central cross at (0,0). Set to 0 to remove it,
# - `label_offset=(0,0)` - x and y offset of the labels to increase readibility over region contours
# - `step_fraction=0.2` - x and y steps to use for tickmarks relative to the full extent.
# - `colorbar=False` - draw a colorbar
# - `axis_labels=True` - draw x and y axis labels: 'x (in m)'
# - `interpolation='none'` - interpolation of the image, can be one of 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
#
# Notes:
# - The plotted dimensiones are always given in meters.
# - The routine returns an `ax` object which can be used to make further changes to the plot. 
# - If you need to save the plot, use: `ax.figure.savefig('my_plot.pdf', bbox_inches="tight")`

# %%
# Plot Material data
U.plot(image='Materials', annotate='Materials', contour=True)

# %%
# Plot Soil Moisture pattern
U.plot(image='SM', annotate='SM', contour=True,
       cmap='Blues', colorbar=True)

# %% [markdown]
# ## 4. Neutron density map
# Automatically recognize the files starting with 'densityMapSelected*' and read them in as a matrix. If there are more than one file, treat them as results from parallel URANOS simulations with identical configurations. This means that the values from multiple matrices will be averaged, which greatly reduces the noise.

# %%
# Read in Density maps
U = U.read_density('densityMapSelected*', pad=True)

# %% [markdown]
# The results can optionally padded by 1 (for compensation legacy URANOS behaviour). If necessary, the results will be resampled to match the resolution of the input matrices and regions.

# %%
# This is how the density matrix looks like:
U.Density

# %% [raw]
# # Plot the density, including region contours and interpolation
# ax = U.plot(image='Density', annotate='Density', contour=True, 
#         cmap='Spectral_r', cmap_scale=1, colorbar=True,
#        interpolation='bicubic')

# %% [markdown]
# ## 5. Neutron Origins
# Neutron origins count how many neutrons had their first soil contact at pixel (i,j) before they got detected in the central detector. The script automatically recognizes the files starting with 'detectorOrigins*' and read them in as a matrix. If there are more than one file, treat them as results from parallel URANOS simulations with identical configurations. This means that the values from multiple matrices will be averaged, which greatly reduces the noise.

# %%
# Read in Origin maps
U = U.read_origins('detectorOrigins*', pad=True)

# %%
# Plot Origins above the SM map
U.plot(image='SM', annotate='Origins', overlay='Origins', contour=True, colorbar=True)

# %% [markdown]
# ## 6. Histogramm data
# The collected data can also be shown in a number of histograms to change the perspective. Note that 'simulated contributions' are a synonym for 'Origins'.

# %%
ax = U.histogram(var=['area', 'Materials', 'SM', 'Density', 'Origins'],
                 layout=(2,3), figsize=(10,5))

# %% [markdown]
# Are you looking for a specific region? Get all the available information by using the region_data DataFrame:

# %%
# Examine a certain region:
U.region_data.iloc[8]

# %%

# %%

# %%
