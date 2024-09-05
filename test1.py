import tidy3d as td
import numpy as np
import matplotlib.pyplot as plt
from tidy3d import web
from typing import Callable
# defining a random seed for reproducibility
np.random.seed(12)

def find_source_decay(source):
    # source bandwidth
    fwidth = source.source_time.fwidth

    # time offset to start the source
    time_offset = source.source_time.offset / (2 * np.pi * fwidth)

    # time width for a gaussian pulse
    pulse_time = 0.44 / fwidth

    decay_time = time_offset + pulse_time
    return decay_time

def taper_factor(
    max_dip,
    structure_index,
    n_tapered_structures,
    cubic_factor=2,
    quadratic_factor=-3,
    linear_factor=0,
    offsetFactor=1,
):
    cubic = cubic_factor
    quadratic = quadratic_factor
    linear = linear_factor
    offset = offsetFactor

    """
    Compute a taper factor using a cubic polynomial.

    Parameters:
    - max_dip: float, maximum dip in taper factor.
    - structure_index: int, index of the current structure.
    - n_tapered_structures: int, total number of tapered structures.
    - cubic_factor: float, coefficient for cubic term (default 2).
    - quadratic_factor: float, coefficient for quadratic term (default -3).
    - linear_factor: float, coefficient for linear term (default 0).
    - offsetFactor: float, constant offset (default 1).

    Returns:
    - float, the taper factor for the given structure.
    """

    # to avoid division for 0 in a case where there is no tapered region
    if n_tapered_structures == 0:
        x = 1
    else:
        x = structure_index / np.ceil(n_tapered_structures / 2)

    # factor is 1 if the structure is not in the tapered region
    factor = (
        1
        if x >= 1
        else 1 - max_dip * (cubic * x**3 + quadratic * x**2 + linear * x + offset)
    )

    return factor

def translate_geometry_group(geometry_group, x=0, y=0, z=0):
    translated_geometries = []
    for geom in geometry_group.geometries:
        if isinstance(geom, td.PolySlab):
            # Translate the vertices of PolySlab
            translated_vertices = geom.vertices + np.array([x, y])
            # Keep slab_bounds as a tuple
            translated_slab_bounds = (geom.slab_bounds[0] + z, geom.slab_bounds[1] + z)
            translated_geom = td.PolySlab(
                vertices=translated_vertices,
                slab_bounds=translated_slab_bounds,
                axis=geom.axis
            )
            translated_geometries.append(translated_geom)
        elif isinstance(geom, td.Box):
            # Translate the center of Box
            translated_center = np.array(geom.center) + np.array([x, y, z])
            translated_geom = td.Box(center=translated_center, size=geom.size)
            translated_geometries.append(translated_geom)
        else:
            raise TypeError(f"Translation for {type(geom)} is not implemented.")
    
    return td.GeometryGroup(geometries=translated_geometries)

def rotate_geometry_group(geometry_group, angle, axis):
    if axis != 2:
        raise ValueError("Only rotation around the z-axis (axis=2) is supported in this function.")
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotated_geometries = []
    for geom in geometry_group.geometries:
        if isinstance(geom, td.PolySlab):
            # Extract the vertices
            vertices = geom.vertices
            
            # Apply rotation matrix around the z-axis
            rotated_vertices = np.dot(vertices, np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]))
            
            # Create a new PolySlab with rotated vertices
            rotated_geom = td.PolySlab(
                vertices=rotated_vertices,
                slab_bounds=geom.slab_bounds,
                axis=geom.axis
            )
            rotated_geometries.append(rotated_geom)
        else:
            raise TypeError(f"Rotation for {type(geom)} is not implemented.")
    
    return td.GeometryGroup(geometries=rotated_geometries)

def get_symmetry(polarization):
    if polarization == "Ex":
        sym = [-1, 1, 1]
    elif polarization == "Ey":
        sym = [1, -1, 1]
    elif polarization == "Ez":
        sym = [1, 1, -1]
    elif polarization == "Hz":
        sym = [-1, -1, 1]
    elif polarization == "Hy":
        sym = [-1, 1, -1]
    elif polarization == "Hx":
        sym = [1, -1, -1]
    return sym

def assemble_unit_cells(
    unit_cell_function: Callable,
    lattice_constant: float,
    max_dip_lattice: float,
    n_constant_structures: int,
    n_tapered_structures: int,
    original_params: list,
    max_dip_params: list,
    central_gap: float = 0,
):
    """
    Assemble unit cells into a structure with tapered lattice constants and parameters.

    Parameters:
    - unit_cell_function: callable, function to generate a unit cell given position and parameters.
    - lattice_constant: float, base lattice constant for the unit cells.
    - max_dip_lattice: float, maximum dip for tapering the lattice constant.
    - n_constant_structures: int, number of constant lattice structures on either side.
    - n_tapered_structures: int, total number of tapered structures.
    - original_params: list of floats, original parameters to be tapered (default empty list).
    - max_dip_params: list of floats, maximum dips for each parameter to be tapered (default empty list).
    - central_gap: float, gap between central unit cells (only valid for an odd number of tapered structures).

    Returns:
    - unit_cells: combined unit cells as generated by the unit_cell_function.
    - pos_x: final position of the last unit cell in the x-direction.
    """

    unit_cells = None

    # start position
    pos_x = 0

    # defining the number of unit cells for each side
    positions = range(
        n_constant_structures
        + int(np.floor(n_tapered_structures / 2))
        + n_constant_structures % 2
    )

    # iterating over the unit cells
    for i in positions:
        # defining the current value of the unit cell given the position
        current_lattice_constant = lattice_constant * taper_factor(
            max_dip_lattice, i, n_tapered_structures
        )

        # condition for odd or even number of unit cells
        if i == 0:
            if n_tapered_structures % 2 != 0:
                pos_x = 0
            else:
                pos_x = central_gap / 2 + current_lattice_constant / 2
        else:
            pos_x += current_lattice_constant / 2 + previous_lattice_constant / 2

        # now we create a list of parameters to input to the unit_cell_function
        parameters = []
        for idx in range(len(original_params)):
            original = original_params[idx]
            max_dip = max_dip_params[idx]
            tapered_param = original * taper_factor(
                max_dip, i, n_tapered_structures
            )
            parameters.append(tapered_param)

        # defining the center
        center = (pos_x, 0, 0)

        # creating the unit cell calling the unit_cell_function
        new_unit_cell = unit_cell_function(center, *parameters)

        # adding the unit cell to the geometry (or initializing the variable if is the first unit cell)
        if unit_cells is None:
            unit_cells = new_unit_cell
        else:
            unit_cells += new_unit_cell

        # the separation to the next unit cell is the mean value of the lattice constants of the current and next unit cells
        previous_lattice_constant = current_lattice_constant

    # return the geometry and the position of the last unit cell
    return unit_cells, pos_x

# function for creating ellipse unit cells
def ellipse_uc(center, x_axis, y_axis, height=0.5):
    theta = np.linspace(0, np.pi * 2, 1001)
    x = x_axis * np.cos(theta) + center[0]
    y = y_axis * np.sin(theta) + center[1]

    geometry = td.PolySlab(
        vertices=np.array([x, y]).T,
        slab_bounds=(-height / 2 + center[2], height / 2 + center[2]),
        axis=2,
    )
    return geometry


# plot the geometries

args = dict(
    lattice_constant=1,
    max_dip_lattice=0.5,
    n_constant_structures=15,
    n_tapered_structures=8,
)

ellipse_geometry, _ = assemble_unit_cells(
    unit_cell_function=ellipse_uc,
    original_params=[0.2, 0.3],
    max_dip_params=[0.4, 0.3],
    **args
)
# ellipse_geometry.plot(z=0)

# plt.show()



def create_source(
    polarization="Ey",
    wl1=0.7,
    wl2=1.0,
    center=(0, 0, 0),
):

    # defining the source
    source = td.PointDipole(
        center=center,
        name="pointDipole",
        polarization=polarization,
        source_time=td.GaussianPulse(
            freq0=freq0, fwidth=fwidth, phase=2 * np.pi * np.random.random()
        ),
    )

    # source1_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth, amplitude=1)
    # source = td.ModeSource(
    #     center=center,
    #     size=[0, td.inf, td.inf],
    #     mode_index=0,
    #     direction="+",
    #     source_time=source1_time,
    #     mode_spec=td.ModeSpec(),    
    # )

    return source

def create_monitors(
    simulation,
    n_point_monitors=5,
    center=(0, 0, 0),
    deviation=(0.2, 0, 0),
):
    simulation_size = simulation.size

    freq0 = simulation.sources[0].source_time.freq0

    # creating random positions around the center of the cavity
    positions = np.random.random((n_point_monitors, 3)) * np.array(deviation) + center

    # creating the randomly positioned monitors
    point_monitors = [
        td.FieldTimeMonitor(
            center=tuple(positions[i]),
            name="pointMon%i" % i,
            start=0,
            size=(0, 0, 0),
            interval=1,
        )
        for i in range(n_point_monitors)
    ]

    # defining size for the 3D field monitor
    size = tuple(np.array(simulation_size) * 0.9)

    # defining the start time of the monitors to record only the last two oscillations of the field
    start = simulation.run_time - 2 / freq0

    # 3D field monitor for energy density calculation
    field_monitor = td.FieldTimeMonitor(
        center=(0, 0, 0),
        size=size,
        start=start,
        name="fieldTimeMon",
        interval_space=(5, 5, 5),
    )

    # 2D field monitor for visualizing the resonant mode profile
    field_profile_monitor = td.FieldTimeMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        start=start,
        name="fieldProfileMon",
    )

    monitor_flux_transmission = td.FluxMonitor(
    center=[(n_constant_cells+n_tapered_cells/2)*lattice_constant, 0, 0],
    size=[0, width * 2, height * 2],
    freqs=freqs,
    name="flux_transmission",
)

    # setup planar flux monitors surrounding the simulation volume
    flux_monitors = []
    for i in [-1, 1]:
        for j in [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]:
            j = np.array(j)
            # defining the center
            mon_center = tuple(np.array(size) * j * i + -1 * i * j * (0.95, 0.95, 0.95))

            # defining the size
            j2 = np.array([0 if l == 0.5 else 1 for l in j])
            mon_size = tuple(np.array(size) * j2 * 2)

            # defining the name
            mon_name = np.array(["x", "y", "z"])[np.ceil(j).astype(bool)][0]
            mon_name = ("-" + mon_name) if i == -1 else mon_name

            # creating the monitors
            flux_monitors.append(
                td.FluxTimeMonitor(
                    center=mon_center, size=mon_size, start=start, name=mon_name
                )
            )

    return (
        point_monitors
        + flux_monitors
        + [
            field_monitor,
            field_profile_monitor,
            monitor_flux_transmission
        ]
    )

def create_simulation(
    wl1: float,
    wl2: float,
    width: float,
    height: float,
    unit_cell_function: Callable,
    lattice_constant: float,
    n_constant_structures_right: int,
    n_constant_structures_left: int,
    n_tapered_structures: int,
    n_tapered_output: int,
    max_dip_lattice: float,
    original_params: list,
    max_dip_params: list,
    polarization: str,
    unit_cell_index: int,
    waveguide_index: float,
    run_time: float = 2e-12,
    central_gap: float = 0,
    delta_override: float = 0.01,
    substrate_index: float = 1,
    medium_index: float = 1,
    sidewall_angle: float = 0,
):
    """
    create a simulation with specified parameters and unit cells.

    Parameters:
    - wl1: float, wavelength 2.
    - wl2: float, wavelength 2 for the simulation.
    - width: float, width of the waveguide structure.
    - height: float, height of the structure.
    - unit_cell_function: callable, function to generate unit cells.
    - delta_override: float, spacing for mesh override.
    - lattice_constant: float, lattice constant.
    - n_constant_structures_right: int, number of constant structures on the right side of the nanobeam.
    - n_constant_structures_left: int, number of constant structures on the left side of the nanobeam.
    - n_tapered_structures: int, number of tapered structures at the cavity region.
    - n_tapered_waveguide: int, number of tapered unit cells at the right end of the nanobeam.
    - max_dip_lattice: float, maximum dip for the lattice constant tapering.
    - original_params: list, list with the unit cell params.
    - max_dip_params: list, list with the tapering params
    - polarization: str, polarization of the source.
    - unit_cell_index: int, index for the unit cell structure.
    - waveguide_index: float, index for the waveguide.
    - run_time: float, total run time for the simulation.
    - central_gap: float, central gap between structures (if n_tapered_structures is odd).
    - substrate_index: float, index for the substrate material.
    - medium_index: float, index for the surrounding medium.
    - sidewall_angle: float, waveguide sidewall angle.

    Returns:
    - sim: the simulation object with defined structures, sources, and monitors.
    """

    # define the mesh override at the cavity region
    mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 0, 0), size=(4, width, height)),
        dl=(delta_override,) * 3,
    )

    # define grid specification for the simulation
    grid_spec = td.GridSpec.auto(
        min_steps_per_wvl=15,
        override_structures=[mesh_override],
    )

    kwargs = dict(unit_cell_function=unit_cell_function,
                  lattice_constant=lattice_constant,
                  max_dip_lattice=max_dip_lattice,
                  original_params=original_params,
                  max_dip_params=max_dip_params,
                  central_gap=central_gap)

    # assemble unit cells for the right side of the nanobeam
    unit_cells_right, pos_x_right = assemble_unit_cells(
        n_constant_structures=n_constant_structures_right,
        n_tapered_structures=n_tapered_structures,
        **kwargs
    )

    # assemble unit cells for the left side of the nanobeam
    unit_cells_left, pos_x_left = assemble_unit_cells(
        n_constant_structures=n_constant_structures_left,
        n_tapered_structures = n_tapered_structures,
        **kwargs
    )



    # # rotate and adjust the position of the left unit cells
    # unit_cells_left = unit_cells_left.rotated(np.pi, axis=2)
    # pos_x_left *= -1

    # if unit_cells_left is not None:
    unit_cells_left = rotate_geometry_group(unit_cells_left, np.pi, axis=2)

    # assemble and position the tapered output if applicable
   # assemble and position the tapered output if applicable
    if n_tapered_output > 0:
        waveguide_tapered, pos_tapered = assemble_unit_cells(
            n_constant_structures=0,
            n_tapered_structures=n_tapered_output * 2,
            **kwargs
        )

        # rotate and position the tapered output
        waveguide_tapered = rotate_geometry_group(waveguide_tapered, np.pi, axis=2)
        waveguide_tapered = translate_geometry_group(
            waveguide_tapered, x=pos_x_right + lattice_constant + pos_tapered, y=0, z=0
        )

        # combine the left, right, and tapered waveguide geometries
        nanobeam_geometry = unit_cells_left + unit_cells_right + waveguide_tapered

    else:
        pos_tapered = 0
        nanobeam_geometry = unit_cells_left + unit_cells_right

    # calculate sizes and defining the center of the nanobeam geometry
    size_left = abs(pos_x_left) + lattice_constant + wl2 / 2
    size_right = abs(pos_x_right + pos_tapered) + lattice_constant + wl2 / 2
    nanobeam_size = size_left + size_right
    nanobeam_center = size_left - nanobeam_size / 2

    # recenter the geometry to optimize the simulation space
    # nanobeam_geometry = nanobeam_geometry.translated(x=nanobeam_center, y=0, z=0)

    nanobeam_geometry = translate_geometry_group(nanobeam_geometry, x=nanobeam_center, y=0, z=0)

    # define mediums for the substrate and nanobeam materials
    nanobeam_medium = td.Medium(permittivity=unit_cell_index**2)

    # create the nanobeam structure
    nanobeam_structure = td.Structure(
        geometry=nanobeam_geometry,
        medium=nanobeam_medium,
    )

    # define the waveguide geometry
    cell_size = (
        nanobeam_size,
        width + 2 * wl2,
        height + 2 * wl2,
    )

    # defining the waveguide. Use a PolySlab for the sidewall angle
    geometry_wvg = td.PolySlab(
        vertices=[
            [-2 * cell_size[0], width / 2],
            [2 * cell_size[0], width / 2],
            [2 * cell_size[0], -width / 2],
            [-2 * cell_size[0], -width / 2],
        ],
        axis=2,
        slab_bounds=(-height / 2, height / 2),
        sidewall_angle=sidewall_angle,
    )
    waveguide = td.Structure(
        geometry=geometry_wvg,
        medium=td.Medium(permittivity=waveguide_index**2),
    )

    # substrate
    substrate = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-999, -999, -999), rmax=(999, 999, -height / 2)
        ),
        medium=td.Medium(permittivity=substrate_index**2),
    )

    # defining the source
    sources = [
        create_source(
            polarization=polarization,
            wl1=1.300,  # Central wavelength at 1550 nm
            wl2=1.320,       # Wavelength span of 20 nm
            center=(nanobeam_center, 0, 0),
        )
    ]

    # boundary conditions
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()
    )

    # determine symmetry based on structure configuration
    if (n_constant_structures_left == n_constant_structures_right) and (
        n_tapered_output == 0
    ):
        symmetry = get_symmetry(polarization=polarization)
    else:
        symmetry = [0] + get_symmetry(polarization=polarization)[1:]

    # create the simulation object
    sim = td.Simulation(
        size=cell_size,
        structures=[substrate, waveguide, nanobeam_structure],
        sources=sources,
        monitors=[],
        run_time=run_time,
        boundary_spec=boundary_spec,
        grid_spec=grid_spec,
        symmetry=symmetry,
        medium=td.Medium(permittivity=medium_index**2),
    )

    # add monitors to the simulation
    monitors = create_monitors(
        sim, n_point_monitors=5, center=(nanobeam_center, 0, 0), deviation=(0.5, 0, 0)
    )
    return sim.updated_copy(monitors=monitors)

height = 0.4
width = 0.75
# lattice_constant = 0.430
lattice_constant = 0.480
max_dip_lattice = 1 - 0.330 / lattice_constant
# radius_x = 0.28 * lattice_constant
# radius_x = 0.3 * lattice_constant
radius_x = 0.28 * lattice_constant
radius_y = radius_x
max_dip_x = 0
max_dip_y = 0
n_constant_cells = 35  # each side
n_tapered_cells = 16  # total
central_gap = 0
wl1 = 1.300
wl2 = 1.320
run_time = 5e-12
polarization = "Ey"
substrate_index = 1.44
unit_cell_index = 1
waveguide_index = 2.0
sidewall_angle = 0
# defining the frequencies based on the given wavelengths
freq1 = td.C_0 / wl1
freq2 = td.C_0 / wl2
freq0 = (freq1 - freq2) / 2 + freq2
fwidth = (freq1 - freq2) / 2
freqs = np.linspace(freq0 - 3 * fwidth, freq0 + 3 * fwidth, 50000)
# freqs = np.linspace(freq0 - td.C_0/1.305, freq0 + td.C_0/1.315, 10000)

print(f"width: {width}")
print(f"lattice constant: {lattice_constant}") 
print(f"n_constant: {n_constant_cells}")
print(f"n_taper: {n_tapered_cells}")
print(f"gap: {central_gap}")
print(f"run time: {run_time}")

# defining a unit cell function depending only on the taper parameters
unit_cell_function = lambda center, x, y: ellipse_uc(
    center, x_axis=x, y_axis=y, height=height
)


sim = create_simulation(
    unit_cell_function=unit_cell_function,
    wl1=wl1,
    wl2=wl2,
    width=width,
    height=height,
    run_time=run_time,
    lattice_constant=lattice_constant,
    n_constant_structures_right=n_constant_cells,
    n_constant_structures_left=n_constant_cells,
    n_tapered_output=0,
    n_tapered_structures=n_tapered_cells,
    max_dip_lattice=max_dip_lattice,
    polarization=polarization,
    substrate_index=substrate_index,
    unit_cell_index=unit_cell_index,
    waveguide_index=waveguide_index,
    sidewall_angle=sidewall_angle,
    original_params=[radius_x, radius_y],
    max_dip_params=[max_dip_lattice, max_dip_lattice],
)

sim.plot(z=0, monitor_alpha=0)
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
sim.plot(x=0, ax=ax1)
sim.plot(z=0, ax=ax2)
# ax2.set_xlim(0, 20 * period)
plt.show()

# estimating the cost
task_id = web.upload(sim, "nanobeam_test")
cost = web.estimate_cost(task_id)

# running the simulation
sim_data = web.run(simulation=sim, task_name="ellipse_nanobeam")

sim_data.plot_field(
    "fieldProfileMon",
    "Ey",
    val="abs",
    z=0,
    t=sim_data["fieldProfileMon"].Ez.t[-1],
)
plt.show()

# sim_data.plot_field(
#     "fieldProfileMon",
#     "Ey",
#     val="real",
#     z=0,
#     t=sim_data["fieldProfileMon"].Ez.t[-1],
# )
# plt.show()

# sim_data.plot_field(
#     "fieldProfileMon",
#     "E",
#     val="abs",
#     z=0,
#     t=sim_data["fieldProfileMon"].Ez.t[-1],
# )
# plt.show()

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

# plot transmitted flux for each waveguide
ax.plot(td.C_0 / freqs, sim_data["flux_transmission"].flux, label="Transmission")
# ax.plot(td.C_0 / freqs, sim_data["flux_misaligned"].flux, label="Misaligned corrugation")

# vertical line at design frequency
ax.axvline(td.C_0 / freq0, ls="--", color="k")

ax.set(xlabel="Wavelength (nm)", ylabel="Transmission")

ax.grid(True)
plt.legend()
plt.show()

# Assuming `freqs` and `sim_data` have already been defined
# Normalize the transmitted flux
transmission_flux = sim_data["flux_transmission"].flux
normalized_flux = transmission_flux / np.max(transmission_flux)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

# Plot normalized transmitted flux
ax.plot(td.C_0 / freqs, normalized_flux, label="Normalized Transmission")

# Vertical line at the design frequency
ax.axvline(td.C_0 / freq0, ls="--", color="k", label="Design Frequency")

# Set axis labels
ax.set(xlabel="Wavelength (nm)", ylabel="Normalized Transmission")

# Enable grid and legend
ax.grid(True)
plt.legend()

# Show the plot
plt.show()

def analyse_resonance_monitors(sim_data, start_time=1e-12, freq_window=None):
    # importing needed libraries
    from tidy3d.plugins.resonance import ResonanceFinder

    combinedSignal = 0
    polarization = sim_data.simulation.sources[0].polarization

    # iterate through all monitors and combine the signal
    i = 0
    name = "pointMon%i" % i
    while name in sim_data.monitor_data:
        combinedSignal += sim_data["pointMon0"].field_components[polarization].squeeze()
        i += 1
        name = "pointMon%i" % i

    # create the ResonanceFinder instance
    rf = ResonanceFinder(freq_window=freq_window)

    # boolean mask to set the data after source decay
    bm = combinedSignal.t >= start_time

    # data containing the resonance information
    data = rf.run_raw_signal(combinedSignal[bm], sim_data.simulation.dt)

    # creating a DataFrame
    df = data.to_dataframe()
    df["wl"] = (td.C_0 / df.index) * 10**3

    return df, combinedSignal

# defining the frequency window
freq0 = sim_data.simulation.sources[0].source_time.freq0
fwidth = sim_data.simulation.sources[0].source_time.fwidth
freq_window = (freq0 - fwidth / 2, freq0 + fwidth / 2)

# start time to analyze the exponential decaying fields after source decay
start_time = 2 * find_source_decay(
    sim.sources[0],
)

df, signal = analyse_resonance_monitors(sim_data, start_time, freq_window)
print(f"width: {width}")
print(f"lattice constant: {lattice_constant}")
print(f"n_constant: {n_constant_cells}")
print(f"n_taper: {n_tapered_cells}")
print(f"gap: {central_gap}")
print(f"run time: {run_time}")
print("------------------------------------------------------------------------------------")
print(df.head(10))

def get_energy_density(sim_data):
    # retrieving monitor data
    Efield = sim_data["fieldTimeMon"]

    # retrieving permittivity data and interpolating it at the same grid points of the monitor data
    eps = abs(sim.epsilon(box=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf))))
    eps = eps.interp(coords=dict(x=Efield.Ex.x, y=Efield.Ex.y, z=Efield.Ex.z))

    # calculating the energy density
    energy_density = np.abs(Efield.Ex**2 + Efield.Ey**2 + Efield.Ez**2) * eps

    # calculating the temporal mean value
    delta_t = energy_density.t[-1] - energy_density.t[0] # signal duration
    energy_density_temporal_mean = energy_density.integrate(coord=("t")) / delta_t

    return energy_density_temporal_mean, eps


energy_density, eps = get_energy_density(sim_data)

def mode_volume(energy_density, eps):
    # volume integral of the energy density
    Integrated = np.abs(energy_density).integrate(coord=("x", "y", "z"))

    # scaling factor to present in common units of (wl/n)^3
    wl = abs(df.sort_values("Q").iloc[-1]["wl"] / 1000)  # um
    index = np.sqrt(abs(eps.max()))

    Vmode = Integrated / np.max(energy_density) / (wl / index) ** 3

    return Vmode


Vmode = mode_volume(energy_density, eps)
print("Mode volume: %.2f (lambda/n)^3" % Vmode)