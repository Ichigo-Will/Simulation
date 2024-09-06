import sys, os
import matplotlib as plt
#default path for current release 
sys.path.append("/Machintosh HD/Applications/Lumerical v211/Contents/API/Python") 
sys.path.append(os.path.dirname(__file__)) #Current directory

import lumapi

# def run_fdtd():
#     if __name__ == '__main__':
#         run_fdtd()

def run_fdtd(mesh_res, sim_time, block_dim, material):
    if __name__ == '__main__':
        block_dim = 1e-6 # x/y block dimensions
        material = "Si3N4 (Silicon Nitride) - Kischkat"
        mesh_res = block_dim / 100
        print(mesh_res)
        sim_time = 1e-10 # sim time must be long enough to reach autoshutoff
        [wavelengths, s11_transmission, s21_transmission] = run_fdtd(mesh_res, sim_time, block_dim, material)

# fdtd = lumapi.FDTD(hide=False)

# ## add initial geometry
# fdtd.addrect()
# fdtd.set("name", "si3n4_block")
# fdtd.set("x", 0)
# fdtd.set("x span", 1e-6)
# fdtd.set("y", 0)
# fdtd.set("y span", 1e-6)
# fdtd.set("z", 0)
# fdtd.set("z span", 1e-6)
# fdtd.set("material", "Si3N4 (Silicon Nitride) - Kischkat")
# breakpoint()

# ## mesh
# fdtd.addmesh()
# fdtd.set("name", "si3n4_mesh")
# fdtd.set("dx", 1e-8)
# fdtd.set("dy", 1e-8)
# fdtd.set("dz", 1e-8)
# fdtd.set("x", 0)
# fdtd.set("x span", 1e-6)
# fdtd.set("y", 0)
# fdtd.set("y span", 1e-6)
# fdtd.set("z", 0)
# fdtd.set("z span", 1e-6)

# ## FDTD solver
# fdtd.addfdtd()
# fdtd.set("x", 0)
# fdtd.set("x span", 1e-6 + 1e-6 / 2) # 1e-6 / 2 buffer on fdtd region
# fdtd.set("y", 0)
# fdtd.set("y span", 1e-6 + 1e-6 / 2)
# fdtd.set("z", 0)
# fdtd.set("z span", 1e-6 + 1e-6 / 2)
# fdtd.set("pml layers", 8)
# fdtd.set("x min bc", "PML")
# fdtd.set("x max bc", "PML")
# fdtd.set("y min bc", "PML")
# fdtd.set("y max bc", "PML")
# fdtd.set("z min bc", "PML")
# fdtd.set("z max bc", "PML")
# fdtd.set("simulation time", 1e-10)
# fdtd.setglobalsource("center wavelength", 1.55e-6) # set wavelength in global source settings
# fdtd.setglobalsource("wavelength span", 0.1e-6)
# fdtd.setglobalmonitor("frequency points", 20)
# breakpoint()

# ## add ports
# fdtd.addport()
# fdtd.select("FDTD::ports::port 1")
# fdtd.set("direction", "forward")
# fdtd.set("frequency dependent profile", 0)
# fdtd.set("x", -1e-6 / 2)
# fdtd.set("y", 0)
# fdtd.set("y span", 1e-6)
# fdtd.set("z", 0)
# fdtd.set("z span", 1e-6)
# fdtd.set("mode selection", "fundamental TM mode")
# fdtd.addport()
# fdtd.select("FDTD::ports::port 2")
# fdtd.set("direction", "backward")
# fdtd.set("frequency dependent profile", 0)
# fdtd.set("x", 1e-6 / 2)
# fdtd.set("y", 0)
# fdtd.set("y span", 1e-6)
# fdtd.set("z", 0)
# fdtd.set("z span", 1e-6)
# fdtd.set("mode selection", "fundamental TM mode")

# ## save file prior to running (file must be saved before it can be run)
# fdtd.save("script_overview_fdtd")
# ## run simulation
# fdtd.run()
# autoshutoff = fdtd.getresult("FDTD", "status")
# print("autoshutoff status (want 2)", autoshutoff)

# ## s-parameter sweep to extract s parameters
# fdtd.addsweep(3)
# fdtd.setsweep("s-parameter sweep", "Excite all ports", 0)
# fdtd.setsweep("S sweep", "auto symmetry", True)
# fdtd.runsweep("s-parameter sweep")
# breakpoint()

# ## extract s-parameters
# s_parameters = fdtd.getsweepresult("s-parameter sweep", "S parameters")
# wavelengths = s_parameters['lambda']
# s11_transmission = abs(s_parameters['S11']) ** 2
# s21_transmission = abs(s_parameters['S12']) ** 2
# total = s11_transmission + s21_transmission
# ## plot data
# plt.plot(wavelengths * 1e9, s11_transmission, linewidth=2, label='S11')
# plt.plot(wavelengths * 1e9, s21_transmission, linewidth=2, label='S21')
# plt.plot(wavelengths * 1e9, total, linewidth=2, label='total')
# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Power [W]')
# plt.ylim([0, 1])
# plt.legend(loc=1)
# plt.grid('on')
# plt.savefig("s_parameters.png")
# breakpoint()