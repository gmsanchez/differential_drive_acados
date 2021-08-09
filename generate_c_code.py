from acados_template import *
import acados_template as at
from export_ode_model import *
import numpy as np
import scipy.linalg
from ctypes import *
from os.path import dirname, join, abspath

# ACADOS_PATH = join(dirname(abspath(__file__)), "../../../acados")

run_simulation = True

# create render arguments
ra = AcadosOcp()

# export model
model = export_ode_model()

Tf = 1.0
N = 10
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx

# set ocp_nlp_dimensions
nlp_dims     = ra.dims
nlp_dims.nx  = nx
nlp_dims.ny  = ny
nlp_dims.ny_e = ny_e
nlp_dims.nbx = 0
nlp_dims.nbu = nu
nlp_dims.nbx_e = 0
nlp_dims.nu  = model.u.size()[0]
nlp_dims.N   = N

# set weighting matrices
nlp_cost = ra.cost
Q = np.eye(nx)
Q[0,0] = 10.0      # x
Q[1,1] = 10.0      # y
Q[2,2] = 0.0001      # psi

R = np.eye(nu)
R[0,0] = 0.1         # v
R[1,1] = 0.1       # w

nlp_cost.W = scipy.linalg.block_diag(Q, R)

Vx = np.zeros((ny, nx))
Vx[0,0] = 1.0
Vx[1,1] = 1.0
Vx[2,2] = 1.0
nlp_cost.Vx = Vx

Vu = np.zeros((ny, nu))
Vu[3,0] = 1.0
Vu[4,1] = 1.0
nlp_cost.Vu = Vu

Q_e = 10*np.eye(nx)
Q_e[2,2] = 0.0001
nlp_cost.W_e = Q_e

Vx_e = np.zeros((ny_e, nx))
Vx_e[0,0] = 1.0
Vx_e[1,1] = 1.0
Vx_e[2,2] = 1.0
nlp_cost.Vx_e = Vx_e

nlp_cost.yref   = np.array([0, 0, 0, 0, 0])
nlp_cost.yref_e = np.array([0, 0, 0])

nlp_con = ra.constraints

nlp_con.lbu = np.array([-1, -2])
nlp_con.ubu = np.array([+1, +2])
nlp_con.x0  = np.array([0, 0, 0])
nlp_con.idxbu = np.array([0, 1])

## set QP solver
#ra.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ra.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ra.solver_options.hessian_approx = 'GAUSS_NEWTON'
ra.solver_options.integrator_type = 'ERK'

# set prediction horizon
ra.solver_options.tf = Tf
ra.solver_options.nlp_solver_type = 'SQP_RTI'
#ra.solver_options.nlp_solver_type = 'SQP'

# set header path
# ra.acados_include_path  = f'{ACADOS_PATH}/include'
# ra.acados_lib_path      = f'{ACADOS_PATH}/lib'

ra.model = model

acados_ocp_solver = AcadosOcpSolver(ra, json_file = 'acados_ocp_' + model.name + '.json')

print('>> NMPC exported')

if run_simulation:
    print(">> Runnig simulation")
    import time
    x_ref = np.array([1.0, 1.0, 1.0])
    n_sim = 30
    x0 = np.array([0.0, 0.0, 0.0])
    acados_integrator = AcadosSimSolver(ra, json_file = 'acados_ocp_' + model.name + '.json')

    simX = np.ndarray((n_sim+1, nx))
    simU = np.ndarray((n_sim, nu))

    xcurrent = x0
    simX[0,:] = xcurrent

    # closed loop
    for i in range(n_sim):

        # solve ocp
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        for k in range(N):
            # yref = np.zeros((ny, ))
            yref = np.hstack([x_ref, np.zeros((nu,))]).ravel()
            acados_ocp_solver.set(k, "yref", yref)
        yref = np.hstack([x_ref])
        acados_ocp_solver.set(N, "yref", yref)

        solvetime = -time.time()
        status = acados_ocp_solver.solve()
        solvetime += time.time()
        print("Step %d took %.4f ms" % (i, solvetime*1e3))

        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        simU[i,:] = acados_ocp_solver.get(0, "u")

        # simulate system
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", simU[i,:])

        status = acados_integrator.solve()
        if status != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update state
        xcurrent = acados_integrator.get("x")
        simX[i+1,:] = xcurrent

    # plot results
    import matplotlib.pyplot as plt

    plt.subplot(3,2,1)
    plt.plot(simX[:, 0])
    plt.grid()
    plt.ylabel("x")

    plt.subplot(3,2,2)
    plt.plot(simX[:, 1])
    plt.grid()
    plt.ylabel("y")

    plt.subplot(3,2,3)
    plt.plot(simX[:, 2])
    plt.grid()
    plt.ylabel("psi")
    plt.subplot(3,2,4)

    plt.plot(simX[:, 0], simX[:, 1])
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(3,2,5)
    plt.plot(simU[:, 0])
    plt.grid()
    plt.ylabel("v")

    plt.subplot(3,2,6)
    plt.plot(simU[:, 1])
    plt.grid()
    plt.ylabel("w")

    plt.tight_layout()
    plt.show()
