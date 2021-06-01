from acados_template import *
def export_ode_model():

    model_name = 'skidbot_control'

    # states (f_exp)
    x = SX.sym('x')  # Position in x-direction [m]
    y = SX.sym('y')  # Position in y-direction [m]
    psi = SX.sym('psi')  # Yaw orientation angle [rad]
    x = vertcat(x, y, psi)

    # controls
    v = SX.sym('v')  # Linear velocity [m/s]
    w = SX.sym('w')  # Angular velocity [rad/s]
    u = vertcat(v, w)

    # for f_impl
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    psi_dot = SX.sym('psi_dot')
    xdot = vertcat(x_dot, y_dot, psi_dot)


    # Model equations
    dx = v * cos(psi)
    dy = v * sin(psi)
    dpsi  = w

    # Explicit and Implicit functions
    f_expl = vertcat(dx, dy, dpsi)
    f_impl = xdot - f_expl

    # algebraic variables
    z = []

    # parameters
    p = []

    # dynamics
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    return model
