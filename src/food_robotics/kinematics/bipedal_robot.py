import numpy as np
def calculate_zero_moment_point(link_data, g):
    num_x = sum(d['I']*d['w_dot'] - d['m']*d['x_ddot']*d['z'] - d['m']*d['x']*(g+d['z_ddot']) for d in link_data)
    num_y = sum(d['I']*d['w_dot'] - d['m']*d['y_ddot']*d['z'] - d['m']*d['y']*(g+d['z_ddot']) for d in link_data)
    den = sum(d['m']*(d['z_ddot']+g) for d in link_data)
    return (num_x/den, num_y/den) if den != 0 else (0,0)
def calculate_dynamic_balance_margin(zmp, foot_dims, angle):
    x_zmp, y_zmp = zmp; f_s, f_w = foot_dims
    x_dbm = (f_s/2)*np.cos(angle) - abs(x_zmp)
    y_dbm = (f_w/2)*np.cos(angle) - abs(y_zmp)
    return x_dbm, y_dbm
