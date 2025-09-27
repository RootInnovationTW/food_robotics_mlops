import numpy as np
def calculate_robot_cycle_time(t_o, t_h, t_w): return t_o + t_h + t_w
def calculate_robot_productivity(t_c_minutes): return 60.0 / t_c_minutes if t_c_minutes > 0 else 0.0
def calculate_payback_period(inv_cost, income, op_cost):
    net_flow = income - op_cost
    return inv_cost / net_flow if net_flow > 0 else float('inf')
