import numpy as np
def estimate_force_from_stiffness(K_c, delta_x): return np.dot(K_c, delta_x)
def calculate_performance_metric(F_measured, F_estimated):
    errors = [np.linalg.norm(m - e) for m, e in zip(F_measured, F_estimated)]
    return np.mean(errors) if errors else 0.0
