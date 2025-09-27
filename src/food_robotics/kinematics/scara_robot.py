import numpy as np

# --- Functions from previous step (Productivity Analysis) ---
def calculate_robot_cycle_time(t_o: float, t_h: float, t_w: float) -> float:
    """Calculates the SCARA robot cycle time for a pick-and-place task."""
    return t_o + t_h + t_w

def calculate_robot_productivity(t_c_minutes: float) -> float:
    """Calculates the SCARA robot's production rate (productivity)."""
    if t_c_minutes <= 0: return 0.0
    return 60.0 / t_c_minutes

# --- NEW: Kinematics from scara_fish_processing ---
class SCARAKinematics:
    """Handles inverse kinematics for a 2-link SCARA robot."""
    def __init__(self, l1: float, l2: float):
        self.l1 = l1
        self.l2 = l2
        print(f"SCARA kinematics initialized with L1={l1}m, L2={l2}m")

    def inverse_kinematics(self, x: float, y: float) -> (tuple[float, float] | None):
        """
        Calculates the joint angles (theta1, theta2) for a given (X, Y) coordinate.
        
        Returns:
            A tuple of (theta1, theta2) in radians or None if the point is unreachable.
        """
        # Distance from origin squared
        d_sq = x**2 + y**2
        
        # Check if the point is within the reachable workspace
        if d_sq > (self.l1 + self.l2)**2 or d_sq < (self.l1 - self.l2)**2:
            print(f"Warning: Point ({x}, {y}) is unreachable.")
            return None

        # Angle for joint 2 (elbow)
        cos_theta2 = (d_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0)) # Elbow-up solution

        # Angle for joint 1 (shoulder)
        alpha = np.arctan2(y, x)
        beta_num = self.l2 * np.sin(theta2)
        beta_den = self.l1 + self.l2 * np.cos(theta2)
        beta = np.arctan2(beta_num, beta_den)
        theta1 = alpha - beta
        
        return (theta1, theta2)
