# src/control/pid_optimizer.py
# PID 優化控制器

class PIDOptimizer:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
