#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Dynamics MPC for Racing
- State: [e_y, e_ψ, v] (errors relative to path)
- Reference: [κ_ref, v_ref] (path curvature & velocity)
- Feedforward: Built into dynamics
"""

import sys
import math
import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from collections import deque
import pickle

try:
    import carla
except ImportError:
    print("CARLA Python API not found!")
    sys.exit(1)

def load_raceline(filename='routes/town04_raceline_mincurv.pkl'):
    """Load CommonRoad raceline with yaw correction"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    raceline = data['raceline']
    metadata = data.get('metadata', {})
    
    print(f"Loaded raceline: {len(raceline)} waypoints")
    
    # Yaw offset 계산
    first_wp = raceline[0]
    second_wp = raceline[1]
    
    dx = second_wp['x'] - first_wp['x']
    dy = second_wp['y'] - first_wp['y']
    actual_yaw = np.arctan2(dy, dx)
    
    yaw_offset = actual_yaw - first_wp['yaw']
    yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
    
    print(f"  YAW offset: {np.rad2deg(yaw_offset):.2f}°")
    
    # 모든 waypoint에 offset 적용
    for wp in raceline:
        wp['yaw_original'] = wp['yaw']
        wp['yaw'] = wp['yaw'] + yaw_offset
        wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
    
    return raceline, metadata


class ErrorDynamicsMPC:
    """
    Error-Dynamics MPC Controller
    
    State: x = [e_y, e_ψ, v]
    - e_y: lateral error (m)
    - e_ψ: heading error (rad)
    - v: velocity (m/s)
    
    Control: u = [a, Δκ]
    - a: acceleration (m/s²)
    - Δκ: curvature correction (1/m)
    
    Reference: r = [κ_ref, v_ref]
    - κ_ref: path curvature (1/m) ← Feedforward!
    - v_ref: target velocity (m/s)
    """
    
    def __init__(self, vehicle, raceline=None, config=None):
        self.vehicle = vehicle
        
        if config is None:
            config = {}
        
        self.wheel_base = config.get('wheelbase', 2.875)
        self.horizon = config.get('horizon', 15)
        self.dt = config.get('dt', 0.1)
        
        # Error state weights: [e_y, e_ψ, v]
        self.Q = ca.diag(config.get('Q', [100.0, 50.0, 10.0]))
        
        # Control weights: [a, Δκ]
        self.R = ca.diag(config.get('R', [1.0, 10.0]))
        
        # Terminal weights
        self.Qf = ca.diag(config.get('Qf', [200.0, 100.0, 20.0]))
        
        # Control limits
        self.a_min = config.get('a_min', -5.0)
        self.a_max = config.get('a_max', 5.0)
        self.kappa_min = config.get('kappa_min', -0.2)  # ← Changed!
        self.kappa_max = config.get('kappa_max', 0.2)   # ← Changed!
        
        # State limits
        self.v_min = config.get('v_min', 0.0)
        self.v_max = config.get('v_max', 30.0)
        self.e_y_max = config.get('e_y_max', 5.0)  # Maximum lateral error
        self.e_psi_max = config.get('e_psi_max', np.pi/3)  # Maximum heading error
        
        # Physical limits
        self.kappa_max = config.get('kappa_max', 0.2)
        self.ay_max = config.get('ay_max', 5.0)
        self.max_steer_angle = config.get('max_steer_angle', 1.22)
        
        self.discount_rate = config.get('discount_rate', 0.95)
        self.velocity_scale = config.get('velocity_scale', 0.8)
        
        # Lap tracking
        self.lap_count = 0
        self.last_waypoint_idx = 0
        self.target_laps = config.get('target_laps', 1)
        self.path_length = len(raceline) if raceline else 0
        self.lap_threshold = int(self.path_length * 0.9)
        
        self.prev_solution = None
        self.trajectory_history = deque(maxlen=1000)
        
        # Visualization
        self.enable_viz = config.get('visualization', False)
        if self.enable_viz:
            self.setup_visualization()
        
        self.raceline = raceline
        
        # Setup MPC
        self.setup_error_dynamics_mpc()
        
        print("Error-Dynamics MPC Controller initialized")
        print(f"  State:   [e_y, e_ψ, v]")
        print(f"  Control: [a, Δκ]")
        print(f"  Horizon: {self.horizon}, dt: {self.dt}s")
    
    def setup_error_dynamics_mpc(self):
        """Setup Error-Dynamics MPC optimization"""
        
        # State: [e_y, e_ψ, v]
        x = ca.SX.sym('x', 3)
        
        # Control: [a, κ]  ← Changed from Δκ to κ!
        u = ca.SX.sym('u', 2)
        
        # Reference: [κ_ref, v_ref]
        ref = ca.SX.sym('ref', 2)
        
        e_y = x[0]
        e_psi = x[1]
        v = x[2]
        
        a = u[0]
        kappa = u[1]  # ← Absolute curvature!
        
        kappa_ref = ref[0]
        v_ref = ref[1]
        
        # ==================== Error Dynamics ====================
        # Lateral error dynamics
        e_y_dot = v * ca.sin(e_psi)
        
        # Heading error dynamics
        # ė_ψ = v·κ - v_ref·κ_ref
        #     = (yaw_rate_actual) - (yaw_rate_reference)
        e_psi_dot = v * kappa - v_ref * kappa_ref
        
        # Velocity dynamics
        v_dot = a
        
        # State derivative
        x_dot = ca.vertcat(e_y_dot, e_psi_dot, v_dot)
        
        # Discrete-time dynamics
        x_next = x + self.dt * x_dot
        # =======================================================
        
        # Dynamics function: x_next = f(x, u, ref)
        self.f = ca.Function('f', [x, u, ref], [x_next])
        
        # ==================== Optimization Setup ====================
        X = ca.SX.sym('X', 3, self.horizon + 1)
        U = ca.SX.sym('U', 2, self.horizon)
        
        X0 = ca.SX.sym('X0', 3)
        REF = ca.SX.sym('REF', 2, self.horizon + 1)
        
        # Cost function
        cost = 0
        
        for k in range(self.horizon):
            discount = self.discount_rate ** k
            
            # State cost (error minimization)
            state_cost = ca.mtimes([X[:, k].T, self.Q, X[:, k]])
            
            # Control cost - Simple regularization!
            # Feedforward is ONLY in dynamics, NOT in cost!
            control_cost = ca.mtimes([U[:, k].T, self.R, U[:, k]])
            
            cost += discount * (state_cost + control_cost)
        
        # Terminal cost
        terminal_cost = ca.mtimes([X[:, self.horizon].T, self.Qf, X[:, self.horizon]])
        cost += terminal_cost
        
        # Constraints
        g = []
        g.append(X[:, 0] - X0)
        
        for k in range(self.horizon):
            g.append(X[:, k+1] - self.f(X[:, k], U[:, k], REF[:, k]))
        
        g = ca.vertcat(*g)
        
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.vertcat(X0, ca.reshape(REF, -1, 1))
        
        nlp = {'x': Z, 'f': cost, 'g': g, 'p': params}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 150,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_eq = 3 + 3 * self.horizon
        
        print("  Error-dynamics model:")
        print("    ė_y = v·sin(e_ψ)")
        print("    ė_ψ = v·κ - v_ref·κ_ref  ← Feedforward ONLY here!")
        print("    v̇   = a")
        print("  Cost function:")
        print("    J = Σ (e_y² + e_ψ² + v² + a² + κ²)")
        print("    → NO (κ - κ_ref)² term! Pure correction!")
    
    def get_carla_state(self):
        """Get vehicle state in CARLA coordinates"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        v = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return x, y, yaw, v
    
    def compute_error_state(self, waypoints):
        """
        Compute error state relative to path
        
        Returns:
            error_state: [e_y, e_ψ, v]
            closest_idx: index of closest waypoint
        """
        x, y, yaw, v = self.get_carla_state()
        
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(waypoints):
            dx = wp['x'] - x
            dy = wp['y'] - y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        wp_closest = waypoints[closest_idx]
        
        # Lateral error (signed distance to path)
        dx = x - wp_closest['x']
        dy = y - wp_closest['y']
        
        # Path tangent direction
        yaw_path = wp_closest['yaw']
        
        # Rotate to path frame
        e_y = -dx * np.sin(yaw_path) + dy * np.cos(yaw_path)
        
        # Heading error
        e_psi = yaw - yaw_path
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))  # Wrap to [-π, π]
        
        error_state = np.array([e_y, e_psi, v])
        
        return error_state, closest_idx
    
    def get_reference_trajectory(self, waypoints):
        """
        Get reference trajectory: [κ_ref, v_ref]
        
        Returns:
            ref_traj: 2 × (horizon+1) array
        """
        ref_traj = np.zeros((2, self.horizon + 1))
        
        for i in range(self.horizon + 1):
            if i >= len(waypoints):
                wp = waypoints[-1]
            else:
                wp = waypoints[i]
            
            ref_traj[0, i] = wp['curvature']  # κ_ref (feedforward!)
            ref_traj[1, i] = wp['velocity'] * self.velocity_scale  # v_ref
        
        return ref_traj
    
    def solve(self, error_state, ref_traj):
        """Solve error-dynamics MPC"""
        
        # Initial guess
        X0 = np.zeros((3, self.horizon + 1))
        U0 = np.zeros((2, self.horizon))
        
        Z0 = np.concatenate((X0.reshape(-1, order='F'), U0.reshape(-1, order='F')))
        
        # Bounds on error states
        lbx_states = []
        ubx_states = []
        
        for _ in range(self.horizon + 1):
            lbx_states.extend([-self.e_y_max, -self.e_psi_max, self.v_min])
            ubx_states.extend([self.e_y_max, self.e_psi_max, self.v_max])
        
        # Bounds on controls: [a, κ]
        lbx_controls = []
        ubx_controls = []
        
        for k in range(self.horizon):
            lbx_controls.extend([self.a_min, self.kappa_min])
            ubx_controls.extend([self.a_max, self.kappa_max])
        
        lbx = lbx_states + lbx_controls
        ubx = ubx_states + ubx_controls
        
        # Equality constraints (dynamics)
        lbg = [0.0] * self.n_eq
        ubg = [0.0] * self.n_eq
        
        # Parameter values
        p_val = np.concatenate((error_state, ref_traj.reshape(-1, order='F')))
        
        # Warm start
        if self.prev_solution is not None:
            x0 = self.prev_solution['x']
            lam_x0 = self.prev_solution['lam_x']
            lam_g0 = self.prev_solution['lam_g']
        else:
            x0 = Z0
            lam_x0 = np.zeros(len(lbx))
            lam_g0 = np.zeros(self.n_eq)
        
        try:
            sol = self.solver(
                x0=x0,
                lam_x0=lam_x0,
                lam_g0=lam_g0,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=p_val
            )
            
            Z = sol['x'].full().flatten()
            
            n_states = 3 * (self.horizon + 1)
            X_opt = Z[:n_states].reshape((3, self.horizon+1), order='F')
            U_opt = Z[n_states:].reshape((2, self.horizon), order='F')
            
            self.prev_solution = {
                'x': Z,
                'lam_x': sol['lam_x'].full().flatten(),
                'lam_g': sol['lam_g'].full().flatten(),
                'X': X_opt,
                'U': U_opt
            }
            
            # Extract first control
            acceleration = float(U_opt[0, 0])
            kappa = float(U_opt[1, 0])  # ← Absolute curvature!
            
            # Reference curvature (for comparison)
            kappa_ref = ref_traj[0, 0]
            
            # Curvature error
            kappa_error = kappa - kappa_ref
            
            return acceleration, kappa, kappa_ref, kappa_error, True
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return 0.0, 0.0, 0.0, 0.0, False
    
    def get_lookahead_waypoints(self, current_location, lookahead=50):
        """Get lookahead waypoints from raceline"""
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.raceline):
            dx = wp['x'] - current_location.x
            dy = wp['y'] - current_location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Lap tracking
        if self.last_waypoint_idx > self.lap_threshold and closest_idx < int(self.path_length * 0.1):
            self.lap_count += 1
            print(f"\n🏁 Lap {self.lap_count} completed!")
        
        self.last_waypoint_idx = closest_idx
        
        # Extract lookahead waypoints
        lookahead_wps = []
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.raceline)
            lookahead_wps.append(self.raceline[idx])
        
        return lookahead_wps, closest_idx
    
    def is_finished(self):
        """Check if finished"""
        return self.lap_count >= self.target_laps
    
    def step(self, waypoints):
        """Execute one control step"""
        
        # Compute error state
        error_state, _ = self.compute_error_state(waypoints)
        
        # Get reference trajectory
        ref_traj = self.get_reference_trajectory(waypoints)
        
        # Solve MPC
        accel, kappa, kappa_ref, kappa_error, success = self.solve(error_state, ref_traj)
        
        # Apply control
        control = carla.VehicleControl()
        
        if success:
            # Curvature → Steering
            steering_angle = np.arctan(kappa * self.wheel_base)
            steering_angle = np.clip(steering_angle, -self.max_steer_angle, self.max_steer_angle)
            control.steer = float(steering_angle / self.max_steer_angle)
            
            # Acceleration → Throttle/Brake
            if accel > 0:
                control.throttle = float(np.clip(accel / 3.0, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-accel / 3.0, 0.0, 1.0))
        else:
            control.throttle = 0.0
            control.brake = 0.5
            control.steer = 0.0
        
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        
        return control, accel, kappa, kappa_ref, kappa_error, error_state
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.current_pos_plot, = self.ax.plot([], [], 'ro', markersize=12, label='Vehicle')
        self.trajectory_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='History')
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Error-Dynamics MPC - Path Tracking')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        plt.tight_layout()
    
    def update_visualization(self, current_location):
        """Update visualization"""
        if not self.enable_viz:
            return
        
        self.current_pos_plot.set_data([current_location.x], [current_location.y])
        
        if len(self.trajectory_history) > 1:
            traj_x = [pos[0] for pos in self.trajectory_history]
            traj_y = [pos[1] for pos in self.trajectory_history]
            self.trajectory_plot.set_data(traj_x, traj_y)
        
        zoom_range = 100
        self.ax.set_xlim([current_location.x - zoom_range, current_location.x + zoom_range])
        self.ax.set_ylim([current_location.y - zoom_range, current_location.y + zoom_range])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    """Main function"""
    
    print("Connecting to CARLA...")
    client = carla.Client('172.22.39.179', 2000)
    client.set_timeout(10.0)
    
    world = client.load_world('Town04')
    print(f"Connected: {world.get_map().name}")
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    
    # Load raceline
    raceline, metadata = load_raceline('routes/town04_raceline_mincurv.pkl')
    
    # Spawn vehicle
    first_wp = raceline[0]
    
    map_obj = world.get_map()
    test_location = carla.Location(x=first_wp['x'], y=first_wp['y'], z=0.0)
    waypoint_on_road = map_obj.get_waypoint(test_location, project_to_road=True)
    
    if waypoint_on_road is not None:
        spawn_z = waypoint_on_road.transform.location.z + 1.0
    else:
        spawn_z = 1.0
    
    spawn_transform = carla.Transform(
        carla.Location(x=first_wp['x'], y=first_wp['y'], z=spawn_z),
        carla.Rotation(yaw=np.rad2deg(first_wp['yaw']))
    )
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    
    if vehicle is None:
        print("Failed to spawn vehicle!")
        return
    
    print(f"✅ Vehicle spawned at ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
    
    time.sleep(0.5)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    time.sleep(0.5)
    
    spectator = world.get_spectator()
    step = 0
    
    try:
        # MPC Configuration
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 15,
            'dt': 0.1,
            'Q': [100.0, 50.0, 10.0],      # [e_y, e_ψ, v]
            'R': [1.0, 10.0],               # [a, κ]
            'Qf': [200.0, 100.0, 20.0],
            'a_min': -5.0,
            'a_max': 5.0,
            'kappa_min': -0.2,    # ← Absolute curvature limits!
            'kappa_max': 0.2,     # ← Absolute curvature limits!
            'v_min': 0.0,
            'v_max': 30.0,
            'e_y_max': 5.0,
            'e_psi_max': np.pi/3,
            'ay_max': 5.0,
            'max_steer_angle': 1.22,
            'discount_rate': 0.95,
            'velocity_scale': 0.7,
            'visualization': True,
            'target_laps': 1
        }
        
        mpc = ErrorDynamicsMPC(vehicle, raceline, config=mpc_config)
        
        print("\n🏁 Starting race...")
        
        while True:
            current_location = vehicle.get_location()
            
            if mpc.is_finished():
                print(f"\n🏁 Race finished! {mpc.lap_count} lap(s) completed")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                break
            
            # Get lookahead waypoints
            lookahead_waypoints, closest_idx = mpc.get_lookahead_waypoints(current_location, lookahead=50)
            
            # MPC step
            control, accel, kappa, kappa_ref, kappa_error, error_state = mpc.step(lookahead_waypoints)
            vehicle.apply_control(control)
            
            # Update spectator
            if step % 10 == 0:
                vehicle_transform = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=60),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
            
            # Logging
            if step % 5 == 0:
                mpc.update_visualization(current_location)
                
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                speed_kmh = speed * 3.6
                
                e_y, e_psi, v = error_state
                
                print(f"\n{'='*80}")
                print(f"Step {step:4d} | Progress: {closest_idx}/{len(raceline)} ({100*closest_idx/len(raceline):.1f}%)")
                print(f"{'='*80}")
                print(f"Error State:")
                print(f"  e_y (lateral):   {e_y:+7.3f} m")
                print(f"  e_ψ (heading):   {e_psi:+7.3f} rad = {np.rad2deg(e_psi):+6.2f}°")
                print(f"  v (velocity):    {speed_kmh:6.1f} km/h")
                
                print(f"\nControl (Error-Dynamics):")
                print(f"  a (acceleration):  {accel:+6.2f} m/s²")
                print(f"  κ (curvature):     {kappa:+7.4f} (1/m)")
                print(f"  κ_ref (FF):        {kappa_ref:+7.4f} (1/m)")
                print(f"  Correction:        {kappa_error:+7.4f} (1/m)")
                print(f"    → FF only in dynamics: ė_ψ = v·κ - v_ref·κ_ref")
                
                print(f"\nCARLA Control:")
                print(f"  Throttle:  {control.throttle:.3f}")
                print(f"  Brake:     {control.brake:.3f}")
                print(f"  Steer:     {control.steer:+.3f}")
                print(f"{'='*80}")
            
            time.sleep(0.1)
            step += 1
            
            if step > 3000:
                print("\n⏱️ Timeout")
                break
        
        print(f"\nFinished after {step} steps")
        
        if mpc.enable_viz:
            input("\nPress Enter to exit...")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        if vehicle is not None:
            vehicle.destroy()
        print("Done!")


if __name__ == '__main__':
    main()