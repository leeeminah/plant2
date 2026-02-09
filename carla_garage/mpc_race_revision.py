#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA MPC Agent with CommonRoad Raceline (Pure MPC - No Curvature Preview)
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

def load_raceline(filename='town04_raceline_mincurv.pkl'):
    """
    CommonRoad raceline 로드 (position, yaw, velocity만 사용)
    
    Returns:
        raceline: list of dict with keys [x, y, z, yaw, velocity, s]
        metadata: dict with method and source info
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    raceline = data['raceline']
    metadata = data.get('metadata', {})
    
    print(f"Loaded raceline: {len(raceline)} waypoints")
    print(f"  Method: {metadata.get('method', 'unknown')}")
    print(f"  Source: {metadata.get('source', 'unknown')}")
    
    # ==================== YAW OFFSET 계산 ====================
    first_wp = raceline[0]
    second_wp = raceline[1]
    
    dx = second_wp['x'] - first_wp['x']
    dy = second_wp['y'] - first_wp['y']
    actual_yaw = np.arctan2(dy, dx)
    
    yaw_offset = actual_yaw - first_wp['yaw']
    yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
    
    print(f"  YAW coordinate transform:")
    print(f"    CommonRoad yaw: {np.rad2deg(first_wp['yaw']):.2f}°")
    print(f"    Actual direction: {np.rad2deg(actual_yaw):.2f}°")
    print(f"    Offset: {np.rad2deg(yaw_offset):.2f}°")
    
    # 모든 waypoint에 offset 적용
    for wp in raceline:
        wp['yaw_original'] = wp['yaw']
        wp['yaw'] = wp['yaw'] + yaw_offset
        wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
    
    print(f"    ✅ Applied offset to all {len(raceline)} waypoints")
    # ====================================================
    
    # 통계 출력
    velocities = [wp['velocity'] for wp in raceline]
    
    print(f"  Velocity: {min(velocities):.1f} - {max(velocities):.1f} m/s "
          f"({min(velocities)*3.6:.1f} - {max(velocities)*3.6:.1f} km/h)")
    print(f"  Note: Only position, yaw, velocity used (pure MPC)")
    
    return raceline, metadata

class MPCRaceController:
    """Pure MPC Controller (No Curvature Preview)"""
    
    def __init__(self, vehicle, raceline=None, config=None):
        self.vehicle = vehicle
        
        if config is None:
            config = {}
        
        self.wheel_base = config.get('wheelbase', 2.875)
        self.horizon = config.get('horizon', 20)
        self.dt = config.get('dt', 0.1)
        
        # State weights: [x, y, θ, v]        
        self.Q = ca.diag(config.get('Q', [200.0, 200.0, 50.0, 100.0]))
        # Control weights: [a, κ]        
        self.R = ca.diag(config.get('R', [0.5, 150.0]))
        # Terminal weights
        self.Qf = ca.diag(config.get('Qf', [300.0, 300.0, 100.0, 200.0]))
        
        # Control limits
        self.a_min = config.get('a_min', -10.0)
        self.a_max = config.get('a_max', 10.0)
        self.kappa_min = config.get('kappa_min', -0.2)
        self.kappa_max = config.get('kappa_max', 0.2)
        
        # State limits
        self.v_min = config.get('v_min', 0.0)
        self.v_max = config.get('v_max', 50.0)
        self.ay_max = config.get('ay_max', 12.0)
        self.discount_rate = config.get('discount_rate', 0.95)
        self.max_steer_angle = config.get('max_steer_angle', 1.22)
        
        # Soft constraint weights
        self.w_ay = config.get('w_ay', 500.0)  # Lateral acceleration penalty
        self.w_dkappa = config.get('w_dkappa', 100.0)  # Curvature rate penalty
        
        # Velocity Scaling
        self.velocity_scale = config.get('velocity_scale', 1.0)
        print(f"  Velocity scaling: {self.velocity_scale * 100:.0f}%")
        
        # Lap Counter
        self.lap_count = 0
        self.last_waypoint_idx = 0
        self.target_laps = config.get('target_laps', 1)
        self.path_length = len(raceline) if raceline else 0
        self.lap_threshold = int(self.path_length * 0.9)
        
        self.prev_solution = None

        # Logging
        self.trajectory_history = deque(maxlen=1000)
        self.optimal_trajectory = None
        
        # Visualization
        self.enable_viz = config.get('visualization', False)
        if self.enable_viz:
            self.setup_visualization()

        self.raceline = raceline
        
        # Setup MPC
        self.setup_mpc()
        
        print("MPC Controller initialized (Pure MPC - No Curvature Preview)")
        print(f"  Horizon: {self.horizon}")
        print(f"  dt: {self.dt}s")
        print(f"  v_max: {self.v_max} m/s ({self.v_max*3.6:.1f} km/h)")
        print(f"  Soft constraints: w_ay={self.w_ay}, w_dkappa={self.w_dkappa}")

    def setup_mpc(self):
        """Setup CasADi optimization (Pure MPC - 4D reference only)"""
        # State: [x, y, θ, v]        
        x = ca.SX.sym('x', 4)
        # Control: [a, κ]
        u = ca.SX.sym('u', 2)

        pos_x = x[0]
        pos_y = x[1]
        theta = x[2]
        v = x[3]

        a = u[0]
        kappa = u[1]

        # Dynamics (Trapezoidal integration)
        v_next = v + self.dt * a
        theta_next = theta + self.dt * kappa * v + (self.dt**2 / 2) * kappa * a
        
        x_next = ca.vertcat(
            pos_x + (self.dt / 2) * (v * ca.cos(theta) + v_next * ca.cos(theta_next)),
            pos_y + (self.dt / 2) * (v * ca.sin(theta) + v_next * ca.sin(theta_next)),
            theta_next,
            v_next
        )

        self.f = ca.Function('f', [x, u], [x_next])

        # Optimization variables
        X = ca.SX.sym('X', 4, self.horizon + 1)
        U = ca.SX.sym('U', 2, self.horizon)
        P = ca.SX.sym('P', 4)
        REF = ca.SX.sym('REF', 4, self.horizon + 1)  # ← 4D only: [x, y, θ, v]

        # Cost function
        cost = 0
        for k in range(self.horizon):
            discount = self.discount_rate ** k 
            
            # State tracking error
            state_error = ca.vertcat(
                X[0, k] - REF[0, k],
                X[1, k] - REF[1, k],
                ca.atan2(ca.sin(X[2, k] - REF[2, k]), ca.cos(X[2, k] - REF[2, k])),
                X[3, k] - REF[3, k]
            )
            
            cost += discount * ca.mtimes([state_error.T, self.Q, state_error])
            
            # Control effort
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])

            # ==================== Soft Constraints ====================
            # 1. Lateral acceleration penalty
            v_k = X[3, k]
            kappa_k = U[1, k]
            ay_k = v_k**2 * ca.fabs(kappa_k)
            ay_violation = ca.fmax(0, ay_k - self.ay_max)
            cost += self.w_ay * ay_violation**2
            
            # 2. Curvature rate penalty (smooth steering)
            if k > 0:
                dkappa = U[1, k] - U[1, k-1]
                cost += self.w_dkappa * dkappa**2
            # ==========================================================
        
        # Terminal cost
        state_error_final = ca.vertcat(
            X[0, self.horizon] - REF[0, self.horizon],
            X[1, self.horizon] - REF[1, self.horizon],
            ca.atan2(ca.sin(X[2, self.horizon] - REF[2, self.horizon]), 
                    ca.cos(X[2, self.horizon] - REF[2, self.horizon])),
            X[3, self.horizon] - REF[3, self.horizon]
        )
        cost += ca.mtimes([state_error_final.T, self.Qf, state_error_final])
        
        # Constraints
        g = []
        g.append(X[:, 0] - P)  # Initial condition
        
        for k in range(self.horizon):
            g.append(X[:, k+1] - self.f(X[:, k], U[:, k]))  # Dynamics
        
        g = ca.vertcat(*g)

        # Decision variables
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.vertcat(P, ca.reshape(REF, -1, 1))

        nlp = {'x': Z, 'f': cost, 'g': g, 'p': params}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 200,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_eq = 4 + 4 * self.horizon
    
    def solve(self, current_state, ref_traj):
        """Solve MPC optimization"""
        
        X0 = np.zeros((4, self.horizon + 1))
        
        for i in range(self.horizon + 1):
            X0[:, i] = ref_traj[:, i]
        X0[:, 0] = current_state
        
        U0 = np.zeros((2, self.horizon))
        Z0 = np.concatenate((X0.reshape(-1, order='F'), U0.reshape(-1, order='F')))
        
        # State bounds
        lbx_states = []
        ubx_states = []
        for _ in range(self.horizon + 1):
            lbx_states.extend([-ca.inf, -ca.inf, -ca.pi, self.v_min])
            ubx_states.extend([ca.inf, ca.inf, ca.pi, self.v_max])
        
        # ==================== Control bounds ====================
        lbx_controls = []
        ubx_controls = []
        
        for k in range(self.horizon):
            # Dynamic curvature limit based on current velocity
            v_k = max(current_state[3], 3.0)  # Conservative: use current velocity
            kappa_dynamic_max = self.ay_max / (v_k ** 2)
            
            # Apply both static and dynamic limits
            kappa_max_k = min(kappa_dynamic_max, self.kappa_max)
            kappa_min_k = max(-kappa_dynamic_max, self.kappa_min)
            
            lbx_controls.extend([self.a_min, kappa_min_k])
            ubx_controls.extend([self.a_max, kappa_max_k])
        # ========================================================

        lbx = lbx_states + lbx_controls
        ubx = ubx_states + ubx_controls
        
        lbg = [0.0] * self.n_eq
        ubg = [0.0] * self.n_eq

        p_val = np.concatenate((
            current_state,
            ref_traj.reshape(-1, order='F')
        ))
        
        # Warm start
        if self.prev_solution is not None:
            X_prev = self.prev_solution['X']
            U_prev = self.prev_solution['U']
            
            X_warm = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
            U_warm = np.hstack([U_prev[:, 1:], U_prev[:, -1:]])
            
            x0 = np.concatenate((X_warm.reshape(-1, order='F'), U_warm.reshape(-1, order='F')))
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
            
            n_states = 4 * (self.horizon + 1)
            X_opt = Z[:n_states].reshape((4, self.horizon+1), order='F')
            U_opt = Z[n_states:].reshape((2, self.horizon), order='F')
            
            self.prev_solution = {
                'x': Z,
                'lam_x': sol['lam_x'].full().flatten(),
                'lam_g': sol['lam_g'].full().flatten(),
                'X': X_opt,
                'U': U_opt
            }
            
            acceleration = float(U_opt[0, 0])
            kappa_cmd = float(U_opt[1, 0])
            
            self.optimal_trajectory = (X_opt[0, :], X_opt[1, :])
            
            return acceleration, kappa_cmd, True
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return 0.0, 0.0, False 
    
    def step(self, waypoints):
        """Execute one control step"""
        current_state = self.get_state()
        ref_traj = self.get_reference_trajectory(waypoints)
        
        acceleration, kappa_cmd, success = self.solve(current_state, ref_traj)
        
        self.trajectory_history.append((current_state[0], current_state[1]))
        
        control = carla.VehicleControl()
                
        if success:
            # Safety clip
            kappa_cmd = np.clip(kappa_cmd, self.kappa_min, self.kappa_max)

            # Steering
            steering_angle = np.arctan(kappa_cmd * self.wheel_base)
            steering_angle = np.clip(
                steering_angle,
                -self.max_steer_angle,
                self.max_steer_angle
            )
            control.steer = float(steering_angle / self.max_steer_angle)

            # Throttle / Brake
            if acceleration > 0:
                control.throttle = float(np.clip(acceleration / self.a_max, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-acceleration / abs(self.a_min), 0.0, 1.0))
            
            return control, acceleration, kappa_cmd
        else:
            control.throttle = 0.0
            control.brake = 0.5
            control.steer = 0.0
            control.hand_brake = False
            control.reverse = False
            control.manual_gear_shift = False
            return control, 0.0, 0.0

    def get_state(self):
        """Get current vehicle state: [x, y, θ, v]"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        theta = np.deg2rad(transform.rotation.yaw)
        v = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return np.array([x, y, theta, v])

    def get_reference_trajectory(self, waypoints):
        """
        Convert raceline waypoints to reference trajectory
        4D only: [x, y, θ, v] - NO curvature
        """
        ref_traj = np.zeros((4, self.horizon + 1))
        
        if len(waypoints) < self.horizon + 1:
            waypoints_extended = list(waypoints) + [waypoints[-1]] * (self.horizon + 1 - len(waypoints))
        else:
            waypoints_extended = waypoints[:self.horizon + 1]
        
        for i in range(self.horizon + 1):
            wp = waypoints_extended[i]
            
            ref_traj[0, i] = wp['x']
            ref_traj[1, i] = wp['y']
            ref_traj[2, i] = wp['yaw']  # Already corrected
            ref_traj[3, i] = wp['velocity'] * self.velocity_scale
        
        return ref_traj

    def get_lookahead_waypoints(self, current_location, lookahead=25):
        """Extract lookahead waypoints from raceline"""
        
        min_dist = float('inf')
        closest_idx = 0
        
        # ✅ Search within ±100 range of last index (prevent jumps)
        search_start = max(0, self.last_waypoint_idx - 50)
        search_end = min(len(self.raceline), self.last_waypoint_idx + 100)
        
        for i in range(search_start, search_end):
            wp = self.raceline[i % len(self.raceline)]
            dx = wp['x'] - current_location.x
            dy = wp['y'] - current_location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i % len(self.raceline)
        
        # ✅ Warn if vehicle is far off track
        if min_dist > 20.0:
            print(f"⚠️ WARNING: Large CTE ({min_dist:.1f}m) - possible off-track!")

        # Lap check
        if self.last_waypoint_idx > self.lap_threshold and closest_idx < int(self.path_length * 0.1):
            self.lap_count += 1
            print(f"\n🏁 Lap {self.lap_count} completed!")
        
        self.last_waypoint_idx = closest_idx
        
        # Lookahead extraction
        lookahead_wps = []
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.raceline)
            lookahead_wps.append(self.raceline[idx])
        
        return lookahead_wps, closest_idx 

    def is_finished(self):
        """Check if race is finished"""
        return self.lap_count >= self.target_laps

    def setup_visualization(self):
        """Setup matplotlib"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.current_pos_plot, = self.ax.plot([], [], 'ro', markersize=12, label='Vehicle', zorder=5)
        self.trajectory_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='History', alpha=0.7)
        self.optimal_path_plot, = self.ax.plot([], [], 'r--', linewidth=3, label='MPC Path', alpha=0.8)
        self.waypoints_plot, = self.ax.plot([], [], 'g*', markersize=10, label='Raceline', alpha=0.6)
        
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('Pure MPC Controller - Raceline Tracking', fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        plt.tight_layout()

    def update_visualization(self, waypoints=None):
        """Update visualization"""
        if not self.enable_viz:
            return
        
        current_state = self.get_state()
        
        self.current_pos_plot.set_data([current_state[0]], [current_state[1]])
        
        if len(self.trajectory_history) > 1:
            traj_x = [pos[0] for pos in self.trajectory_history]
            traj_y = [pos[1] for pos in self.trajectory_history]
            self.trajectory_plot.set_data(traj_x, traj_y)
        
        if self.optimal_trajectory is not None:
            opt_x, opt_y = self.optimal_trajectory
            self.optimal_path_plot.set_data(opt_x, opt_y)
        
        if waypoints is not None:
            wp_x = [wp['x'] for wp in waypoints]
            wp_y = [wp['y'] for wp in waypoints]
            self.waypoints_plot.set_data(wp_x, wp_y)
        
        zoom_range = 100
        self.ax.set_xlim([current_state[0] - zoom_range, current_state[0] + zoom_range])
        self.ax.set_ylim([current_state[1] - zoom_range, current_state[1] + zoom_range])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    """Main function"""
    
    print("Connecting to CARLA...")
    client = carla.Client('172.22.39.175', 2000)
    client.set_timeout(10.0)
    
    world = client.load_world('Town04')
    print(f"Connected: {world.get_map().name}")
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    for actor in world.get_actors():
        tid = actor.type_id.lower()
        if "traffic" in tid or "speed" in tid or "sign" in tid:
            try:
                actor.set_collision_enabled(carla.CollisionEnabled.NoCollision)
            except:
                pass

    # Load raceline
    raceline, metadata = load_raceline('routes/town04_raceline_mincurv13_1.pkl')

    # Spawn setup
    first_wp = raceline[0]
    
    print(f"\n📍 Spawn location:")
    print(f"   Position: ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
    print(f"   Yaw: {np.rad2deg(first_wp['yaw']):.2f}° (corrected)")
    
    map_obj = world.get_map()
    test_location = carla.Location(x=first_wp['x'], y=first_wp['y'], z=0.0)
    waypoint_on_road = map_obj.get_waypoint(test_location, project_to_road=True)
    
    if waypoint_on_road is not None:
        spawn_z = waypoint_on_road.transform.location.z + 1.0
        print(f"   Road height: {waypoint_on_road.transform.location.z:.2f}m")
    else:
        spawn_z = 1.0
        print(f"   ⚠️ Could not find road waypoint, using default z=1.0")
    
    spawn_transform = carla.Transform(
        carla.Location(x=first_wp['x'], y=first_wp['y'], z=spawn_z),
        carla.Rotation(yaw=np.rad2deg(first_wp['yaw']))
    )
    
    vehicle = None
    spawn_attempts = 0
    max_attempts = 5
    
    while vehicle is None and spawn_attempts < max_attempts:
        try:
            spawn_z_attempt = spawn_z + (spawn_attempts * 0.5)
            spawn_transform.location.z = spawn_z_attempt
            
            print(f"\nSpawn attempt {spawn_attempts + 1}/{max_attempts} at z={spawn_z_attempt:.2f}m...")
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if vehicle is None:
                spawn_attempts += 1
                time.sleep(0.5)
            else:
                print(f"✅ Vehicle spawned successfully")
                break
                
        except Exception as e:
            print(f"   Spawn error: {e}")
            spawn_attempts += 1
            time.sleep(0.5)
    
    if vehicle is None:
        print("\n❌ Failed to spawn vehicle!")
        return
    
    time.sleep(0.5)
    
    spectator = world.get_spectator()
    step = 0
    
    try:
        time.sleep(1.0)
        
        # MPC Config (Pure MPC - 4D reference)
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 20,
            'dt': 0.1,
            
            # Tracking weights
            'Q': [200.0, 200.0, 50.0, 100.0],   # [x, y, θ, v]
            'R': [0.5, 150.0],                   # [a, κ]
            'Qf': [300.0, 300.0, 100.0, 200.0],
            
            # Soft constraint weights
            'w_ay': 500.0,      # Lateral acceleration penalty
            'w_dkappa': 100.0,  # Curvature rate penalty
            
            # Limits
            'a_min': -10.0,
            'a_max': 10.0,
            'kappa_min': -0.2,
            'kappa_max': 0.2,
            'ay_max': 12.0,
            'v_min': 0.0,
            'v_max': 50.0,
            'max_steer_angle': 1.22,
            
            # Other
            'discount_rate': 0.95,
            'velocity_scale': 1.0,
            'visualization': True,
            'target_laps': 1
        }

        mpc = MPCRaceController(vehicle, raceline, config=mpc_config)
        lookahead = 50
        
        print("\n🏁 Starting race...")
        
        while True:
            current_location = vehicle.get_location()
            
            if mpc.is_finished():
                print(f"\n🏁 Race finished! {mpc.lap_count} lap(s) completed")
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0
                ))
                break
            
            # Get waypoints
            lookahead_waypoints, closest_idx = mpc.get_lookahead_waypoints(
                current_location, 
                lookahead=lookahead
            )
            
            # MPC step
            control, accel, kappa_cmd = mpc.step(lookahead_waypoints)
            vehicle.apply_control(control)
            
            # Spectator update
            if step % 10 == 0:
                vehicle_transform = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=60),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
            
            # Logging
            if step % 5 == 0:
                mpc.update_visualization(lookahead_waypoints)
                
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                curr_x = current_location.x
                curr_y = current_location.y
                
                # CTE
                cte = float('inf')
                for wp in lookahead_waypoints[:5]:
                    dx = current_location.x - wp['x']
                    dy = current_location.y - wp['y']
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < cte:
                        cte = dist
                
                # Steering
                steering_angle_rad = np.arctan(kappa_cmd * mpc.wheel_base)
                steering_angle_deg = np.rad2deg(steering_angle_rad)
                
                # Lateral ay
                actual_ay = speed**2 * abs(kappa_cmd)
                
                # Reference
                ref_traj = mpc.get_reference_trajectory(lookahead_waypoints)
                v_ref = ref_traj[3, 0]
                
                # Print
                print(f"\n{'='*80}")
                print(f"Step {step:4d} | Progress: {closest_idx}/{len(raceline)} ({100*closest_idx/len(raceline):.1f}%)")
                print(f"{'='*80}")
                print(f"Position:  ({curr_x:6.1f}, {curr_y:6.1f}) | CTE: {cte:5.2f}m")
                print(f"Velocity:  {speed:5.1f} m/s (ref: {v_ref:5.1f} m/s)")
                
                print(f"\nControl Inputs:")
                print(f"  Acceleration:  {accel:+6.2f} m/s²")
                print(f"  Curvature:     {kappa_cmd:+7.4f} (1/m)")
                print(f"      → Radius:  {1/abs(kappa_cmd) if abs(kappa_cmd) > 1e-6 else float('inf'):7.1f} m")
                print(f"  Lateral ay:    {actual_ay:6.2f} m/s²")
                
                print(f"\nSteering:")
                print(f"  Angle:   {steering_angle_deg:+6.2f}°")
                print(f"  Normalized: {control.steer:+6.3f}")
                
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
        
        print(f"\nFinished after {step} steps ({step * 0.1:.1f}s)")
        
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