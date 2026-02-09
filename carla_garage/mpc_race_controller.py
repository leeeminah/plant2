#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA MPC Agent with CommonRoad Raceline (velocity + curvature)
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
    CommonRoad raceline 로드 (velocity, curvature 포함)
    
    Returns:
        raceline: list of dict with keys [x, y, z, yaw, velocity, curvature, s]
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
    # 첫 두 waypoint로 실제 진행 방향 계산
    first_wp = raceline[0]
    second_wp = raceline[1]
    
    dx = second_wp['x'] - first_wp['x']
    dy = second_wp['y'] - first_wp['y']
    actual_yaw = np.arctan2(dy, dx)
    
    # CommonRoad yaw와의 차이 = offset
    yaw_offset = actual_yaw - first_wp['yaw']
    
    # Wrap to [-pi, pi]
    yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
    
    print(f"  YAW coordinate transform:")
    print(f"    CommonRoad yaw: {np.rad2deg(first_wp['yaw']):.2f}°")
    print(f"    Actual direction: {np.rad2deg(actual_yaw):.2f}°")
    print(f"    Offset: {np.rad2deg(yaw_offset):.2f}°")
    
    # 모든 waypoint에 offset 적용
    for wp in raceline:
        wp['yaw_original'] = wp['yaw']  # 원본 백업
        wp['yaw'] = wp['yaw'] + yaw_offset  # offset 적용
        # Wrap to [-pi, pi]
        wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
    
    print(f"    ✅ Applied offset to all {len(raceline)} waypoints")
    # ====================================================
    
    # 통계 출력
    velocities = [wp['velocity'] for wp in raceline]
    curvatures = [abs(wp['curvature']) for wp in raceline]
    
    print(f"  Velocity: {min(velocities):.1f} - {max(velocities):.1f} m/s "
          f"({min(velocities)*3.6:.1f} - {max(velocities)*3.6:.1f} km/h)")
    print(f"  Max |curvature|: {max(curvatures):.4f} (1/m)")
    
    return raceline, metadata

class MPCRaceController:
    """MPC Controller for CommonRoad Raceline"""
    
    def __init__(self, vehicle, raceline=None, config=None):
        self.vehicle = vehicle
        
        if config is None:
            config = {}
        
        self.wheel_base = config.get('wheelbase', 2.875)
        self.horizon = config.get('horizon', 10)
        self.dt = config.get('dt', 0.1)
        
        # State weights: [x, y, θ, v]        
        self.Q = ca.diag(config.get('Q', [100.0, 100.0, 50.0, 10.0]))
        # Control weights: [a, κ]        
        self.R = ca.diag(config.get('R', [0.1, 1.0]))
        # Terminal weights
        self.Qf = ca.diag(config.get('Qf', [200.0, 200.0, 100.0, 100.0]))

        # Control limits
        self.a_min = config.get('a_min', -5.0)
        self.a_max = config.get('a_max', 5.0)
        self.kappa_min = config.get('kappa_min', -0.2)
        self.kappa_max = config.get('kappa_max', 0.2)
        
        # State limits
        self.v_min = config.get('v_min', 0.0)
        self.v_max = config.get('v_max', 30.0)

        self.discount_rate = config.get('discount_rate', 0.95)
        self.max_steer_angle = config.get('max_steer_angle', 1.22)
        
        # Lateral acceleration limit
        self.ay_max = config.get('ay_max', 5.0)
        # EMA for steering curvature (execution-level only)
        # ==================== EMA for curvature ====================
        self.kappa_ema_alpha = config.get('kappa_ema_alpha', 0.3)  # 0.2~0.4 추천
        self.kappa_cmd_ema = 0.0
        # ============================================================

        
        # ==================== Velocity Scaling ====================
        # CommonRoad의 속도가 너무 빠를 수 있으므로 스케일링
        self.velocity_scale = config.get('velocity_scale', 0.8)  # 80% 사용
        print(f"  Velocity scaling: {self.velocity_scale * 100:.0f}%")
        # ==========================================================
        
        # ==================== Lap Counter ====================
        self.lap_count = 0
        self.last_waypoint_idx = 0
        self.target_laps = config.get('target_laps', 1)
        self.path_length = len(raceline) if raceline else 0
        self.lap_threshold = int(self.path_length * 0.9)
        # ====================================================
        
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
        
        print("MPC Controller initialized")
        print(f"  Horizon: {self.horizon}")
        print(f"  dt: {self.dt}s")
        print(f"  v_max: {self.v_max} m/s ({self.v_max*3.6:.1f} km/h)")

    # def setup_mpc(self):
    #     """Setup CasADi optimization"""
    #     # State: [x, y, θ, v]        
    #     x = ca.SX.sym('x', 4)
    #     # Control: [a, κ]
    #     u = ca.SX.sym('u', 2)

    #     pos_x = x[0]
    #     pos_y = x[1]
    #     theta = x[2]
    #     v = x[3]

    #     a = u[0]
    #     kappa = u[1]

    #     # Dynamics (Trapezoidal integration)
    #     v_next = v + self.dt * a
    #     theta_next = theta + self.dt * kappa * v + (self.dt**2 / 2) * kappa * a
        
    #     x_next = ca.vertcat(
    #         pos_x + (self.dt / 2) * (v * ca.cos(theta) + v_next * ca.cos(theta_next)),
    #         pos_y + (self.dt / 2) * (v * ca.sin(theta) + v_next * ca.sin(theta_next)),
    #         theta_next,
    #         v_next
    #     )

    #     self.f = ca.Function('f', [x, u], [x_next])

    #     # Optimization variables
    #     X = ca.SX.sym('X', 4, self.horizon + 1)
    #     U = ca.SX.sym('U', 2, self.horizon)
    #     P = ca.SX.sym('P', 4)
    #     REF = ca.SX.sym('REF', 4, self.horizon + 1)

    #     # Cost function with discounting
    #     cost = 0
    #     for k in range(self.horizon):
    #         discount = self.discount_rate ** k 
    #         state_error = ca.vertcat(
    #             X[0, k] - REF[0, k],
    #             X[1, k] - REF[1, k],
    #             ca.atan2(ca.sin(X[2, k] - REF[2, k]), ca.cos(X[2, k] - REF[2, k])),
    #             X[3, k] - REF[3, k]
    #         )
            
    #         cost += discount * ca.mtimes([state_error.T, self.Q, state_error])
    #         cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
        
    #     # Terminal cost
    #     state_error_final = ca.vertcat(
    #         X[0, self.horizon] - REF[0, self.horizon],
    #         X[1, self.horizon] - REF[1, self.horizon],
    #         ca.atan2(ca.sin(X[2, self.horizon] - REF[2, self.horizon]), 
    #                 ca.cos(X[2, self.horizon] - REF[2, self.horizon])),
    #         X[3, self.horizon] - REF[3, self.horizon]
    #     )
    #     cost += ca.mtimes([state_error_final.T, self.Qf, state_error_final])
        
    #     # Constraints
    #     g = []
    #     g.append(X[:, 0] - P)  # Initial condition
        
    #     for k in range(self.horizon):
    #         # g.append(X[:, k+1] - self.f(X[:, k], U[:, k]))  # Dynamics
    #         g.append(X[:, k+1] - self.f(X[:, k], U[:, k], KAPPA_FF[k]))
            
    #         # ==================== 속도 제약을 constraint로 추가 ====================
    #         # Soft constraint 대신 hard constraint로 적용
    #         # g.append(X[3, k+1] - self.v_max)  # v <= v_max (inequality)
    #         # g.append(self.v_min - X[3, k+1])  # v >= v_min (inequality)
    #         # ======================================================================
        
    #     g = ca.vertcat(*g)

    #     # Decision variables
    #     Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    #     params = ca.vertcat(P, ca.reshape(REF, -1, 1))

    #     nlp = {'x': Z, 'f': cost, 'g': g, 'p': params}
        
    #     opts = {
    #         'ipopt.print_level': 0,
    #         'ipopt.max_iter': 150,
    #         'ipopt.warm_start_init_point': 'yes',
    #         'ipopt.tol': 1e-4,
    #         'ipopt.acceptable_tol': 1e-3,
    #         'print_time': 0
    #     }

    #     self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    #     self.n_eq = 4 + 4 * self.horizon  # Equality constraints만 (dynamics)
    
    def setup_mpc(self):
        """Setup CasADi optimization with feedforward inside"""
        # State: [x, y, θ, v]        
        x = ca.SX.sym('x', 4)
        # Control: [a, κ_fb]  ← κ_fb만 (feedback만)
        u = ca.SX.sym('u', 2)

        pos_x = x[0]
        pos_y = x[1]
        theta = x[2]
        v = x[3]

        a = u[0]
        kappa_fb = u[1]  # ← Feedback curvature만
        
        # ✅ Feedforward는 parameter로 받음
        kappa_ff = ca.SX.sym('kappa_ff')  # ← 외부에서 제공
        kappa_total = kappa_fb + kappa_ff  # ← 합산

        # Dynamics (Trapezoidal integration)
        v_next = v + self.dt * a
        theta_next = theta + self.dt * kappa_total * v + (self.dt**2 / 2) * kappa_total * a
        
        x_next = ca.vertcat(
            pos_x + (self.dt / 2) * (v * ca.cos(theta) + v_next * ca.cos(theta_next)),
            pos_y + (self.dt / 2) * (v * ca.sin(theta) + v_next * ca.sin(theta_next)),
            theta_next,
            v_next
        )

        # ✅ Function with feedforward parameter
        self.f = ca.Function('f', [x, u, kappa_ff], [x_next])

        # Optimization variables
        X = ca.SX.sym('X', 4, self.horizon + 1)
        U = ca.SX.sym('U', 2, self.horizon)
        P = ca.SX.sym('P', 4)
        REF = ca.SX.sym('REF', 4, self.horizon + 1)
        
        # ✅ Feedforward sequence (parameter)
        KAPPA_FF = ca.SX.sym('KAPPA_FF', self.horizon)

        # Cost function
        cost = 0
        for k in range(self.horizon):
            discount = self.discount_rate ** k 
            state_error = ca.vertcat(
                X[0, k] - REF[0, k],
                X[1, k] - REF[1, k],
                ca.atan2(ca.sin(X[2, k] - REF[2, k]), ca.cos(X[2, k] - REF[2, k])),
                X[3, k] - REF[3, k]
            )
            
            cost += discount * ca.mtimes([state_error.T, self.Q, state_error])
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
        
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
            # ✅ Dynamics with feedforward
            g.append(X[:, k+1] - self.f(X[:, k], U[:, k], KAPPA_FF[k]))
        
        g = ca.vertcat(*g)

        # Decision variables
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        # ✅ Parameters: initial state + reference + feedforward
        params = ca.vertcat(P, ca.reshape(REF, -1, 1), KAPPA_FF)

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
        self.n_eq = 4 + 4 * self.horizon

    def solve(self, current_state, ref_traj, kappa_ff_sequence):
        """Solve MPC optimization with feedforward constraint shifting"""
        
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
        
        lbx_controls = []
        ubx_controls = []
        
        constraint_debug = {
            'velocities': [],
            'kappa_ff': [],
            'kappa_dynamic_max': [],
            'kappa_fb_bounds': [],
            'kappa_total_bounds': [],
        }
        
        for k in range(self.horizon):
            # ✅ Use reference velocity (conservative upper bound)
            # This ensures constraint satisfaction even as vehicle accelerates
            if k == 0:
                # Current step: use actual velocity
                v_k = max(current_state[3], 2.0)
            else:
                # Future steps: use reference velocity (target)
                # v_k = max(ref_traj[3, k] * 3, current_state[3], 2.0)
                v_k = max(ref_traj[3, k] * 3, current_state[3], 2.0)
                # v_k = max(ref_traj[3, k]*0.5 + 0.5*current_state[3], 2.0)

            # Dynamic curvature limit: ay_max / v²
            kappa_dynamic_max = self.ay_max / (v_k ** 2)
            
            # Apply static limit
            kappa_max_k = min(kappa_dynamic_max, self.kappa_max)
            kappa_min_k = max(-kappa_dynamic_max, self.kappa_min)
            
            # Feedforward curvature
            kappa_ff_k = kappa_ff_sequence[k]
            
            # ✅✅✅ Constraint shifting ✅✅✅
            kappa_fb_min_raw = kappa_min_k - kappa_ff_k
            kappa_fb_max_raw = kappa_max_k - kappa_ff_k
            
            # Apply static bounds
            kappa_fb_min = max(kappa_fb_min_raw, self.kappa_min)
            kappa_fb_max = min(kappa_fb_max_raw, self.kappa_max)
            
            # Debug
            constraint_debug['velocities'].append(v_k)
            constraint_debug['kappa_ff'].append(kappa_ff_k)
            constraint_debug['kappa_dynamic_max'].append(kappa_dynamic_max)
            constraint_debug['kappa_fb_bounds'].append((kappa_fb_min, kappa_fb_max))
            constraint_debug['kappa_total_bounds'].append((kappa_min_k, kappa_max_k))
            
            lbx_controls.extend([self.a_min, kappa_fb_min])
            ubx_controls.extend([self.a_max, kappa_fb_max])

        lbx = lbx_states + lbx_controls
        ubx = ubx_states + ubx_controls
        
        lbg = [0.0] * self.n_eq
        ubg = [0.0] * self.n_eq

        p_val = np.concatenate((
            current_state,
            ref_traj.reshape(-1, order='F'),
            kappa_ff_sequence
        ))
        
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        if self._debug_counter < 3:
            print(f"\n  [MPC DEBUG] Constraint computation:")
            print(f"    Current velocity = {current_state[3]:.1f} m/s ({current_state[3]*3.6:.1f} km/h)")
            print(f"    Reference velocity = {ref_traj[3, 0]:.1f} m/s ({ref_traj[3, 0]*3.6:.1f} km/h)")
            
            # k=0 constraint
            v_0 = max(current_state[3], 2.0)
            kappa_dynamic_max_0 = self.ay_max / (v_0 ** 2)
            kappa_max_0 = min(kappa_dynamic_max_0, self.kappa_max)
            kappa_ff_0 = kappa_ff_sequence[0]
            
            print(f"    k=0: v={v_0:.1f} m/s, κ_max={kappa_max_0:.4f}, κ_ff={kappa_ff_0:+.4f}")
            
            # k=5 constraint (future)
            if len(ref_traj[3]) > 5:
                v_5 = max(ref_traj[3, 5], current_state[3], 2.0)
                kappa_dynamic_max_5 = self.ay_max / (v_5 ** 2)
                kappa_max_5 = min(kappa_dynamic_max_5, self.kappa_max)
                kappa_ff_5 = kappa_ff_sequence[min(5, len(kappa_ff_sequence)-1)]
                
                print(f"    k=5: v={v_5:.1f} m/s, κ_max={kappa_max_5:.4f}, κ_ff={kappa_ff_5:+.4f}")
            
            self._debug_counter += 1
        
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
                'U': U_opt,
                'constraint_debug': constraint_debug
            }
            
            acceleration = float(U_opt[0, 0])
            kappa_fb = float(U_opt[1, 0])
            
            self.optimal_trajectory = (X_opt[0, :], X_opt[1, :])
            
            return acceleration, kappa_fb, True
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return 0.0, 0.0, False 
    
    def step(self, waypoints):
        """Execute one control step"""
        current_state = self.get_state()
        ref_traj = self.get_reference_trajectory(waypoints)
        
        # ✅ Feedforward sequence 준비
        kappa_ff_sequence = np.array([
            waypoints[i]['curvature'] for i in range(min(self.horizon, len(waypoints)))
        ])
        
        # ✅ MPC solve with feedforward
        acceleration, kappa_fb, success = self.solve(current_state, ref_traj, kappa_ff_sequence)
        
        self.trajectory_history.append((current_state[0], current_state[1]))
        
        control = carla.VehicleControl()
                
        if success:
            # ==================== Curvature composition ====================
            kappa_ff = waypoints[0]['curvature']

            # ① 직선 판단 (raceline 기준)
            is_straight = abs(kappa_ff) < 5e-4  # ≈ radius > 1000m

            # ② feedback gain scheduling
            if is_straight:
                # 직선에서는 MPC를 "미세 보정기"로
                kappa_cmd_raw = kappa_fb + kappa_ff
            else:
                # 곡선에서는 MPC 풀 파워
                kappa_cmd_raw = kappa_fb + kappa_ff

            # ③ EMA (고주파 억제)
            # alpha = self.kappa_ema_alpha
            # self.kappa_cmd_ema = (
            #     0.6 * kappa_cmd_raw
            #     + 0.4 * self.kappa_cmd_ema
            # )

            # kappa_cmd = self.kappa_cmd_ema

            # ④ safety clip (보험)
            kappa_cmd = np.clip(kappa_cmd_raw, self.kappa_min, self.kappa_max)

            # ==================== Steering ====================
            steering_angle = np.arctan(kappa_cmd * self.wheel_base)
            steering_angle = np.clip(
                steering_angle,
                -self.max_steer_angle,
                self.max_steer_angle
            )
            control.steer = float(steering_angle / self.max_steer_angle)

            # ==================== Throttle / Brake ====================
            if acceleration > 0:
                control.throttle = float(np.clip(acceleration / self.a_max, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-acceleration / abs(self.a_min), 0.0, 1.0))
            
            return control, acceleration, kappa_fb, kappa_ff, kappa_cmd
        else:
            control.throttle = 0.0
            control.brake = 0.5
            control.steer = 0.0
            control.hand_brake = False
            control.reverse = False
            control.manual_gear_shift = False
            return control, acceleration, 0.0, 0.0, 0.0
        # if success:
        #     kappa_ff = waypoints[0]['curvature']
        # #     kappa_cmd = kappa_fb + kappa_ff
        # #     return control, acceleration, kappa_fb, kappa_ff, kappa_cmd
        # else:
        #     return control, acceleration, 0.0, 0.0, 0.0
        
    def get_dynamic_kappa_limits(self, velocity):
        """
        속도 기반 동적 곡률 제약: ay = V² × κ ≤ ay_max
        """
        velocity = max(velocity, 2.0)  # 저속 보호
        kappa_dynamic_max = self.ay_max / (velocity ** 2)
        
        kappa_max_actual = min(kappa_dynamic_max, self.kappa_max)
        kappa_min_actual = max(-kappa_dynamic_max, self.kappa_min)
        
        return kappa_min_actual, kappa_max_actual

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
        
        CommonRoad raceline은 이미 최적 속도와 곡률을 포함!
        """
        ref_traj = np.zeros((4, self.horizon + 1))
        
        if len(waypoints) < self.horizon + 1:
            waypoints_extended = list(waypoints) + [waypoints[-1]] * (self.horizon + 1 - len(waypoints))
        else:
            waypoints_extended = waypoints[:self.horizon + 1]
        
        for i in range(self.horizon + 1):
            wp = waypoints_extended[i]
            
            # ==================== CommonRoad 정보 직접 사용 ====================
            ref_traj[0, i] = wp['x']
            ref_traj[1, i] = wp['y']
            ref_traj[2, i] = wp['yaw']
            
            # CommonRoad의 최적 속도를 스케일링하여 사용
            # ref_traj[3, i] = wp['velocity'] * self.velocity_scale
            ref_traj[3, i] = wp['velocity'] * self.velocity_scale
            # print(wp['velocity'])

            # v_max 제한 적용(m/s)
            ref_traj[3, i] = min(ref_traj[3, i], self.v_max)
            # ref_traj[3, i] = 150
            # print(ref_traj[3, i]) 
            # ===================================================================
        
        return ref_traj

    def get_lookahead_waypoints(self, current_location, lookahead=25):
        """Raceline에서 lookahead waypoints 추출"""
        
        # 가장 가까운 waypoint 찾기
        min_dist = float('inf')
        closest_idx = 0
        
        # ==================== 디버깅: 처음 10개 waypoint 거리 체크 ====================
        debug_distances = []
        for i in range(min(10, len(self.raceline))):
            wp = self.raceline[i]
            dx = wp['x'] - current_location.x
            dy = wp['y'] - current_location.y
            dist = np.sqrt(dx**2 + dy**2)
            debug_distances.append((i, dist, wp['x'], wp['y']))
        # ===========================================================================
        
        for i, wp in enumerate(self.raceline):
            dx = wp['x'] - current_location.x
            dy = wp['y'] - current_location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # ==================== 디버깅 출력 (처음 몇 스텝만) ====================
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 0
            
        if self._debug_step_count < 3:  # 처음 3번만 출력
            print(f"\n  [DEBUG] Waypoint search:")
            print(f"    Current pos: ({current_location.x:.2f}, {current_location.y:.2f})")
            print(f"    First 5 waypoints:")
            for i, dist, wx, wy in debug_distances[:5]:
                marker = " <-- CLOSEST" if i == closest_idx else ""
                print(f"      [{i}] ({wx:.2f}, {wy:.2f}) - dist: {dist:.2f}m{marker}")
            print(f"    Selected: waypoint[{closest_idx}] at {min_dist:.2f}m")
        # ====================================================================

        # ==================== Lap 체크 ====================
        if self.last_waypoint_idx > self.lap_threshold and closest_idx < int(self.path_length * 0.1):
            self.lap_count += 1
            print(f"\n🏁 Lap {self.lap_count} completed!")
        
        self.last_waypoint_idx = closest_idx
        # =================================================
        
        # Lookahead 구간 추출 (순환)
        lookahead_wps = []
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.raceline)
            lookahead_wps.append(self.raceline[idx])
        
        return lookahead_wps, closest_idx
    
    def is_finished(self):
        """완주 여부"""
        return self.lap_count >= self.target_laps

    # def solve(self, current_state, ref_traj):
    #     """Solve MPC optimization"""
    #     X0 = np.zeros((4, self.horizon + 1))
        
    #     for i in range(self.horizon + 1):
    #         X0[:, i] = ref_traj[:, i]
    #     X0[:, 0] = current_state
        
    #     U0 = np.zeros((2, self.horizon))
    #     Z0 = np.concatenate((X0.reshape(-1, order='F'), U0.reshape(-1, order='F')))
        
    #     # State bounds
    #     lbx_states = []
    #     ubx_states = []
    #     for _ in range(self.horizon + 1):
    #         lbx_states.extend([-ca.inf, -ca.inf, -ca.pi, self.v_min])
    #         ubx_states.extend([ca.inf, ca.inf, ca.pi, self.v_max])
        
    #     # ==================== CRITICAL FIX ====================
    #     if self.v_max <= 0:
    #         raise ValueError(f"Invalid v_max: {self.v_max}")
    #     # ====================================================

    #     # Control bounds (동적 곡률 제약)
    #     lbx_controls = []
    #     ubx_controls = []
        
    #     # ==================== 제약 디버깅 정보 수집 ====================
    #     constraint_debug = {
    #         'static_kappa': (self.kappa_min, self.kappa_max),
    #         'dynamic_kappa': [],
    #         'actual_kappa': [],
    #         'velocities': []
    #     }
    #     # ===========================================================
        
    #     for k in range(self.horizon):
    #         v_k = max(ref_traj[3, k], current_state[3], 2.0)  # 최소 2 m/s
    #         kappa_min_k, kappa_max_k = self.get_dynamic_kappa_limits(v_k)
            
    #         # ==================== 디버깅 정보 저장 ====================
    #         constraint_debug['velocities'].append(v_k)
    #         constraint_debug['dynamic_kappa'].append((kappa_min_k, kappa_max_k))
    #         constraint_debug['actual_kappa'].append((
    #             max(kappa_min_k, self.kappa_min),
    #             min(kappa_max_k, self.kappa_max)
    #         ))
    #         # ========================================================
            
    #         lbx_controls.extend([self.a_min, kappa_min_k])
    #         ubx_controls.extend([self.a_max, kappa_max_k])

    #     lbx = lbx_states + lbx_controls
    #     ubx = ubx_states + ubx_controls
        
    #     lbg = [0.0] * self.n_eq
    #     ubg = [0.0] * self.n_eq

    #     p_val = np.concatenate((current_state, ref_traj.reshape(-1, order='F')))
        
    #     # Warm start
    #     if self.prev_solution is not None:
    #         X_prev = self.prev_solution['X']
    #         U_prev = self.prev_solution['U']
            
    #         X_warm = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
    #         U_warm = np.hstack([U_prev[:, 1:], U_prev[:, -1:]])
            
    #         x0 = np.concatenate((X_warm.reshape(-1, order='F'), U_warm.reshape(-1, order='F')))
    #         lam_x0 = self.prev_solution['lam_x']
    #         lam_g0 = self.prev_solution['lam_g']
    #     else:
    #         x0 = Z0
    #         lam_x0 = np.zeros(len(lbx))
    #         lam_g0 = np.zeros(self.n_eq)

    #     try:
    #         sol = self.solver(
    #             x0=x0,
    #             lam_x0=lam_x0,
    #             lam_g0=lam_g0,
    #             lbx=lbx,
    #             ubx=ubx,
    #             lbg=lbg,
    #             ubg=ubg,
    #             p=p_val
    #         )
            
    #         Z = sol['x'].full().flatten()
            
    #         n_states = 4 * (self.horizon + 1)
    #         X_opt = Z[:n_states].reshape((4, self.horizon+1), order='F')
    #         U_opt = Z[n_states:].reshape((2, self.horizon), order='F')
            
    #         self.prev_solution = {
    #             'x': Z,
    #             'lam_x': sol['lam_x'].full().flatten(),
    #             'lam_g': sol['lam_g'].full().flatten(),
    #             'X': X_opt,
    #             'U': U_opt,
    #             'constraint_debug': constraint_debug  # 디버깅 정보 저장
    #         }
            
    #         acceleration = float(U_opt[0, 0])
    #         curvature = float(U_opt[1, 0])
            
    #         self.optimal_trajectory = (X_opt[0, :], X_opt[1, :])
            
    #         return acceleration, curvature, True
            
    #     except Exception as e:
    #         print(f"MPC solve failed: {e}")
    #         return 0.0, 0.0, False

    # def step(self, waypoints):
    #     """Execute one control step"""
    #     current_state = self.get_state()
    #     ref_traj = self.get_reference_trajectory(waypoints)
        
    #     acceleration, curvature, success = self.solve(current_state, ref_traj)
        
    #     self.trajectory_history.append((current_state[0], current_state[1]))
        
    #     control = carla.VehicleControl()
        
    #     if success:
    #         # ==================== Curvature Feedforward ====================
    #         # MPC curvature (feedback for error correction)
    #         kappa_fb = curvature
            
    #         # Raceline curvature (feedforward for path geometry)
    #         kappa_ff = waypoints[0]['curvature']
            
    #         # Total curvature command
    #         kappa_cmd = kappa_fb + kappa_ff
    #         # kappa_cmd = kappa_ff + 0.7 * kappa_fb
            
    #         # Safety clipping (중요!)
    #         kappa_cmd = np.clip(kappa_cmd, self.kappa_min, self.kappa_max)

    #         # ================================================================
            
    #         # Curvature → Steering angle
    #         steering_angle = np.arctan(kappa_cmd * self.wheel_base)
    #         steering_angle = np.clip(steering_angle, -self.max_steer_angle, self.max_steer_angle)
    #         control.steer = float(steering_angle / self.max_steer_angle)

    #         # ==================== 수정: Throttle 매핑 ====================
    #         # if acceleration > 0:
    #         #     # a_max를 기준으로 정규화 (더 공격적인 가속)
    #         #     control.throttle = float(np.clip(acceleration / self.a_max, 0.0, 1.0))
    #         #     control.brake = 0.0
    #         # else:
    #         #     control.throttle = 0.0
    #         #     # 제동도 더 강하게
    #         #     control.brake = float(np.clip(-acceleration / self.a_min, 0.0, 1.0))
    #         #     # control.brake = np.clip(-acceleration / abs(self.a_min), 0.0, 1.0)
    #         if acceleration > 0:
    #             # ✅ 더 공격적인 throttle
    #             throttle_raw = acceleration / self.a_max
                
    #             # Boost 추가 (저속에서)
    #             if current_state[3] < 10.0:  # 36 km/h 미만
    #                 throttle_boost = 1.5  # 50% 증폭
    #             else:
    #                 throttle_boost = 1.0
                
    #             control.throttle = float(np.clip(throttle_raw * throttle_boost, 0.0, 1.0))
    #             control.brake = 0.0
    #         else:
    #             control.throttle = 0.0
    #             control.brake = float(np.clip(-acceleration / abs(self.a_min), 0.0, 1.0))
    #         # ===========================================================
    #     else:
    #         control.throttle = 0.0
    #         control.brake = 0.5
    #         control.steer = 0.0
        
    #     control.hand_brake = False
    #     control.reverse = False
    #     control.manual_gear_shift = False
        
    #     # Return both components for debugging
    #     if success:
    #         return control, acceleration, kappa_fb, kappa_ff, kappa_cmd
    #     else:
    #         return control, acceleration, 0.0, 0.0, 0.0

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
        self.ax.set_title('MPC Controller - CommonRoad Raceline Tracking', fontsize=14, fontweight='bold')
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
    # client = carla.Client('172.22.39.145', 2000)
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


    # ==================== CommonRoad Raceline 로드 ====================
    raceline, metadata = load_raceline('routes/town04_raceline_mincurv13_1.pkl')
    # (yaw는 이미 재계산됨)
    # ===================================================================

    # Raceline 시작점에 spawn
    first_wp = raceline[0]
    
    print(f"\n📍 Spawn location:")
    print(f"   Position: ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
    print(f"   Yaw: {np.rad2deg(first_wp['yaw']):.2f}° (corrected)")
    
    # ==================== Z 좌표 수정 ====================
    map_obj = world.get_map()
    test_location = carla.Location(x=first_wp['x'], y=first_wp['y'], z=0.0)
    waypoint_on_road = map_obj.get_waypoint(test_location, project_to_road=True)
    
    if waypoint_on_road is not None:
        spawn_z = waypoint_on_road.transform.location.z + 1.0
        print(f"   Road height: {waypoint_on_road.transform.location.z:.2f}m")
    else:
        spawn_z = 1.0
        print(f"   ⚠️ Could not find road waypoint, using default z=1.0")
    # ===================================================
    
    spawn_transform = carla.Transform(
        carla.Location(x=first_wp['x'], y=first_wp['y'], z=spawn_z),
        carla.Rotation(yaw=np.rad2deg(first_wp['yaw']))  # 이미 수정된 yaw
    )
    
    vehicle = None
    spawn_attempts = 0
    max_attempts = 5
    
    # ==================== Spawn with retry ====================
    while vehicle is None and spawn_attempts < max_attempts:
        try:
            # 높이를 점진적으로 증가시키며 시도
            spawn_z_attempt = spawn_z + (spawn_attempts * 0.5)
            spawn_transform.location.z = spawn_z_attempt
            
            print(f"\nSpawn attempt {spawn_attempts + 1}/{max_attempts} at z={spawn_z_attempt:.2f}m...")
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if vehicle is None:
                spawn_attempts += 1
                time.sleep(0.5)
            else:
                print(f"✅ Vehicle spawned successfully at:")
                print(f"   Location: ({first_wp['x']:.2f}, {first_wp['y']:.2f}, {spawn_z_attempt:.2f})")
                print(f"   Yaw: {np.rad2deg(first_wp['yaw']):.2f}°")
                break
                
        except Exception as e:
            print(f"   Spawn error: {e}")
            spawn_attempts += 1
            time.sleep(0.5)
    
    if vehicle is None:
        print("\n❌ Failed to spawn vehicle after all attempts!")
        print("Possible issues:")
        print("  1. Location is out of bounds")
        print("  2. Location is occupied")
        print("  3. Location is not on a road")
        print(f"\nTried location: ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
        print("Try manually checking this location in CARLA")
        return
    # ==========================================================
    
    print(f"\nVehicle spawned at raceline start:")
    print(f"   Location: ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
    
    # ==================== 물리 엔진 레벨에서 속도 제한 ====================
    # physics_control = vehicle.get_physics_control()
    
    # # 최대 RPM 제한으로 속도 cap
    # physics_control.max_rpm = 12000.0  # 기본값보다 낮게
    
    # # 기어비 조정으로 최고속도 제한
    # for wheel in physics_control.wheels:
    #     wheel.max_steer_angle = 70.0  # 기본 조향각
    
    # vehicle.apply_physics_control(physics_control)
    # print("   Physics control applied (RPM limited)")
    # ====================================================================
    
    # ==================== 초기 속도 0으로 설정 ====================
    # time.sleep(0.5)
    # vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    time.sleep(0.5)
    print("   Initial velocity set to 0")
    # ===========================================================
    
    spectator = world.get_spectator()
    step = 0
    
    try:
        time.sleep(1.0)
        
        # ==================== MPC Config ====================
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 15,
            'dt': 0.1,
            'Q': [100.0, 100.0, 10.0, 100.0],
            'R': [0.5, 5.0],
            'Qf': [200.0, 200.0, 100.0, 200.0],
            'a_min': -12.0,
            'a_max': 12.0,
            'kappa_min': -0.2,
            'kappa_max': 0.2,
            'ay_max': 13.0,
            'v_min': 0.0,
            'v_max': 50.0,
            'max_steer_angle': 1.22,
            'discount_rate': 0.95,
            'velocity_scale': 1.2,  # CommonRoad 속도의 70% 사용 (안전 마진)
            'visualization': True,
            'target_laps': 1
        }

        mpc = MPCRaceController(vehicle, raceline, config=mpc_config)
        lookahead = 50
        
        print("\n🏁 Starting race...")
        
        while True:
            current_location = vehicle.get_location()
            
            # ==================== 완주 체크 ====================
            if mpc.is_finished():
                print(f"\n🏁 Race finished! {mpc.lap_count} lap(s) completed")
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0
                ))
                break
            # ==================================================
            
            # Lookahead waypoints
            lookahead_waypoints, closest_idx = mpc.get_lookahead_waypoints(
                current_location, 
                lookahead=lookahead
            )
            
            # MPC step
            control, accel, kappa_fb, kappa_ff, kappa_cmd = mpc.step(lookahead_waypoints)
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
                speed_kmh = speed * 3.6
                
                # ==================== 디버깅: 현재 위치 출력 ====================
                curr_x = current_location.x
                curr_y = current_location.y
                closest_wp = lookahead_waypoints[0]
                wp_x = closest_wp['x']
                wp_y = closest_wp['y']
                dist_to_closest = np.sqrt((curr_x - wp_x)**2 + (curr_y - wp_y)**2)
                # ===========================================================
            
                
                # CTE 계산
                cte = float('inf')
                for wp in lookahead_waypoints[:5]:
                    dx = current_location.x - wp['x']
                    dy = current_location.y - wp['y']
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < cte:
                        cte = dist
                
                # ==================== 제약 및 Steering 디버깅 ====================
                # Curvature → Steering 변환
                steering_angle_rad = np.arctan(kappa_cmd * mpc.wheel_base)
                steering_angle_deg = np.rad2deg(steering_angle_rad)
                
                # 실제 적용된 steering (normalized)
                steering_normalized = control.steer
                
                # Lateral acceleration (with feedforward)
                actual_ay = speed**2 * abs(kappa_cmd)
                
                # # 제약 정보 (첫 번째 step만)
                # if mpc.prev_solution and 'constraint_debug' in mpc.prev_solution:
                #     debug_info = mpc.prev_solution['constraint_debug']
                    
                #     # 정적 제약
                #     static_min, static_max = debug_info['static_kappa']
                    
                #     # 동적 제약 (첫 번째 step)
                #     if len(debug_info['dynamic_kappa']) > 0:
                #         dyn_min, dyn_max = debug_info['dynamic_kappa'][0]
                #         v_constraint = debug_info['velocities'][0]
                        
                #         # 실제 적용된 제약
                #         actual_min, actual_max = debug_info['actual_kappa'][0]
                #     else:
                #         dyn_min, dyn_max = static_min, static_max
                #         v_constraint = speed
                #         actual_min, actual_max = static_min, static_max
                # else:
                #     static_min, static_max = mpc.kappa_min, mpc.kappa_max
                #     dyn_min, dyn_max = static_min, static_max
                #     v_constraint = speed
                #     actual_min, actual_max = static_min, static_max
                
                # Reference velocity
                ref_traj = mpc.get_reference_trajectory(lookahead_waypoints)
                v_ref = ref_traj[3, 0] * 3.6  # km/h
                
                # Raceline 정보
                raceline_v = closest_wp['velocity'] * 3.6  # km/h
                raceline_kappa = closest_wp['curvature']
                # ================================================================
                # 직선 판정
                is_straight = abs(kappa_cmd) < 0.005  # 200m 이상 반경
                
                if is_straight:
                    print(f"\n🚗 STRAIGHT SECTION DETECTED")
                    print(f"  κ_cmd = {kappa_cmd:+.6f} (1/m)")
                    print(f"  Radius = {1/abs(kappa_cmd) if abs(kappa_cmd) > 1e-6 else float('inf'):.0f} m")
                    print(f"  steer = {control.steer:+.6f} ← Should be ≈ 0")
                    
                    if abs(control.steer) > 0.01:
                        print(f"  ⚠️ WARNING: Large steering on straight!")
                        print(f"     κ_fb = {kappa_fb:+.6f}")
                        print(f"     κ_ff = {kappa_ff:+.6f}")
                        print(f"     CTE = {cte:.2f}m")

                # ==================== 상세 출력 ====================
                print(f"\n{'='*80}")
                print(f"Step {step:4d} | Progress: {closest_idx}/{len(raceline)} ({100*closest_idx/len(raceline):.1f}%)")
                print(f"{'='*80}")
                
                # 위치 및 속도
                print(f"Position:  ({curr_x:6.1f}, {curr_y:6.1f}) | CTE: {cte:5.2f}m")
                print(f"Velocity:  {speed_kmh:5.1f} km/h (v_max={mpc.v_max*3.6:.0f} km/h)")
                print(f"  Ref:     {v_ref:5.1f} km/h | Raceline: {raceline_v:5.1f} km/h")
                
                # # 제어 입력
                print(f"\nControl Inputs:")
                print(f"  Acceleration:  {accel:+6.2f} m/s²")
                print(f"  Curvature (Breakdown):")
                print(f"    κ_fb (MPC):      {kappa_fb:+7.4f} (1/m) ← Error correction")
                print(f"    κ_ff (Raceline): {kappa_ff:+7.4f} (1/m) ← Path geometry")
                print(f"    κ_cmd (Total):   {kappa_cmd:+7.4f} (1/m) ← Applied to vehicle")
                print(f"      → Radius:      {1/abs(kappa_cmd) if abs(kappa_cmd) > 1e-6 else float('inf'):7.1f} m")
                print(f"  Lateral ay:    {actual_ay:6.2f} m/s² (max={mpc.ay_max:.1f})")
                
                # # Steering 변환
                print(f"\nSteering Conversion:")
                print(f"  κ_cmd = {kappa_cmd:+7.4f} (1/m)")
                print(f"  δ = atan(κ_cmd × L) = atan({kappa_cmd:+.4f} × {mpc.wheel_base:.3f})")
                print(f"    = {steering_angle_rad:+7.4f} rad = {steering_angle_deg:+6.2f}°")
                print(f"  steer = δ / δ_max = {steering_angle_rad:.4f} / {mpc.max_steer_angle:.4f}")
                print(f"        = {steering_normalized:+6.3f} (CARLA input)")
                
                # # 제약 정보
                # print(f"\nConstraints (Curvature):")
                # print(f"  Static:    [{static_min:+.4f}, {static_max:+.4f}] (1/m)")
                # print(f"  Dynamic:   [{dyn_min:+.4f}, {dyn_max:+.4f}] (1/m) @ v={v_constraint:.1f} m/s")
                # print(f"    (from ay_max / v² = {mpc.ay_max:.1f} / {v_constraint:.1f}² = {mpc.ay_max/(v_constraint**2):.4f})")
                # print(f"  Applied:   [{actual_min:+.4f}, {actual_max:+.4f}] (1/m) ← min of above")
                
                # # Raceline과 비교
                # print(f"\nFeedforward Analysis:")
                # print(f"  Raceline κ (FF):   {raceline_kappa:+7.4f} (1/m)")
                # print(f"  MPC κ (FB):        {kappa_fb:+7.4f} (1/m)")
                # print(f"  Combined κ (CMD):  {kappa_cmd:+7.4f} (1/m)")
                # print(f"  FF contribution:   {abs(kappa_ff)/(abs(kappa_cmd)+1e-6)*100:.1f}%")
                
                # # CARLA 제어
                print(f"\nCARLA Control:")
                print(f"  Throttle:  {control.throttle:.3f}")
                print(f"  Brake:     {control.brake:.3f}")
                print(f"  Steer:     {control.steer:+.3f}")
                print(f"{'='*80}")
                # ==================================================

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