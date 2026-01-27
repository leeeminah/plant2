#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA MPC Agent with Exponential Smoothing for Steering
"""

import sys
import math
import time
import numpy as np
import casadi as ca
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque
import pickle

try:
    import carla
except ImportError:
    print("CARLA Python API not found!")
    sys.exit(1)

def load_global_path(filename='town02_centerline.pkl'):
    """저장된 global path 로드"""
    with open(filename, 'rb') as f:
        centerline = pickle.load(f)
    print(f"Loaded {len(centerline)} waypoints")
    return centerline

class MPCController:
    """MPC Controller with EMA Steering Smoothing"""
    
    def __init__(self, vehicle, global_path=None, config=None):
        self.vehicle = vehicle
        # self.world = vehicle.get_world()
        
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
        self.a_min = config.get('a_min', -3.0)  # m/s²
        self.a_max = config.get('a_max', 3.0)   # m/s²
        self.kappa_min = config.get('kappa_min', -0.05)  # 1/m
        self.kappa_max = config.get('kappa_max', 0.05)   # 1/m
        
        # State limits
        self.v_min = config.get('v_min', 0.0)
        self.v_max = config.get('v_max', 15.0)

        self.discount_rate = config.get('discount_rate', 1.0)
        
        self.max_steer_angle = np.deg2rad(70)

        #ay 제약 추가 
        self.ay_max = config.get('ay_max', 3.0)

        self.cfg_velocity = config.get('velocity_profile')
        
        # ==================== Lap Counter ====================
        self.lap_count = 0
        self.last_waypoint_idx = 0
        self.target_laps = config.get('target_laps', 1)
        self.path_length = len(global_path) if global_path else 0
        self.lap_threshold = int(self.path_length * 0.9)  # 90% 지점
        # ====================================================

        # ==================== EMA Steering Smoothing ====================
        # self.steering_alpha = config.get('steering_alpha', 0.3)
        # Alpha ∈ [0, 1]
        # 0: 완전히 이전 값 유지 (변화 없음)
        # 1: 완전히 새 값 사용 (smoothing 없음)
        # 0.3: 권장값 (균형)
        
        self.prev_steering = 0.0
        self.smoothed_steering = 0.0
        # ================================================================
        
        self.prev_solution = None
        
        # Logging
        self.trajectory_history = deque(maxlen=1000)
        self.optimal_trajectory = None
        
        # Visualization
        self.enable_viz = config.get('visualization', False)
        if self.enable_viz:
            self.setup_visualization()

        # self.global_path = global_path
        # print(f"Global path loaded: {len(global_path)} waypoints")

        # ==================== 곡률 계산 ====================
        # print("Calculating path curvatures...")
        # curvatures = calculate_path_curvatures(global_path)
        
        # waypoint에 곡률 추가
        # for i, wp in enumerate(global_path):
        #     wp['curvature'] = curvatures[i]
        
        # print(f"  Max |curvature|: {max(abs(k) for k in curvatures):.4f} rad/m")
        # print(f"  Mean |curvature|: {np.mean([abs(k) for k in curvatures]):.4f} rad/m")
        # ====================================================
        
        
        # Setup MPC
        self.setup_mpc()
        
        print("MPC Controller initialized")
        print(f"  Horizon: {self.horizon}")
        print(f"  dt: {self.dt}s")
        print(f"  v_max: {self.v_max} m/s ({self.v_max*3.6:.1f} km/h)")
        # print(f"  Steering alpha: {self.steering_alpha} (EMA smoothing)")

    def setup_mpc(self):
        """Setup CasADi optimization"""
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

        # Dynamics
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
        REF = ca.SX.sym('REF', 4, self.horizon + 1)

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
        
        # Terminal Cost
        state_error_final = ca.vertcat(
            X[0, self.horizon] - REF[0, self.horizon],
            X[1, self.horizon] - REF[1, self.horizon],
            ca.atan2(ca.sin(X[2, self.horizon] - REF[2, self.horizon]), 
                    ca.cos(X[2, self.horizon] - REF[2, self.horizon])),
            X[3, self.horizon] - REF[3, self.horizon]
        )
        cost += ca.mtimes([state_error_final.T, self.Qf, state_error_final])
        
        # constranint 정의 
        g = []
        g.append(X[:, 0] - P) # 초기 조건: X_0 = P (현재 상태)
        
        for k in range(self.horizon):
            g.append(X[:, k+1] - self.f(X[:, k], U[:, k])) #동역학
        
        g = ca.vertcat(*g) # 모든 constraint를 ㅎㅏ나의 벡터로 

        # Decisioin variables
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.vertcat(P, ca.reshape(REF, -1, 1))

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
    
    def get_dynamic_kappa_limits(self, velocity):
        """
        속도 기반 동적 곡률 제약
        
        ay = V² × κ ≤ ay_max  →  κ ≤ ay_max / V²
        
        Args:
            velocity: 현재 속도 [m/s]
            
        Returns:
            kappa_min, kappa_max: 동적 곡률 제약 [1/m]
        """
        # 저속에서는 급격한 회전 허용
        # if velocity < 1.0:
        #     return self.kappa_min, self.kappa_max
        # V_eff = max(velocity, 2.0)   # 핵심: 저속 보호
        # print(velocity)
        kappa_dynamic_max = self.ay_max / (velocity ** 2)
        # print(velocity)
        # κ_max = ay_max / V²
        # kappa_dynamic_max = self.ay_max / (velocity ** 2)
        
        # 물리적 한계와 동적 한계 중 작은 값 선택
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
        """Convert waypoints to reference trajectory"""
        ref_traj = np.zeros((4, self.horizon + 1))
        
        if len(waypoints) < self.horizon + 1:
            waypoints_extended = list(waypoints) + [waypoints[-1]] * (self.horizon + 1 - len(waypoints))
        else:
            waypoints_extended = waypoints[:self.horizon + 1]
        
        for i in range(self.horizon + 1):
            wp = waypoints_extended[i]
            
            # 이미 dict에 있는 정보 직접 사용!
            ref_traj[0, i] = wp['x']
            ref_traj[1, i] = wp['y']
            ref_traj[2, i] = wp['yaw']  # ← 탐색 필요 없음!
        
            # 마지막 waypoint는 이전 속도 유지
            if i >= len(waypoints_extended) - 1:
                ref_traj[3, i] = ref_traj[3, i-1] if i > 0 else 0.5 * self.v_max
                continue

            wp_next = waypoints_extended[i + 1]

            dx = wp_next['x'] - wp['x']
            dy = wp_next['y'] - wp['y']
            ds = np.hypot(dx, dy) + 1e-6

            dyaw = wp_next['yaw'] - wp['yaw']
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # wrap

            kappa_est = abs(dyaw) / ds

            # print(f"[Ego Frame] Step {i}: ds={ds:.3f}, dθ={dyaw:.4f}, κ={kappa_est:.4f}")

            # 논문 수식: ay = v^2 * kappa
            # v_safe = np.sqrt(self.ay_max / (kappa_est + 1e-3))

            # ==================== 곡률 기반 속도 결정 ====================
            v_ref = self._velocity_from_curvature(kappa_est)
            ref_traj[3, i] = v_ref
            # ===========================================================
        
        return ref_traj

            # 속도 범위 제한 (너무 느리거나 빠르지 않게)
            # ref_traj[3, i] = np.clip(
            #     v_safe,
            #     0.1 * self.v_max,
            #     self.v_max
            # )
            # print(ref_traj[3, i])
            # ref_traj[3, i] = v_safe

        #     if v_safe < 20:
        #         v = 0.3 * self.v_max
        #     elif v_safe < 30:
        #         v = 0.7 * self.v_max
        #     elif v_safe < 35:
        #         v = 0.8 * self.v_max
        #     else:
        #         v = self.v_max

        #     ref_traj[3, i] = v
        # # print(ref_traj[3, :])
        # return ref_traj
    
    def _velocity_from_curvature(self, kappa):
        """
        곡률 → 속도 매핑
        """
        kappa_breakpoints = self.cfg_velocity.get('kappa_breakpoints', 
                                                [0.0, 0.02, 0.2])
        velocity_ratios = self.cfg_velocity.get('velocity_ratios', 
                                                [1.0, 0.9, 0.3])
        
        # 가장 작은 곡률보다 작으면 최고속
        if kappa <= kappa_breakpoints[0]:
            return velocity_ratios[0] * self.v_max
        
        # 가장 큰 곡률보다 크면 최저속
        if kappa >= kappa_breakpoints[-1]:
            return velocity_ratios[-1] * self.v_max
        
        # 구간별 선형 보간
        for i in range(len(kappa_breakpoints) - 1):
            if kappa_breakpoints[i] <= kappa < kappa_breakpoints[i + 1]:
                # 선형 보간
                t = (kappa - kappa_breakpoints[i]) / \
                    (kappa_breakpoints[i + 1] - kappa_breakpoints[i])
                v_ratio = velocity_ratios[i] + t * (velocity_ratios[i + 1] - velocity_ratios[i])
                return v_ratio * self.v_max
        
        return velocity_ratios[-1] * self.v_max
        
    def get_lookahead_waypoints(self, current_location, lookahead=25):
        """Global path에서 lookahead waypoints 추출"""
        
        # 1. 가장 가까운 waypoint 찾기
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.global_path):
            dx = wp['x'] - current_location.x
            dy = wp['y'] - current_location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # # 2. Lookahead 구간 추출
        # start_idx = closest_idx
        # end_idx = min(start_idx + lookahead, len(self.global_path))

        # # Global path 그대로 반환 (dict 형태, yaw 포함)
        # lookahead_wps = self.global_path[start_idx:end_idx]     
        
        # return lookahead_wps, closest_idx
        # ==================== Lap 체크 ====================
        # 마지막 10%에서 첫 10%로 넘어가면 랩 완주
        if self.last_waypoint_idx > self.lap_threshold and closest_idx < int(self.path_length * 0.1):
            self.lap_count += 1
            print(f"\n🏁 Lap {self.lap_count} completed!")
        
        self.last_waypoint_idx = closest_idx
        # =================================================
        
        # Lookahead 구간 추출 (순환 처리)
        lookahead_wps = []
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.global_path)  # ← 순환!
            lookahead_wps.append(self.global_path[idx])
        
        return lookahead_wps, closest_idx
    
    def is_finished(self):
        """완주 여부 체크"""
        return self.lap_count >= self.target_laps

    def solve(self, current_state, ref_traj):
        """Solve MPC optimization"""
        X0 = np.zeros((4, self.horizon + 1))
        
        for i in range(self.horizon + 1):
            X0[:, i] = ref_traj[:, i]
        X0[:, 0] = current_state
        
        U0 = np.zeros((2, self.horizon))
        
        Z0 = np.concatenate((X0.reshape(-1, order='F'), U0.reshape(-1, order='F')))
        
        # State bounds: [x, y, θ, v]
        lbx_states = []
        ubx_states = []
        for _ in range(self.horizon + 1):
            lbx_states.extend([-ca.inf, -ca.inf, -ca.pi, self.v_min])
            ubx_states.extend([ca.inf, ca.inf, ca.pi, self.v_max])

        # ==================== Horizon별 동적 제약 ====================
        # # Warm start에서 예상 속도 궤적 추출
        # if self.prev_solution is not None:
        #     X_prev = self.prev_solution['X']
        #     # Shift forward
        #     V_predicted = np.append(X_prev[3, 1:], X_prev[3, -1])
        #     # V_predicted = X_opt[3,:]
        # else:
        #     # 첫 스텝: 현재 속도 유지 가정
        #     V_predicted = np.full(self.horizon, current_state[3])
                

        lbx_controls = []
        ubx_controls = []
        # print(current_state[3])
        for k in range(self.horizon):
            v_k = max(ref_traj[3, k], current_state[3])  # 보수적 설정
            # print(v_k)
            kappa_min_k, kappa_max_k = self.get_dynamic_kappa_limits(ref_traj[3, k])
            # print(kappa_max_k)
            lbx_controls.extend([self.a_min, kappa_min_k])
            ubx_controls.extend([self.a_max, kappa_max_k])

        lbx = lbx_states + lbx_controls
        ubx = ubx_states + ubx_controls
        
        lbg = [0.0] * self.n_eq
        ubg = [0.0] * self.n_eq

        p_val = np.concatenate((current_state, ref_traj.reshape(-1, order='F')))
        
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
            # print(X_opt[3,:])
            
            self.prev_solution = {
                'x': Z,
                'lam_x': sol['lam_x'].full().flatten(),
                'lam_g': sol['lam_g'].full().flatten(),
                'X': X_opt,
                'U': U_opt
            }
            
            # 첫 번째 제어 입력: [a, κ]
            acceleration = float(U_opt[0, 0])
            curvature = float(U_opt[1, 0])
            
            # ==================== EMA Steering Smoothing ====================
            # # Step 1: Calculate target steering from MPC
            # target_steering = self.prev_steering + steer_dot * self.dt
            # target_steering = np.clip(target_steering, self.delta_min, self.delta_max)
            
            # # Step 2: Apply Exponential Moving Average
            # # smoothed = α * new + (1 - α) * old
            # self.smoothed_steering = (
            #     self.steering_alpha * target_steering + 
            #     (1.0 - self.steering_alpha) * self.smoothed_steering
            # )
            
            # # Step 3: Update previous steering for next iteration
            # self.prev_steering = target_steering
            # # ================================================================
            
            self.optimal_trajectory = (X_opt[0, :], X_opt[1, :])
            
            return acceleration, curvature, True
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return 0.0, 0.0, False

    def step(self, waypoints):
        """Execute one control step"""
        current_state = self.get_state()
        ref_traj = self.get_reference_trajectory(waypoints)
        
        acceleration, curvature, success = self.solve(current_state, ref_traj)
        
        self.trajectory_history.append((current_state[0], current_state[1]))
        
        control = carla.VehicleControl()
        
        if success:
            # Curvature → Steering angle
            # κ = tan(δ) / L  →  δ = atan(κ * L)
            steering_angle = np.arctan(curvature * self.wheel_base)
            # print(steering_angle)
            steering_angle = np.clip(steering_angle, -self.max_steer_angle, self.max_steer_angle)
            # print(steering_angle)
            control.steer = float(steering_angle / self.max_steer_angle)
            # control.steer = float(steering_angle / 0.495)
            # print(control.steer)


            # print(acceleration)
            # Acceleration → Throttle/Brake
            if acceleration > 0:
                control.throttle = float(np.clip(acceleration / 3.0, 0.0, 1.0))
                # print(control.throttle)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-acceleration / 3.0, 0.0, 1.0))
        else:
            control.throttle = 0.0
            control.brake = 0.5
            control.steer = 0.0
        
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        
        return control, acceleration, curvature

    def setup_visualization(self):
        """Setup matplotlib"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.current_pos_plot, = self.ax.plot([], [], 'ro', markersize=12, label='Vehicle', zorder=5)
        self.trajectory_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='History', alpha=0.7)
        self.optimal_path_plot, = self.ax.plot([], [], 'r--', linewidth=3, label='MPC Path', alpha=0.8)
        self.waypoints_plot, = self.ax.plot([], [], 'g*', markersize=10, label='Waypoints', alpha=0.6)
        
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('MPC Controller - Path Tracking', fontsize=14, fontweight='bold')
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
        
        self.ax.relim()
        self.ax.autoscale_view()

        # 차량 중심으로 범위 설정
        zoom_range = 50  # 100m 범위
        self.ax.set_xlim([current_state[0] - zoom_range, current_state[0] + zoom_range])
        self.ax.set_ylim([current_state[1] - zoom_range, current_state[1] + zoom_range])
        
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    """Main function"""
    
    # print("Connecting to CARLA...")
    # # client = carla.Client('localhost', 2000)
    # client = carla.Client('172.22.39.145', 2000)
    # client.set_timeout(10.0)
    
    # # world = client.get_world()
    # world = client.load_world('Town04')

    # print(f"Connected: {world.get_map().name}")
    
    # blueprint_library = world.get_blueprint_library()
    # vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # global_path = load_global_path('town04_full_centerline2.pkl')

    # # Global path 시작점에 spawn
    # first_wp = global_path[0]
    # spawn_transform = carla.Transform(
    #     carla.Location(x=first_wp['x'], y=first_wp['y'], z=first_wp['z'] + 0.5),
    #     carla.Rotation(yaw=np.rad2deg(first_wp['yaw']))
    # )    
    # # start_idx = 0
    # # spawn_point = spawn_points[start_idx]
    # vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    # print(f"\nVehicle spawned at global path start:")
    # print(f"   Location: ({first_wp['x']:.2f}, {first_wp['y']:.2f})")
    
    # spectator = world.get_spectator()        
    # step = 0
    
    # try:
    #     time.sleep(2.0)
        
    #     # 목표 지점
    #     goal_wp = global_path[-1]
    #     print(f"\nDestination (end of global path):")
    #     print(f"   Location: ({goal_wp['x']:.2f}, {goal_wp['y']:.2f})")
        
    #     # 초기 거리 확인
    #     current_location = vehicle.get_location()
    #     initial_distance = np.sqrt(
    #         (current_location.x - goal_wp['x'])**2 + 
    #         (current_location.y - goal_wp['y'])**2
    #     )
    #     print(f"\n Initial distance to goal: {initial_distance:.2f}m")
         
    #     # ==================== MPC Config ====================
    #     mpc_config = {
    #         'wheelbase': 2.875,
    #         'horizon': 15,
    #         'dt': 0.1,
    #         'Q': [100.0, 100.0, 100.0, 1.0],
    #         'R': [1.0, 5.0],
    #         'Qf': [150.0, 150.0, 150.0, 1.0],

    #         'a_min': -3.0,
    #         'a_max': 3.0,
            
    #         'kappa_min': -0.3,
    #         'kappa_max': 0.3,

    #         # ==================== 논문 방식 추가 ====================
    #         'ay_max': 5.0,  # 허용 가능한 최대 횡가속도 [m/s²]
    #         # =======================================================

    #         'v_min':0.0,
    #         'v_max': 30.0,
    #         'discount_rate' : 0.95,

    #         'visualization':  True,

    #         'velocity_profile': {
    #             # ==================== 곡률 기준 ====================
    #             # κ (1/m) | 반경 (m) | 속도 비율
    #             # -----------------------------------------------
    #             'kappa_breakpoints': [
    #                 0.00,   # R = ∞ (직선)
    #                 0.01,   # R = 100m
    #                 0.02,   # R = 50m
    #             ],
    #             'velocity_ratios': [
    #                 1.0,    # 100% (직선)
    #                 0.65,   # 65%
    #                 0.25,   # 25% (급커브)
    #             ],
    #             # ==================================================
    #         }
    #     }
    #     mpc = MPCController(vehicle, global_path, config=mpc_config)
    #     lookahead = 50
        
    #     while True:
    #         current_location = vehicle.get_location()
            
    #         # 목표까지 거리
    #         distance_to_goal = np.sqrt(
    #             (current_location.x - goal_wp['x'])**2 + 
    #             (current_location.y - goal_wp['y'])**2
    #         )

    #         # ==================== 완주 체크 ====================
    #         if mpc.is_finished():
    #             print(f"\n🏁 Race finished! {mpc.lap_count} lap(s) completed")
    #             vehicle.apply_control(carla.VehicleControl(
    #                 throttle=0.0,
    #                 steer=0.0,
    #                 brake=1.0
    #             ))
    #             break
    #         # ==================================================
            
    #         # #  첫 스텝 정보
    #         # if step == 0:
    #         #     print(f" Starting control loop (distance: {distance_to_goal:.2f}m)")
            
    #         # 목표 도달 체크 (인덴트 수정!)
    #         # if distance_to_goal < 5.0:
    #         #     print(f"\n Goal reached! (distance: {distance_to_goal:.2f}m)")
    #         #     vehicle.apply_control(carla.VehicleControl(
    #         #         throttle=0.0,
    #         #         steer=0.0,
    #         #         brake=1.0
    #         #     ))
    #         #     break 
            
    #         # Lookahead waypoints
    #         lookahead_waypoints, closest_idx = mpc.get_lookahead_waypoints(
    #             current_location, 
    #             lookahead=lookahead
    #         )
            
    #         # MPC step
    #         control, accel, kappa = mpc.step(lookahead_waypoints)
    #         vehicle.apply_control(control)
            
    #         # Spectator update
    #         if step % 10 == 0:
    #             vehicle_transform = vehicle.get_transform()
    #             spectator_transform = carla.Transform(
    #                 vehicle_transform.location + carla.Location(z=60),
    #                 carla.Rotation(pitch=-90)
    #             )
    #             spectator.set_transform(spectator_transform)
            
    #         # Visualization & logging
    #         if step % 5 == 0:
    #             mpc.update_visualization(lookahead_waypoints)
                
    #             velocity = vehicle.get_velocity()
    #             speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    #             speed_kmh = speed * 3.6
    #             # ay 계산
    #             actual_ay = speed**2 * abs(kappa)

    #             #  CTE 계산
    #             cte = float('inf')
    #             for wp in lookahead_waypoints[:5]:
    #                 # Dict에서 좌표 추출
    #                 dx = current_location.x - wp['x']
    #                 dy = current_location.y - wp['y']
    #                 dist = np.sqrt(dx**2 + dy**2)
    #                 if dist < cte:
    #                     cte = dist
                
    #             kmin, kmax = mpc.get_dynamic_kappa_limits(speed)

    #             # ==================== 추가 출력 ====================
    #             ref_traj = mpc.get_reference_trajectory(lookahead_waypoints)
    #             v_ref_first = ref_traj[3, 0] * 3.6  # km/h
    #             v_ref_avg = np.mean(ref_traj[3, :]) * 3.6  # km/h
    #             # ===================================================
                
    #             print(f"Step {step:4d} | Dist: {distance_to_goal:6.2f}m | "
    #                 f"Speed: {speed_kmh:5.1f} km/h | "
    #                 f"CTE: {cte:5.2f}m | "
    #                 f"κ_mpc: {kappa:+6.3f} (+- {kmax:.3f}) | "
    #                 f"ay: {actual_ay:5.2f} | "
    #                 f"Steer: {control.steer:+5.2f}")
                
    #             # ==================== 추가 출력 ====================
    #             print(f"       | accel: {accel:+.2f} m/s² | "
    #                 f"v_ref: {v_ref_first:.1f} km/h (avg: {v_ref_avg:.1f}) | "
    #                 f"v_error: {speed_kmh - v_ref_first:+.1f} km/h")
    #             # ===================================================

    #         time.sleep(0.1)
    #         step += 1
            
    #         if step > 2000:
    #             print("\n Timeout")
    #             break
        
    #     print(f"\nFinished after {step} steps")
        
    #     if mpc.enable_viz:
    #         input("\nPress Enter to exit...")
    
    # except KeyboardInterrupt:
    #     print("\n Interrupted by user")
    
    # except Exception as e:
    #     print(f"\n Error: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # finally:
    #     print("\nCleaning up...")
    #     if step > 0:
    #         print(f"Total steps: {step}")
    #     vehicle.destroy()
    #     print("Done!")

if __name__ == '__main__':
    main()