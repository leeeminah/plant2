#!/usr/bin/env python3
"""
MPC Data Collection - Multi-Scenario with Strict Quality Control
실무 기준: feasible & on-track만 저장, 실패 시 즉시 중단
"""

import sys
import os
sys.path.append('/workspace/plant2/carla_garage')

import carla
import numpy as np
import time
import json
import gzip
import pickle
import shutil
from pathlib import Path

from mpc_race_controller import MPCRaceController


class QualityChecker:
    """데이터 품질 검증기"""
    
    def __init__(self, off_track_threshold=10.0, min_success_rate=0.8):
        self.off_track_threshold = off_track_threshold
        self.min_success_rate = min_success_rate
        
        self.total_frames = 0
        self.success_frames = 0
        self.failure_reasons = {
            'off_track': 0,
            'optimization_failed': 0,
            'collision': 0,
            'timeout': 0,
        }
    
    def check_frame(self, mpc_gt, route_dist, collision_detected):
        """
        프레임별 품질 검증
        
        Returns:
            (is_valid, should_continue, reason)
        """
        self.total_frames += 1
        
        # ❌ 1. Collision (최우선)
        if collision_detected:
            self.failure_reasons['collision'] += 1
            return False, False, 'collision'
        
        # ❌ 2. Off-track
        if route_dist > self.off_track_threshold:
            self.failure_reasons['off_track'] += 1
            return False, False, 'off_track'
        
        # ❌ 3. MPC infeasible
        if not mpc_gt or not mpc_gt.get('feasible', False):
            reason = mpc_gt.get('reason', 'unknown') if mpc_gt else 'no_mpc_gt'
            self.failure_reasons['optimization_failed'] += 1
            return False, False, f'mpc_failed_{reason}'
        
        # ✅ Valid frame
        self.success_frames += 1
        return True, True, 'success'
    
    def get_success_rate(self):
        """성공률 계산"""
        if self.total_frames == 0:
            return 0.0
        return self.success_frames / self.total_frames
    
    def is_route_valid(self):
        """Route 전체가 유효한가?"""
        return self.get_success_rate() >= self.min_success_rate
    
    def get_summary(self):
        """통계 요약"""
        return {
            'total_frames': self.total_frames,
            'success_frames': self.success_frames,
            'success_rate': self.get_success_rate(),
            'failure_reasons': self.failure_reasons.copy()
        }


class CollisionSensor:
    """충돌 감지 센서"""
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.collision_detected = False
        self.collision_history = []
        
        # Collision sensor 설정
        bp_lib = world.get_blueprint_library()
        collision_bp = bp_lib.find('sensor.other.collision')
        
        self.sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )
        
        # Callback 설정
        self.sensor.listen(lambda event: self._on_collision(event))
    
    def _on_collision(self, event):
        """충돌 이벤트 콜백"""
        self.collision_detected = True
        self.collision_history.append({
            'frame': event.frame,
            'actor': event.other_actor.type_id if event.other_actor else 'unknown',
            'intensity': event.normal_impulse.length()
        })
    
    def reset(self):
        """충돌 플래그 리셋"""
        self.collision_detected = False
    
    def has_collision(self):
        """충돌 발생했는가?"""
        return self.collision_detected
    
    def destroy(self):
        """센서 제거"""
        if self.sensor:
            self.sensor.destroy()


class ScenarioGenerator:
    """출발점 시나리오 생성기 - start_idx=0 주변 집중"""
    
    def __init__(self, raceline):
        self.raceline = raceline
        self.total_waypoints = len(raceline)
    
    def generate_scenarios(self, num_scenarios=50, base_start_idx=0, longitudinal_range=20):
        """
        시나리오 생성 - 특정 start_idx 주변에서만 출발
        
        Args:
            num_scenarios: 생성할 시나리오 수
            base_start_idx: 기준 출발 waypoint index (기본: 0)
            longitudinal_range: 진행 방향 offset 범위 (waypoints 단위, 기본: ±20)
        
        Returns:
            scenarios: list of scenario dicts
        """
        scenarios = []
        
        # ==================== 1. 기본 시나리오 (base_start_idx에서 정확히 출발) ====================
        
        # 정상 출발 (offset 없음)
        scenarios.append({
            'start_idx': base_start_idx,
            'lateral_offset': 0.0,
            'yaw_offset': 0.0,
            'description': 'Normal_Start'
        })
        
        # ==================== 2. Lateral Offset 시나리오 ====================
        
        # 좌측 offset (base_start_idx에서)
        for offset in [-2.0, -1.5, -1.0, -0.5]:
            scenarios.append({
                'start_idx': base_start_idx,
                'lateral_offset': offset,
                'yaw_offset': 0.0,
                'description': f'Left_{abs(offset):.1f}m'
            })
        
        # 우측 offset (base_start_idx에서)
        for offset in [0.5, 1.0, 1.5, 2.0]:
            scenarios.append({
                'start_idx': base_start_idx,
                'lateral_offset': offset,
                'yaw_offset': 0.0,
                'description': f'Right_{offset:.1f}m'
            })
        
        # ==================== 3. Yaw Offset 시나리오 ====================
        
        # 각도 offset (base_start_idx에서)
        for yaw_deg in [-15, -10, -5, 5, 10, 15]:
            scenarios.append({
                'start_idx': base_start_idx,
                'lateral_offset': 0.0,
                'yaw_offset': np.deg2rad(yaw_deg),
                'description': f'Yaw_{yaw_deg:+d}deg'
            })
        
        # ==================== 4. 혼합 시나리오 (Lateral + Yaw) ====================
        
        # 좌측 + 각도
        for lat, yaw in [(-1.0, 5), (-1.5, 10), (-2.0, 15)]:
            scenarios.append({
                'start_idx': base_start_idx,
                'lateral_offset': lat,
                'yaw_offset': np.deg2rad(yaw),
                'description': f'Left{abs(lat):.1f}m_Yaw{yaw:+d}deg'
            })
        
        # 우측 + 각도
        for lat, yaw in [(1.0, -5), (1.5, -10), (2.0, -15)]:
            scenarios.append({
                'start_idx': base_start_idx,
                'lateral_offset': lat,
                'yaw_offset': np.deg2rad(yaw),
                'description': f'Right{lat:.1f}m_Yaw{yaw:+d}deg'
            })
        
        # ==================== 5. 진행 방향 Offset (base_start_idx 주변) ====================
        
        # 앞쪽 출발 (5-20 waypoints 앞)
        for forward_wps in [5, 10, 15, 20]:
            scenarios.append({
                'start_idx': (base_start_idx + forward_wps) % self.total_waypoints,
                'lateral_offset': 0.0,
                'yaw_offset': 0.0,
                'description': f'Forward_{forward_wps}wps'
            })
        
        # 뒤쪽 출발 (5-20 waypoints 뒤)
        for backward_wps in [5, 10, 15, 20]:
            scenarios.append({
                'start_idx': (base_start_idx - backward_wps) % self.total_waypoints,
                'lateral_offset': 0.0,
                'yaw_offset': 0.0,
                'description': f'Backward_{backward_wps}wps'
            })
        
        # ==================== 6. Random 시나리오 (base_start_idx 주변만) ====================
        
        # np.random.seed(42)
        
        # remaining = num_scenarios - len(scenarios)
        
        # for i in range(remaining):
        #     # ✅ base_start_idx ± longitudinal_range 범위 내에서만
        #     random_offset = np.random.randint(-longitudinal_range, longitudinal_range + 1)
        #     random_start_idx = (base_start_idx + random_offset) % self.total_waypoints
            
        #     # Random lateral offset (-2 ~ 2m)
        #     random_lateral = np.random.uniform(-2.0, 2.0)
            
        #     # Random yaw offset (-15 ~ 15 degrees)
        #     random_yaw = np.deg2rad(np.random.uniform(-15, 15))
            
        #     scenarios.append({
        #         'start_idx': random_start_idx,
        #         'lateral_offset': random_lateral,
        #         'yaw_offset': random_yaw,
        #         'description': f'Random_{i:03d}'
        #     })
        
        # ==================== 통계 출력 ====================
        
        print(f"\n{'='*60}")
        print(f"Generated {len(scenarios)} scenarios around waypoint {base_start_idx}")
        print(f"{'='*60}")
        
        # Start index 분포
        start_indices = [s['start_idx'] for s in scenarios]
        min_idx = min(start_indices)
        max_idx = max(start_indices)
        
        print(f"\nStart index range: [{min_idx}, {max_idx}]")
        print(f"  Base: {base_start_idx}")
        print(f"  Spread: ±{max(abs(min_idx - base_start_idx), abs(max_idx - base_start_idx))} waypoints")
        
        # 처음 10개만 출력
        print(f"\nFirst 10 scenarios:")
        for i, s in enumerate(scenarios[:10]):
            print(f"  {i:2d}: {s['description']:25s} "
                  f"start_idx={s['start_idx']:4d}, "
                  f"lat={s['lateral_offset']:+5.2f}m, "
                  f"yaw={np.rad2deg(s['yaw_offset']):+6.1f}°")
        
        if len(scenarios) > 10:
            print(f"  ... and {len(scenarios) - 10} more")
        
        print(f"{'='*60}\n")
        
        return scenarios
    
    def get_spawn_transform(self, scenario, map_obj):
        """시나리오에서 spawn transform 계산"""
        start_idx = scenario['start_idx'] % self.total_waypoints
        wp = self.raceline[start_idx]
        
        base_x = wp['x']
        base_y = wp['y']
        base_yaw = wp['yaw']
        
        # Lateral offset (좌우)
        lateral_offset = scenario['lateral_offset']
        perpendicular_yaw = base_yaw + np.pi / 2
        
        offset_x = base_x + lateral_offset * np.cos(perpendicular_yaw)
        offset_y = base_y + lateral_offset * np.sin(perpendicular_yaw)
        
        # Yaw offset
        final_yaw = base_yaw + scenario['yaw_offset']
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        # Z coordinate from map
        test_location = carla.Location(x=offset_x, y=offset_y, z=0.0)
        waypoint_on_road = map_obj.get_waypoint(
            test_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint_on_road is not None:
            spawn_z = waypoint_on_road.transform.location.z + 0.5
        else:
            spawn_z = wp.get('z', 0.5) + 0.5
        
        spawn_transform = carla.Transform()
        spawn_transform.location = carla.Location(x=offset_x, y=offset_y, z=spawn_z)
        spawn_transform.rotation = carla.Rotation(yaw=np.rad2deg(final_yaw))
        
        return spawn_transform

class MPCPlanTAgent:
    """MPC Agent (기존과 동일)"""
    
    def __init__(self, mpc_config):
        self.mpc_config = mpc_config
        self.mpc = None
        self.raceline = None
        self.wheel_base = mpc_config.get('wheelbase', 2.875)
        self.max_steer = mpc_config.get('max_steer_angle', 1.22)
        self.a_max = mpc_config.get('a_max', 8.0)
        self.a_min = mpc_config.get('a_min', -8.0)
        self.prev_closest_idx = None
        
        self.step_counter = 0
        self.cached_control = None
        self.cached_mpc_gt = None

        self.horizon = mpc_config.get('horizon', 15)

    def setup(self, vehicle, raceline):
        self.vehicle = vehicle
        self.raceline = raceline
        self.prev_closest_idx = None
        
        self.mpc = MPCRaceController(
            vehicle=None,
            raceline=raceline,
            config=self.mpc_config
        )
    
    def get_lookahead_waypoints(self, ego_transform, lookahead=20):
        """
        Raceline에서 lookahead waypoints 추출
        
        수정: 검색 범위 대폭 확대 (150 → 전체)
        """
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # 전체 검색 (가장 확실한 방법)
        min_dist = float('inf')
        closest_idx = 0
        
        for i in range(len(self.raceline)):
            wp = self.raceline[i]
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 진행 확인 (디버깅)
        if self.prev_closest_idx is not None and closest_idx < self.prev_closest_idx:
            # Wrapping (정상) or 역행 (비정상)
            wrap_dist = (len(self.raceline) - self.prev_closest_idx) + closest_idx
            if wrap_dist > 100:  # 너무 크면 역행
                print(f"  Large backward jump: {self.prev_closest_idx} → {closest_idx}")
        
        self.prev_closest_idx = closest_idx
        
        # Lookahead waypoints 추출
        lookahead_wps = []
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.raceline)
            wp = self.raceline[idx]
            
            # Global → Ego frame 변환
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            local_yaw = wp['yaw'] - ego_yaw
            
            lookahead_wps.append({
                'x': local_x,
                'y': local_y,
                'yaw': local_yaw,
                'velocity': wp['velocity'],
                'curvature': wp['curvature']
            })
        
        return lookahead_wps, min_dist, closest_idx
    
    def run_step(self, ego_velocity):
        ego_transform = self.vehicle.get_transform()
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        lookahead_wps, route_dist, route_idx = self.get_lookahead_waypoints(ego_transform, lookahead=20)
        
        # Off-track 체크 (QualityChecker로 이동)
        
        should_solve = (
            self.step_counter % 2 == 0 or
            self.cached_control is None
        )
        
        if should_solve:
            current_state = np.array([0.0, 0.0, 0.0, speed])
            ref_traj = self.mpc.get_reference_trajectory(lookahead_wps)

            # ✅ Feedforward sequence 준비
            kappa_ff_sequence = np.array([
                lookahead_wps[i]['curvature'] for i in range(min(self.horizon, len(lookahead_wps)))
            ])
            
            # ✅ MPC solve with feedforward
            acceleration, kappa_fb, success = self.mpc.solve(current_state, ref_traj, kappa_ff_sequence)
            
                
            # acceleration, curvature, success = self.mpc.solve(current_state, ref_traj)
            
            control = carla.VehicleControl()
            
            if success:
                # ==================== Curvature composition ====================
                kappa_ff = lookahead_wps[0]['curvature']

                # ① 직선 판단 (raceline 기준)
                is_straight = abs(kappa_ff) < 5e-4  # ≈ radius > 1000m

                # ② feedback gain scheduling
                if is_straight:
                    # 직선에서는 MPC를 "미세 보정기"로
                    kappa_cmd_raw = 0.1 * kappa_fb + kappa_ff
                else:
                    # 곡선에서는 MPC 풀 파워
                    kappa_cmd_raw = kappa_fb + kappa_ff

                # kappa_fb = curvature
                # kappa_ff = lookahead_wps[0]['curvature']
                
                # kappa_cmd = kappa_fb + kappa_ff
                kappa_cmd = np.clip(kappa_cmd_raw, self.mpc.kappa_min, self.mpc.kappa_max)
                
                steering_angle = np.arctan(kappa_cmd * self.wheel_base)
                steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
                control.steer = float(steering_angle / self.max_steer)
                
                if acceleration > 0:
                    control.throttle = float(np.clip(acceleration / self.a_max, 0.0, 1.0))
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = float(np.clip(-acceleration / abs(self.a_min), 0.0, 1.0))
            else:
                control.throttle = 0.0
                control.brake = 1.0
            
            if success and self.mpc.optimal_trajectory:
                opt_x, opt_y = self.mpc.optimal_trajectory
                
                acceleration_seq = []
                curvature_seq = []
                curvature_ff_seq = []
                curvature_total_seq = []
                
                if hasattr(self.mpc, 'prev_solution') and self.mpc.prev_solution:
                    U_opt = self.mpc.prev_solution['U']
                    X_opt = self.mpc.prev_solution['X']
                    
                    acceleration_seq = U_opt[0, :].tolist()
                    curvature_seq = U_opt[1, :].tolist()
                    velocity_seq = X_opt[3, :].tolist()
                    
                    horizon_len = len(acceleration_seq)
                    for i in range(horizon_len):
                        if i < len(lookahead_wps):
                            kappa_ff_i = lookahead_wps[i]['curvature']
                        else:
                            kappa_ff_i = lookahead_wps[-1]['curvature']
                        
                        curvature_ff_seq.append(float(kappa_ff_i))
                    
                    curvature_total_seq = [
                        float(np.clip(fb + ff, self.mpc.kappa_min, self.mpc.kappa_max))
                        for fb, ff in zip(curvature_seq, curvature_ff_seq)
                    ]
                else:
                    horizon_len = self.mpc_config.get('horizon', 15)
                    acceleration_seq = [float(acceleration)] * horizon_len
                    curvature_seq = [float(kappa_fb)] * horizon_len
                    curvature_ff_seq = [float(kappa_ff)] * horizon_len
                    curvature_total_seq = [float(kappa_cmd)] * horizon_len
                    velocity_seq = [float(ref_traj[3, 0])] * (horizon_len + 1)
                
                mpc_gt = {
                    'trajectory': np.stack([opt_x, opt_y], axis=1).tolist(),
                    'accelerations': acceleration_seq,
                    'curvatures': curvature_seq,
                    'curvatures_feedforward': curvature_ff_seq,
                    'curvatures_total': curvature_total_seq,
                    'velocities': velocity_seq,
                    'target_speed': float(ref_traj[3, 0]),
                    'acceleration': float(acceleration),
                    'curvature': float(kappa_fb),
                    'curvature_feedforward': float(kappa_ff),
                    'curvature_total': float(kappa_cmd),
                    'feasible': True,
                    'horizon_length': len(acceleration_seq),
                }
            else:
                mpc_gt = {'feasible': False, 'reason': 'optimization_failed'}
            
            self.cached_control = control
            self.cached_mpc_gt = mpc_gt
            
        else:
            control = self.cached_control
            mpc_gt = self.cached_mpc_gt
        
        self.step_counter += 1
        
        return control, mpc_gt, lookahead_wps, route_idx, route_dist


class PlanTDataCollector:
    """Quality-controlled data collector"""
    
    def __init__(self, save_dir='./mpc_plant_dataset_quality', town='Town04'):
        # self.client = carla.Client('172.22.39.179', 2000)
        self.client = carla.Client('172.22.39.145', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.town = town
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.mpc_agent = None
        self.vehicle = None
        self.collision_sensor = None
        
        print(f"Collector initialized for {town}")
        print(f"Save directory: {self.save_dir}")
    
    def load_raceline(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        raceline = data['raceline']
        
        first_wp = raceline[0]
        second_wp = raceline[1]
        
        dx = second_wp['x'] - first_wp['x']
        dy = second_wp['y'] - first_wp['y']
        actual_yaw = np.arctan2(dy, dx)
        
        yaw_offset = actual_yaw - first_wp['yaw']
        yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
        
        for wp in raceline:
            wp['yaw_original'] = wp['yaw']
            wp['yaw'] = wp['yaw'] + yaw_offset
            wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
        
        print(f"Loaded raceline: {len(raceline)} waypoints")
        
        return raceline
    
    def spawn_vehicle(self, spawn_transform):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        
        for z_offset in [0.0, 0.5, 1.0, 1.5]:
            try:
                test_transform = carla.Transform(spawn_transform.location, spawn_transform.rotation)
                test_transform.location.z += z_offset
                
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, test_transform)
                
                if self.vehicle is not None:
                    print(f"  Spawned at z={test_transform.location.z:.2f}")
                    time.sleep(0.5)
                    
                    # ✅ Collision sensor 추가
                    self.collision_sensor = CollisionSensor(self.world, self.vehicle)
                    
                    return True
                
            except Exception as e:
                continue
        
        return False
    
    def get_bounding_boxes(self, ego_transform):
        boxes = []
        
        ego_extent = self.vehicle.bounding_box.extent
        ego_velocity = self.vehicle.get_velocity()
        ego_speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        boxes.append({
            'class': 'ego_car',
            'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
            'position': [0.0, 0.0, 0.0],
            'yaw': 0.0,
            'speed': ego_speed,
            'id': int(self.vehicle.id),
            'matrix': ego_transform.get_matrix()
        })
        
        return boxes
    
    def save_frame_data(self, frame_idx, route_dir, ego_transform, ego_velocity,
                        local_route, mpc_gt, boxes):
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        measurement = {
            'ego_matrix': ego_transform.get_matrix(),
            'pos_global': [ego_transform.location.x,
                          ego_transform.location.y,
                          ego_transform.location.z],
            'theta': np.deg2rad(ego_transform.rotation.yaw),
            'speed': speed,
            'target_speed': mpc_gt.get('target_speed', speed) if mpc_gt else speed,
            'speed_limit': 50.0 / 3.6,
            'route': [[wp['x'], wp['y']] for wp in local_route],
            'route_original': [[wp['x'], wp['y']] for wp in local_route],
            'brake': False,
            'augmentation_translation': 0.0,
            'augmentation_rotation': 0.0,
            'mpc_gt': mpc_gt,
        }
        
        measurements_file = route_dir / 'measurements' / f'{frame_idx:04d}.json.gz'
        measurements_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(measurements_file, 'wt', encoding='utf-8') as f:
            json.dump(measurement, f)
        
        boxes_file = route_dir / 'boxes' / f'{frame_idx:04d}.json.gz'
        boxes_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(boxes_file, 'wt', encoding='utf-8') as f:
            json.dump(boxes, f)
        
        if mpc_gt is not None:
            mpc_gt_file = route_dir / 'mpc_gt' / f'{frame_idx:04d}.json.gz'
            mpc_gt_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(mpc_gt_file, 'wt', encoding='utf-8') as f:
                json.dump(mpc_gt, f)
    
    def run_scenario(self, scenario_idx, scenario, raceline, raceline_name, max_steps=5000):
        """✅ 완주한 route만 저장"""
        
        print(f"\n{'='*60}")
        print(f"Scenario {scenario_idx}: {scenario['description']}")
        
        # CARLA sync
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # ✅ 임시 디렉토리 (완주 전까지)
        final_route_name = f"{self.town}_{raceline_name}_S{scenario_idx:03d}_{scenario['description']}"
        temp_route_name = f"temp_{final_route_name}"
        
        temp_route_dir = self.save_dir / temp_route_name
        temp_route_dir.mkdir(parents=True, exist_ok=True)
        
        # Spawn
        scenario_gen = ScenarioGenerator(raceline)
        spawn_transform = scenario_gen.get_spawn_transform(scenario, self.world.get_map())
        
        if not self.spawn_vehicle(spawn_transform):
            shutil.rmtree(temp_route_dir, ignore_errors=True)
            return False
        
        # MPC setup
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

        
        self.mpc_agent = MPCPlanTAgent(mpc_config)
        self.mpc_agent.setup(self.vehicle, raceline)
        
        quality_checker = QualityChecker(
            off_track_threshold=100.0,
            min_success_rate=0.8
        )
        
        spectator = self.world.get_spectator()
        
        frame_count = 0
        success_frames = 0
        route_completed = False
        failure_reason = None
        
        max_progress = 0.0
        min_steps = 500


        dt = settings.fixed_delta_seconds

        last_time = time.time()
        
        try:
            for step in range(max_steps):
                self.world.tick()

                now = time.time()
                elapsed = now - last_time

                if elapsed < dt:
                    time.sleep(dt - elapsed)

                last_time = time.time()
                
                ego_transform = self.vehicle.get_transform()
                ego_velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
                
                # MPC step
                control, mpc_gt, lookahead_wps, route_idx, route_dist = self.mpc_agent.run_step(ego_velocity)
                
                # ✅ Quality check
                is_valid, should_continue, reason = quality_checker.check_frame(
                    mpc_gt,
                    route_dist,
                    self.collision_sensor.has_collision()
                )
                
                if not should_continue:
                    failure_reason = reason
                    print(f"  ❌ Failed: {reason} at step {step}")
                    break
                
                # Apply control
                self.vehicle.apply_control(control)
                
                # ✅ 임시 디렉토리에 저장
                if is_valid:
                    boxes = self.get_bounding_boxes(ego_transform)
                    self.save_frame_data(
                        success_frames,
                        temp_route_dir,  # ← 임시!
                        ego_transform, ego_velocity,
                        lookahead_wps, mpc_gt, boxes
                    )
                    success_frames += 1
                
                frame_count += 1
                
                # Spectator
                if step % 10 == 0:
                        vehicle_transform = self.vehicle.get_transform()
                        spectator_transform = carla.Transform(
                            vehicle_transform.location + carla.Location(z=60),
                            carla.Rotation(pitch=-90)
                        )
                        spectator.set_transform(spectator_transform)
                
                # Logging
                if step % 20 == 0:
                    progress = (route_idx / len(raceline)) * 100 if route_idx else 0
                    success_rate = quality_checker.get_success_rate()
                    
                    print(f"  Step {step:5d} | "
                          f"Speed: {speed*3.6:5.1f} km/h | "
                          f"Progress: {progress:5.1f}% | "
                          f"Valid: {success_frames:5d} | "
                          f"Rate: {success_rate:.1%}")
                
                # ✅ 완주 조건
                if route_idx is not None:
                    progress = route_idx / len(raceline)
                    
                    if progress > max_progress:
                        max_progress = progress
                    
                    # 99% 이상
                    if progress >= 0.99:
                        print(f"  ✅ Reached 90%+ of raceline!")
                        route_completed = True
                        break
        
        except KeyboardInterrupt:
            print("\n  Interrupted")
        
        finally:
            # Cleanup
            #배속
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            
            if self.collision_sensor:
                self.collision_sensor.destroy()
            
            if self.vehicle:
                self.vehicle.destroy()
            
            # ✅ 완주 판정
            if route_completed:
                # 임시 → 최종 디렉토리
                final_route_dir = self.save_dir / final_route_name
                temp_route_dir.rename(final_route_dir)
                
                # Results 저장
                quality_summary = quality_checker.get_summary()
                results = {
                    'scores': {
                        'score_composed': 100.0,
                        'score_route': 100.0,
                    },
                    'status': 'Completed',
                    'progress': max_progress,
                    'timestamp': final_route_name,
                    'total_frames': frame_count,
                    'success_frames': success_frames,
                    'quality_summary': quality_summary,
                    'scenario': scenario,
                }
                
                results_file = final_route_dir / 'results.json.gz'
                with gzip.open(results_file, 'wt', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                
                print(f"  ✅ Status: Completed ({max_progress*100:.1f}%)")
                print(f"  ✅ Saved: {success_frames} frames")
                
                return True
            
            else:
                # ✅ 실패 시 임시 디렉토리 삭제
                shutil.rmtree(temp_route_dir, ignore_errors=True)
                
                print(f"  ❌ Status: Failed ({max_progress*100:.1f}%)")
                print(f"  ❌ Discarded: {success_frames} frames")
                
                if failure_reason:
                    print(f"  ❌ Reason: {failure_reason}")
                
                return False
               
    def run_all_scenarios(self, raceline_file, num_scenarios=50, base_start_idx=1):
        """Run all scenarios"""
        
        print(f"\n{'='*60}")
        print(f"Quality-Controlled Multi-Scenario Collection")
        print(f"Raceline: {raceline_file}")
        print(f"Target scenarios: {num_scenarios}")
        print(f"Base start index: {base_start_idx}")
        print(f"{'='*60}\n")
        
        raceline = self.load_raceline(raceline_file)
        raceline_name = Path(raceline_file).stem
        
        scenario_gen = ScenarioGenerator(raceline)
        scenarios = scenario_gen.generate_scenarios(
            num_scenarios=num_scenarios,
            base_start_idx=base_start_idx,
            longitudinal_range=20
        )
        
        completed = 0
        
        for idx, scenario in enumerate(scenarios):
            success = self.run_scenario(
                idx,
                scenario,
                raceline,
                raceline_name,
                max_steps=5000  # 충분한 시간
            )
            
            if success:
                completed += 1
            
            time.sleep(1.0)
        
        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"Completed: {completed}/{len(scenarios)}")
        print(f"Success rate: {100*completed/len(scenarios):.1f}%")
        print(f"Data saved to: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    
    collector = PlanTDataCollector(
        save_dir='./mpc_plant_dataset_completed',  # ✅ 완주만
        town='Town04'
    )
    
    raceline_files = [
        'routes/town04_raceline_mincurv13.pkl',
    ]
    
    for raceline_file in raceline_files:
        collector.run_all_scenarios(
            raceline_file,
            num_scenarios=50,
            base_start_idx=13  # ✅ Raceline spawn 위치
        )


if __name__ == '__main__':
    main()