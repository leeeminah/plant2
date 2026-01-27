#!/usr/bin/env python3
"""
MPC Data Collection for PlanT Training (Pre-saved Route Version)
미리 저장된 경로를 로드해서 MPC 데이터 수집
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
from pathlib import Path
from datetime import datetime

from mpc_controller import MPCController


class MPCPlanTAgent:
    """MPC Agent that provides control + GT trajectory"""
    
    def __init__(self, mpc_config):
        self.mpc_config = mpc_config
        self.mpc = None
        self.global_route = None
        self.wheel_base = mpc_config.get('wheelbase', 2.875)
        self.max_steer = mpc_config.get('max_steer_angle', 1.22)
        self.prev_closest_idx = None  

        
        self.step_counter = 0
        self.cached_control = None
        self.cached_mpc_gt = None
        
    def setup(self, vehicle, global_route):
        """Initialize with vehicle and route"""
        self.vehicle = vehicle
        self.global_route = global_route
        self.prev_closest_idx = None  # 초기화
        
        self.mpc = MPCController(
            vehicle=None,
            global_path=None,
            config=self.mpc_config
        )
        
        print("MPC Agent setup complete")
    
    def get_local_route(self, ego_transform, lookahead=20):
        """Global route → Ego frame local route (개선된 검색)"""
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # ============== 개선: 범위 제한 검색 ==============
        if self.prev_closest_idx is None:
            # 첫 호출: 전체 검색
            search_start = 0
            search_end = len(self.global_route)
        else:
            # 이후: 이전 index 주변만 검색
            search_start = max(0, self.prev_closest_idx - 50)
            search_end = min(len(self.global_route), self.prev_closest_idx + 150)
        
        min_dist = float('inf')
        closest_idx = self.prev_closest_idx if self.prev_closest_idx is not None else 0
        
        for i in range(search_start, search_end):
            wp = self.global_route[i]
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        self.prev_closest_idx = closest_idx
        # ==============================================
        
        local_route = []
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        for i in range(lookahead):
            idx = min((closest_idx + i), len(self.global_route) - 1)
            wp = self.global_route[idx]
            
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            local_yaw = wp['yaw'] - ego_yaw
            
            local_route.append({
                'x': local_x,
                'y': local_y,
                'yaw': local_yaw
            })
        
        return local_route, min_dist, closest_idx
    
    def run_step(self, ego_velocity):
        """MPC control step"""
        ego_transform = self.vehicle.get_transform()
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        local_route, route_dist, route_idx = self.get_local_route(ego_transform, lookahead=20)
        
        if route_dist > 10.0:
            print(f"Off-track: {route_dist:.1f}m")
            control = carla.VehicleControl()
            control.brake = 1.0
            mpc_gt = {'feasible': False, 'reason': 'off_track'}
            return control, mpc_gt, local_route, route_idx
        
        # MPC 10Hz solving (2 스텝마다)
        should_solve = (
            self.step_counter % 2 == 0 or
            self.cached_control is None
        )
        
        if should_solve:
            # === Fresh MPC solve ===
            current_state = np.array([0.0, 0.0, 0.0, speed])
            ref_traj = self.mpc.get_reference_trajectory(local_route)
            
            acceleration, curvature, success = self.mpc.solve(current_state, ref_traj)
            
            control = carla.VehicleControl()
            
            if success:
                steering_angle = np.arctan(curvature * self.wheel_base)
                steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
                control.steer = float(steering_angle / self.max_steer)
                
                if acceleration > 0:
                    control.throttle = float(np.clip(acceleration / 3.0, 0.0, 1.0))
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = float(np.clip(-acceleration / 3.0, 0.0, 1.0))
            else:
                control.throttle = 0.0
                control.brake = 1.0
            
            # MPC GT data
            if success and self.mpc.optimal_trajectory:
                opt_x, opt_y = self.mpc.optimal_trajectory
                
                velocities = None
                if hasattr(self.mpc, 'prev_solution') and self.mpc.prev_solution:
                    X_opt = self.mpc.prev_solution['X']
                    velocities = X_opt[3, :].tolist()
                
                mpc_gt = {
                    'trajectory': np.stack([opt_x, opt_y], axis=1).tolist(),
                    'velocities': velocities,
                    'target_speed': float(ref_traj[3, 0]),
                    'acceleration': float(acceleration),
                    'curvature': float(curvature),
                    'feasible': True
                }
            else:
                mpc_gt = {'feasible': False, 'reason': 'optimization_failed'}
            
            # Cache for reuse
            self.cached_control = control
            self.cached_mpc_gt = mpc_gt
            
        else:
            # === Reuse cached control ===
            control = self.cached_control
            mpc_gt = self.cached_mpc_gt
        
        self.step_counter += 1
        
        return control, mpc_gt, local_route, route_idx


class PlanTDataCollector:
    """Collects data in PlanT format using MPC with pre-saved routes"""
    
    def __init__(self, save_dir='./mpc_plant_dataset', town='Town04'):
        self.client = carla.Client('172.22.39.145', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.town = town
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.routes_dir = Path('./routes')  # 경로 저장 디렉토리
        
        self.mpc_agent = None
        self.vehicle = None
        
        print(f"Collector initialized for {town}")
        print(f"Save directory: {self.save_dir}")
        print(f"Routes directory: {self.routes_dir}")
    
    def load_global_path(self, filename):
        """저장된 global path 로드"""
        filepath = self.routes_dir / filename
        
        if not filepath.exists():
            print(f"Route file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Dict 형식으로 저장된 경우
        if isinstance(data, dict):
            centerline = data.get('route', data.get('centerline', None))
            print(f"Loaded route from {filename}")
            print(f"Waypoints: {len(centerline)}")
            if 'town' in data:
                print(f"   Town: {data['town']}")
            if 'start_idx' in data:
                print(f"   Start idx: {data['start_idx']}")
        else:
            # List 형식으로 저장된 경우(이거네)
            centerline = data
            print(f"Loaded {len(centerline)} waypoints from {filename}")
        
        return centerline
    
    def spawn_vehicle(self, spawn_point):
        """Spawn vehicle at given spawn point"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        time.sleep(0.5)
        
        print(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def get_spawn_from_route(self, route):
        """경로의 첫 waypoint에서 spawn point 생성"""
        first_wp = route[0]
        
        spawn_transform = carla.Transform()
        spawn_transform.location = carla.Location(
            x=first_wp['x'],
            y=first_wp['y'],
            z=first_wp['z'] + 0.5  # 약간 위로
        )
        spawn_transform.rotation = carla.Rotation(
            yaw=np.rad2deg(first_wp['yaw'])
        )
        
        return spawn_transform
    
    def get_bounding_boxes(self, ego_transform):
        """Get bounding boxes of nearby actors (PlanT format)"""
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
        """Save frame data in PlanT format"""
        
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        # 1. Measurements 저장
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
        }
        
        measurements_file = route_dir / 'measurements' / f'{frame_idx:04d}.json.gz'
        measurements_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(measurements_file, 'wt', encoding='utf-8') as f:
            json.dump(measurement, f)
        
        # 2. Boxes 저장
        boxes_file = route_dir / 'boxes' / f'{frame_idx:04d}.json.gz'
        boxes_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(boxes_file, 'wt', encoding='utf-8') as f:
            json.dump(boxes, f)
        
        # 3. MPC GT 저장
        if mpc_gt is not None:
            mpc_gt_file = route_dir / 'mpc_gt' / f'{frame_idx:04d}.json.gz'
            mpc_gt_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(mpc_gt_file, 'wt', encoding='utf-8') as f:
                json.dump(mpc_gt, f)
    
    def run_route(self, route_idx, route_filename, max_steps=None):
        """Run single route collection with pre-saved route"""
        
        print(f"\n{'='*60}")
        print(f"Route {route_idx}: {route_filename}")
        
        # ============== 경로 로드 ==============
        global_route = self.load_global_path(route_filename)
        
        if global_route is None:
            print("Failed to load route")
            return False
        # =====================================
        
        # ============== CARLA Settings ==============
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)
        
        print(f"CARLA Settings:")
        print(f"   Synchronous mode: True")
        print(f"   Fixed delta: 0.05s (20 Hz)")
        # ==========================================
        
        # ============== max_steps 자동 계산 ==============
        if max_steps is None:
            spacing = 2.0
            average_speed = 5.0  # m/s (v_max=8.0, 평균 5.0)
            fps = 20
            margin = 1.5
            
            total_distance = len(global_route) * spacing
            total_time = total_distance / average_speed
            required_steps = total_time * fps
            max_steps = int(required_steps * margin)
            
            print(f"Auto-calculated max_steps:")
            print(f"   Distance: {total_distance:.0f} m")
            print(f"   Est. time: {total_time:.0f} s ({total_time/60:.1f} min)")
            print(f"   Max steps: {max_steps}")
        # ===============================================
        
        # Create route directory
        route_name = f"{self.town}_Rep0_{route_idx}"
        route_dir = self.save_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        
        # ============== Spawn vehicle ==============
        spawn_point = self.get_spawn_from_route(global_route)
        
        try:
            self.spawn_vehicle(spawn_point)
        except RuntimeError as e:
            print(f"Spawn failed: {e}")
        
        # ============== MPC Setup ==============
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 15,
            'dt': 0.1,
            'Q': [100.0, 100.0, 50.0, 10.0],
            'R': [1.0, 5.0],
            'Qf': [200.0, 200.0, 100.0, 100.0],
            'a_min': -5.0,
            'a_max': 5.0,
            'kappa_min': -0.2,
            'kappa_max': 0.2,
            'ay_max': 5.0,
            'v_min': 0.0,
            'v_max': 30.0,
            'max_steer_angle': 1.22,
            'discount_rate': 0.95,
            'velocity_profile': {
                'kappa_breakpoints': [0.00, 0.01, 0.02],
                'velocity_ratios': [1.0, 0.65, 0.25],
            }
        }
        
        self.mpc_agent = MPCPlanTAgent(mpc_config)
        self.mpc_agent.setup(self.vehicle, global_route)
        # =====================================
        
        spectator = self.world.get_spectator()
        
        frame_count = 0
        success_frames = 0
        start_time = time.time()
        route_completed = False
        
        # ============== 완료 조건 변수 ==============
        max_progress = 0.0  # 최대 도달 progress
        min_steps = 500     # 최소 500 스텝
        # ========================================
        
        try:
            for step in range(max_steps):
                self.world.tick()
                # time.sleep(0.05) #debug를 위함 ?

                ego_transform = self.vehicle.get_transform()
                ego_velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
                
                # MPC control
                control, mpc_gt, local_route, route_idx_current = self.mpc_agent.run_step(ego_velocity)
                
                if not mpc_gt or not mpc_gt.get('feasible', False):
                    if mpc_gt and mpc_gt.get('reason') == 'off_track':
                        print(f"Off-track at step {step}, stopping")
                        break
                
                self.vehicle.apply_control(control)
                boxes = self.get_bounding_boxes(ego_transform)
                
                # 데이터 저장
                if mpc_gt and mpc_gt.get('feasible', False):
                    self.save_frame_data(
                        success_frames,
                        route_dir, ego_transform, ego_velocity,
                        local_route, mpc_gt, boxes
                    )
                    success_frames += 1
                
                frame_count += 1
                

                # 차량 위 시점 
                if step % 10 == 0:
                    spectator_transform = carla.Transform(
                        ego_transform.location + carla.Location(z=60),  # 이미 가져온 transform 사용
                        carla.Rotation(pitch=-90)
                    )
                    spectator.set_transform(spectator_transform)

                # Spectator (차량 뒤 시점)
                # if step % 2 == 0:
                #     spectator_transform = carla.Transform(
                #         ego_transform.location + carla.Location(z=3.0),
                #         ego_transform.rotation
                #     )
                #     spectator_transform.location += -6.0 * spectator_transform.get_forward_vector()
                #     spectator_transform.rotation.pitch = -15
                #     spectator.set_transform(spectator_transform)
                # 로깅
                
                if step % 20 == 0:
                    # elapsed = time.time() - start_time
                    # fps = frame_count / elapsed if elapsed > 0 else 0
                    progress = (route_idx_current / len(global_route)) * 100 if route_idx_current else 0

                    # MPC GT에서 가져오기
                    curvature = mpc_gt.get('curvature', 0.0) if mpc_gt else 0.0
                    acceleration = mpc_gt.get('acceleration', 0.0) if mpc_gt else 0.0
                    
                    print(f"Step {step:5d} | "
                        f"Speed: {speed*3.6:5.1f} km/h | "
                        f"Progress: {progress:5.1f}% | "
                        f"Saved: {success_frames:5d} | "\
                        f"Throttle: {control.throttle:4.2f} | "
                        f"Steer: {control.steer:5.2f} | "
                        f"Acc: {acceleration:5.2f} | "
                        f"Curv: {curvature:6.3f}")
                
                # ============== 완료 조건 체크 ==============
                progress = 0.0 
                if route_idx_current is not None:
                    progress = route_idx_current / len(global_route)
                    # 최대 progress 업데이트
                    if progress > max_progress:
                        max_progress = progress
                    
                    # 90% 완료
                    if progress >= 0.9:
                        print(f"Completed 90%+ of route!")
                        route_completed = True
                        break
                    
                    # Wrapping 감지
                    if step >= min_steps and max_progress > 0.2:
                        if progress < max_progress - 0.2:
                            print(f"Route completed (wrapping detected)")
                            print(f"   Max: {max_progress:.1%} → Current: {progress:.1%}")
                            route_completed = True
                            break
                # ==========================================
                           
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Settings 복원
            settings.synchronous_mode = False 
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            
            if self.vehicle:
                self.vehicle.destroy()
            
            # Save results
            results = {
                'scores': {
                    'score_composed': 100.0 if route_completed else 0.0,
                    'score_route': 100.0 if route_completed else 0.0,
                    'score_penalty': 0.0,
                },
                'status': 'Completed' if route_completed else 'Failed',
                'num_infractions': 0,
                'infractions': {'min_speed_infractions': []},
                'timestamp': route_name,
                'total_frames': frame_count,
                'success_frames': success_frames,
                'route_file': route_filename,
            }
            
            results_file = route_dir / 'results.json.gz'
            with gzip.open(results_file, 'wt', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n Route {route_idx} complete:")
            print(f"   Status: {' Completed' if route_completed else ' Failed'}")
            print(f"   Frames: {frame_count} (Success: {success_frames})")
            print(f"   Saved to: {route_dir}")
        
        return route_completed
    
    def run_all_routes(self, route_files):
        """Run multiple pre-saved routes"""
        
        print(f"\n{'='*60}")
        print(f"Starting PlanT-compatible MPC data collection")
        print(f"Town: {self.town}")
        print(f"Routes: {len(route_files)}")
        print(f"{'='*60}\n")
        
        completed_routes = 0
        
        for route_idx, route_file in enumerate(route_files):
            success = self.run_route(route_idx, route_file, max_steps=None)
            
            if success:
                completed_routes += 1
            
            time.sleep(2.0)
        
        print(f"\n{'='*60}")
        print(f" Collection complete!")
        print(f" Completed routes: {completed_routes}/{len(route_files)}")
        print(f" Data saved to: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    
    collector = PlanTDataCollector(
        save_dir='./mpc_plant_dataset',
        town='Town04'  # 또는 'Town02'
    )
    
    # ============== 미리 저장된 경로 파일 목록 ==============
    route_files = [  
        'town04_max30_start1.pkl',   # 경로 1
        'town04_max25_start4.pkl',   # 경로 2
        'town04_max30_start10.pkl',  # 경로 3
    ]
    # ====================================================
    
    collector.run_all_routes(route_files)


if __name__ == '__main__':
    main()