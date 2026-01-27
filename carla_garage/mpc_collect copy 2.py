#!/usr/bin/env python3
"""
MPC Data Collection for PlanT Training (Full Map Loop Version)
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
import networkx as nx
from pathlib import Path
from datetime import datetime

from mpc_controller import MPCController
from route_extractor import CARLARouteExtractor

class MPCPlanTAgent:
    """MPC Agent that provides control + GT trajectory"""
    
    def __init__(self, mpc_config):
        self.mpc_config = mpc_config
        self.mpc = None
        self.global_route = None
        self.wheel_base = mpc_config.get('wheelbase', 2.875)
        self.max_steer = mpc_config.get('max_steer_angle', 1.22)

        # ============== 추가 ==============
        self.prev_closest_idx = None  # 이전 closest index 저장
        # =================================
        
    def setup(self, vehicle, global_route):
        """Initialize with vehicle and route"""
        self.vehicle = vehicle
        self.global_route = global_route
        
        self.mpc = MPCController(
            vehicle=None,
            global_path=None,
            config=self.mpc_config
        )

        # ============== 추가 ==============
        self.prev_closest_idx = None  # 초기화
        # =================================
        
        print("MPC Agent setup complete")
    
    def get_local_route(self, ego_transform, lookahead=20): #Global-> Ego 변환 
        """Global route → Ego frame local route"""
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # min_dist = float('inf')
        # closest_idx = 0
        
        # for i, wp in enumerate(self.global_route):
        #     dx = wp['x'] - ego_x
        #     dy = wp['y'] - ego_y
        #     dist = np.sqrt(dx**2 + dy**2)
        #     if dist < min_dist:
        #         min_dist = dist
        #         closest_idx = i
        
        # ============== 개선: 범위 제한 검색 ==============
        if self.prev_closest_idx is None:
            # 첫 호출: 전체 검색
            search_start = 0
            search_end = len(self.global_route)
        else:
            # 이후: 이전 index 주변만 검색
            search_window = 100  # 앞뒤 100개 waypoint만
            search_start = max(0, self.prev_closest_idx - 50)
            search_end = min(len(self.global_route), self.prev_closest_idx + search_window)
        
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
        
        # 저장
        self.prev_closest_idx = closest_idx
        # ================================================
        
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
            print(f"ff-track: {route_dist:.1f}m")
            control = carla.VehicleControl()
            control.brake = 1.0
            return control, None, None, None
        
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
        mpc_gt = None
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
            mpc_gt = {'feasible': False}
        
        return control, mpc_gt, local_route, route_idx


# ============== RouteManager==============

class RouteManager:
    """Manages CARLA route generation using full map loop"""
    
    def __init__(self, world, town):
        self.world = world
        self.town = town
        self.extractor = CARLARouteExtractor(world, town)
    
    def generate_full_map_route(self, start_idx=0, spacing=2.0):
        """맵 전체를 도는 긴 경로 생성"""
        return self.extractor.extract_route(
            start_idx=start_idx,
            spacing=spacing,
            max_nodes=500
        )
    
    def visualize_route(self, centerline, lifetime=300.0):
        """경로 시각화"""
        self.extractor.visualize_route(centerline, lifetime=lifetime)


# ============== PlanTDataCollector==============

class PlanTDataCollector:
    """Collects data in PlanT format using MPC with full map route"""
    
    def __init__(self, save_dir='./mpc_plant_dataset', town='Town04'):
        self.client = carla.Client('172.22.39.145', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.town = town
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.route_manager = RouteManager(self.world, town)
        
        self.mpc_agent = None
        self.vehicle = None
        
        print(f"Collector initialized for {town}")
        print(f"Save directory: {self.save_dir}")
    
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
    
    def get_bounding_boxes(self, ego_transform): #주변 객체들의 3D 위치/크기 정보
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
        
        # 1. Measurements 저장 "자차의 상태 + 주행 정보"
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
                
    def run_route(self, route_idx, start_idx, max_steps=2000):
        """Run single route collection with full map loop"""
        
        print(f"\n{'='*60}")
        print(f"Route {route_idx}: Full map loop from spawn {start_idx}")
        
        # 전체 맵 경로 생성
        global_route, start_spawn = self.route_manager.generate_full_map_route(
            start_idx=start_idx, 
            spacing=2.0
        )
        
        if global_route is None:
            print("Failed to generate route")
            return False
        
        print(f"Route length: {len(global_route)} waypoints")

        # ============== 🔧 CARLA FPS 설정 추가! ==============
        settings = self.world.get_settings()
        original_settings = settings  # 나중에 복원용
        
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS 설정!
        self.world.apply_settings(settings)
        
        print(f"CARLA Settings:")
        print(f"   Synchronous mode: {settings.synchronous_mode}")
        print(f"   Fixed delta: {settings.fixed_delta_seconds}s")
        print(f"   Target FPS: {1.0/settings.fixed_delta_seconds:.1f}")
        # ====================================================
        
        # ============== max_steps 자동 계산 ==============
        if max_steps is None:
            spacing = 2.0
            average_speed = 4  # m/s (보수적 추정)
            fps = 20
            margin = 1.5
            
            total_distance = len(global_route) * spacing
            total_time = total_distance / average_speed
            required_steps = total_time * fps
            max_steps = int(required_steps * margin)
            
            print(f"Auto-calculated max_steps:")
            print(f"   Distance: {total_distance:.0f} m")
            print(f"   Est. time: {total_time:.0f} s ({total_time/60:.1f} min)")
            print(f"   Max steps: {max_steps} (with 50% margin)")
        else:
            print(f"Using provided max_steps: {max_steps}")
            # 경고: max_steps가 충분한지 체크
            estimated_min_steps = len(global_route) * 8  # waypoint당 평균 8 스텝
            if max_steps < estimated_min_steps:
                print(f"WARNING: max_steps ({max_steps}) might be too small!")
                print(f"Estimated minimum: {estimated_min_steps}")
        # ==================================================
        
        # Create route directory
        route_name = f"{self.town}_Rep0_{route_idx}"
        route_dir = self.save_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.spawn_vehicle(start_spawn)
        except RuntimeError as e:
            print(f"Spawn failed: {e}")
            return False
        
        # Setup MPC
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 15,
            'dt': 0.1,
            'Q': [100.0, 100.0, 50.0, 10.0],
            'R': [1.0, 5.0],
            'Qf': [200.0, 200.0, 100.0, 100.0],
            'a_min': -5.0,
            'a_max': 5.0,
            'kappa_min': -0.5,
            'kappa_max': 0.5,
            'ay_max': 5.0,
            'v_min': 0.0,
            'v_max': 30.0,
            'max_steer_angle': 1.22,
            'discount_rate': 0.95,
            'velocity_profile': {
                # ==================== 곡률 기준 ====================
                # κ (1/m) | 반경 (m) | 속도 비율
                # -----------------------------------------------
                'kappa_breakpoints': [
                    0.00,   # R = ∞ (직선)
                    0.01,   # R = 100m
                    0.02,   # R = 50m
                ],
                'velocity_ratios': [
                    1.0,    # 100% (직선)
                    0.65,   # 65%
                    0.25,   # 25% (급커브)
                ],
                # ==================================================
            }
        }
        
        self.mpc_agent = MPCPlanTAgent(mpc_config)
        self.mpc_agent.setup(self.vehicle, global_route)
        
        spectator = self.world.get_spectator()
        
        frame_count = 0
        success_frames = 0
        start_time = time.time()
        
        route_completed = False
        min_steps_before_check = 100  # ← 추가! 최소 100 스텝 후 체크
        
        try:
            for step in range(max_steps):
                ego_transform = self.vehicle.get_transform()
                ego_velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
                
                # Spectator camera
                spectator_transform = carla.Transform(
                    ego_transform.location + carla.Location(z=3.0),
                    ego_transform.rotation
                )
                spectator_transform.location += -6.0 * spectator_transform.get_forward_vector()
                spectator_transform.rotation.pitch = -15
                spectator.set_transform(spectator_transform)
                
                # MPC control
                control, mpc_gt, local_route, route_idx_current = self.mpc_agent.run_step(ego_velocity)
                
                if control is None:
                    print("Control failed, stopping")
                    break
                
                self.vehicle.apply_control(control)
                
                boxes = self.get_bounding_boxes(ego_transform)
                
                if mpc_gt and mpc_gt.get('feasible', False):
                    # self.save_frame_data(
                    #     frame_count, route_dir, ego_transform, ego_velocity,
                    #     local_route, mpc_gt, boxes
                    # )
                    success_frames += 1
                
                frame_count += 1
                
                if step % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    progress = (route_idx_current / len(global_route)) * 100 if route_idx_current else 0
                    
                    print(f"Step {step:4d} | Speed: {speed*3.6:5.1f} km/h | "
                          f"Progress: {progress:5.1f}% | "
                          f"Frames: {success_frames:4d} | FPS: {fps:.1f}")
                
                self.world.tick()
                # time.sleep(0.05)
                
                # 경로의 90% 이상 완료하면 성공으로 간주
                if route_idx_current and route_idx_current > len(global_route) * 0.9:
                    print(f"Completed 90%+ of route!")
                    route_completed = True
                    break
                
                if route_idx_current:
                    progress = route_idx_current / len(global_route)
                    
                    # Progress가 줄어들면 (wrapping 감지)
                    if step >= min_steps_before_check:
                        if hasattr(self, 'prev_progress'):
                            if progress < self.prev_progress * 0.5:  # 50% 이상 급감
                                print(f"✅ Route completed (detected wrapping)")
                                route_completed = True
                                break
                    
                    self.prev_progress = progress
                    
                    # 90% 완료
                    if progress >= 0.9:
                        print(f"✅ Completed 90%+ of route!")
                        route_completed = True
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # ============== 🔧 Settings 복원 ==============
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            print("CARLA settings restored")
            # =============================================

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
                'infractions': {
                    'min_speed_infractions': []
                },
                'timestamp': route_name,
                'total_frames': frame_count,
                'success_frames': success_frames,
            }
            
            results_file = route_dir / 'results.json.gz'
            with gzip.open(results_file, 'wt', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Route {route_idx} complete:")
            print(f"   Status: {'✅ Completed' if route_completed else '❌ Failed'}")
            print(f"   Frames: {frame_count} (Success: {success_frames})")
            print(f"   Saved to: {route_dir}")
        
        return route_completed
    
    def run_all_routes(self, max_routes=3, start_indices=None):
        """Run multiple routes from different start points"""
        
        if start_indices is None:
            # 기본값: 첫 3개 spawn point
            start_indices = list(range(max_routes))
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting PlanT-compatible MPC data collection (Full Map)")
        print(f"Town: {self.town}")
        print(f"Routes: {len(start_indices)}")
        print(f"{'='*60}\n")
        
        completed_routes = 0
        
        for route_idx, start_idx in enumerate(start_indices):
            success = self.run_route(route_idx, start_idx, max_steps=None)
            
            if success:
                completed_routes += 1
            
            time.sleep(2.0)
        
        print(f"\n{'='*60}")
        print(f"✅ Collection complete!")
        print(f"📊 Completed routes: {completed_routes}/{len(start_indices)}")
        print(f"📁 Data saved to: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    
    collector = PlanTDataCollector(
        save_dir='./mpc_plant_dataset_fullmap',
        town='Town04'
    )
    
    # 3개의 다른 시작점에서 긴 경로 수집
    collector.run_all_routes(
        max_routes=3,
        start_indices=[1, 5, 10]  # 다양한 시작점
    )


if __name__ == '__main__':
    main()