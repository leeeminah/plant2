#!/usr/bin/env python3
"""
MPC Data Collection for PlanT Training (Raceline-based Version)
CommonRoad Raceline을 로드해서 MPC 데이터 수집 - 시퀀스 형태 GT
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

from mpc_race_controller import MPCRaceController


class MPCPlanTAgent:
    """MPC Agent that provides control + GT trajectory (Raceline version)"""
    
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
        
    def setup(self, vehicle, raceline):
        """Initialize with vehicle and raceline"""
        self.vehicle = vehicle
        self.raceline = raceline
        self.prev_closest_idx = None
        
        self.mpc = MPCRaceController(
            vehicle=None,
            raceline=raceline,
            config=self.mpc_config
        )
        
        print("MPC Agent setup complete")
    
    def get_lookahead_waypoints(self, ego_transform, lookahead=20):
        """Raceline에서 lookahead waypoints 추출 (ego frame 변환)"""
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # 범위 제한 검색
        if self.prev_closest_idx is None:
            search_start = 0
            search_end = len(self.raceline)
        else:
            search_start = max(0, self.prev_closest_idx - 50)
            search_end = min(len(self.raceline), self.prev_closest_idx + 150)
        
        min_dist = float('inf')
        closest_idx = self.prev_closest_idx if self.prev_closest_idx is not None else 0
        
        for i in range(search_start, search_end):
            wp = self.raceline[i]
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
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
        """MPC control step with SEQUENCE GT"""
        ego_transform = self.vehicle.get_transform()
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        lookahead_wps, route_dist, route_idx = self.get_lookahead_waypoints(
            ego_transform, 
            lookahead=20
        )
        
        if route_dist > 10.0:
            print(f"Off-track: {route_dist:.1f}m")
            control = carla.VehicleControl()
            control.brake = 1.0
            mpc_gt = {'feasible': False, 'reason': 'off_track'}
            return control, mpc_gt, lookahead_wps, route_idx
        
        # MPC 10Hz solving
        should_solve = (
            self.step_counter % 2 == 0 or
            self.cached_control is None
        )
        
        if should_solve:
            # === Fresh MPC solve ===
            current_state = np.array([0.0, 0.0, 0.0, speed])
            ref_traj = self.mpc.get_reference_trajectory(lookahead_wps)
            
            acceleration, curvature, success = self.mpc.solve(current_state, ref_traj)
            
            control = carla.VehicleControl()
            
            if success:
                # ==================== Curvature Feedforward ====================
                kappa_fb = curvature
                kappa_ff = lookahead_wps[0]['curvature']
                kappa_cmd = kappa_fb + kappa_ff
                kappa_cmd = np.clip(kappa_cmd, self.mpc.kappa_min, self.mpc.kappa_max)
                # ================================================================
                
                # Steering
                steering_angle = np.arctan(kappa_cmd * self.wheel_base)
                steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
                control.steer = float(steering_angle / self.max_steer)
                
                # Throttle/Brake
                if acceleration > 0:
                    control.throttle = float(np.clip(acceleration / self.a_max, 0.0, 1.0))
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = float(np.clip(-acceleration / abs(self.a_min), 0.0, 1.0))
            else:
                control.throttle = 0.0
                control.brake = 1.0
            
            # ==================== 시퀀스 형태 MPC GT ====================
            if success and self.mpc.optimal_trajectory:
                opt_x, opt_y = self.mpc.optimal_trajectory
                
                # ✅ 1. Control sequence 추출
                acceleration_seq = []
                curvature_seq = []
                curvature_ff_seq = []
                curvature_total_seq = []
                
                if hasattr(self.mpc, 'prev_solution') and self.mpc.prev_solution:
                    U_opt = self.mpc.prev_solution['U']  # (2, N-1)
                    X_opt = self.mpc.prev_solution['X']  # (4, N)
                    
                    # Acceleration sequence (horizon 길이)
                    acceleration_seq = U_opt[0, :].tolist()  # [a0, a1, ..., a_{N-2}]
                    
                    # Curvature feedback sequence
                    curvature_seq = U_opt[1, :].tolist()  # [κ0, κ1, ..., κ_{N-2}]
                    
                    # Velocity sequence
                    velocity_seq = X_opt[3, :].tolist()  # [v0, v1, ..., v_{N-1}]
                    
                    # ✅ 2. Feedforward curvature sequence (raceline에서)
                    horizon_len = len(acceleration_seq)
                    for i in range(horizon_len):
                        if i < len(lookahead_wps):
                            kappa_ff_i = lookahead_wps[i]['curvature']
                        else:
                            kappa_ff_i = lookahead_wps[-1]['curvature']
                        
                        curvature_ff_seq.append(float(kappa_ff_i))
                    
                    # ✅ 3. Total curvature sequence (feedback + feedforward)
                    curvature_total_seq = [
                        float(np.clip(fb + ff, self.mpc.kappa_min, self.mpc.kappa_max))
                        for fb, ff in zip(curvature_seq, curvature_ff_seq)
                    ]
                else:
                    # Fallback: single value를 시퀀스로 확장
                    horizon_len = self.mpc_config.get('horizon', 15)
                    acceleration_seq = [float(acceleration)] * horizon_len
                    curvature_seq = [float(curvature)] * horizon_len
                    curvature_ff_seq = [float(kappa_ff)] * horizon_len
                    curvature_total_seq = [float(kappa_cmd)] * horizon_len
                    velocity_seq = [float(ref_traj[3, 0])] * (horizon_len + 1)
                
                mpc_gt = {
                    # Trajectory (spatial)
                    'trajectory': np.stack([opt_x, opt_y], axis=1).tolist(),
                    
                    # Control sequences (temporal) - 핵심!
                    'accelerations': acceleration_seq,        # [a0, a1, ..., a_{N-2}]
                    'curvatures': curvature_seq,              # [κ_fb0, κ_fb1, ...]
                    'curvatures_feedforward': curvature_ff_seq,  # [κ_ff0, κ_ff1, ...]
                    'curvatures_total': curvature_total_seq,  # [κ_cmd0, κ_cmd1, ...]
                    'velocities': velocity_seq,               # [v0, v1, ..., v_N]
                    
                    # Scalar values (backward compatibility)
                    'target_speed': float(ref_traj[3, 0]),
                    'acceleration': float(acceleration),      # 첫 번째 값
                    'curvature': float(curvature),            # 첫 번째 feedback
                    'curvature_feedforward': float(kappa_ff), # 첫 번째 feedforward
                    'curvature_total': float(kappa_cmd),      # 첫 번째 total
                    
                    'feasible': True,
                    'horizon_length': len(acceleration_seq),
                }
            else:
                mpc_gt = {'feasible': False, 'reason': 'optimization_failed'}
            # ============================================================
            
            # Cache for reuse
            self.cached_control = control
            self.cached_mpc_gt = mpc_gt
            
        else:
            # === Reuse cached control ===
            control = self.cached_control
            mpc_gt = self.cached_mpc_gt
        
        self.step_counter += 1
        
        return control, mpc_gt, lookahead_wps, route_idx


class PlanTDataCollector:
    """Collects data in PlanT format using MPC with CommonRoad Raceline"""
    
    def __init__(self, save_dir='./mpc_plant_dataset', town='Town04'):
        self.client = carla.Client('172.22.39.179', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.town = town
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.routes_dir = Path('./routes')
        
        self.mpc_agent = None
        self.vehicle = None
        
        print(f"Collector initialized for {town}")
        print(f"Save directory: {self.save_dir}")
    
    def load_raceline(self, filename='town04_raceline_mincurv.pkl'):
        """CommonRoad raceline 로드"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        raceline = data['raceline']
        metadata = data.get('metadata', {})
        
        # YAW OFFSET 계산
        first_wp = raceline[0]
        second_wp = raceline[1]
        
        dx = second_wp['x'] - first_wp['x']
        dy = second_wp['y'] - first_wp['y']
        actual_yaw = np.arctan2(dy, dx)
        
        yaw_offset = actual_yaw - first_wp['yaw']
        yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
        
        # 모든 waypoint에 offset 적용
        for wp in raceline:
            wp['yaw_original'] = wp['yaw']
            wp['yaw'] = wp['yaw'] + yaw_offset
            wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
        
        print(f"Loaded raceline: {len(raceline)} waypoints")
        print(f"  Applied YAW offset: {np.rad2deg(yaw_offset):.2f}°")
        
        # 통계
        velocities = [wp['velocity'] for wp in raceline]
        curvatures = [abs(wp['curvature']) for wp in raceline]
        
        print(f"  Velocity: {min(velocities)*3.6:.1f} - {max(velocities)*3.6:.1f} km/h")
        print(f"  Max |curvature|: {max(curvatures):.4f} (1/m)")
        
        return raceline
    
    def spawn_vehicle(self, spawn_point):
        """Spawn vehicle"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        time.sleep(0.5)
        
        print(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def get_spawn_from_raceline(self, raceline):
        """Raceline 첫 waypoint에서 spawn"""
        first_wp = raceline[0]
        
        # Z 좌표 찾기
        map_obj = self.world.get_map()
        test_location = carla.Location(x=first_wp['x'], y=first_wp['y'], z=0.0)
        waypoint_on_road = map_obj.get_waypoint(test_location, project_to_road=True)
        
        if waypoint_on_road is not None:
            spawn_z = waypoint_on_road.transform.location.z + 0.5
        else:
            spawn_z = first_wp.get('z', 0.5) + 0.5
        
        spawn_transform = carla.Transform()
        spawn_transform.location = carla.Location(
            x=first_wp['x'],
            y=first_wp['y'],
            z=spawn_z
        )
        spawn_transform.rotation = carla.Rotation(
            yaw=np.rad2deg(first_wp['yaw'])
        )
        
        return spawn_transform
    
    def get_bounding_boxes(self, ego_transform):
        """Get bounding boxes"""
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
        """Save frame data (MPC GT with sequences)"""
        
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        # 1. Measurements
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
            'mpc_gt': mpc_gt,  # ✅ measurements에 직접 포함!
        }
        
        measurements_file = route_dir / 'measurements' / f'{frame_idx:04d}.json.gz'
        measurements_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(measurements_file, 'wt', encoding='utf-8') as f:
            json.dump(measurement, f)
        
        # 2. Boxes
        boxes_file = route_dir / 'boxes' / f'{frame_idx:04d}.json.gz'
        boxes_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(boxes_file, 'wt', encoding='utf-8') as f:
            json.dump(boxes, f)
        
        # 3. MPC GT (별도 저장도 유지 - 옵션)
        if mpc_gt is not None:
            mpc_gt_file = route_dir / 'mpc_gt' / f'{frame_idx:04d}.json.gz'
            mpc_gt_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(mpc_gt_file, 'wt', encoding='utf-8') as f:
                json.dump(mpc_gt, f)
    
    def run_route(self, route_idx, raceline_filename, max_steps=None):
        """Run single route"""
        
        print(f"\n{'='*60}")
        print(f"Route {route_idx}: {raceline_filename}")
        
        raceline = self.load_raceline(raceline_filename)
        if raceline is None:
            print("Failed to load raceline")
            return False
        
        # CARLA Settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        print(f"CARLA: Sync mode ON, 20 Hz")
        
        # Auto-calculate max_steps
        if max_steps is None:
            total_distance = len(raceline) * 2.0
            average_speed = 12.0
            fps = 20
            margin = 1.5
            
            total_time = total_distance / average_speed
            required_steps = total_time * fps
            max_steps = int(required_steps * margin)
            
            print(f"Max steps: {max_steps} (est. {total_time:.0f}s)")
        
        # Create route directory
        route_name = f"{self.town}_Rep0_{route_idx}"
        route_dir = self.save_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        
        # Spawn
        spawn_point = self.get_spawn_from_raceline(raceline)
        
        try:
            self.spawn_vehicle(spawn_point)
        except RuntimeError as e:
            print(f"Spawn failed: {e}")
            return False
        
        # MPC Setup
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': 15,
            'dt': 0.1,
            'Q': [100.0, 100.0, 50.0, 10.0],
            'R': [1.0, 5.0],
            'Qf': [200.0, 200.0, 100.0, 100.0],
            'a_min': -8.0,
            'a_max': 8.0,
            'kappa_min': -0.2,
            'kappa_max': 0.2,
            'ay_max': 10.0,
            'v_min': 0.0,
            'v_max': 50.0,
            'max_steer_angle': 1.22,
            'discount_rate': 0.95,
            'velocity_scale': 1.0,
            'visualization': False,
            'target_laps': 1
        }
        
        self.mpc_agent = MPCPlanTAgent(mpc_config)
        self.mpc_agent.setup(self.vehicle, raceline)
        
        spectator = self.world.get_spectator()
        
        frame_count = 0
        success_frames = 0
        route_completed = False
        
        max_progress = 0.0
        min_steps = 500
        
        try:
            for step in range(max_steps):
                self.world.tick()

                ego_transform = self.vehicle.get_transform()
                ego_velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
                
                # MPC control
                control, mpc_gt, lookahead_wps, route_idx_current = self.mpc_agent.run_step(ego_velocity)
                
                if not mpc_gt or not mpc_gt.get('feasible', False):
                    if mpc_gt and mpc_gt.get('reason') == 'off_track':
                        print(f"Off-track at step {step}")
                        break
                
                self.vehicle.apply_control(control)
                boxes = self.get_bounding_boxes(ego_transform)
                
                # Save data
                if mpc_gt and mpc_gt.get('feasible', False):
                    self.save_frame_data(
                        success_frames,
                        route_dir, ego_transform, ego_velocity,
                        lookahead_wps, mpc_gt, boxes
                    )
                    success_frames += 1
                
                frame_count += 1
                
                # Spectator
                if step % 10 == 0:
                    spectator_transform = carla.Transform(
                        ego_transform.location + carla.Location(z=60),
                        carla.Rotation(pitch=-90)
                    )
                    spectator.set_transform(spectator_transform)

                # Logging
                if step % 20 == 0:
                    progress = (route_idx_current / len(raceline)) * 100 if route_idx_current else 0
                    
                    # ✅ 시퀀스 첫 값 사용
                    acceleration = mpc_gt.get('accelerations', [0.0])[0] if mpc_gt else 0.0
                    curvature = mpc_gt.get('curvatures_total', [0.0])[0] if mpc_gt else 0.0
                    
                    print(f"Step {step:5d} | "
                        f"Speed: {speed*3.6:5.1f} km/h | "
                        f"Progress: {progress:5.1f}% | "
                        f"Saved: {success_frames:5d} | "
                        f"Acc: {acceleration:5.2f} | "
                        f"Curv: {curvature:6.3f}")
                
                # Completion check
                if route_idx_current is not None:
                    progress = route_idx_current / len(raceline)
                    if progress > max_progress:
                        max_progress = progress
                    
                    if progress >= 0.9:
                        print(f"Completed 90%+ of raceline!")
                        route_completed = True
                        break
                    
                    if step >= min_steps and max_progress > 0.2:
                        if progress < max_progress - 0.2:
                            print(f"Raceline completed (wrapping)")
                            route_completed = True
                            break
                           
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Restore settings
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
                'raceline_file': raceline_filename,
            }
            
            results_file = route_dir / 'results.json.gz'
            with gzip.open(results_file, 'wt', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nRoute {route_idx} complete:")
            print(f"   Status: {'Completed' if route_completed else 'Failed'}")
            print(f"   Frames: {frame_count} (Success: {success_frames})")
            print(f"   Saved to: {route_dir}")
        
        return route_completed
    
    def run_all_routes(self, raceline_files):
        """Run multiple racelines"""
        
        print(f"\n{'='*60}")
        print(f"MPC Data Collection - Sequence GT Version")
        print(f"Town: {self.town}")
        print(f"Racelines: {len(raceline_files)}")
        print(f"{'='*60}\n")
        
        completed_routes = 0
        
        for route_idx, raceline_file in enumerate(raceline_files):
            success = self.run_route(route_idx, raceline_file, max_steps=None)
            
            if success:
                completed_routes += 1
            
            time.sleep(2.0)
        
        print(f"\n{'='*60}")
        print(f"Collection complete!")
        print(f"Completed: {completed_routes}/{len(raceline_files)}")
        print(f"Data saved to: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    
    collector = PlanTDataCollector(
        save_dir='./mpc_plant_dataset_seq', 
        town='Town04'
    )
    
    raceline_files = [  
        'town04_raceline_mincurv.pkl',
        # 'town04_raceline_shortest.pkl',
    ]
    
    collector.run_all_routes(raceline_files)


if __name__ == '__main__':
    main()