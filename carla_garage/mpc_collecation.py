#!/usr/bin/env python3
"""
PlanT-Compatible MPC Data Collection
DataAgent (sensors/boxes/bev) + MPCAgent (control/GT) + Route Loop
"""

import sys
import os
# sys.path.append('/workspace/plant2/carla_garage')

import carla
import numpy as np
import time
import json
import gzip
from pathlib import Path
from datetime import datetime

from data_agent import DataAgent
from mpc_controller import MPCController


class MPCPlanTAgent:
    """MPC Agent that provides control + GT trajectory"""
    
    def __init__(self, mpc_config):
        self.mpc_config = mpc_config
        self.mpc = None
        self.global_route = None
        self.wheel_base = mpc_config.get('wheelbase', 2.875)
        self.max_steer = mpc_config.get('max_steer_angle', 1.22)
        
    def setup(self, vehicle, global_route):
        """Initialize with vehicle and route"""
        self.vehicle = vehicle
        self.global_route = global_route
        
        self.mpc = MPCController(
            vehicle=None,
            global_path=None,
            config=self.mpc_config
        )
        
        print("✅ MPC Agent setup complete")
    
    def get_local_route(self, ego_transform, lookahead=20):
        """Global route → Ego frame local route"""
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.global_route):
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
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
        
        return local_route, min_dist
    
    def run_step(self, ego_velocity):
        """MPC control step"""
        ego_transform = self.vehicle.get_transform()
        speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
        
        local_route, route_dist = self.get_local_route(ego_transform, lookahead=20)
        
        if route_dist > 10.0:
            print(f"⚠️ Off-track: {route_dist:.1f}m")
            control = carla.VehicleControl()
            control.brake = 1.0
            return control, None
        
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
        
        return control, mpc_gt


class RouteManager:
    """Manages CARLA route generation"""
    
    def __init__(self, world, town):
        self.world = world
        self.town = town
        self.carla_map = world.get_map()
        self.spawn_points = self.carla_map.get_spawn_points()
        
    def get_route_by_index(self, start_idx, end_idx, spacing=10.0):
        """Generate route between two spawn points"""
        if start_idx >= len(self.spawn_points) or end_idx >= len(self.spawn_points):
            raise ValueError(f"Invalid spawn point indices")
        
        start_wp = self.spawn_points[start_idx]
        end_wp = self.spawn_points[end_idx]
        
        current_wp = self.carla_map.get_waypoint(start_wp.location)
        target_wp = self.carla_map.get_waypoint(end_wp.location)
        
        route = []
        max_waypoints = 200
        
        for _ in range(max_waypoints):
            route.append({
                'x': current_wp.transform.location.x,
                'y': current_wp.transform.location.y,
                'z': current_wp.transform.location.z,
                'yaw': np.deg2rad(current_wp.transform.rotation.yaw)
            })
            
            next_wps = current_wp.next(spacing)
            if not next_wps:
                break
            
            current_wp = next_wps[0]
            
            dist_to_target = current_wp.transform.location.distance(
                target_wp.transform.location
            )
            if dist_to_target < 10.0:
                route.append({
                    'x': target_wp.transform.location.x,
                    'y': target_wp.transform.location.y,
                    'z': target_wp.transform.location.z,
                    'yaw': np.deg2rad(target_wp.transform.rotation.yaw)
                })
                break
        
        return route, start_wp, end_wp
    
    def get_all_route_pairs(self, max_routes=10):
        """Generate list of route pairs"""
        route_pairs = []
        n_spawns = len(self.spawn_points)
        
        for i in range(min(max_routes, n_spawns - 5)):
            end_idx = min(i + 5, n_spawns - 1)
            route_pairs.append((i, end_idx))
        
        return route_pairs


class MPCDataCollectionRunner:
    """Main runner for MPC data collection with PlanT format"""
    
    def __init__(self, save_dir='./mpc_plant_dataset', town='Town01'):
        self.client = carla.Client('172.22.39.145', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.town = town
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.route_manager = RouteManager(self.world, town)
        
        self.data_agent = None
        self.mpc_agent = None
        self.vehicle = None
        
        print(f"✅ Runner initialized for {town}")
        print(f"📁 Save directory: {self.save_dir}")
    
    def spawn_vehicle(self, spawn_point):
        """Spawn vehicle at given spawn point"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        # ⚠️ 핵심: role_name을 'hero'로 설정!
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        time.sleep(0.5)
        
        print(f"✅ Vehicle spawned at {spawn_point.location}")
        print(f"🔍 Role name: {self.vehicle.attributes.get('role_name', 'NONE')}")
        return self.vehicle
    
    def setup_agents(self, global_route, route_dir):
        """Setup DataAgent and MPCAgent"""
        
        # Dummy config 파일 먼저 생성
        dummy_config = route_dir / 'config_dummy.txt'
        dummy_config.touch()
        
        # 환경변수 설정 (DataAgent가 읽음)
        os.environ['SAVE_PATH'] = str(route_dir.parent)
        os.environ['TOWN'] = self.town
        os.environ['REPETITION'] = '0'
        os.environ['DATAGEN'] = '1'

        # CarlaDataProvider 초기화 (Leaderboard가 하던 일)
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from agents.navigation.local_planner import RoadOption

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(8000)
        
        # setup_agents 안에 디버깅 코드 추가
        print(f"🔍 Vehicle ID: {self.vehicle.id}")
        print(f"🔍 Vehicle type: {self.vehicle.type_id}")

        CarlaDataProvider.register_actor(self.vehicle, transform=self.vehicle.get_transform())
        CarlaDataProvider._carla_actor_pool[self.vehicle.id] = self.vehicle

        # 등록 확인
        hero = CarlaDataProvider.get_hero_actor()
        print(f"🔍 Hero actor from provider: {hero}")

        # DataAgent 초기화 (config 파일 경로 전달)
        self.data_agent = DataAgent(path_to_conf_file=str(dummy_config))

        # ⚠️ Global plan 설정 (Leaderboard가 하던 일)
        # global_route를 CARLA waypoint + RoadOption 형식으로 변환
        
        # Sparse waypoints (전체 route의 일부만)
        sparse_plan = []
        for i in range(0, len(global_route), 10):  # 10개마다 하나씩
            wp = global_route[i]
            location = carla.Location(x=wp['x'], y=wp['y'], z=wp['z'])
            waypoint = self.world.get_map().get_waypoint(location)
            sparse_plan.append((waypoint.transform, RoadOption.LANEFOLLOW))
        
        # Dense waypoints (모든 route points)
        dense_plan = []
        for wp in global_route:
            location = carla.Location(x=wp['x'], y=wp['y'], z=wp['z'])
            waypoint = self.world.get_map().get_waypoint(location)
            dense_plan.append((waypoint.transform, RoadOption.LANEFOLLOW))
        
        # DataAgent에 route 주입
        self.data_agent._global_plan = sparse_plan
        self.data_agent._global_plan_world_coord = sparse_plan
        self.data_agent.org_dense_route_world_coord = dense_plan
            
        # Setup 호출
        self.data_agent.setup(
            path_to_conf_file=str(dummy_config),
            route_index=0,
            traffic_manager=None
        )

        # ⚠️ 핵심: Vehicle 수동 주입
        # if not hasattr(self.data_agent, '_vehicle') or self.data_agent._vehicle is None:
        self.data_agent._vehicle = self.vehicle
        self.data_agent._world = self.world
        self.data_agent.world_map = self.world.get_map()

        # _init() 수동 호출 (센서 setup)
        self.data_agent._init(hd_map=None)
        
        print(f"✅ DataAgent setup complete")
        
        # MPCAgent config
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
            'v_max': 8.0,
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
        
        print("✅ MPCAgent setup complete")
    
    def run_route(self, route_idx, start_idx, end_idx, max_steps=1000):
        """Run single route collection"""
        
        print(f"\n{'='*60}")
        print(f"🚗 Route {route_idx}: Spawn {start_idx} → {end_idx}")
        
        global_route, start_spawn, end_spawn = self.route_manager.get_route_by_index(
            start_idx, end_idx, spacing=10.0
        )
        
        print(f"📍 Route length: {len(global_route)} waypoints")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        route_dir = self.save_dir / f"route_{route_idx:03d}_{timestamp}"
        route_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.spawn_vehicle(start_spawn)
        except RuntimeError as e:
            print(f"❌ Spawn failed: {e}")
            return False
        
        self.setup_agents(global_route, route_dir)

        spectator = self.world.get_spectator()
        
        frame_count = 0
        success_frames = 0
        start_time = time.time()
        
        try:
            for step in range(max_steps):
                ego_transform = self.vehicle.get_transform()
                ego_velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])

                # ⚠️ Spectator를 차량 뒤 위쪽에 배치
                spectator_transform = carla.Transform(
                    ego_transform.location + carla.Location(z=3.0),  # 3m 위
                    ego_transform.rotation
                )
                # 뒤에서 보기
                spectator_transform.location += (
                    -6.0 * spectator_transform.get_forward_vector()  # 6m 뒤
                )
                spectator_transform.rotation.pitch = -15  # 약간 아래로
                spectator.set_transform(spectator_transform)
                
                # MPC control step
                control, mpc_gt = self.mpc_agent.run_step(ego_velocity)
                
                # Apply control
                self.vehicle.apply_control(control)
                
                # DataAgent tick - 센서 데이터 저장
                # DataAgent.run_step() 대신 tick() 직접 호출
                input_data = {}  # DataAgent가 센서에서 자동으로 받음
                try:
                    # DataAgent의 내부 센서에서 데이터 수집
                    # 이 부분은 Leaderboard가 자동으로 해주는 부분
                    # 우리는 world.tick()으로 센서 데이터 업데이트
                    pass
                except Exception as e:
                    print(f"⚠️ DataAgent tick error: {e}")
                
                # Save MPC GT data
                if mpc_gt and mpc_gt.get('feasible', False):
                    gt_file = route_dir / 'measurements' / f'{frame_count:04d}_mpc_gt.json.gz'
                    gt_file.parent.mkdir(exist_ok=True)
                    
                    with gzip.open(gt_file, 'wt', encoding='utf-8') as f:
                        json.dump(mpc_gt, f)
                    
                    success_frames += 1
                
                frame_count += 1
                
                if step % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"Step {step:4d} | Speed: {speed*3.6:5.1f} km/h | "
                          f"Frames: {frame_count:4d} | Success: {success_frames:4d} | "
                          f"steer: {control.steer} | "
                          f"FPS: {fps:.1f}")
                
                self.world.tick()
                time.sleep(0.05)
                
                dist_to_end = ego_transform.location.distance(end_spawn.location)
                if dist_to_end < 5.0:
                    print(f"✅ Reached destination!")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            if self.data_agent:
                self.data_agent.destroy()
            
            if self.vehicle:
                self.vehicle.destroy()
            
            meta = {
                'town': self.town,
                'route_idx': route_idx,
                'start_spawn': start_idx,
                'end_spawn': end_idx,
                'total_frames': frame_count,
                'success_frames': success_frames,
                'route_length': len(global_route),
                'duration': time.time() - start_time,
            }
            
            meta_file = route_dir / 'metadata.json'
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"\n📊 Route {route_idx} complete:")
            print(f"   Frames: {frame_count} (Success: {success_frames})")
            print(f"   Saved to: {route_dir}")
        
        return True
    
    def run_all_routes(self, max_routes=5):
        """Run multiple routes"""
        route_pairs = self.route_manager.get_all_route_pairs(max_routes=max_routes)
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting MPC data collection")
        print(f"Town: {self.town}")
        print(f"Routes: {len(route_pairs)}")
        print(f"{'='*60}\n")
        
        for route_idx, (start_idx, end_idx) in enumerate(route_pairs):
            success = self.run_route(route_idx, start_idx, end_idx, max_steps=1000)
            
            if not success:
                print(f"⚠️ Route {route_idx} failed, skipping...")
                continue
            
            time.sleep(2.0)
        
        print(f"\n{'='*60}")
        print(f"✅ All routes complete!")
        print(f"📁 Data saved to: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    
    runner = MPCDataCollectionRunner(
        save_dir='./mpc_plant_dataset',
        town='Town01'
    )
    
    runner.run_all_routes(max_routes=5)


if __name__ == '__main__':
    main()