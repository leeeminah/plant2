#!/usr/bin/env python3
"""
visualize_mpc.py
MPC 주행 시각화: Reference path, 차량 위치, Optimal trajectory
"""

import sys
sys.path.append('/workspace/plant2/carla_garage')

import carla
import numpy as np
import pickle
import time
from pathlib import Path


class MPCVisualizer:
    """MPC 주행 실시간 시각화"""
    
    def __init__(self, world, route_file='routes/town04_racepath_0126_1.pkl'):
        self.world = world
        self.debug = world.debug
        
        # 경로 로드
        self.global_route = self.load_route(route_file)
        print(f"✅ Loaded {len(self.global_route)} waypoints")
        
        # 시각화 설정
        self.ref_path_lifetime = 300.0  # 5분
        self.vehicle_lifetime = 0.1     # 0.1초 (갱신)
        self.optimal_lifetime = 0.1     # 0.1초 (갱신)
        
        # Hero vehicle 찾기
        self.vehicle = None
        self.find_hero_vehicle()
    
    def load_route(self, filename):
        """경로 로드"""
        with open(filename, 'rb') as f:
            route = pickle.load(f)
        return route
    
    def find_hero_vehicle(self):
        """Hero 차량 찾기"""
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                if actor.attributes.get('role_name') == 'hero':
                    self.vehicle = actor
                    print(f"✅ Found hero vehicle: {actor.type_id}")
                    return
        print("⚠️ Hero vehicle not found")
    
    def draw_reference_path(self):
        """Reference path 그리기 (한 번만)"""
        print("🎨 Drawing reference path...")
        
        for i in range(len(self.global_route) - 1):
            p1 = carla.Location(
                x=self.global_route[i]['x'],
                y=self.global_route[i]['y'],
                z=self.global_route[i]['z'] + 0.2
            )
            p2 = carla.Location(
                x=self.global_route[i+1]['x'],
                y=self.global_route[i+1]['y'],
                z=self.global_route[i+1]['z'] + 0.2
            )
            
            self.debug.draw_line(
                p1, p2,
                thickness=0.1,
                color=carla.Color(0, 255, 0),  # 초록색
                life_time=self.ref_path_lifetime
            )
            
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(self.global_route)}")
        
        print(f"✅ Reference path drawn (green)")
    
    def draw_vehicle_position(self):
        """차량 현재 위치 표시"""
        if not self.vehicle:
            return
        
        loc = self.vehicle.get_location()
        
        # 차량 위치에 빨간 원
        self.debug.draw_point(
            loc + carla.Location(z=1.0),
            size=0.3,
            color=carla.Color(255, 0, 0),  # 빨간색
            life_time=self.vehicle_lifetime
        )
        
        # 차량 방향 화살표
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        
        end_loc = loc + carla.Location(
            x=forward.x * 3.0,
            y=forward.y * 3.0,
            z=1.0
        )
        
        self.debug.draw_arrow(
            loc + carla.Location(z=1.0),
            end_loc,
            thickness=0.2,
            arrow_size=0.3,
            color=carla.Color(255, 0, 0),  # 빨간색
            life_time=self.vehicle_lifetime
        )
    
    def draw_optimal_trajectory(self, trajectory):
        """MPC Optimal trajectory 그리기"""
        if not self.vehicle or not trajectory:
            return
        
        # Ego frame → Global frame 변환
        ego_transform = self.vehicle.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_z = ego_transform.location.z
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        
        global_traj = []
        for point in trajectory:
            local_x, local_y = point
            
            # Ego → Global
            global_x = ego_x + (cos_yaw * local_x - sin_yaw * local_y)
            global_y = ego_y + (sin_yaw * local_x + cos_yaw * local_y)
            
            global_traj.append(carla.Location(
                x=global_x,
                y=global_y,
                z=ego_z + 0.5
            ))
        
        # Optimal trajectory 그리기
        for i in range(len(global_traj) - 1):
            self.debug.draw_line(
                global_traj[i],
                global_traj[i + 1],
                thickness=0.15,
                color=carla.Color(0, 0, 255),  # 파란색
                life_time=self.optimal_lifetime
            )
    
    def get_mpc_data_from_vehicle(self):
        """파일에서 MPC 데이터 읽기"""
        viz_file = Path('/tmp/mpc_viz_data.pkl')
        
        if not viz_file.exists():
            return None
        
        try:
            with open(viz_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except:
            return None
    
    def run(self, update_hz=10):
        """시각화 루프 실행"""
        
        # Reference path 한 번만 그리기
        self.draw_reference_path()
        
        print(f"\n🎨 Starting visualization loop (Hz: {update_hz})")
        print("Press Ctrl+C to stop")
        
        dt = 1.0 / update_hz
        
        try:
            # while True:
            #     # Hero vehicle 다시 찾기 (respawn 대비)
            #     if not self.vehicle or not self.vehicle.is_alive:
            #         self.find_hero_vehicle()
                
            #     if self.vehicle:
            #         # 차량 위치 그리기
            #         self.draw_vehicle_position()
                    
            while True:
                if self.vehicle:
                    self.draw_vehicle_position()
                    
                    # ============== MPC 데이터 읽기 ==============
                    mpc_data = self.get_mpc_data_from_vehicle()
                    if mpc_data:
                        self.draw_optimal_trajectory(mpc_data['trajectory'])
                    # ==========================================
                
                time.sleep(dt)
                
                time.sleep(dt)
        
        except KeyboardInterrupt:
            print("\n✅ Visualization stopped")


def main():
    """Main entry point"""
    
    # CARLA 연결
    client = carla.Client('172.22.39.145', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 시각화 시작
    visualizer = MPCVisualizer(
        world,
        route_file='routes/town04_racepath_0126_1.pkl'
    )
    
    visualizer.run(update_hz=10)  # 10Hz 업데이트


if __name__ == '__main__':
    main()