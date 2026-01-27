#!/usr/bin/env python3
"""
extract_global_path.py
CARLA에서 global centerline 추출해서 저장
"""

import carla
import numpy as np
import pickle

def extract_global_centerline(world, start_idx=0, path_length=500, spacing=2.0):
    """Global centerline 추출"""
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    
    # 시작점
    start_wp = map.get_waypoint(spawn_points[start_idx].location)
    
    centerline = []
    current_wp = start_wp
    
    for i in range(path_length):
        loc = current_wp.transform.location
        rot = current_wp.transform.rotation
        
        centerline.append({
            'x': loc.x,
            'y': loc.y,
            'z': loc.z,
            'yaw': np.deg2rad(rot.yaw)
        })
        
        next_wps = current_wp.next(spacing)
        if not next_wps:
            print(f"Warning: Reached end at {i} waypoints")
            break
        current_wp = next_wps[0]
    # print(centerline)
    return centerline

def save_centerline(centerline, filename='centerline.pkl'):
    """저장"""
    with open(filename, 'wb') as f:
        pickle.dump(centerline, f)
    print(f"✅ Saved {len(centerline)} waypoints to {filename}")
    

def main():
    # client = carla.Client('localhost', 2000)
    client = carla.Client('172.22.39.145', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    print("Extracting centerline...")
    centerline = extract_global_centerline(
        world, 
        start_idx=0,      # spawn point index
        path_length=500,  # 몇 개 추출할지
        spacing=4.0       # 간격 (m)
    )
    
    save_centerline(centerline, 'town04_centerline.pkl')
    
    # 확인용 출력
    print(f"\nFirst waypoint: x={centerline[0]['x']:.2f}, y={centerline[0]['y']:.2f}")
    print(f"Last waypoint: x={centerline[-1]['x']:.2f}, y={centerline[-1]['y']:.2f}")

if __name__ == '__main__':
    main()