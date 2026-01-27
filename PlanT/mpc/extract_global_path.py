#!/usr/bin/env python3
"""
extract_global_path.py
CARLA에서 맵 전체를 크게 도는 global centerline 추출
"""

import carla
import numpy as np
import pickle
import networkx as nx
import time

def build_topology_graph(carla_map):
    """맵의 topology를 그래프로 변환"""
    topology = carla_map.get_topology()
    
    G = nx.DiGraph()
    
    # waypoint ID -> 그래프 노드 매핑
    wp_to_node = {}
    node_to_wp = {}
    node_id = 0
    
    for wp_start, wp_end in topology:
        # 시작 waypoint
        if wp_start.id not in wp_to_node:
            wp_to_node[wp_start.id] = node_id
            node_to_wp[node_id] = wp_start
            node_id += 1
        
        # 끝 waypoint
        if wp_end.id not in wp_to_node:
            wp_to_node[wp_end.id] = node_id
            node_to_wp[node_id] = wp_end
            node_id += 1
        
        # 엣지 추가
        start_node = wp_to_node[wp_start.id]
        end_node = wp_to_node[wp_end.id]
        
        dist = wp_start.transform.location.distance(wp_end.transform.location)
        
        G.add_edge(start_node, end_node, 
                   weight=dist, 
                   start_wp=wp_start, 
                   end_wp=wp_end)
    
    return G, wp_to_node, node_to_wp

def find_nearest_node(carla_map, target_wp, node_to_wp):
    """주어진 waypoint에 가장 가까운 그래프 노드 찾기"""
    target_loc = target_wp.transform.location
    
    min_dist = float('inf')
    nearest_node = None
    
    for node_id, wp in node_to_wp.items():
        dist = target_loc.distance(wp.transform.location)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node_id
    
    return nearest_node

def greedy_longest_path_improved(G, start_node, max_nodes=500):
    """개선된 Greedy: 방문 횟수 기반 exploration"""
    path = [start_node]
    visit_count = {node: 0 for node in G.nodes()}
    visit_count[start_node] = 1
    current = start_node
    
    for step in range(max_nodes):
        neighbors = list(G.successors(current))
        
        if not neighbors:
            print(f"Dead end at step {step}")
            break
        
        # 방문 횟수가 적은 이웃 우선
        next_node = min(neighbors, key=lambda n: visit_count[n])
        
        path.append(next_node)
        visit_count[next_node] += 1
        current = next_node
        
        # 시작점으로 돌아올 수 있고 충분히 길면 종료
        if step > 100 and start_node in neighbors:
            path.append(start_node)
            print(f"Completed loop with {len(path)} segments")
            break
        
        if step % 50 == 0:
            print(f"  {step} segments processed...")
    
    return path

def find_longest_loop(G, start_node):
    """시작점에서 가장 긴 경로 찾기"""
    print("Using greedy approach for large graph...")
    return greedy_longest_path_improved(G, start_node, max_nodes=30)

def path_to_centerline(G, path, spacing=2.0):
    """노드 경로 → 보간된 centerline"""
    centerline = []
    
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        
        if not G.has_edge(start_node, end_node):
            print(f"Warning: No edge {start_node} -> {end_node}")
            continue
        
        start_wp = G.edges[start_node, end_node]['start_wp']
        end_wp = G.edges[start_node, end_node]['end_wp']
        
        # 이 segment를 spacing 간격으로 보간
        current_wp = start_wp
        segment_points = []
        
        max_iters = 1000  # 무한 루프 방지
        iters = 0
        
        while iters < max_iters:
            loc = current_wp.transform.location
            rot = current_wp.transform.rotation
            
            segment_points.append({
                'x': loc.x,
                'y': loc.y,
                'z': loc.z,
                'yaw': np.deg2rad(rot.yaw)
            })
            
            # end_wp에 도달했는지 체크
            if current_wp.transform.location.distance(end_wp.transform.location) < spacing * 0.5:
                break
            
            next_wps = current_wp.next(spacing)
            if not next_wps:
                break
            
            current_wp = next_wps[0]
            iters += 1
        
        centerline.extend(segment_points)
    
    return centerline

def extract_global_centerline_full_map(world, start_idx=0, spacing=2.0):
    """맵 전체를 크게 도는 centerline 추출"""
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    print("🔨 Building topology graph...")
    G, wp_to_node, node_to_wp = build_topology_graph(carla_map)
    print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 시작점 찾기
    start_loc = spawn_points[start_idx].location
    start_wp = carla_map.get_waypoint(start_loc)
    start_node = find_nearest_node(carla_map, start_wp, node_to_wp)
    
    print(f"🔍 Finding longest path from spawn point {start_idx} (node {start_node})...")
    path = find_longest_loop(G, start_node)
    
    if not path:
        print("❌ Failed to find path")
        return []
    
    print(f"✅ Found path with {len(path)} topology segments")
    
    print("📍 Interpolating waypoints...")
    centerline = path_to_centerline(G, path, spacing)
    
    return centerline

def save_centerline(centerline, filename='centerline.pkl'):
    """저장"""
    with open(filename, 'wb') as f:
        pickle.dump(centerline, f)
    print(f"✅ Saved {len(centerline)} waypoints to {filename}")

def visualize_path(world, centerline, lifetime=300.0):
    """디버깅용: 경로 시각화"""
    debug = world.debug
    
    for i in range(len(centerline) - 1):
        p1 = carla.Location(x=centerline[i]['x'], 
                           y=centerline[i]['y'], 
                           z=centerline[i]['z'] + 0.5)
        p2 = carla.Location(x=centerline[i+1]['x'], 
                           y=centerline[i+1]['y'], 
                           z=centerline[i+1]['z'] + 0.5)
        
        debug.draw_line(p1, p2, 
                       thickness=1,
                       color=carla.Color(255, 0, 0),
                       life_time=lifetime)
        
        # 진행 상황 출력 (100개마다)
        if i % 100 == 0:
            print(f"  Drawing... {i}/{len(centerline)}")
    
    print(f"✅ Visualized {len(centerline)} waypoints (green lines)")
    print(f"   Lifetime: {lifetime:.0f} seconds")
    # ====================================================

def main():
    client = carla.Client('172.22.39.145', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    print("🚀 Extracting full map centerline...")
    centerline = extract_global_centerline_full_map(
        world, 
        start_idx=1,      # spawn point index
        spacing=2.0       # 간격 (m)
    )
    
    if not centerline:
        print("Failed to extract centerline")
        return
    
    save_centerline(centerline, 'town04_max30_start1.pkl')
    
    # 확인용 출력
    print(f"\nStatistics:")
    print(f"  Total waypoints: {len(centerline)}")
    print(f"  First: x={centerline[0]['x']:.2f}, y={centerline[0]['y']:.2f}")
    print(f"  Last:  x={centerline[-1]['x']:.2f}, y={centerline[-1]['y']:.2f}")
    
    # 시각화
    print("\nVisualizing path...")
    visualize_path(world, centerline, lifetime=300.0)
    
    # Spectator를 경로 시작점으로 이동
    print("\n📍 Moving spectator to route start...")
    spectator = world.get_spectator()
    start_wp = centerline[0]
    spectator_transform = carla.Transform(
        carla.Location(
            x=start_wp['x'],
            y=start_wp['y'],
            z=start_wp['z'] + 50.0  # 50m 위에서
        ),
        carla.Rotation(pitch=-45, yaw=0)  # 아래를 향해
    )
    spectator.set_transform(spectator_transform)
    # ===================================

    print("Done! Check CARLA window for green lines")

if __name__ == '__main__':
    main()