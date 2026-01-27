#!/usr/bin/env python3
"""
route_extractor.py
CARLA에서 맵 전체를 크게 도는 global centerline 추출 (클래스 버전)
"""

import carla
import numpy as np
import pickle
import networkx as nx
from pathlib import Path


class CARLARouteExtractor:
    """CARLA 맵에서 전체 경로를 추출하는 클래스"""
    
    def __init__(self, world, town_name):
        """
        Args:
            world: CARLA world 객체
            town_name: Town 이름 (예: 'Town04')
        """
        self.world = world
        self.town_name = town_name
        self.carla_map = world.get_map()
        self.spawn_points = self.carla_map.get_spawn_points()
        
        # Graph 캐싱
        self.topology_graph = None
        self.wp_to_node = None
        self.node_to_wp = None
        
        print(f"✅ RouteExtractor initialized for {town_name}")
        print(f"📍 Found {len(self.spawn_points)} spawn points")
    
    def build_topology_graph(self):
        """맵의 topology를 그래프로 변환"""
        if self.topology_graph is not None:
            print("📦 Using cached topology graph")
            return self.topology_graph, self.wp_to_node, self.node_to_wp
        
        print("🔨 Building topology graph...")
        topology = self.carla_map.get_topology()
        
        G = nx.DiGraph()
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
        
        # 캐싱
        self.topology_graph = G
        self.wp_to_node = wp_to_node
        self.node_to_wp = node_to_wp
        
        print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, wp_to_node, node_to_wp
    
    def find_nearest_node(self, target_wp):
        """주어진 waypoint에 가장 가까운 그래프 노드 찾기"""
        if self.node_to_wp is None:
            raise RuntimeError("Topology graph not built. Call build_topology_graph() first.")
        
        target_loc = target_wp.transform.location
        
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, wp in self.node_to_wp.items():
            dist = target_loc.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        
        return nearest_node
    
    def greedy_longest_path(self, start_node, max_nodes=500):
        """개선된 Greedy: 방문 횟수 기반 exploration"""
        if self.topology_graph is None:
            raise RuntimeError("Topology graph not built. Call build_topology_graph() first.")
        
        G = self.topology_graph
        path = [start_node]
        visit_count = {node: 0 for node in G.nodes()}
        visit_count[start_node] = 1
        current = start_node
        
        for step in range(max_nodes):
            neighbors = list(G.successors(current))
            
            if not neighbors:
                print(f"⚠️ Dead end at step {step}")
                break
            
            # 방문 횟수가 적은 이웃 우선
            next_node = min(neighbors, key=lambda n: visit_count[n])
            
            path.append(next_node)
            visit_count[next_node] += 1
            current = next_node
            
            # 시작점으로 돌아올 수 있고 충분히 길면 종료
            if step > 100 and start_node in neighbors:
                path.append(start_node)
                print(f"✅ Completed loop with {len(path)} segments")
                break
            
            if step % 50 == 0:
                print(f"  {step} segments processed...")
        
        return path
    
    def path_to_centerline(self, path, spacing=2.0):
        """노드 경로 → 보간된 centerline"""
        if self.topology_graph is None:
            raise RuntimeError("Topology graph not built. Call build_topology_graph() first.")
        
        G = self.topology_graph
        centerline = []
        
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            
            if not G.has_edge(start_node, end_node):
                print(f"⚠️ No edge {start_node} -> {end_node}")
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
    
    def extract_route(self, start_idx=0, spacing=2.0, max_nodes=500):
        """
        맵 전체를 크게 도는 centerline 추출
        
        Args:
            start_idx: Spawn point 인덱스
            spacing: Waypoint 간격 (m)
            max_nodes: 최대 topology 노드 수
            
        Returns:
            centerline: List of waypoint dicts
            start_spawn: 시작 spawn point
        """
        # Graph 빌드 (캐싱됨)
        G, wp_to_node, node_to_wp = self.build_topology_graph()
        
        # 시작점 찾기
        if start_idx >= len(self.spawn_points):
            print(f"⚠️ Invalid start_idx {start_idx}, using 0")
            start_idx = 0
        
        start_loc = self.spawn_points[start_idx].location
        start_wp = self.carla_map.get_waypoint(start_loc)
        start_node = self.find_nearest_node(start_wp)
        
        print(f"🔍 Finding longest path from spawn point {start_idx} (node {start_node})...")
        path = self.greedy_longest_path(start_node, max_nodes=max_nodes)
        
        if not path:
            print("❌ Failed to find path")
            return None, None
        
        print(f"✅ Found path with {len(path)} topology segments")
        
        print("📍 Interpolating waypoints...")
        centerline = self.path_to_centerline(path, spacing)
        
        print(f"✅ Generated route with {len(centerline)} waypoints")
        
        return centerline, self.spawn_points[start_idx]
    
    def save_route(self, centerline, filename=None):
        """경로를 파일로 저장"""
        if filename is None:
            filename = f'{self.town_name.lower()}_centerline.pkl'
        
        filepath = Path(filename)
        with open(filepath, 'wb') as f:
            pickle.dump(centerline, f)
        
        print(f"✅ Saved {len(centerline)} waypoints to {filepath}")
        return filepath
    
    def load_route(self, filename):
        """저장된 경로 로드"""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Route file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            centerline = pickle.load(f)
        
        print(f"✅ Loaded {len(centerline)} waypoints from {filepath}")
        return centerline
    
    def visualize_route(self, centerline, lifetime=300.0, color=None):
        """디버깅용: 경로 시각화"""
        if color is None:
            color = carla.Color(0, 255, 0)  # 기본: 초록색
        
        debug = self.world.debug
        
        for i in range(len(centerline) - 1):
            p1 = carla.Location(x=centerline[i]['x'], 
                               y=centerline[i]['y'], 
                               z=centerline[i]['z'] + 0.5)
            p2 = carla.Location(x=centerline[i+1]['x'], 
                               y=centerline[i+1]['y'], 
                               z=centerline[i+1]['z'] + 0.5)
            
            debug.draw_line(p1, p2, 
                           thickness=0.1,
                           color=color,
                           life_time=lifetime)
        
        print(f"✅ Visualized {len(centerline)} waypoints")
    
    def get_route_stats(self, centerline):
        """경로 통계 출력"""
        if not centerline:
            print("❌ Empty centerline")
            return
        
        # 총 길이 계산
        total_length = 0.0
        for i in range(len(centerline) - 1):
            dx = centerline[i+1]['x'] - centerline[i]['x']
            dy = centerline[i+1]['y'] - centerline[i]['y']
            total_length += np.hypot(dx, dy)
        
        print(f"\n📊 Route Statistics:")
        print(f"  Total waypoints: {len(centerline)}")
        print(f"  Total length: {total_length:.1f} m")
        print(f"  Average spacing: {total_length / (len(centerline) - 1):.2f} m")
        print(f"  First point: x={centerline[0]['x']:.2f}, y={centerline[0]['y']:.2f}")
        print(f"  Last point:  x={centerline[-1]['x']:.2f}, y={centerline[-1]['y']:.2f}")


def main():
    """테스트용 메인 함수"""
    client = carla.Client('172.22.39.145', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    # RouteExtractor 생성
    extractor = CARLARouteExtractor(world, 'Town04')
    
    # 경로 추출
    print("\n🚀 Extracting full map centerline...")
    centerline, start_spawn = extractor.extract_route(
        start_idx=1,
        spacing=2.0,
        max_nodes=500
    )
    
    if not centerline:
        print("❌ Failed to extract centerline")
        return
    
    # 통계 출력
    extractor.get_route_stats(centerline)
    
    # 저장
    extractor.save_route(centerline, 'town04_full_centerline.pkl')
    
    # 시각화
    print("\n🎨 Visualizing path...")
    extractor.visualize_route(centerline, lifetime=300.0)
    print("✅ Done! Check CARLA window for green lines")


if __name__ == '__main__':
    main()