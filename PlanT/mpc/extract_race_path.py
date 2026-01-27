import carla
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import minimize, differential_evolution

class CarlaRacelinePlanner:
    def __init__(self, world, track_width=10.0, routes_dir='./routes'):
        """
        Args:
            world: CARLA world object
            track_width: 트랙 폭 (양쪽 각각, meter)
            routes_dir: route 파일이 저장된 디렉토리
        """
        self.world = world
        self.map = world.get_map()
        self.track_width = track_width
        self.routes_dir = Path(routes_dir)
        
    def load_global_path(self, filename):
        """
        저장된 global path (centerline) 로드
        
        pkl 파일 형식: List[dict] with keys 'x', 'y', 'z', 'yaw'
        
        Returns:
            centerline: nx2 numpy array (x, y)
            headings: n numpy array (radians)
        """
        filepath = self.routes_dir / filename
        
        if not filepath.exists():
            print(f"❌ Route file not found: {filepath}")
            return None, None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded route from {filename}")
        print(f"   Data type: {type(data)}")
        print(f"   Number of points: {len(data)}")
        
        # List of dictionaries에서 x, y, yaw 추출
        centerline = []
        headings = []
        
        for point in data:
            centerline.append([point['x'], point['y']])
            headings.append(point['yaw'])
        
        centerline = np.array(centerline)
        headings = np.array(headings)
        
        print(f"   ✅ Centerline shape: {centerline.shape}")
        print(f"   ✅ Headings shape: {headings.shape}")
        print(f"   First point: x={centerline[0][0]:.2f}, y={centerline[0][1]:.2f}, yaw={headings[0]:.2f} rad")
        
        return centerline, headings
    
    def compute_normal_vectors(self, centerline, headings):
        """
        중심선에 수직인 normal vector 계산
        
        Args:
            centerline: nx2 array
            headings: n array (radians)
        Returns:
            normal_vectors: nx2 array (normalized)
        """
        normal_vectors = np.zeros_like(centerline)
        normal_vectors[:, 0] = -np.sin(headings)
        normal_vectors[:, 1] = np.cos(headings)
        
        return normal_vectors
    
    def compute_track_boundaries(self, centerline, normal_vectors, 
                                 track_width_left=None, track_width_right=None):
        """
        트랙 경계 계산
        """
        if track_width_left is None:
            track_width_left = self.track_width
        if track_width_right is None:
            track_width_right = self.track_width
            
        left_boundary = centerline + normal_vectors * track_width_left
        right_boundary = centerline - normal_vectors * track_width_right
        
        return left_boundary, right_boundary
    
    def plan_shortest_path_raceline(self, centerline, normal_vectors,
                                   track_width_left, track_width_right,
                                   num_iterations=3):
        """
        Step 2: Shortest path raceline planning
        
        목적: 각 점 사이의 거리 제곱의 합을 최소화
        """
        n_points = len(centerline)
        alpha = np.zeros(n_points)
        
        print(f"\n🏁 Starting SHORTEST PATH optimization...")
        print(f"   Points: {n_points}")
        print(f"   Track width: L={track_width_left}m, R={track_width_right}m")
        
        for iteration in range(num_iterations):
            import time
            start_time = time.time()
            
            print(f"\n   Iteration {iteration+1}/{num_iterations} starting...")
            
            iter_count = [0]
            def callback(xk):
                iter_count[0] += 1
                if iter_count[0] % 10 == 0:
                    print(f"      ... optimizer iteration {iter_count[0]}")
            
            def objective(alpha_var):
                """각 점 사이 거리의 제곱의 합을 최소화"""
                test_raceline = centerline + alpha_var[:, np.newaxis] * normal_vectors
                
                # 인접한 점들 사이의 거리 제곱
                diffs = np.diff(test_raceline, axis=0)
                squared_distances = np.sum(diffs**2, axis=1)
                
                return np.sum(squared_distances)
            
            bounds = [(-track_width_right, track_width_left) 
                    for _ in range(n_points)]
            
            result = minimize(
                objective, 
                alpha, 
                method='SLSQP', 
                bounds=bounds,
                callback=callback,
                options={
                    'maxiter': 100,
                    'disp': True,
                    'ftol': 1e-4
                }
            )
            alpha = result.x
            
            elapsed = time.time() - start_time
            print(f"   Iteration {iteration+1}: Cost = {result.fun:.4f} (took {elapsed:.1f}s)")
        
        raceline = centerline + alpha[:, np.newaxis] * normal_vectors
        
        # 경로 길이 계산
        centerline_length = np.sum(np.sqrt(np.sum(np.diff(centerline, axis=0)**2, axis=1)))
        raceline_length = np.sum(np.sqrt(np.sum(np.diff(raceline, axis=0)**2, axis=1)))
        
        print(f"\nSHORTEST PATH Optimization complete!")
        print(f"   Centerline length: {centerline_length:.2f}m")
        print(f"   Shortest path length: {raceline_length:.2f}m")
        print(f"   Length reduction: {centerline_length - raceline_length:.2f}m ({100*(centerline_length-raceline_length)/centerline_length:.1f}%)")
        print(f"   Max lateral offset: {np.max(np.abs(alpha)):.2f}m")
        
        return raceline, alpha
    
    def plan_minimum_curvature_raceline(self, centerline, normal_vectors, 
                                    track_width_left, track_width_right,
                                    num_iterations=3):
        """
        Step 3: Minimum curvature raceline planning
        
        목적: 곡률의 제곱의 합을 최소화 (속도 최대화)
        """
        n_points = len(centerline)
        alpha = np.zeros(n_points)
        
        print(f"\n🏁 Starting MINIMUM CURVATURE optimization...")
        print(f"   Points: {n_points}")
        print(f"   Track width: L={track_width_left}m, R={track_width_right}m")
        
        for iteration in range(num_iterations):
            import time
            start_time = time.time()
            
            print(f"\n   Iteration {iteration+1}/{num_iterations} starting...")
            
            iter_count = [0]
            def callback(xk):
                iter_count[0] += 1
                if iter_count[0] % 10 == 0:
                    print(f"      ... optimizer iteration {iter_count[0]}")
            
            def objective(alpha_var):
                """곡률의 제곱의 합을 최소화"""
                test_raceline = centerline + alpha_var[:, np.newaxis] * normal_vectors
                kappa = self._compute_curvature(test_raceline)
                return np.sum(kappa**2)
            
            bounds = [(-track_width_right, track_width_left) 
                    for _ in range(n_points)]
            
            result = minimize(
                objective, 
                alpha, 
                method='SLSQP', 
                bounds=bounds,
                callback=callback,
                options={
                    'maxiter': 100,
                    'disp': True,
                    'ftol': 1e-4
                }
            )
            alpha = result.x
            
            elapsed = time.time() - start_time
            print(f"   ✅ Iteration {iteration+1}: Cost = {result.fun:.4f} (took {elapsed:.1f}s)")
        
        raceline = centerline + alpha[:, np.newaxis] * normal_vectors
        kappa = self._compute_curvature(raceline)
        
        print(f"\n✅ MINIMUM CURVATURE Optimization complete!")
        print(f"   Max curvature: {np.max(np.abs(kappa)):.4f} (1/m)")
        print(f"   Avg curvature: {np.mean(np.abs(kappa)):.4f} (1/m)")
        print(f"   Max lateral offset: {np.max(np.abs(alpha)):.2f}m")
        
        return raceline, alpha
    
    def find_intersection_sectors(self, alpha_shortest, alpha_mincurv):
        """
        Step 4: shortest path와 minimum curvature path가 교차하는 지점을 찾아 sector 분할
        
        Returns:
            sectors: List of sector indices [0, idx1, idx2, ..., n]
        """
        n_points = len(alpha_shortest)
        sectors = [0]  # 시작점
        
        # 두 경로의 차이 부호가 바뀌는 지점 찾기
        diff = alpha_shortest - alpha_mincurv
        
        for i in range(1, n_points - 1):
            # 부호가 바뀌는 지점 = 교차점
            if diff[i-1] * diff[i] < 0:
                sectors.append(i)
        
        sectors.append(n_points - 1)  # 끝점
        
        print(f"\n📍 Found {len(sectors)-1} sectors from intersection points:")
        for i in range(len(sectors)-1):
            sector_length = sectors[i+1] - sectors[i]
            print(f"   Sector {i+1}: points {sectors[i]} to {sectors[i+1]} (length: {sector_length})")
        
        return sectors
    
    def plan_optimal_raceline(self, centerline, normal_vectors,
                            track_width_left, track_width_right,
                            alpha_shortest, alpha_mincurv,
                            v_max=30.0, a_lat_max=8.0):
        """
        Step 4: 두 경로를 섞어서 optimal raceline 생성
        
        각 sector에서 shortest와 mincurv의 가중치를 최적화
        목표: 랩타임 최소화
        
        Returns:
            raceline_optimal: nx2 array
            alpha_optimal: n array
            weights: sector별 가중치 array
        """
        sectors = self.find_intersection_sectors(alpha_shortest, alpha_mincurv)
        n_sectors = len(sectors) - 1
        
        print(f"\n🏁 Starting OPTIMAL RACELINE optimization...")
        print(f"   Number of sectors: {n_sectors}")
        print(f"   Optimizing sector weights for minimum lap time...")
        
        def compute_lap_time(weights):
            """
            가중치에 따른 랩타임 추정
            
            weight=0: shortest path (거리 짧음, 곡률 큼)
            weight=1: minimum curvature path (거리 길음, 곡률 작음)
            """
            # 각 sector에 가중치 적용
            alpha_mixed = np.zeros_like(alpha_shortest)
            
            for i in range(n_sectors):
                start_idx = sectors[i]
                end_idx = sectors[i+1]
                w = weights[i]
                
                # Linear interpolation between shortest and mincurv
                alpha_mixed[start_idx:end_idx] = \
                    (1 - w) * alpha_shortest[start_idx:end_idx] + \
                    w * alpha_mincurv[start_idx:end_idx]
            
            # 경로 생성
            raceline = centerline + alpha_mixed[:, np.newaxis] * normal_vectors
            
            # 곡률 계산
            kappa = self._compute_curvature(raceline)
            
            # 속도 프로파일 계산 (곡률 제약)
            v_kappa = np.sqrt(a_lat_max / (np.abs(kappa) + 1e-6))
            v_profile = np.minimum(v_kappa, v_max)
            
            # 각 구간별 시간 계산
            diffs = np.diff(raceline, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            
            # 평균 속도로 시간 계산
            segment_times = segment_lengths / ((v_profile[:-1] + v_profile[1:]) / 2 + 1e-6)
            
            total_time = np.sum(segment_times)
            
            return total_time
        
        # 초기 가중치: 모두 0.5 (중간)
        initial_weights = np.ones(n_sectors) * 0.5
        
        # Differential Evolution으로 전역 최적화
        print(f"   Using Differential Evolution (global optimizer)...")
        
        bounds = [(0.0, 1.0) for _ in range(n_sectors)]
        
        result = differential_evolution(
            compute_lap_time,
            bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=15,
            tol=0.01,
            disp=True,
            seed=42
        )
        
        optimal_weights = result.x
        optimal_lap_time = result.fun
        
        # 최적 경로 생성
        alpha_optimal = np.zeros_like(alpha_shortest)
        for i in range(n_sectors):
            start_idx = sectors[i]
            end_idx = sectors[i+1]
            w = optimal_weights[i]
            
            alpha_optimal[start_idx:end_idx] = \
                (1 - w) * alpha_shortest[start_idx:end_idx] + \
                w * alpha_mincurv[start_idx:end_idx]
        
        raceline_optimal = centerline + alpha_optimal[:, np.newaxis] * normal_vectors
        
        # 비교를 위한 shortest/mincurv 랩타임 계산
        shortest_time = compute_lap_time(np.zeros(n_sectors))
        mincurv_time = compute_lap_time(np.ones(n_sectors))
        
        print(f"\nOPTIMAL RACELINE complete!")
        print(f"\n Lap Time Comparison:")
        print(f"   Shortest path:      {shortest_time:.3f}s")
        print(f"   Min curvature path: {mincurv_time:.3f}s")
        print(f"   Optimal raceline:   {optimal_lap_time:.3f}s ⭐")
        print(f"\nImprovement over shortest: {shortest_time - optimal_lap_time:.3f}s ({100*(shortest_time-optimal_lap_time)/shortest_time:.1f}%)")
        print(f"   Improvement over mincurv:  {mincurv_time - optimal_lap_time:.3f}s ({100*(mincurv_time-optimal_lap_time)/mincurv_time:.1f}%)")
        
        print(f"\nSector Weights (0=shortest, 1=mincurv):")
        for i, w in enumerate(optimal_weights):
            sector_type = "SHORTEST-like" if w < 0.3 else "MINCURV-like" if w > 0.7 else "MIXED"
            print(f"   Sector {i+1}: {w:.3f} ({sector_type})")
        
        return raceline_optimal, alpha_optimal, optimal_weights, sectors
    
    def _compute_curvature(self, path):
        """
        경로의 곡률 계산 (finite difference 근사)
        """
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**(3/2) + 1e-6
        
        kappa = numerator / denominator
        
        return kappa
    
    def compute_velocity_profile(self, raceline, kappa, v_max=50.0, 
                                 a_lat_max=10.0):
        """
        velocity profile 계산
        """
        v_kappa = np.sqrt(a_lat_max / (np.abs(kappa) + 1e-6))
        v_profile = np.minimum(v_kappa, v_max)
        
        print(f"\nVelocity profile computed:")
        print(f"   Max velocity: {np.max(v_profile):.2f} m/s ({np.max(v_profile)*3.6:.1f} km/h)")
        print(f"   Min velocity: {np.min(v_profile):.2f} m/s ({np.min(v_profile)*3.6:.1f} km/h)")
        print(f"   Avg velocity: {np.mean(v_profile):.2f} m/s ({np.mean(v_profile)*3.6:.1f} km/h)")
        
        return v_profile
    
    def save_raceline(self, filename, raceline, alpha, v_profile, kappa, headings, metadata=None):
        """
        최적화된 raceline 저장
        """
        filepath = self.routes_dir / filename
        
        raceline_data = []
        for i in range(len(raceline)):
            raceline_data.append({
                'x': raceline[i, 0],
                'y': raceline[i, 1],
                'z': 0.0,
                'yaw': headings[i],
                'velocity': v_profile[i],
                'curvature': kappa[i],
                'alpha': alpha[i]
            })
        
        data = {
            'raceline': raceline_data,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Raceline saved to {filepath}")

def visualize_all_racelines(world, centerline, raceline_shortest, raceline_mincurv, 
                            raceline_optimal, left_bound, right_bound, sectors=None):
    """CARLA에서 모든 raceline 시각화"""
    print("\nVisualizing all racelines...")
    
    # Track boundaries (green)
    for i in range(len(left_bound)-1):
        start = carla.Location(x=left_bound[i][0], y=left_bound[i][1], z=0.3)
        end = carla.Location(x=left_bound[i+1][0], y=left_bound[i+1][1], z=0.3)
        world.debug.draw_line(start, end, thickness=0.05, 
                            color=carla.Color(0,255,0), life_time=120.0)
    
    for i in range(len(right_bound)-1):
        start = carla.Location(x=right_bound[i][0], y=right_bound[i][1], z=0.3)
        end = carla.Location(x=right_bound[i+1][0], y=right_bound[i+1][1], z=0.3)
        world.debug.draw_line(start, end, thickness=0.05, 
                            color=carla.Color(0,255,0), life_time=120.0)
    
    # Centerline (blue, thin)
    for i in range(len(centerline)-1):
        start = carla.Location(x=centerline[i][0], y=centerline[i][1], z=0.4)
        end = carla.Location(x=centerline[i+1][0], y=centerline[i+1][1], z=0.4)
        world.debug.draw_line(start, end, thickness=0.03, 
                            color=carla.Color(100,100,255), life_time=120.0)
    
    # Shortest path (cyan)
    for i in range(len(raceline_shortest)-1):
        start = carla.Location(x=raceline_shortest[i][0], y=raceline_shortest[i][1], z=0.5)
        end = carla.Location(x=raceline_shortest[i+1][0], y=raceline_shortest[i+1][1], z=0.5)
        world.debug.draw_line(start, end, thickness=0.08,
                            color=carla.Color(0,255,255), life_time=120.0)
    
    # Minimum curvature (yellow)
    for i in range(len(raceline_mincurv)-1):
        start = carla.Location(x=raceline_mincurv[i][0], y=raceline_mincurv[i][1], z=0.6)
        end = carla.Location(x=raceline_mincurv[i+1][0], y=raceline_mincurv[i+1][1], z=0.6)
        world.debug.draw_line(start, end, thickness=0.08,
                            color=carla.Color(255,255,0), life_time=120.0)
    
    # Optimal raceline (RED, thick)
    for i in range(len(raceline_optimal)-1):
        start = carla.Location(x=raceline_optimal[i][0], y=raceline_optimal[i][1], z=0.7)
        end = carla.Location(x=raceline_optimal[i+1][0], y=raceline_optimal[i+1][1], z=0.7)
        world.debug.draw_line(start, end, thickness=0.15,
                            color=carla.Color(255,0,0), life_time=120.0)
    
    # Sector split points (magenta spheres)
    if sectors is not None:
        for sector_idx in sectors:
            loc = carla.Location(x=raceline_optimal[sector_idx][0], 
                               y=raceline_optimal[sector_idx][1], z=1.0)
            world.debug.draw_point(loc, size=0.2, 
                                 color=carla.Color(255,0,255), life_time=120.0)
    
    print("Visualization complete!")
    print("   Green = Track boundaries")
    print("   Blue (thin) = Centerline (original)")
    print("   Cyan = Shortest path")
    print("   Yellow = Minimum curvature path")
    print("   RED (thick) = OPTIMAL raceline ⭐")
    if sectors:
        print("    Magenta points = Sector splits")

def main():
    print("=" * 80)
    print("CARLA Raceline Planner - Complete Implementation")
    print("Step 2: Shortest Path | Step 3: Minimum Curvature | Step 4: Optimal Blend")
    print("=" * 80)
    
    # CARLA 초기화
    client = carla.Client('172.22.39.179', 2000)
    world = client.get_world()
    
    # Raceline Planner 생성
    planner = CarlaRacelinePlanner(world, track_width=5.0, routes_dir='./routes')
    
    # 1. Global path (centerline) 로드
    centerline, headings = planner.load_global_path('town04_max30_start1.pkl')
    
    if centerline is None:
        print("❌ Failed to load route!")
        return

    # 2. Normal vectors 계산
    normal_vectors = planner.compute_normal_vectors(centerline, headings)
    
    # 3. Track boundaries 계산
    track_width_left = 3.5
    track_width_right = 7.0
    
    left_bound, right_bound = planner.compute_track_boundaries(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right
    )
    
    print("\n" + "=" * 80)
    print("STEP 2: Computing SHORTEST PATH")
    print("=" * 80)
    
    # 4. Shortest Path 계획
    raceline_shortest, alpha_shortest = planner.plan_shortest_path_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        num_iterations=3
    )
    
    print("\n" + "=" * 80)
    print("STEP 3: Computing MINIMUM CURVATURE PATH")
    print("=" * 80)
    
    # 5. Minimum Curvature 계획
    raceline_mincurv, alpha_mincurv = planner.plan_minimum_curvature_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        num_iterations=3
    )
    
    print("\n" + "=" * 80)
    print("STEP 4: Computing OPTIMAL RACELINE (Blending)")
    print("=" * 80)
    
    # 6. Optimal Raceline (두 경로 혼합)
    raceline_optimal, alpha_optimal, weights, sectors = planner.plan_optimal_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        alpha_shortest=alpha_shortest,
        alpha_mincurv=alpha_mincurv,
        v_max=30.0,
        a_lat_max=8.0
    )
    
    # 7. Velocity Profile 계산 (optimal raceline 기준)
    kappa_optimal = planner._compute_curvature(raceline_optimal)
    v_profile = planner.compute_velocity_profile(
        raceline_optimal, kappa_optimal,
        v_max=30.0,
        a_lat_max=8.0
    )
    
    # 8. 시각화
    visualize_all_racelines(
        world, 
        centerline, 
        raceline_shortest,
        raceline_mincurv,
        raceline_optimal,
        left_bound, 
        right_bound,
        sectors
    )
    
    # 9. 결과 저장 (3가지 모두)
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    # Shortest path 저장
    kappa_shortest = planner._compute_curvature(raceline_shortest)
    v_shortest = planner.compute_velocity_profile(raceline_shortest, kappa_shortest, v_max=30.0, a_lat_max=8.0)
    planner.save_raceline(
        'town04_raceline_shortest.pkl',
        raceline=raceline_shortest,
        alpha=alpha_shortest,
        v_profile=v_shortest,
        kappa=kappa_shortest,
        headings=headings,
        metadata={'method': 'shortest_path', 'track_width_left': track_width_left, 'track_width_right': track_width_right}
    )
    
    # Minimum curvature 저장
    kappa_mincurv = planner._compute_curvature(raceline_mincurv)
    v_mincurv = planner.compute_velocity_profile(raceline_mincurv, kappa_mincurv, v_max=30.0, a_lat_max=8.0)
    planner.save_raceline(
        'town04_raceline_mincurv.pkl',
        raceline=raceline_mincurv,
        alpha=alpha_mincurv,
        v_profile=v_mincurv,
        kappa=kappa_mincurv,
        headings=headings,
        metadata={'method': 'minimum_curvature', 'track_width_left': track_width_left, 'track_width_right': track_width_right}
    )
    
    # Optimal 저장
    planner.save_raceline(
        'town04_raceline_optimal.pkl',
        raceline=raceline_optimal,
        alpha=alpha_optimal,
        v_profile=v_profile,
        kappa=kappa_optimal,
        headings=headings,
        metadata={
            'method': 'optimal_blend',
            'track_width_left': track_width_left,
            'track_width_right': track_width_right,
            'v_max': 30.0,
            'a_lat_max': 8.0,
            'sector_weights': weights.tolist(),
            'n_sectors': len(sectors) - 1
        }
    )
    
    print("\n" + "=" * 80)
    print("✅ Raceline planning complete!")
    print("=" * 80)
    print("\n📁 Generated files:")
    print("   - town04_raceline_shortest.pkl (shortest path)")
    print("   - town04_raceline_mincurv.pkl (minimum curvature)")
    print("   - town04_raceline_optimal.pkl (optimal blend) ⭐")

if __name__ == '__main__':
    main()