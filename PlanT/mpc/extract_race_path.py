import carla
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d

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
        
        centerline = []
        headings = []
        
        for point in data:
            centerline.append([point['x'], point['y']])
            headings.append(point['yaw'])
        
        centerline = np.array(centerline)
        headings = np.array(headings)
        
        print(f"   ✅ Centerline shape: {centerline.shape}")
        print(f"   First point: x={centerline[0][0]:.2f}, y={centerline[0][1]:.2f}")
        
        return centerline, headings
    
    def compute_normal_vectors(self, centerline, headings):
        """
        중심선에 수직인 normal vector 계산
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
    
    def _subsample_path(self, centerline, normal_vectors, headings, n_opt_points=200):
        """
        최적화를 위해 경로를 subsampling
        
        Returns:
            centerline_sub, normal_sub, headings_sub, indices
        """
        n_points = len(centerline)
        indices = np.linspace(0, n_points-1, n_opt_points, dtype=int)
        
        return (centerline[indices], 
                normal_vectors[indices], 
                None,
                indices)
    
    def _interpolate_alpha(self, alpha_sub, indices, n_full):
        """
        Subsampled alpha를 full resolution으로 interpolation
        """
        f = interp1d(indices, alpha_sub, kind='cubic', fill_value='extrapolate')
        full_indices = np.arange(n_full)
        alpha_full = f(full_indices)
        
        return alpha_full
    
    def plan_shortest_path_raceline(self, centerline, normal_vectors,
                                   track_width_left, track_width_right,
                                   n_opt_points=200):
        """
        Step 2: Shortest path raceline planning
        
        목적: 전체 경로 길이를 최소화
        """
        n_points = len(centerline)
        
        print(f"\n🏁 Starting SHORTEST PATH optimization...")
        print(f"   Full points: {n_points}")
        print(f"   Optimization points: {n_opt_points}")
        print(f"   Track width: L={track_width_left}m, R={track_width_right}m")
        
        # Subsample for optimization
        centerline_sub, normal_sub, _, indices = self._subsample_path(
            centerline, normal_vectors, None, n_opt_points
        )
        
        # 코너 감지 (subsampled)
        kappa_center_sub = self._compute_curvature(centerline_sub)
        corner_threshold = np.percentile(np.abs(kappa_center_sub), 70)
        is_corner = np.abs(kappa_center_sub) > corner_threshold
        
        # 초기값: 코너에서 inside
        alpha_init = np.zeros(n_opt_points)
        alpha_init[is_corner] = np.sign(kappa_center_sub[is_corner]) * track_width_left * 0.8
        
        print(f"   Initial guess: {np.sum(is_corner)} corner points")
        print(f"   Centerline length: {self._compute_path_length(centerline):.2f}m")
        
        # 초기 경로 길이
        raceline_init = centerline_sub + alpha_init[:, np.newaxis] * normal_sub
        init_length = self._compute_path_length(raceline_init)
        print(f"   Initial raceline length: {init_length:.2f}m")
        
        def objective(alpha_var):
            """전체 경로 길이를 최소화"""
            raceline = centerline_sub + alpha_var[:, np.newaxis] * normal_sub
            return self._compute_path_length(raceline)
        
        bounds = [(-track_width_right, track_width_left) for _ in range(n_opt_points)]
        
        print(f"   Optimizing with SLSQP...")
        
        result = minimize(
            objective,
            alpha_init,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': 200,
                'disp': True,
                'ftol': 1e-6
            }
        )
        
        alpha_sub = result.x
        
        # Interpolate to full resolution
        print(f"   Interpolating to full resolution...")
        alpha = self._interpolate_alpha(alpha_sub, indices, n_points)
        
        # Smooth alpha to avoid oscillations
        from scipy.ndimage import gaussian_filter1d
        alpha = gaussian_filter1d(alpha, sigma=3.0, mode='nearest')
        
        # Clip to bounds
        alpha = np.clip(alpha, -track_width_right, track_width_left)
        
        raceline = centerline + alpha[:, np.newaxis] * normal_vectors
        
        centerline_length = self._compute_path_length(centerline)
        raceline_length = self._compute_path_length(raceline)
        
        print(f"\n✅ SHORTEST PATH Optimization complete!")
        print(f"   Centerline length: {centerline_length:.2f}m")
        print(f"   Shortest path length: {raceline_length:.2f}m")
        print(f"   Length reduction: {centerline_length - raceline_length:.2f}m ({100*(centerline_length-raceline_length)/centerline_length:.1f}%)")
        print(f"   Max lateral offset: {np.max(np.abs(alpha)):.2f}m")
        print(f"   Avg lateral offset: {np.mean(np.abs(alpha)):.2f}m")
        
        return raceline, alpha
    
    # def plan_minimum_curvature_raceline(self, centerline, normal_vectors, 
    #                                 track_width_left, track_width_right,
    #                                 n_opt_points=200):
    #     """
    #     Step 3: Minimum curvature raceline planning
        
    #     목적: 곡률의 제곱의 합을 최소화
    #     """
    #     n_points = len(centerline)
        
    #     print(f"\nStarting MINIMUM CURVATURE optimization...")
    #     print(f"   Full points: {n_points}")
    #     print(f"   Optimization points: {n_opt_points}")
    #     print(f"   Track width: L={track_width_left}m, R={track_width_right}m")
        
    #     # Subsample
    #     centerline_sub, normal_sub, _, indices = self._subsample_path(
    #         centerline, normal_vectors, None, n_opt_points
    #     )
        
    #     # 코너 감지
    #     kappa_center_sub = self._compute_curvature(centerline_sub)
    #     kappa_center = self._compute_curvature(centerline)
        
    #     corner_threshold = np.percentile(np.abs(kappa_center_sub), 70)
    #     is_corner = np.abs(kappa_center_sub) > corner_threshold
        
    #     # 초기값: 코너에서 outside (곡률 최소화)
    #     alpha_init = np.zeros(n_opt_points)
    #     alpha_init[is_corner] = -np.sign(kappa_center_sub[is_corner]) * track_width_right * 0.8
        
    #     print(f"   Initial guess: {np.sum(is_corner)} corner points")
    #     print(f"   Centerline max curvature: {np.max(np.abs(kappa_center)):.4f} (1/m)")
        
    #     # 초기 곡률
    #     raceline_init = centerline_sub + alpha_init[:, np.newaxis] * normal_sub
    #     kappa_init = self._compute_curvature(raceline_init)
    #     print(f"   Initial raceline max curvature: {np.max(np.abs(kappa_init)):.4f} (1/m)")
        
    #     def objective(alpha_var):
    #         """곡률의 제곱의 합을 최소화"""
    #         raceline = centerline_sub + alpha_var[:, np.newaxis] * normal_sub
    #         kappa = self._compute_curvature(raceline)
    #         return np.sum(kappa**2)
        
    #     bounds = [(-track_width_right, track_width_left) for _ in range(n_opt_points)]
        
    #     print(f"   Optimizing with SLSQP...")
        
    #     result = minimize(
    #         objective,
    #         alpha_init,
    #         method='SLSQP',
    #         bounds=bounds,
    #         options={
    #             'maxiter': 200,
    #             'disp': True,
    #             'ftol': 1e-6
    #         }
    #     )
        
    #     alpha_sub = result.x
        
    #     # Interpolate to full resolution
    #     print(f"   Interpolating to full resolution...")
    #     alpha = self._interpolate_alpha(alpha_sub, indices, n_points)
        
    #     # Smooth
    #     from scipy.ndimage import gaussian_filter1d
    #     alpha = gaussian_filter1d(alpha, sigma=3.0, mode='nearest')
        
    #     # Clip
    #     alpha = np.clip(alpha, -track_width_right, track_width_left)
        
    #     raceline = centerline + alpha[:, np.newaxis] * normal_vectors
    #     kappa = self._compute_curvature(raceline)
        
    #     print(f"\n✅ MINIMUM CURVATURE Optimization complete!")
    #     print(f"   Centerline max curvature: {np.max(np.abs(kappa_center)):.4f} (1/m)")
    #     print(f"   Raceline max curvature: {np.max(np.abs(kappa)):.4f} (1/m)")
    #     print(f"   Curvature reduction: {100*(np.max(np.abs(kappa_center))-np.max(np.abs(kappa)))/np.max(np.abs(kappa_center)):.1f}%")
    #     print(f"   Max lateral offset: {np.max(np.abs(alpha)):.2f}m")
    #     print(f"   Avg lateral offset: {np.mean(np.abs(alpha)):.2f}m")
        
    #     return raceline, alpha
    
    def plan_minimum_curvature_raceline(
        self,
        centerline,
        normal_vectors,
        track_width_left,
        track_width_right,
        n_opt_points=200):
        """
        Step 3: Minimum curvature raceline planning (FIXED VERSION)

        핵심 아이디어:
        - 곡률 직접 계산 
        - 2차 차분(second difference) 최소화 
        """

        n_points = len(centerline)

        print(f"\n🏁 Starting MINIMUM CURVATURE optimization (FIXED)...")
        print(f"   Full points: {n_points}")
        print(f"   Optimization points: {n_opt_points}")
        print(f"   Track width: L={track_width_left}m, R={track_width_right}m")

        # --- Subsample ---
        indices = np.linspace(0, n_points - 1, n_opt_points, dtype=int)
        center_sub = centerline[indices]
        normal_sub = normal_vectors[indices]

        # --- 초기값: centerline ---
        alpha_init = np.zeros(n_opt_points)

        def objective(alpha):
            """
            곡률 최소화 ≈ 2차 차분 최소화
            """
            path = center_sub + alpha[:, None] * normal_sub

            # second difference
            d2 = path[2:] - 2 * path[1:-1] + path[:-2]

            smooth_cost = np.sum(np.linalg.norm(d2, axis=1) ** 2)
            offset_cost = 0.01 * np.sum(alpha**2)   # ⭐ 중요
            return smooth_cost + offset_cost

        bounds = [(-track_width_right, track_width_left)] * n_opt_points

        print("   Optimizing with SLSQP (second-difference objective)...")

        result = minimize(
            objective,
            alpha_init,
            method="SLSQP",
            bounds=bounds,
            options=dict(maxiter=200, ftol=1e-6, disp=True),
        )

        alpha_sub = result.x

        # --- Interpolate to full resolution ---
        f = interp1d(indices, alpha_sub, kind="cubic", fill_value="extrapolate")
        alpha = f(np.arange(n_points))

        # --- Smooth (NO wrap!) ---
        from scipy.ndimage import gaussian_filter1d
        alpha = gaussian_filter1d(alpha, sigma=3.0, mode="nearest")

        # --- Clip to track ---
        alpha = np.clip(alpha, -track_width_right, track_width_left)

        raceline = centerline + alpha[:, None] * normal_vectors
        kappa = self._compute_curvature(raceline)
        kappa_center = self._compute_curvature(centerline)

        print(f"\n✅ MINIMUM CURVATURE Optimization complete!")
        print(f"   Centerline max curvature: {np.max(np.abs(kappa_center)):.4f}")
        print(f"   Raceline max curvature:   {np.max(np.abs(kappa)):.4f}")
        print(
            f"   Curvature reduction: "
            f"{100 * (np.max(np.abs(kappa_center)) - np.max(np.abs(kappa))) / np.max(np.abs(kappa_center)):.1f}%"
        )

        return raceline, alpha

    def find_intersection_sectors(self, alpha_shortest, alpha_mincurv):
        """
        Step 4: shortest path와 minimum curvature path가 교차하는 지점을 찾아 sector 분할
        """
        n_points = len(alpha_shortest)
        sectors = [0]
        
        diff = alpha_shortest - alpha_mincurv
        
        for i in range(1, n_points - 1):
            if diff[i-1] * diff[i] < 0:
                sectors.append(i)
        
        sectors.append(n_points - 1)
        
        # 최소 섹터 크기
        min_sector_size = max(10, n_points // 20)
        
        filtered_sectors = [sectors[0]]
        for i in range(1, len(sectors)):
            if sectors[i] - filtered_sectors[-1] >= min_sector_size:
                filtered_sectors.append(sectors[i])
        
        if filtered_sectors[-1] != n_points - 1:
            filtered_sectors.append(n_points - 1)
        
        print(f"\n📍 Found {len(filtered_sectors)-1} sectors from intersection points:")
        for i in range(len(filtered_sectors)-1):
            sector_length = filtered_sectors[i+1] - filtered_sectors[i]
            print(f"   Sector {i+1}: points {filtered_sectors[i]} to {filtered_sectors[i+1]} (length: {sector_length})")
        
        return filtered_sectors
    
    def plan_optimal_raceline(self, centerline, normal_vectors,
                            track_width_left, track_width_right,
                            alpha_shortest, alpha_mincurv,
                            v_max=30.0, a_lat_max=8.0):
        """
        Step 4: 두 경로를 섞어서 optimal raceline 생성
        """
        sectors = self.find_intersection_sectors(alpha_shortest, alpha_mincurv)
        n_sectors = len(sectors) - 1
        
        print(f"\n🏁 Starting OPTIMAL RACELINE optimization...")
        print(f"   Number of sectors: {n_sectors}")
        
        def compute_lap_time(weights):
            """가중치에 따른 랩타임 추정"""
            alpha_mixed = np.zeros_like(alpha_shortest)
            
            for i in range(n_sectors):
                start_idx = sectors[i]
                end_idx = sectors[i+1]
                w = weights[i]
                
                alpha_mixed[start_idx:end_idx] = \
                    (1 - w) * alpha_shortest[start_idx:end_idx] + \
                    w * alpha_mincurv[start_idx:end_idx]
            
            raceline = centerline + alpha_mixed[:, np.newaxis] * normal_vectors
            kappa = self._compute_curvature(raceline)
            
            # 속도 계산
            v_kappa = np.sqrt(a_lat_max / (np.abs(kappa) + 1e-6))
            v_profile = np.minimum(v_kappa, v_max)
            
            # 시간 계산
            diffs = np.diff(raceline, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            segment_times = segment_lengths / ((v_profile[:-1] + v_profile[1:]) / 2 + 1e-6)
            
            return np.sum(segment_times)
        
        print(f"   Using Differential Evolution for sector weights...")
        
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
        
        # 비교
        shortest_time = compute_lap_time(np.zeros(n_sectors))
        mincurv_time = compute_lap_time(np.ones(n_sectors))
        
        print(f"\n✅ OPTIMAL RACELINE complete!")
        print(f"\n   📊 Lap Time Comparison:")
        print(f"   Shortest path:      {shortest_time:.3f}s")
        print(f"   Min curvature path: {mincurv_time:.3f}s")
        print(f"   Optimal raceline:   {optimal_lap_time:.3f}s ⭐")
        
        if optimal_lap_time < shortest_time:
            print(f"   Improvement over shortest: {shortest_time - optimal_lap_time:.3f}s ({100*(shortest_time-optimal_lap_time)/shortest_time:.1f}%)")
        if optimal_lap_time < mincurv_time:
            print(f"   Improvement over mincurv:  {mincurv_time - optimal_lap_time:.3f}s ({100*(mincurv_time-optimal_lap_time)/mincurv_time:.1f}%)")
        
        print(f"\n   🎯 Sector Weights (0=shortest, 1=mincurv):")
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
    
    def _compute_path_length(self, path):
        """
        경로의 전체 길이 계산
        """
        diffs = np.diff(path, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(lengths)
    
    def compute_velocity_profile(self, raceline, kappa, v_max=50.0, a_lat_max=10.0):
        """
        velocity profile 계산
        """
        v_kappa = np.sqrt(a_lat_max / (np.abs(kappa) + 1e-6))
        v_profile = np.minimum(v_kappa, v_max)
        
        print(f"\n🚗 Velocity profile computed:")
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
    print("\n Visualizing all racelines...")
    
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
    
    # Centerline (blue)
    for i in range(len(centerline)-1):
        start = carla.Location(x=centerline[i][0], y=centerline[i][1], z=0.4)
        end = carla.Location(x=centerline[i+1][0], y=centerline[i+1][1], z=0.4)
        world.debug.draw_line(start, end, thickness=0.03, 
                            color=carla.Color(100,100,255), life_time=120.0)
    
    # Shortest (cyan)
    for i in range(len(raceline_shortest)-1):
        start = carla.Location(x=raceline_shortest[i][0], y=raceline_shortest[i][1], z=0.5)
        end = carla.Location(x=raceline_shortest[i+1][0], y=raceline_shortest[i+1][1], z=0.5)
        world.debug.draw_line(start, end, thickness=0.08,
                            color=carla.Color(0,255,255), life_time=120.0)
    
    # Mincurv (yellow)
    for i in range(len(raceline_mincurv)-1):
        start = carla.Location(x=raceline_mincurv[i][0], y=raceline_mincurv[i][1], z=0.6)
        end = carla.Location(x=raceline_mincurv[i+1][0], y=raceline_mincurv[i+1][1], z=0.6)
        world.debug.draw_line(start, end, thickness=0.08,
                            color=carla.Color(255,255,0), life_time=120.0)
    
    # Optimal (RED)
    for i in range(len(raceline_optimal)-1):
        start = carla.Location(x=raceline_optimal[i][0], y=raceline_optimal[i][1], z=0.7)
        end = carla.Location(x=raceline_optimal[i+1][0], y=raceline_optimal[i+1][1], z=0.7)
        world.debug.draw_line(start, end, thickness=0.15,
                            color=carla.Color(255,0,0), life_time=120.0)
    
    # Sectors
    if sectors is not None:
        for sector_idx in sectors:
            loc = carla.Location(x=raceline_optimal[sector_idx][0], 
                               y=raceline_optimal[sector_idx][1], z=1.0)
            world.debug.draw_point(loc, size=0.2, 
                                 color=carla.Color(255,0,255), life_time=120.0)
    
    print(" Visualization complete!")
    print("   🟢 Green = Track boundaries")
    print("   🔵 Blue = Centerline")
    print("   🔷 Cyan = Shortest")
    print("   🟡 Yellow = Minimum curvature")
    print("   🔴 RED = OPTIMAL")

def main():
    print("=" * 80)
    print("CARLA Raceline Planner - Optimized Version")
    print("=" * 80)
    
    client = carla.Client('localhost', 2000)
    world = client.load_world('Town04')
    
    planner = CarlaRacelinePlanner(world, track_width=5.0, routes_dir='./routes')
    
    centerline, headings = planner.load_global_path('town04_max30_start13.pkl')
    
    if centerline is None:
        return

    normal_vectors = planner.compute_normal_vectors(centerline, headings)
    
    track_width_left = 3.5
    track_width_right = 7.0
    
    left_bound, right_bound = planner.compute_track_boundaries(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right
    )
    
    print("\n" + "=" * 80)
    print("STEP 2: SHORTEST PATH")
    print("=" * 80)
    
    raceline_shortest, alpha_shortest = planner.plan_shortest_path_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        n_opt_points=200  # Optimization points
    )
    
    print("\n" + "=" * 80)
    print("STEP 3: MINIMUM CURVATURE")
    print("=" * 80)
    
    raceline_mincurv, alpha_mincurv = planner.plan_minimum_curvature_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        n_opt_points=200
    )
    
    print("\n" + "=" * 80)
    print("STEP 4: OPTIMAL BLEND")
    print("=" * 80)
    
    raceline_optimal, alpha_optimal, weights, sectors = planner.plan_optimal_raceline(
        centerline, normal_vectors,
        track_width_left=track_width_left,
        track_width_right=track_width_right,
        alpha_shortest=alpha_shortest,
        alpha_mincurv=alpha_mincurv,
        v_max=30.0,
        a_lat_max=8.0
    )
    
    kappa_optimal = planner._compute_curvature(raceline_optimal)
    v_profile = planner.compute_velocity_profile(raceline_optimal, kappa_optimal, v_max=30.0, a_lat_max=8.0)
    
    visualize_all_racelines(
        world, centerline, raceline_shortest, raceline_mincurv,
        raceline_optimal, left_bound, right_bound, sectors
    )
    
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    # Save all
    kappa_shortest = planner._compute_curvature(raceline_shortest)
    v_shortest = planner.compute_velocity_profile(raceline_shortest, kappa_shortest, v_max=30.0, a_lat_max=8.0)
    planner.save_raceline('town04_raceline_shortest13.pkl', raceline_shortest, alpha_shortest,
                         v_shortest, kappa_shortest, headings, {'method': 'shortest_path'})
    
    kappa_mincurv = planner._compute_curvature(raceline_mincurv)
    v_mincurv = planner.compute_velocity_profile(raceline_mincurv, kappa_mincurv, v_max=30.0, a_lat_max=8.0)
    planner.save_raceline('town04_raceline_mincurv13.pkl', raceline_mincurv, alpha_mincurv,
                         v_mincurv, kappa_mincurv, headings, {'method': 'minimum_curvature'})
    
    planner.save_raceline('town04_raceline_optimal13.pkl', raceline_optimal, alpha_optimal,
                         v_profile, kappa_optimal, headings,
                         {'method': 'optimal_blend', 'sector_weights': weights.tolist()})
    
    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()