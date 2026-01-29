"""
Town04 Racing with MPC Control Agent
"""

import carla
import numpy as np
import time
import pickle
from mpc_control_agent import MPCControlAgent

def load_raceline(filename='town04_raceline_mincurv.pkl'):
    """
    CommonRoad raceline 로드 (velocity, curvature 포함)
    
    Returns:
        raceline: list of dict with keys [x, y, z, yaw, velocity, curvature, s]
        metadata: dict with method and source info
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    raceline = data['raceline']
    metadata = data.get('metadata', {})
    
    print(f"Loaded raceline: {len(raceline)} waypoints")
    print(f"  Method: {metadata.get('method', 'unknown')}")
    print(f"  Source: {metadata.get('source', 'unknown')}")
    
    # ==================== YAW OFFSET 계산 ====================
    # 첫 두 waypoint로 실제 진행 방향 계산
    first_wp = raceline[0]
    second_wp = raceline[1]
    
    dx = second_wp['x'] - first_wp['x']
    dy = second_wp['y'] - first_wp['y']
    actual_yaw = np.arctan2(dy, dx)
    
    # CommonRoad yaw와의 차이 = offset
    yaw_offset = actual_yaw - first_wp['yaw']
    
    # Wrap to [-pi, pi]
    yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
    
    print(f"  YAW coordinate transform:")
    print(f"    CommonRoad yaw: {np.rad2deg(first_wp['yaw']):.2f}°")
    print(f"    Actual direction: {np.rad2deg(actual_yaw):.2f}°")
    print(f"    Offset: {np.rad2deg(yaw_offset):.2f}°")
    
    # 모든 waypoint에 offset 적용
    for wp in raceline:
        wp['yaw_original'] = wp['yaw']  # 원본 백업
        wp['yaw'] = wp['yaw'] + yaw_offset  # offset 적용
        # Wrap to [-pi, pi]
        wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
    
    print(f"    ✅ Applied offset to all {len(raceline)} waypoints")
    # ====================================================
    
    # 통계 출력
    velocities = [wp['velocity'] for wp in raceline]
    curvatures = [abs(wp['curvature']) for wp in raceline]
    
    print(f"  Velocity: {min(velocities):.1f} - {max(velocities):.1f} m/s "
          f"({min(velocities)*3.6:.1f} - {max(velocities)*3.6:.1f} km/h)")
    print(f"  Max |curvature|: {max(curvatures):.4f} (1/m)")
    
    return raceline, metadata

def run_town04_racing():
    """Run racing in Town04 with learned MPC controls"""
    
    # ==================== CARLA Setup ====================
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)
    
    # ==================== Load raceline ====================
    # import pickle
    # with open('mpc/routes/town04_raceline_mincurv.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # raceline = data['raceline']
    raceline, metadata = load_raceline('mpc/routes/town04_raceline_mincurv.pkl')
    
    print(f"Loaded raceline: {len(raceline)} waypoints")
    
    # ==================== Spawn vehicle (SAFE VERSION) ====================
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    first_wp = raceline[0]  # dict: x, y, yaw

    print(f"\n📍 Spawn from raceline[0]")
    print(f"   x={first_wp['x']:.2f}, y={first_wp['y']:.2f}, yaw={np.rad2deg(first_wp['yaw']):.2f}°")

    # --- Project to road to get correct Z ---
    map_obj = world.get_map()
    test_loc = carla.Location(x=first_wp['x'], y=first_wp['y'], z=0.0)
    road_wp = map_obj.get_waypoint(test_loc, project_to_road=True)

    if road_wp is not None:
        spawn_z = road_wp.transform.location.z + 1.0
        print(f"   Road height: {road_wp.transform.location.z:.2f} → spawn_z={spawn_z:.2f}")
    else:
        spawn_z = 1.0
        print("   ⚠️ Road waypoint not found, fallback z=1.0")

    spawn_transform = carla.Transform(
        carla.Location(
            x=first_wp['x'],
            y=first_wp['y'],
            z=spawn_z
        ),
        carla.Rotation(
            yaw=np.rad2deg(first_wp['yaw'])
        )
    )

    vehicle = None
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            spawn_transform.location.z = spawn_z + 0.5 * attempt
            print(f"Spawn attempt {attempt+1}/{max_attempts} at z={spawn_transform.location.z:.2f}")

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
            if vehicle is not None:
                print("✅ Vehicle spawned successfully")
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"Spawn error: {e}")
            time.sleep(0.5)

    if vehicle is None:
        raise RuntimeError("❌ Failed to spawn vehicle after multiple attempts")

    time.sleep(0.5)
    
    # ==================== Initialize agent ====================
    checkpoint_path = "/workspace/plant2/PlanT/checkpoints/epoch=029_3.ckpt"
    agent = MPCControlAgent(checkpoint_path)
    
    # ==================== Helper functions ====================
    def get_lookahead_waypoints(ego_transform, raceline, lookahead=20):
        """Get local waypoints from raceline"""
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # Find closest point
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(raceline):
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Extract lookahead
        local_wps = []
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        for i in range(lookahead):
            idx = (closest_idx + i) % len(raceline)
            wp = raceline[idx]
            
            # Global → Local
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            
            local_wps.append([local_x, local_y])
        
        return np.array(local_wps), closest_idx
    
    # ==================== Main loop ====================
    spectator = world.get_spectator()

    dt = 0.05
    last_tick = time.time()
    
    try:
        for step in range(5000):  # 250 seconds @ 20Hz
            # ✅ Real-time synchronization
            current_time = time.time()
            elapsed = current_time - last_tick
            
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            tick_start = time.time()
            last_tick = tick_start

            world.tick()
            
            # Get vehicle state
            ego_transform = vehicle.get_transform()
            ego_velocity = vehicle.get_velocity()
            speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
            
            # Get local route
            route, route_idx = get_lookahead_waypoints(
                ego_transform, raceline, lookahead=20
            )
            
            # Predict MPC controls
            results = agent.predict(
                ego_speed=speed,
                route=route,
                bounding_boxes=None,  # No objects in racing
                speed_limit=50.0
            )
            
            # Apply control
            control = agent.control_to_carla(
                results['acceleration'],
                results['curvature'],
                speed
            )
            vehicle.apply_control(control)
            
            # Update spectator
            if step % 70 == 0:
                spectator_transform = carla.Transform(
                    ego_transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
            
            # Logging
            if step % 20 == 0:
                progress = (route_idx / len(raceline)) * 100
                print(f"Step {step:5d} | "
                      f"Speed: {speed*3.6:5.1f} km/h | "
                      f"Progress: {progress:5.1f}% | "
                      f"Acc: {results['acceleration']:5.2f} | "
                      f"Curv: {results['curvature']:6.3f} | "
                      f"Steer: {control.steer:5.2f}")

    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        vehicle.destroy()
        print("Simulation ended")


if __name__ == '__main__':
    run_town04_racing()
