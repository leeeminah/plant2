"""
Town04 Racing with MPC Control Agent
"""

import carla
import numpy as np
import time
import pickle
from mpc_control_agent import MPCControlAgent

def run_town04_racing():
    """Run racing in Town04 with learned MPC controls"""
    
    # ==================== CARLA Setup ====================
    # client = carla.Client('localhost', 2000)
    client = carla.Client('172.22.39.175', 2000)
    # client = carla.Client('172.22.39.179', 2000)
    # client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)
    
    # ==================== Initialize agent (with raceline) ====================
    # checkpoint_path = "/workspace/plant2/PlanT/checkpoints_0209_30scenario/epoch=029_2.ckpt"\
    checkpoint_path = "/workspace/plant2/PlanT/checkpoints_0208_2(controlweight=1andtargetspeed=current)/epoch=029_1.ckpt"
    # checkpoint_path = "/workspace/plant2/PlanT/checkpoints/epoch=029_1.ckpt"
    raceline_file = 'mpc/routes/town04_raceline_mincurv13_1.pkl'
    
    agent = MPCControlAgent(checkpoint_path, raceline_file)
    
    # ==================== Spawn vehicle ====================
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Use first waypoint from raceline
    first_wp = agent.raceline[0]

    print(f"\n📍 Spawn from raceline[0]")
    print(f"   x={first_wp['x']:.2f}, y={first_wp['y']:.2f}, yaw={np.rad2deg(first_wp['yaw']):.2f}°")

    # Project to road
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
        carla.Location(x=first_wp['x'], y=first_wp['y'], z=spawn_z),
        carla.Rotation(yaw=np.rad2deg(first_wp['yaw']))
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
     # ✅ 디버깅 플래그
    DEBUG_VERBOSE = True
    
    # ==================== Main loop ====================
    spectator = world.get_spectator()

    dt = 0.05
    last_tick = time.time()

    # ✅ 디버깅 카운터
    override_count = 0
    total_steps = 0
    progress = 0.0  # ← 추가!
    inference_times = []
    try:
        for step in range(5000):  # 250 seconds @ 20Hz
            # Real-time synchronization
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
            
            # ✅ Get lookahead waypoints (from agent's raceline)
            lookahead_wps, route_dist, route_idx = agent.get_lookahead_waypoints(
                ego_transform, 
                lookahead=30
            )
            
            # ✅ Extract local route for model
            route = np.array([[wp['x'], wp['y']] for wp in lookahead_wps])
            curvature_ff = lookahead_wps[0]['curvature']  # ← Not calculated, from raceline!
            
            
            # ✅ Extract feedforward curvature (from raceline!)
            raceline_speed = lookahead_wps[0]['velocity'] 
            target_speed = min(raceline_speed, 35)
            # raceline_speed = lookahead_wps[0]['velocity']
            # current_kappa = abs(curvature_ff)

            # if current_kappa > 0.012:  # 매우 급한 커브만
            #     target_speed = min(raceline_speed, 30.0)  # 90 km/h
            # else:
            #     target_speed = min(raceline_speed, 40.0)  # 기존대로

            # # ① 물리 기반 cap
            # speed_cap_physics = np.sqrt(12.0 / max(abs(curvature_ff), 1e-4))

            # # ② 안전 마진
            # target_speed = min(
            #     raceline_speed * 0.9,     # 레이싱 마진
            #     speed_cap_physics,        # 횡가속 한계
            #     30.0               # 맵 제한
            # )

            # target_speed = min(target_speed, 25.0 / 3.6)
            # print(target_speed)
            # Predict MPC controls (model outputs κ_fb only)
            
            # ✅ 추론 시작
            inference_start = time.perf_counter()

            results = agent.predict(
                ego_speed=speed,
                route=route,
                target_speed=target_speed,
                bounding_boxes=None,
                speed_limit=30.0
            )

            inference_time = (time.perf_counter() - inference_start) * 1000  # ms
            inference_times.append(inference_time)

            # acc = results['acceleration']
            
            # # if speed > target_speed:
            # #     acc = min(acc, 0.0)
            # acc = results['acceleration']
            # v = speed
            # v_ref = target_speed  # raceline speed (m/s)

            # #  핵심: 속도 초과 시 가속 금지
            # if v > v_ref:
            #     acc = min(acc, 0.0)
            # # speed_error = target_speed - ego_speed
            # # if speed_error < 0:
            # #     acc = speed_error * k   # k ≈ 0.3~0.7   
            # === SPEED GOVERNOR DEBUG ===
            # print(
            #     f"[SPEED CTRL] current={speed*3.6:6.1f} km/h | "
            #     f"target={target_speed*3.6:6.1f} km/h | "
            #     f"a_before={results['acceleration']:+6.2f}",
            #     end=" "
            # )

            # ✅ Apply control (with feedforward composition)
            control = agent.control_to_carla(
                acceleration=results['acceleration'],
                curvature_fb=results['curvature_fb'],
                curvature_ff=curvature_ff,
                current_speed=speed,
                target_speed=target_speed   # ← raceline velocity
            )

            vehicle.apply_control(control)

            # Update spectator
            if step % 10 == 0:
                spectator_transform = carla.Transform(
                    ego_transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
            
            # Logging
            # ✅ Logging (개선됨)
            if step % 20 == 0:
                avg_inference = np.mean(inference_times[-100:]) 
                progress = (route_idx / len(agent.raceline)) * 100
                kappa_total = results['curvature_fb'] + curvature_ff
                
                # 기본 로그
                print(f"Step {step:5d} | "
                      f"Speed: {speed*3.6:5.1f} km/h | "
                      f"Prog: {progress:4.1f}% | "
                      f"Infer: {inference_time:5.2f}ms (avg: {avg_inference:5.2f}ms) | "
                      f"Acc: {results['acceleration']:+5.2f} | "
                      f"Acc: {results['acceleration']:+5.2f} | "
                      f"κ_fb: {results['curvature_fb']:+6.4f} | "
                      f"κ_ff: {curvature_ff:+6.4f} | "
                      f"Steer: {control.steer:+5.2f}", end="")
                
            
            # ✅ 주기적 요약 (200 steps마다)
            if step > 0 and step % 200 == 0:
                agent.print_debug_summary()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        if inference_times:
            print(f"\n{'='*80}")
            print(f"INFERENCE TIME STATISTICS")
            print(f"{'='*80}")
            print(f"Mean:   {np.mean(inference_times):6.2f} ms")
            print(f"Median: {np.median(inference_times):6.2f} ms")
            print(f"Std:    {np.std(inference_times):6.2f} ms")
            print(f"Min:    {np.min(inference_times):6.2f} ms")
            print(f"Max:    {np.max(inference_times):6.2f} ms")
            print(f"P95:    {np.percentile(inference_times, 95):6.2f} ms")
            print(f"P99:    {np.percentile(inference_times, 99):6.2f} ms")
            
        # 상세 통계
        agent.print_debug_summary()

        # Cleanup
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        vehicle.destroy()
        print("Simulation ended")


if __name__ == '__main__':
    run_town04_racing()

