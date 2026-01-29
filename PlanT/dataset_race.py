import os
import logging
import json
import numpy as np
from pathlib import Path
from beartype import beartype

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

import gzip
import glob

from plant_variables import PlanTVariables
from util.static_extents import CAR_EXTENTS, STATIC_EXTENTS

from scipy.spatial import cKDTree

def normalize_angle_degree(x):
  x = x % 360.0
  if isinstance(x, np.ndarray):
      x[x > 180] -= 360
  elif x > 180.0:
    x -= 360.0
  return x

def rad2deg(theta):
    return normalize_angle_degree(np.rad2deg(theta).item())

class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None) -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training

        self.plant_vars = PlanTVariables

        self.data_cache = shared_dict

        self.MAX_DISTANCE = self.cfg_train.range
        self.MAX_DISTANCE_DOUBLE = 2*self.MAX_DISTANCE

        self.bev_colors = torch.tensor(self.plant_vars.bev_colors)

        root = root.rstrip("/")

        self.aug_rate = 0.5
        if self.cfg_train.augment:
            self.transform = self.aug_sample
        else:
            self.transform = None

        if self.cfg_train.augment_parked: #미리 저장된 정차 차량 위치 데이터 
            self.parked_locations = {}
            self.parked_rotations = {}
            self.parked_extents = {}
            self.parked_trees = {}
            parked_cars = np.load("/workspace/plant2/PlanT/car_data.npy", allow_pickle=True).item()
            for town, data in parked_cars.items():
                self.parked_locations[town] = data["locations"]
                self.parked_rotations[town] = data["rotations"]
                self.parked_extents[town] = data["extents"]
                self.parked_trees[town] = cKDTree(self.parked_locations[town])

        self.speed_cats = self.plant_vars.speed_cats

        self.car_types = self.plant_vars.car_types
        self.type_nums = self.plant_vars.class_nums

        if not self.cfg_train.get("input_static_cars", False):
            self.type_nums.pop("static_car")

        # PlanTDataset의 본체(index 리스트 초기화)
        self.BEV  = []
        self.labels = [] #(boxes)
        self.measurements = []

        # If you're not using a slurm cluster you can use this line instead of the one after
        label_raw_path_all = glob.glob(os.path.join(root, "**/boxes"), recursive=True)
        # NOTE: FOR SLURM CHANGE TO:
        # label_raw_path_all = subprocess.run(["lfs", "find", root, "-type", "d", "-name", "boxes", "--maxdepth", "3"], capture_output=True, text=True, check=True).stdout.splitlines()

        label_raw_path_all = [p[:-5] for p in label_raw_path_all] 

        label_raw_path = label_raw_path_all # Could filter here if needed (경로에서 /boxes 문자열 지움)

        logging.info(f"Found {len(label_raw_path)} results jsons.")

        # route 필터링(학습에 쓸 수 있는 좋은 주행만 남긴다)
        total_routes = 0
        skipped_routes = 0
        trainable_routes = 0

        for route_dir in label_raw_path:

            route = os.path.basename(route_dir)
            total_routes += 1

            if self.cfg_train.get("filter_routes", True): # Can be set to false for visu 
                if route.startswith('FAILED_') or not os.path.isfile(route_dir + '/results.json.gz'): # or route_dir in manually_kicked:
                    skipped_routes += 1
                    continue

                # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
                with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
                    results_route = json.load(f)
                condition1 = (results_route['scores']['score_composed'] < 100.0 and \
                not (results_route['num_infractions'] == len(results_route['infractions']['min_speed_infractions'])))
                condition2 = results_route['status'] == 'Failed - Agent couldn\'t be set up'
                condition3 = results_route['status'] == 'Failed'
                condition4 = results_route['status'] == 'Failed - Simulation crashed'
                condition5 = results_route['status'] == 'Failed - Agent crashed'
                if condition1 or condition2 or condition3 or condition4 or condition5:
                    continue

                if results_route["timestamp"][:4] == "Town":
                    log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:3]) + ".log"
                else:
                    log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:2]) + ".log"

                log_file = root.rstrip("/")[:-4]+"/slurm/run_files/logs/"+log_file

                # Hacky(겉으로는 성공, 실제로는 실패 -> 이건 논문 재현용 코드에서만 볼 수 있는 디테일)
                silentcrash = False
                with open(log_file, "r", encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    if "SKIPPED" in line:
                        vehicle = line.split(" ")[-1].strip()
                        
                        if vehicle[:6] != "walker" and vehicle not in ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]:
                            silentcrash = True
                            print(line)
                            break
                
                if silentcrash:
                    continue

            trainable_routes += 1

            route_dir = Path(route_dir)
            num_seq = len(os.listdir(route_dir / "boxes"))

            # ignore the first 5 and last two frames  (이중 for문으로 같은 json파일이 쓰이지만 이건 같은 주행을 다른 의사결정 시점에서 학습시키기 위한 의도적인 슬라이딩 윈도우 설계)
            for seq in range(
                5,
                num_seq - self.cfg.model.waypoints.wps_len - self.cfg_train.seq_len - 2,
            ):
                # load input seq and pred seq jointly
                label = []
                measurement = []
                for idx in range(
                    self.cfg_train.seq_len + self.cfg.model.waypoints.wps_len
                ):
                    labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json.gz"
                    measurements_file = (
                        route_dir / "measurements" / f"{seq + idx:04d}.json.gz"
                    )
                    label.append(labels_file)
                    measurement.append(measurements_file)

                self.BEV.append(route_dir / "bev_no_car_semantics" / f"{seq + self.cfg_train.seq_len-1:04d}.png")
                self.labels.append(label)
                self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.BEV          = np.array(self.BEV         ).astype(np.bytes_)
        self.labels       = np.array(self.labels      ).astype(np.bytes_)
        self.measurements = np.array(self.measurements).astype(np.bytes_)

        print(f"Loading {len(self.labels)} samples")
        print('Total amount of routes:', total_routes)
        print('Skipped routes:', skipped_routes)
        print('Trainable routes:', trainable_routes)

    def __len__(self) -> int: #dataloader가 epoch 크기 계산할 때 사용
        """Returns the length of the dataset."""
        return len(self.measurements)

    # 정차 차량을 일부 샘플에 인위적으로 추가 
    def add_parked_cars(self, sample): 
        if self.cfg_train.augment_parked:
            n = np.random.randint(0, 10)
            if len(sample["parked_cars"]) > n:
                if n > 0:
                    idx = np.random.choice(sample["parked_cars"].shape[0], n, replace=False)  
                    sample["input"] += sample["parked_cars"][idx].tolist()
                    sample["output_floating"] += sample["parked_cars"][idx][:, 1:5].tolist()
                    sample["output"] += sample["parked_cars_quant"][idx].tolist()
            elif len(sample["parked_cars"]) > 0:
                sample["input"] += sample["parked_cars"].tolist()
                sample["output_floating"] += sample["parked_cars"][:, 1:5].tolist()
                sample["output"] += sample["parked_cars_quant"].tolist()
            del sample["parked_cars"]
            del sample["parked_cars_quant"]

    # getitem은 DataLoader가 배치를 만들기 위해 샘플을 하나씩 필요로 할 때 호출되고,한 epoch 동안 샘플 개수만큼 반복된다.  
    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]

        sample = {
            "input": []
        }

        #캐시 & 증강 판단 
        augment = self.transform is not None and np.random.rand() < self.aug_rate

        # See if we can use the cache (같은 sample을 여러번 disk에서 다시 만들지 말자)
        if augment and self.data_cache is not None:
            if labels[0].decode()+"_aug" in self.data_cache:
                sample = self.data_cache[labels[0].decode()+"_aug"]
                return sample

            elif labels[0].decode() in self.data_cache:
                sample = self.transform(self.data_cache[labels[0].decode()])
                sample.pop("BEV_aug", None)
                sample.pop("output_floating", None)
                self.data_cache[labels[0].decode()+"_aug"] = sample
                return sample

        elif self.data_cache is not None and labels[0].decode() in self.data_cache:
                sample = self.data_cache[labels[0].decode()]
                sample.pop("BEV_aug", None)
                sample.pop("output_floating", None)
                return sample

        # Load new sample
        loaded_labels = []
        loaded_measurements = []
        loaded_mpc_gts = []

        # json 로딩 (디스크 I/O 발생)
        for i in range(self.cfg_train.seq_len + self.cfg.model.waypoints.wps_len):
            measurements_i = json.load(gzip.open(measurements[i]))
            labels_i = json.load(gzip.open(labels[i]))
            mpc_gt_file = measurements[i].decode().replace('/measurements/', '/mpc_gt/')
            try:
                mpc_gt_i = json.load(gzip.open(mpc_gt_file))
            except FileNotFoundError:
                mpc_gt_i = None

            loaded_labels.append(labels_i)
            loaded_measurements.append(measurements_i)
            loaded_mpc_gts.append(mpc_gt_i) 

        # Extract ego waypoints(Ego waypoint 생성=PlanT의 핵심 GT)
        matrices = [x["ego_matrix"] for x in loaded_measurements[self.cfg_train.seq_len - 1 :]]
        ego_inv = np.linalg.inv(matrices[0]) #월드 좌표를 현재 ego 좌표로 변환 (inv)
        points = np.array(matrices[1:])[:,:,3] #position x,y,z,1
        points = (ego_inv @ points.T).T[:,:2].tolist() # 미래 ego 위치들을 현재 ego 좌표계 기준으로 반환 (최종적으로 x, y 만 씀)
        sample["waypoints"] = points 

        # ==================== MPC Controls 추가 ====================
        mpc_data = loaded_mpc_gts[self.cfg_train.seq_len - 1]  # ✅ 변경!
        
        if mpc_data is not None and mpc_data.get('feasible', False):
            target_len = self.cfg.model.waypoints.wps_len
            
            accelerations = mpc_data.get("accelerations", [])
            curvatures_total = mpc_data.get("curvatures_total", [])
            
            if len(accelerations) >= target_len and len(curvatures_total) >= target_len:
                # Subsample
                indices = np.linspace(0, len(accelerations)-1, target_len, dtype=int)
                acc_seq = [accelerations[i] for i in indices]
                curv_seq = [curvatures_total[i] for i in indices]
            elif len(accelerations) > 0 and len(curvatures_total) > 0:
                # Pad
                acc_seq = list(accelerations) + [accelerations[-1]] * (target_len - len(accelerations))
                curv_seq = list(curvatures_total) + [curvatures_total[-1]] * (target_len - len(curvatures_total))
                acc_seq = acc_seq[:target_len]
                curv_seq = curv_seq[:target_len]
            else:
                # Fallback to scalar
                acc_seq = [mpc_data.get("acceleration", 0.0)] * target_len
                curv_seq = [mpc_data.get("curvature_total", 0.0)] * target_len
            
            sample["mpc_controls"] = [[a, k] for a, k in zip(acc_seq, curv_seq)]
        else:
            # No MPC GT or not feasible
            sample["mpc_controls"] = [[0.0, 0.0]] * self.cfg.model.waypoints.wps_len
        # ===========================================================

        # Route / Speed / Limit 정보 
        # 글로벌 기준 원본 route
        sample["route_original"] = loaded_measurements[self.cfg_train.seq_len - 1]["route_original"][:20]
        # ego 기준으로 변환된 route
        sample["route"] = interpolate_route(loaded_measurements[self.cfg_train.seq_len - 1]["route"][:20])
        # expert가 의도한 목표 속도 
        sample["target_speed"] = loaded_measurements[self.cfg_train.seq_len - 1]["target_speed"]
        # 현재 ego 실제 속도 
        sample["ego_speed"] = loaded_measurements[self.cfg_train.seq_len - 1]["speed"]

        speed_limit = loaded_measurements[self.cfg_train.seq_len - 1]["speed_limit"]
        speed_limit = round(speed_limit*3.6) # TODO
        sample["speed_limit"] = self.speed_cats[speed_limit]

        if loaded_measurements[self.cfg_train.seq_len - 1]["brake"]: # Just in case
            sample["target_speed"] = 0.0

        if self.cfg_train.get("input_bev", False): #bev 추가로 공간 컨텍스트 
            bev = Image.open(self.BEV[index].decode())
            bev = pil_to_tensor(bev)
            bev = torch.rot90(bev, dims=(1, 2))
            sample["BEV"] = self.bev_colors[bev[0, 64:-64, 64:-64].to(torch.int)].permute(2, 0, 1)

            if self.cfg_train.augment:
                aug_path = self.BEV[index].decode().replace("bev_no_car_semantics", "bev_no_car_semantics_augmented")
                bev_aug = Image.open(aug_path)
                bev_aug = pil_to_tensor(bev_aug)
                bev_aug = torch.rot90(bev_aug, dims=(1, 2))
                sample["BEV_aug"] = self.bev_colors[bev_aug[0, 64:-64, 64:-64].to(torch.int)].permute(2, 0, 1)
        
        # autmentation 파라미터 저장 
        sample["augmentation_translation"] = loaded_measurements[self.cfg_train.seq_len - 1]["augmentation_translation"]
        sample["augmentation_rotation"] = loaded_measurements[self.cfg_train.seq_len - 1]["augmentation_rotation"]

        # Load Input objects
        measurements_data = loaded_measurements[self.cfg_train.seq_len - 1]
        labels_data_all = loaded_labels[self.cfg_train.seq_len - 1]

        labels_data = labels_data_all[1:] # remove ego car

        ego_matrix = np.array(measurements_data["ego_matrix"])
        ego_yaw = measurements_data["theta"]

        # Only for viz
        sample["ego_pos"] = measurements_data["pos_global"]
        sample["ego_rot"] = ego_yaw

        # Fix static extents and drop irrelevant objects 객체 필터링 & 정규화 
        for x in labels_data:
            if "position" in x:
                pos_x, pos_y, pos_z = x["position"]

                # 30m radius for tl and stop
                if x["class"] in ["traffic_light", "stop_sign"]:
                    if pos_x**2 + pos_y**2 > 30**2 or abs(pos_z) > 30:
                        x["class"] = "too far"
                # ellipse for others
                else:
                    x_div = self.cfg_train.range_factor_front**2 if pos_x > 0 else 1
                    if pos_x**2/x_div + pos_y**2 > self.MAX_DISTANCE**2 or abs(pos_z) > 30:
                        x["class"] = "too far"

            # Emergency vehicles
            if x["class"]=="car" and x["type_id"] in ["vehicle.dodge.charger_police",
                                                        "vehicle.dodge.charger_police_2020",
                                                        "vehicle.carlamotors.firetruck",
                                                        "vehicle.ford.ambulance"]:
                x["class"] = "emergency"

            # Filter statics and fix extents
            elif x["class"]=="static":
                if "type_id" in x.keys() and x["type_id"] not in ["static.prop.constructioncone", 
                                                                    "static.prop.trafficwarning"]:
                    x["class"] = "irrelevant_static"
                else:
                    # # update static extent
                    if x["type_id"] in STATIC_EXTENTS:
                        x["extent"] = STATIC_EXTENTS[x["type_id"]]
                    else:
                        print(x["type_id"], "was not found in static extents")

            elif x["class"] == "static_car":
                if x["mesh_path"] in CAR_EXTENTS:
                    x["extent"] = CAR_EXTENTS[x["mesh_path"]]
                    if "scale" in x.keys() and x["scale"] is not None:
                        scale = float(x["scale"])
                        x["extent"] = [a*scale for a in x["extent"]]
                else:
                    print("missing static car:", x["mesh_path"])

        #input object tocken 생성 (동적인 객체만 input token으로 사용, 이 줄이 Transformer에 들어가는 객체 토큰 하나)
        input_objects = [
                [
                    self.type_nums[x["class"].lower()],  # type indicator
                    x["position"][0],
                    x["position"][1],
                    rad2deg(x["yaw"]),  # in degrees
                    x["speed"] * 3.6,  # in km/h
                    x["extent"][1]*2 + (0 if "scenario" not in x.keys() or "Door" not in x["scenario"] else 1),
                    x["extent"][0]*2,
                    x["id"],
                ]
                for x in labels_data
                if x["class"].lower() in self.car_types
            ]

        # Add static cars, static objects, traffic lights, stop signs
        input_objects += [[
                self.type_nums[x["class"].lower()], # type indicator
                x["position"][0],
                x["position"][1],
                rad2deg(x["yaw"]),  # in degrees
                0.0,
                x["extent"][1]*2,
                x["extent"][0]*2,
                -1 if x["class"].lower() != "static_car" else -999, #-1 is for all the static objects, -999 denotes static cars, which dont have an id
            ]
            for x in labels_data
            if x["class"].lower() not in self.car_types and x["class"].lower() in self.type_nums.keys() and (x["class"].lower()!="traffic_light" or (x["state"] in ["Red", "Yellow"] and x["affects_ego"])) and (x["class"]!="stop_sign" or x["affects_ego"])
        ]

        # Load output (forecasting) objects (input 객체들과 동일한 ID를 기준으로 미래의 객체 상태를 ego 기준 좌표계로 변환해 object forecasting GT)
        offset = 1
        measurements_data_out = loaded_measurements[self.cfg_train.seq_len - 1 + offset]
        labels_data_all_out = loaded_labels[self.cfg_train.seq_len - 1 + offset]

        output_input_tf =  np.linalg.inv(ego_matrix) @ measurements_data_out["ego_matrix"]
        output_input_yaw = ego_yaw - measurements_data_out["theta"]

        # Generate transformed output objects for forecasting
        output_cars = {x["id"]: [
                                *(output_input_tf @ np.append(x["position"],[1]))[:2],
                                rad2deg(x["yaw"] - output_input_yaw),  # in degrees
                                x["speed"] * 3.6  # in km/h
                                ] for x in labels_data_all_out if "id" in x and "speed" in x}

        output_objects_matched = []
        for x in input_objects:
            object_id = x[-1]
            if object_id == -999: # static_car
                output_objects_matched.append(x[1:5])
            elif object_id in output_cars:
                output_objects_matched.append(output_cars[object_id])
            else:
                output_objects_matched.append([-999., -999., -999., -999.]) # Dummy for traffic lights, statics, etc.

        output_objects_quantized = self.quantize_box(output_objects_matched)

        #output object
        sample["output_floating"] = output_objects_matched
        sample["output"] = output_objects_quantized

        # remove id 
        input_objects = [x[:-1] for x in input_objects]

        sample["input"] = input_objects

        if self.cfg_train.augment_parked:
            town = labels[0].decode().split("/")[-3].split("_")[0]
            if town == "Town10":
                town = "Town10HD"
            ego_pos = loaded_measurements[self.cfg_train.seq_len - 1]["pos_global"]
            ego_theta = loaded_measurements[self.cfg_train.seq_len - 1]["theta"]

            idxs = self.parked_trees[town].query_ball_point(ego_pos, 30)
            if len(idxs) > 0:
                sample["parked_cars"] = np.array([[self.type_nums["car"], x, y, yaw, 0, y_ex*2, x_ex*2]
                                        for ((x, y), yaw, (x_ex, y_ex, _)) in zip(self.parked_locations[town][idxs], self.parked_rotations[town][idxs], self.parked_extents[town][idxs])])
                sample["parked_cars"][:, 1:3] -= ego_pos
                c, s = np.cos(ego_theta), np.sin(ego_theta)
                R = np.array([[c, -s], [s, c]])
                sample["parked_cars"][:, 1:3] = (R.T @ sample["parked_cars"][:, 1:3].T).T
                sample["parked_cars"][:, 3] = normalize_angle_degree(sample["parked_cars"][:, 3] - np.rad2deg(ego_theta))
                sample["parked_cars_quant"] = np.array(self.quantize_box(sample["parked_cars"][:, 1:5]))
            else:
                sample["parked_cars"] = np.array([])
                sample["parked_cars_quant"] = np.array([])

            # For now i fix the parked augmentation per sample with and without aug, could be unique per call but has performance implications
            self.add_parked_cars(sample)

        # Store in data cache
        if self.data_cache is not None:
            self.data_cache[labels[0].decode()] = sample # Save unaugmented sample with BEV_aug so we can use it later for aug

        if augment:
            sample = self.transform(sample)
            if self.data_cache is not None:
                sample.pop("BEV_aug", None) # Augmented sample doesnt need BEV_aug since its the normal BEV
                sample.pop("output_floating", None)
                self.data_cache[labels[0].decode()+"_aug"] = sample

        sample.pop("BEV_aug", None)
        sample.pop("output_floating", None)

        return sample
    
    def aug_sample(self, sample): #실제 데이터 증강 (장면 전체를 rigid transform 하는 증강)
        # In transfuser, translation gets subtracted and applied first
        translate = - np.array([0.0, sample["augmentation_translation"]])
        # In transfuser multiplizieren die R von links und transposen beides?
        rot = np.deg2rad(sample["augmentation_rotation"])

        if self.cfg_train.get("input_bev", False):
            sample["BEV"] = sample["BEV_aug"]

        input = np.array(sample["input"])
        if "output" in sample:
            output = np.array(sample["output_floating"])
            dummy_mask = output == -999.
        else:
            output = []
        waypoints = np.array(sample["waypoints"])
        route = np.array(sample["route"])
        route_original = np.array(sample["route_original"])

        # Translation
        if len(input) > 0:
            input[:, 1:3] += translate
        if len(output) > 0:
            output[:, :2] += translate
        waypoints += translate
        route += translate
        route_original += translate

        # Rotation
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])

        if len(input) > 0:
            input[:, 1:3] = (R.T @ input[:, 1:3].T).T
        if len(output) > 0:
            output[:, :2] = (R.T @ output[:, :2].T).T
        waypoints = (R.T @ waypoints.T).T
        route = (R.T @ route.T).T
        route_original = (R.T @ route_original.T).T

        if len(input) > 0:
            input[:, 3] -= np.rad2deg(rot)
        if len(output) > 0:
            output[:, 2] -= np.rad2deg(rot)

        sample["input"] = input.tolist()
        if "output" in sample:
            output[dummy_mask] = -999.
            sample["output"] = self.quantize_box(output.tolist())
        sample["waypoints"] = waypoints.tolist()
        sample["route"] = route.tolist()
        sample["route_original"] = route_original.tolist()

        return sample

    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        if len(boxes)==0:
            return boxes.tolist()

        # range of xy is [-30, 30]
        # range of yaw is [-360, 0]
        # range of speed is [0, 120]
        # range of extent is [0, 30]

        # Dummy mask:
        dummy_mask = boxes==-999

        # quantize xy
        boxes[:, 0] = (boxes[:, 0] + self.MAX_DISTANCE) / self.MAX_DISTANCE_DOUBLE
        boxes[:, 1] = (boxes[:, 1] + self.MAX_DISTANCE) / self.MAX_DISTANCE_DOUBLE

        # quantize yaw
        boxes[:, 2] = (boxes[:, 2] % 360) / 360

        # quantize speed
        boxes[:, 3] = boxes[:, 3] / 120

        boxes[:, 0] = np.clip(boxes[:, 0], 0, (1 + self.cfg.model.training.get("range_factor_front", 1)) / 2)
        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, :2] = (boxes[:, :2] * (size_pos - 1)).round()
        boxes[:, 2] = (boxes[:, 2] * (size_angle - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_speed - 1)).round()

        boxes[dummy_mask] = -999

        return boxes.astype(np.int32).tolist()
    
    # This is only used for visualization
    def unquantize_box(self, boxes):
        boxes = np.array(boxes).astype(np.float32)

        if len(boxes)==0:
            return boxes.tolist()
        
        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, :2] = boxes[:, :2] / (size_pos - 1)
        boxes[:, 2] = boxes[:, 2] / (size_angle - 1)
        boxes[:, 3] = boxes[:, 3] / (size_speed - 1)

        # unquantize xy
        boxes[:, 0] = boxes[:, 0] * self.MAX_DISTANCE_DOUBLE - self.MAX_DISTANCE
        boxes[:, 1] = boxes[:, 1] * self.MAX_DISTANCE_DOUBLE - self.MAX_DISTANCE

        # unquantize yaw
        boxes[:, 2] = normalize_angle_degree(boxes[:, 2] * 360)

        # unquantize speed
        boxes[:, 3] = boxes[:, 3] * 120 #TODO

        return boxes.tolist()


    # def filter_data_by_town(self, label_raw_path_all, split):
    #     # in case we want to train without T2 and T5
    #     label_raw_path = []
    #     if split == "train":
    #         for path in label_raw_path_all:
    #             if "Town02" in path or "Town05" in path:
    #                 continue
    #             label_raw_path.append(path)
    #     elif split == "val":
    #         for path in label_raw_path_all:
    #             if "Town02" in path or "Town05" in path:
    #                 label_raw_path.append(path)
    #     elif split == "all":
    #         label_raw_path = label_raw_path_all
            
    #     return label_raw_path

def interpolate_route(points):
    route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
    shift = np.roll(route, 1, axis=0) # Shift by 1
    shift[0] = shift[1] # Set wraparound value to 0

    dists = np.linalg.norm(route-shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

    x = np.arange(0, 20, 1)
    interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

    return interp_points

def generate_batch(data_batch):
    maxseq = max([len(sample["input"]) for sample in data_batch])
    B = len(data_batch)

    x_batch_objs = [[0, 0, 0, 0, 0, 0, 0]]  # Padding at idx 0
    y_batch_objs = [[-999, -999, -999, -999]]  # Padding

    batch_idxs = torch.zeros((B, maxseq), dtype=torch.int32)

    keys = [x for x in data_batch[0] if x not in ["input", "output"]]

    if "mpc_controls" not in keys and "mpc_controls" in data_batch[0]:
        keys.append("mpc_controls")

    batches = {key: [] for key in keys}

    n = 1  # Padding is 0

    for i, sample in enumerate(data_batch):
        # Input
        n_sample = len(sample["input"])
        batch_idxs[i, :n_sample] = torch.arange(n, n+n_sample)
        n += n_sample

        x_batch_objs.extend(sample["input"])
        y_batch_objs.extend(sample["output"])

        for key in keys:
            if key == "speed_limit":
                batches[key].append(torch.tensor(sample[key], dtype=torch.int))
            else:
                if torch.is_tensor(sample[key]):
                    batches[key].append(sample[key].type(torch.float32))
                else:
                    batches[key].append(torch.tensor(sample[key], dtype=torch.float32))

    batches = {key: torch.stack(value) for key, value in batches.items()}
    batches["idxs"] = batch_idxs
    batches["x_objs"] = torch.tensor(x_batch_objs, dtype=torch.float32)
    batches["y_objs"] = torch.tensor(y_batch_objs, dtype=torch.long)

    return batches


if __name__=="__main__":
    import yaml
    # Read YAML file
    with open("PlanT/config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    with open("PlanT/config/model/PlanT.yaml", 'r') as stream:
        plnt = yaml.safe_load(stream)

    cfg["model"] = plnt

    cfg["visualize"] = False

    cfg["trainset_size"] = 1
    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    cfg = DictAsMember(cfg)

    ds = PlanTDataset("/workspace/data/PlanT_2_dataset", cfg)

    print(generate_batch([ds[255], ds[256], ds[257]]).keys())
