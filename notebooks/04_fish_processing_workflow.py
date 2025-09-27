import numpy as np
import yaml

# Import our custom modules
from food_robotics.vision.fish_processor import FishVisionProcessor
from food_robotics.kinematics.scara_robot import SCARAKinematics
from food_robotics.control.scara_controller import SCARAController
from food_robotics.utils.common import set_mlflow_tracking_uri

def run_fish_processing_workflow():
    """Main function to orchestrate the entire fish processing task."""
    
    # 1. Load Configuration
    with open('../project_config.yml', 'r') as f:
        config = yaml.safe_load(f)['scara_fish_processor']

    # 2. Initialize Components
    vision = FishVisionProcessor(model_path="/models/fish_yolov8.onnx")
    kinematics = SCARAKinematics(l1=config['link_1_length'], l2=config['link_2_length'])
    controller = SCARAController(kinematics, tool_offset_z=config['tool_offset_z'])

    # 3. Simulate Capturing an Image
    # In a real system: image = camera.capture()
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 4. Vision Processing
    detection = vision.detect_fish_and_keypoints(mock_image)
    if not detection:
        print("No fish detected. Exiting.")
        return

    world_coords = vision.pixels_to_world(
        detection['keypoints'], 
        np.array(config['camera_matrix']), 
        np.array(config['distortion_coeffs'])
    )

    # 5. Generate and Execute Robot Path
    # We will define the cut along the dorsal fin
    start_cut_point = world_coords['dorsal_fin_start']
    end_cut_point = world_coords['dorsal_fin_end']
    
    motion_path = controller.generate_cut_path(
        start_point_xy=start_cut_point,
        end_point_xy=end_cut_point,
        cut_depth=config['cut_depth']
    )
    
    controller.execute_path(motion_path)

    # 6. Log Results with MLflow (Optional)
    # set_mlflow_tracking_uri()
    # with mlflow.start_run(run_name="fish_processing_run") as run:
    #     mlflow.log_params(config)
    #     mlflow.log_metric("fish_detected_confidence", detection['confidence'])
    #     mlflow.log_dict(world_coords, "world_coordinates.json")
    print("Workflow finished successfully.")

if __name__ == '__main__':
    run_fish_processing_workflow()
