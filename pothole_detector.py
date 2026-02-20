import argparse
from ultralytics import YOLO
import cv2
import os

def detect_pothole(image_path, model_path='best.pt', save_output=False, show_result=False):
    try:
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        print(f"Processing image: {image_path}")
        results = model(image_path)
        
        result = results[0]
        
        if show_result:
            result.show()
            print("Displaying result window. Close it to continue...")
        
        if save_output:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = f"{name}_detected{ext}"
            result.save(filename=output_path)
            print(f"Result saved to: {output_path}")
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            num_detections = len(result.boxes)
            print(f"\nDetection Summary:")
            print(f"- Found {num_detections} pothole(s)")
            
            if num_detections > 0:
                print("- Confidence scores:")
                for i, box in enumerate(result.boxes):
                    conf = box.conf[0].item()
                    print(f"  Pothole {i+1}: {conf:.2%}")
        else:
            print("No potholes detected.")
            
        return result
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Detect potholes in images using YOLO')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('--model', '-m', type=str, default='best.pt')
    parser.add_argument('--save', '-s', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output', '-o', type=str)
    
    args = parser.parse_args()
    
    detect_pothole(
        image_path=args.image,
        model_path=args.model,
        save_output=args.save or args.output is not None,
        show_result=args.show
    )

if __name__ == "__main__":
    main()