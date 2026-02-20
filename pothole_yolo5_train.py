import os
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


DATASET_PATH = r"C:\Users\.."

IMG_SIZE = 640         
BATCH_SIZE = 16        
EPOCHS = 100           
DEVICE = 0             

MODEL_TYPE = 'yolov5s'  


def create_dataset_yaml():
    dataset_config = f"""
path: {DATASET_PATH}
train: images/train
val: images/valid
test: images/test

nc: 1
names: ['pothole']
"""
    yaml_path = os.path.join(DATASET_PATH, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(dataset_config.strip())
    
    print(f"✓ Dataset config created: {yaml_path}")
    return yaml_path


def verify_dataset(dataset_path):
    print("\n=== Verifying Dataset ===")
    
    splits = ['train', 'valid', 'test']
    stats = {}
    
    for split in splits:
        img_dir = os.path.join(dataset_path, 'images', split)
        label_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(img_dir):
            print(f" Warning: {img_dir} not found!")
            continue
            
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if os.path.exists(label_dir):
            labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            stats[split] = {'images': len(images), 'labels': len(labels)}
        else:
            print(f" Warning: {label_dir} not found!")
            labels = [f for f in os.listdir(img_dir) if f.endswith('.txt')]
            stats[split] = {'images': len(images), 'labels': len(labels)}
    
    print("\nDataset Statistics:")
    for split, counts in stats.items():
        print(f"  {split:6s}: {counts['images']:3d} images, {counts['labels']:3d} labels")
    
    total_images = sum(s['images'] for s in stats.values())
    print(f"\n  Total: {total_images} images")
    
    return stats


def check_gpu():
    print("\n=== GPU Check ===")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" GPU Available: {gpu_name}")
        print(f" GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print(" No GPU detected! Training will use CPU (very slow)")
        return False


def train_model(yaml_path, model_type=MODEL_TYPE):
    print(f"\n=== Training {model_type.upper()} ===")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    
    model = YOLO(f'{model_type}.pt')
    
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='pothole_detection',
        name='train',
        patience=20,
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n Training Complete!")
    print(f" Best model saved to: /run/pothole_detection/train/weights/best.pt")
    
    return model, results


def evaluate_model(model, yaml_path):
    print("\n=== Evaluating Model ===")
    
    metrics = model.val(data=yaml_path)
    
    print(f"\nValidation Results:")
    print(f"  mAP@0.5: {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")
    
    return metrics


def test_on_images(model, test_dir, num_samples=5):
    print("\n=== Testing on Sample Images ===")
    
    output_dir = 'pothole_detection/test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    if not test_images:
        print("No test images found!")
        return
    
    results = model(test_images)
    
    for i, result in enumerate(results):
        plt.figure(figsize=(10, 8))
        img = result.plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Test Image {i+1} - Pothole Detection")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/test_result_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        boxes = result.boxes
        print(f"\nImage {i+1}: {len(boxes)} pothole(s) detected")
        for j, box in enumerate(boxes):
            conf = box.conf[0].item()
            print(f"  Pothole {j+1}: Confidence {conf:.2%}")
    
    print(f"\n✓ Test results saved to: {output_dir}/")


def export_model(model):
    print("\n=== Exporting Model ===")
    model.export(format='onnx')
    print("✓ Model exported to ONNX format")


def main():
    print("=" * 60)
    print("POTHOLE DETECTION - YOLOV5 TRAINING")
    print("=" * 60)
    
    check_gpu()
    
    stats = verify_dataset(DATASET_PATH)
    
    if not stats:
        print("\n Dataset not found or incomplete!")
        print("Please check your dataset path and structure.")
        return
    
    yaml_path = create_dataset_yaml()
    model, results = train_model(yaml_path)
    metrics = evaluate_model(model, yaml_path)
    
    test_dir = os.path.join(DATASET_PATH, 'images', 'test')
    if os.path.exists(test_dir):
        test_on_images(model, test_dir, num_samples=5)
    
    export_model(model)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nYour trained model is ready!")
    print(f"Location: pothole_detection/train/weights/best.pt")
    print("\nTo use it:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('pothole_detection/train/weights/best.pt')")
    print("  results = model('path/to/your/image.jpg')")


def resume_training():
    model = YOLO('pothole_detection/train/weights/last.pt')
    model.train(resume=True)


def predict_single_image(image_path):
    model = YOLO('pothole_detection/train/weights/best.pt')
    results = model(image_path)
    results[0].show()
    return results


def predict_video(video_path, output_path='output_video.mp4'):
    model = YOLO('pothole_detection/train/weights/best.pt')
    results = model(video_path, stream=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for result in results:
        frame = result.plot()
        out.write(frame)
    
    cap.release()
    out.release()
    print(f" Video saved to: {output_path}")


if __name__ == "__main__":
    main()