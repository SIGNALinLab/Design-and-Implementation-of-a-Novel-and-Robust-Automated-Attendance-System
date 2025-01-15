import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import torch
from ESRGAN import RRDBNet_arch as arch
# from gfpgan import GFPGANer


def run_yolo(image_path):
    # Load YOLOv8 model
    model = YOLO("yolov8x.pt")  

    # Perform detection
    results = model.predict(source=image_path, conf=0.1)  # Set confidence threshold

    # Extract bounding boxes for detected "persons"
    yolo_boxes = []
    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # Class ID 0 corresponds to "person" in COCO
                x1, y1, x2, y2 = map(int, bbox)
                width = x2 - x1
                height = y2 - y1
                yolo_boxes.append([x1, y1, width, height])

    return yolo_boxes


# RetinaFace Face Detection
def detect_faces_retinaface(image, expansion_factor=0.5):
    # Initialize RetinaFace model
    face_detector = FaceAnalysis(allowed_modules=["detection"])
    face_detector.prepare(ctx_id=0)  # Use CPU (-1) or GPU (0, 1, etc.) if available
    
    # Detect faces in the image
    faces = face_detector.get(image)
    
    # Extract bounding boxes for detected faces
    face_boxes = []
    height, width = image.shape[:2]  # Get image dimensions
    
    for face in faces:
        face_box = face.bbox.astype(int)  # Get bounding box and convert to int
        x1, y1, x2, y2 = face_box

        # Calculate the expansion size for each side
        box_width = x2 - x1
        box_height = y2 - y1
        x_expansion = int(box_width * expansion_factor)
        y_expansion = int(box_height * expansion_factor)

        # Expand the bounding box, ensuring it stays within the image boundaries
        x1 = max(0, x1 - x_expansion)  # Ensure we don't go beyond the image boundary
        y1 = max(0, y1 - y_expansion)
        x2 = min(width, x2 + x_expansion)
        y2 = min(height, y2 + y_expansion)

        # Append the expanded bounding box [x1, y1, width, height]
        face_boxes.append([x1, y1, x2 - x1, y2 - y1])
    
    return face_boxes


# Load the ESRGAN model
def load_esrgan_model(model_path, device):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

# Apply super-resolution to enhance the cropped face image
def apply_esrgan(image, model, device):

    img = image * 1.0 / 255  # Normalize image to [0, 1]
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Apply the ESRGAN model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Convert back to image format
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output

"""
def enhance_with_gfpgan(image):
    # Initialize GFPGAN (make sure you've installed the model checkpoint)
    gfpgan_model = GFPGANer(model_path='path_to_gfpgan_model.pth', upscale=2, arch='clean')
    # GFPGAN enhancement function
    _, restored_image, _ = gfpgan_model.enhance(image, has_aligned=False, only_center_face=False)
    return restored_image
"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Load the ESRGAN model for super-resolution
        esrgan_model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'  # Path to the ESRGAN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        esrgan_model = load_esrgan_model(esrgan_model_path, device)
        
        # Step 1: Run YOLO detection to get person bounding boxes
        yolo_boxes = run_yolo(image_path)
        
        # Step 2: Load the image for RetinaFace detection
        image = cv2.imread(image_path)
        
        for i, box in enumerate(yolo_boxes):
            x, y, w, h = box
            cropped_person = image[y:y+h, x:x+w]
            
            # Step 3: Run RetinaFace detection on the cropped person image to get face bounding boxes
            face_boxes = detect_faces_retinaface(cropped_person)
            
            for j, face_box in enumerate(face_boxes):
                fx, fy, fw, fh = face_box
                cropped_face = cropped_person[fy:fy+fh, fx:fx+fw]
                
                # Step 4: Apply super-resolution to the cropped face image
                enhanced_cropped_face = apply_esrgan(cropped_face, esrgan_model, device)
                
                # Save the enhanced image for debugging purposes
                enhanced_face_image_path = f"enhanced_face_{i}_{j}.jpg"
                cv2.imwrite(enhanced_face_image_path, enhanced_cropped_face)
                
                print(f"Person {i}, face {j} enhanced image saved at {enhanced_face_image_path}")
    else:
        print("Please provide an image path")
