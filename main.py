from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2

image = Image.open("sam.jpg")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# read the image with OpenCV
image = cv2.imread("sam.jpg")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    # Get x, y, width and height
    x, y, width, height = box
    # Draw rectangle on image
    cv2.rectangle(image, (x, y), (width, height), (0, 255, 0), 2)
    # Get label text
    text = model.config.id2label[label.item()]
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Add text label
    cv2.putText(image, f'{text} {round(score.item(),2)*100}%', (x, y-5), font, 0.5, (0, 255, 255), 2)
# Display image
cv2.imshow('Image with rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


    # print(
    #         f"Detected {model.config.id2label[label.item()]} with confidence "
    #         f"{round(score.item(), 3)} at location {box}")