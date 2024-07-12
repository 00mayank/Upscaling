import os
import cv2
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

SDvideopath = 'SDvideo/641.mp4'
HDvideopath = 'HDvideo'
HDvideo='HDvideo/1080.mp4'
HDvideoframepath = 'HDvideo/frames'
SDvideoframepath = 'SDvideo/frames'
os.makedirs(SDvideoframepath, exist_ok=True)
os.makedirs(HDvideoframepath, exist_ok=True)

# Extract frames from the SD video
SDvideo = cv2.VideoCapture(SDvideopath)
frame_count = 0
while True:
    ret, frame = SDvideo.read()
    if not ret:
        break
    else:
        frameName = os.path.join(SDvideoframepath, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frameName, frame)
        frame_count += 1

SDvideo.release()

# Setting up CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# Load the up-scaling model
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Defining a prompt for up-scaling the video
prompt = "Upscale video frames"

# Upscale frames and save
for imageName in os.listdir(SDvideoframepath):
    if imageName.endswith('.png'):  # Adjust file extensions as needed
        # Load and preprocess the image
        file_path = os.path.join(SDvideoframepath, imageName)
        image = Image.open(file_path).convert("RGB")

        # Upsample the image
        with torch.no_grad():
            upscaled_image = pipe(image=image, prompt=prompt)["image"].GPU()

        # Save the upscaled image
        HDframe = os.path.join(HDvideoframepath, f"upscaled_{imageName}")
        upscaled_image.save(HDframe)

print("Frame up-scaling completed.")

frame_rate = 60  # frames per second
frames = [frame for frame in sorted(os.listdir(HDvideoframepath)) if frame.endswith('.png')]
frame1 = cv2.imread(os.path.join(HDvideoframepath, frames[0]))
height, width, channels = frame1.shape
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(HDvideo,fourcc, frame_rate, frame_size)
for frame in frames:
    videoFrame = cv2.imread(os.path.join(HDvideoframepath, frame))
    video_writer.write(videoFrame)
video_writer.release()
print("Video up-sampling completed.")

