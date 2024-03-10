import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from ultralytics import YOLO
import torch
import time
import os


def check_graphics_driver():
    print("Is CUDA supported by this system?",torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print("ID of current CUDA device:",torch.cuda.current_device())

    print("Name of current CUDA device:",torch.cuda.get_device_name(cuda_id))


class KinectHandler(object):
    def __init__(self):
        self.kinect_depth_sensor = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinect_depth_sensor.depth_frame_desc.Width, self.kinect_depth_sensor.depth_frame_desc.Height
        self.kinect_color_sensor = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.color_width, self.color_height = self.kinect_color_sensor.color_frame_desc.Width, self.kinect_color_sensor.color_frame_desc.Height
        self.color_frames_buffer = []
        self.depth_frames_buffer = []

    def get_depth_frame(self):
        if self.kinect_depth_sensor.has_new_depth_frame():
            depth_frame = self.kinect_depth_sensor.get_last_depth_frame()
            depth_frame = depth_frame.reshape((self.depth_height, self.depth_width)).astype(np.uint16)
            return depth_frame
        return None

    def get_color_frame(self):
        if self.kinect_color_sensor.has_new_color_frame():
            color_frame = self.kinect_color_sensor.get_last_color_frame()
            color_frame = color_frame.reshape((self.color_height, self.color_width, 4))
            return color_frame
        return None
    
    def close_depth_sensor(self):
        self.kinect_depth_sensor.close()
        
    def close_color_sensor(self):
        self.kinect_color_sensor.close()


def get_depth_frame_with_timestamp(kinect):
    depth_frame = kinect.get_depth_frame()
    return depth_frame, time.time()

    
def get_color_frame_with_timestamp(kinect):
    color_frame = kinect.get_color_frame()
    return color_frame, time.time()

    
def main():
    time_lag = 0.01
    check_graphics_driver()
    model_path = './src/model/last.pt'
    
    kinect = KinectHandler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = YOLO(model_path).to(device)

    while True:

        depth_frame, depth_timestamp = get_depth_frame_with_timestamp(kinect) # (424, 512)
        color_frame, color_timestamp = get_color_frame_with_timestamp(kinect) # (1080, 1920, 4)
        if depth_frame is not None:
            kinect.depth_frames_buffer.append([depth_frame, depth_timestamp])
        if color_frame is not None:
            kinect.color_frames_buffer.append([color_frame, color_timestamp])
        # print(kinect.depth_frames_buffer[-1][0])
        if len(kinect.depth_frames_buffer)>0 and len(kinect.color_frames_buffer)>0:
            if kinect.depth_frames_buffer[-1][0] is not None and kinect.color_frames_buffer[-1][0] is not None:
                color_frame, color_timestamp = kinect.color_frames_buffer[-1]
                depth_frame, depth_timestamp = kinect.depth_frames_buffer[-1]
                timestamp_difference = abs(color_timestamp-depth_timestamp)
                if timestamp_difference<=time_lag:
                    color_frame_RGB = color_frame[:, :, :3]   # Use color_frame directly

                    H, W  = depth_frame.shape #(1080 1920 3)


                    results = model(color_frame_RGB, verbose=False)
                    for result in results:
                        if result.masks is not None:
                            print("Identified a box")
                            for j, mask in enumerate(result.masks.data):
                                mask = mask.to(device)
                                mask = mask.cpu().numpy() * 255 if device == 'cuda' else mask.numpy() * 255
                                                        
                                mask = cv2.resize(mask, (W, H)).astype(np.uint8)
                                
                                depth_frame_with_mask = cv2.bitwise_and(depth_frame, depth_frame, mask=mask)
                                distance_to_object = np.mean(depth_frame_with_mask)
                                # my_path = 'D:/tricalseventhsem/ARoBotXVision/src/outputs'
                                # cv2.imwrite(os.path.join(my_path, 'depth_with_mask.jpg'), depth_frame_with_mask)
                                # cv2.imwrite(os.path.join(my_path, 'depth_frame.jpg'), depth_frame_with_mask)
                                # cv2.imwrite(os.path.join(my_path, 'color_frame.jpg'), color_frame_RGB)
                                # cv2.imwrite(os.path.join(my_path, 'mask.jpg'), mask)
                                # cv2.imshow('Depth Frame with Mask', depth_frame_with_mask)

            
            
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the Kinect resources
    kinect.close_color_sensor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




