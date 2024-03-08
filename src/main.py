import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from ultralytics import YOLO
import torch

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




def main():
    check_graphics_driver()
    model_path = './src/model/last.pt'
    
    kinect = KinectHandler()

    while True:
        # # Capture depth frame
        # depth_frame = kinect.get_depth_frame()

        # #print(depth_frame)
        # if depth_frame is not None:
        #     # Display depth frame
        #     print(depth_frame)
        #     cv2.imshow('Depth Frame', depth_frame.astype(np.uint8))

        color_frame = kinect.get_color_frame()

        if color_frame is not None:
            cv2.imwrite('input.png', color_frame)
            #cv2.imshow('Color Frame', color_frame.astype(np.uint8))
            img = cv2.imread('input.png')
            H, W , _ = img.shape
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Using device: {device}')
            model = YOLO(model_path).to(device)

            results = model(img)
            for result in results:
                if result.masks is not None:
                    print("Identified a box")
                    for j, mask in enumerate(result.masks.data):
                        mask = mask.numpy() * 255
                        mask = cv2.resize(mask, (W, H))
                        cv2.imwrite('output.png', mask)
                        np.savetxt(f'output.csv', mask, delimiter=',', fmt='%d')

        

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the Kinect resources
    kinect.close_color_sensor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




