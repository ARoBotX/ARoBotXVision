import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from ultralytics import YOLO


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
    model_path = 'last.pt'

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

            model = YOLO(model_path)

            results = model(img, verbose=False)
            for result in results:

                if result.masks is not None:
                    print("Identified a box")
                    break
                #     for j, mask in enumerate(result.masks.data):
                #         mask = mask.numpy() * 255
                #         mask = cv2.resize(mask, (W, H))
                #         cv2.imwrite('output.png', mask)




        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the Kinect resources
    kinect.close_color_sensor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



