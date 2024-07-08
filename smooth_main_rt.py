# smooth_main_rt.py
# Real-time object detection using webcam stream.
# Authors: Sasika Amarasinghe, S Thuvaragan

import cv2
import time
from threading import Thread
import torch
from utils import get_device, get_model, get_box_coordinates, get_image_with_box_corners

# Configuration
camera_index = 0  # Change according to your camera (0 for default camera)
buffer_size = 2
model_name = "fastSAM-s"
device = "cpu"
fast_sam_s_checkpoint = "FastSAM-s.pt"

# Initialize model
model = get_model(model_name)


class WebcamStream:
    def __init__(self, stream_id=0, buffer_size=1):
        self.stream_id = stream_id
        self.buffer_size = buffer_size
        self.vcap = cv2.VideoCapture(self.stream_id)
        self.vcap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        
        fps_input_stream = int(self.vcap.get(cv2.CAP_PROP_FPS))
        print("FPS of input stream:", fps_input_stream)
        
        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)
            
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
    
    def start(self):
        self.stopped = False
        self.t.start()
        
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.vcap.read()
            if not self.grabbed:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True


def print_configuration():
    print("="*30)
    print("Real-time Box Segmentation using Camera Stream")
    print("="*30)
    print()
    print("Configuration Settings:")
    print(f"Camera Index: {camera_index}")
    print(f"Buffer Size: {buffer_size}")
    print(f"Model Name: {model_name}")
    print(f"Device: {device}")
    print(f"FastSAM-s Checkpoint: {fast_sam_s_checkpoint}")
    print("="*30)
    print() 
    print("Instructions:")
    print("Press 'q' to exit.")
    print("="*30)
    print()


def main():
    print_configuration()
    
    webcam_stream = WebcamStream(stream_id=camera_index, buffer_size=buffer_size)
    webcam_stream.start()

    num_frames_processed = 0 
    start = time.time()

    while True:
        if webcam_stream.stopped:
            break
        
        frame = webcam_stream.read()

        # Processing Image
        try:
            box_corners_dict = get_box_coordinates(frame, model, device, False, False, False)
            print(box_corners_dict)
            annotated_frame = get_image_with_box_corners(frame, box_corners_dict)
            num_frames_processed += 1

            cv2.imshow("frame", annotated_frame)
            
        except ValueError:  # No box is detected
            cv2.imshow("frame", frame)
        
        num_frames_processed += 1

        if cv2.waitKey(1) == ord('q'):
            break

    elapsed = time.time() - start
    fps = num_frames_processed / elapsed 
    print()
    print("="*30)
    print("Execution Summary:")
    print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed:.2f} seconds, Frames Processed: {num_frames_processed}")
    print("="*30)
    print()

    webcam_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()