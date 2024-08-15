# smooth_main_rt.py
# Real-time object detection using webcam stream.
# Authors: Sasika Amarasinghe, S Thuvaragan

import cv2
import time
from threading import Thread
import torch
import sys
import serial.tools.list_ports
from utils import get_device, get_model, get_box_coordinates, get_image_with_box_corners

# Configuration     
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


def print_configuration(camera_index):
    
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
    print("="*30)
    print("Real-time Box Segmentation using Camera Stream")
    print("="*30)
    print()
    # List to store available camera indexes
    available_cameras = []

    # Test the first few camera indexes (0-9)
    print("Checking available camera indexes...")
    for index in range(4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera
    
    # Print the available camera indexes
    print("Available camera indexes:", available_cameras)
    
    # Change according to your camera (0 for default camera)
    camera_index = input("\nEnter the camera index: ")
    if camera_index.isdigit():
        camera_index = int(camera_index)
    else:
        print("Invalid camera index. Exiting...")
        exit(0)
           
    print_configuration(camera_index=camera_index)
    
    
    ports = serial.tools.list_ports.comports()
    baud_rate = 9600
    
    
    print("Specified baud rate:", baud_rate)

    for port, desc, hwid in sorted(ports):
        print("Port:",port,"\nDesc:",desc,"\nHwid:",hwid,"\n")
    
    print("=====================================")
    # Start Serial communication with the board
    isSerialPortWorking = False
    if len(ports) > 0:
        # port = ports[0].device
        print("Select the port to communicate with the board:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device}")
            
        port = input("Enter the port index: ")
        if port.isdigit():
            port = ports[int(port)].device
        else:
            print("Invalid port number. Exiting...")
            exit(0)
        
        ser = serial.Serial(port, baud_rate, timeout=1)
        isSerialPortWorking = True
    
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
            
            if isSerialPortWorking:
                # Send the box coordinates to the board
                ser.write(
                    str((box_corners_dict["top_left"], 
                        box_corners_dict["top_right"],
                        box_corners_dict["bottom_right"],
                        box_corners_dict["bottom_left"])
                    ).encode('utf-8')
                )

                # Send a newline character to mark the end of the message
                ser.write(b'\n')
            
            annotated_frame = get_image_with_box_corners(frame, box_corners_dict)
            num_frames_processed += 1

            cv2.imshow("frame", annotated_frame)
            
        except ValueError:  # No box is detected
            cv2.imshow("frame", frame)
        
        num_frames_processed += 1

        if cv2.waitKey(1) == ord('q'):
            if isSerialPortWorking:
                ser.close()
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