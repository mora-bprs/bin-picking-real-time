# smooth_main_rt.py
# Real-time object detection using webcam stream.
# Authors: Sasika Amarasinghe, S Thuvaragan

# Binpicking System Flow:
# 1. Capture the image from the camera
# 2. Detect the box in the image
# 3. Process the box coordinates
#   - Calculate the distance to box respect to center of camera frame.
#   - Send the distance to the arm controller system
# 4. Arm adjusts the position so that center of box and camera aligns
# 5. Capture the image again and confirm
# 6. If the box is not in the center, repeat the process
# 7. If the box is in the center, proceed to pick the box ( igmored )
# 8. Ready to pickup the box? ( emulated using a button press )
# 9. End effector picks up the box ( approximated with stepper actuation )
# 10. Send the predefined destination coordinates to arm controller
# 11. Arm moves to the destination ( reverse kinematics, ignored )
# 12. Ready to drop the box? ( emulated using a button hold >1s )
# 13. End effector releases the box ( approximated to stepper actuation in reverse )
# 14. Repeat the process

import time
from threading import Thread

import cv2
import serial.tools.list_ports

from utils2 import (
    get_box_coordinates,
    get_image_with_box_corners,
    get_model,
)

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
            print("[Exiting] No more frames to read")
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
                print("[Exiting] No more frames to read")
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
    print("=" * 30)
    print()
    print("Instructions:")
    print("Press 'q' to exit.")
    print("=" * 30)
    print()


def main():
    print("=" * 30)
    print("Real-time Box Segmentation using Camera Stream")
    print("=" * 30)
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

    # serial configuration
    ports = serial.tools.list_ports.comports()
    baud_rate = 115200

    print("specified baud rate:", baud_rate)

    for port, desc, hwid in sorted(ports):
        print("port:", port, "\ndesc:", desc, "\nhwid:", hwid, "\n")

    print("=====================================")
    # Start Serial communication with the board
    isSerialPortWorking = False
    isSerial2Working = False
    if len(ports) > 0:
        # port = ports[0].device
        print("select the port to communicate with the grabber board:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device}")

        port1 = input("enter the port index: ")
        if port1.isdigit():
            port1 = ports[int(port1)].device
        else:
            print("invalid port number. exiting...")
            exit(0)

        print("select the port to communicate with the arm controller:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device}")

        port2 = input("enter the port index: ")
        if port2.isdigit():
            port2 = ports[int(port2)].device
        else:
            print("invalid port number. exiting...")
            exit(0)

        serial1 = serial.Serial(port1, baud_rate, timeout=1)
        serial2 = serial.Serial(port2, baud_rate, timeout=1)
        isSerialPortWorking = True
        isSerial2Working = True

    webcam_stream = WebcamStream(stream_id=camera_index, buffer_size=buffer_size)
    webcam_stream.start()

    num_frames_processed = 0
    start = time.time()

    while True:
        if webcam_stream.stopped:
            break

        box_grabbed = False
        frame = webcam_stream.read()

        if box_grabbed:
            # send predefined coordinates to arm computer
            pass
        else:
            try:
                # INFO: processing image
                box_corners_dict, armctrl_dict = get_box_coordinates(
                    frame, model, device, False, False, False, DEBUG=True
                )
                
                print("box corners:", box_corners_dict)
                print("arm control:", armctrl_dict)
                # INFO: detecting correct box position for pickup
                threshold_radius = 100

                annotated_frame = get_image_with_box_corners(frame, box_corners_dict)
                num_frames_processed += 1

                cv2.imshow("frame", annotated_frame)
                

                # TODO: try to do the serial work in separate thread
                if (
                    armctrl_dict["ty"] <= threshold_radius
                    and armctrl_dict["tx"] <= threshold_radius
                ):
                    if isSerialPortWorking:
                        print("box is in the center, proceed to pickup")
                        # send the box coordinates to the arm controller board
                        serial1.write(
                            str(
                                (
                                    # box_corners_dict["top_left"],
                                    # box_corners_dict["top_right"],
                                    # box_corners_dict["bottom_right"],
                                    # box_corners_dict["bottom_left"],
                                    "ready to pick"
                                )
                            ).encode("utf-8")
                        )
                        # Send a newline character to mark the end of the message
                        serial1.write(b"\n")
                else:
                    # sending the coordinates to the arm to get it to point to the center of the box
                    if isSerial2Working:
                        # send the transition coordinates to the arm controller board
                        serial2.write(
                            str(
                                (
                                    armctrl_dict["tx"],
                                    armctrl_dict["ty"],
                                    armctrl_dict["theta"],
                                )
                            ).encode("utf-8")
                        )

                        # send a newline character to mark the end of the message
                        serial2.write(b"\n")

            except ValueError or IndexError:  # No box is detected
                cv2.imshow("frame", frame)
                num_frames_processed += 1

        # BUG: fix keyboard interrupt not working in unix
        if cv2.waitKey(1) == ord("q"):
            if isSerialPortWorking:
                serial1.close()
            break

    elapsed = time.time() - start
    fps = num_frames_processed / elapsed
    print()
    print("=" * 30)
    print("Execution Summary:")
    print(
        f"FPS: {fps:.2f}, Elapsed Time: {elapsed:.2f} seconds, Frames Processed: {num_frames_processed}"
    )
    print("=" * 30)
    print()

    webcam_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
