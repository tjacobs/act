#!/usr/bin/env python3

# You can run this file ./robot/recorder.py to record robot movement to a text file
# and then hit CTRL-C and it will play that movement back on the robot
# Like: ./robot/recorder.py
# Or:   ./robot/recorder.py --file a.txt --play --plot

import serial
import time
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2


# USB serial port of robot
port_pattern = '/dev/tty.usbmodem*'
baudrate = 115200


# Standalone running of this file records robot movement and plays it back
def main(play, file, plot):
    port = find_serial_port(port_pattern)
    if not file: file = 'output.txt'
    serial = open_serial_port(port, baudrate)
    if not play: read_from_serial_port(serial, file)
    write_to_serial_port(serial, file, plot)
    close_serial_port(serial)


class Recorder:
    def __init__(self):
        # Open serial port to connect to robot
        port = find_serial_port(port_pattern)
        self.serial = open_serial_port(port, baudrate)

    def get_joint_positions(self):
        # Check serial port
        if self.serial is None:
            print("Serial port not open")
            return None

        # Read joint positions from robot
        pos_joints = read_robot_joints(self.serial)
        return pos_joints

    def set_joint_positions(self, joints):
        # Check serial port
        if self.serial is None:
            print("Serial port not open")
            return

        # Write joint positions to robot
        write_robot_joints(self.serial, joints)


    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        # Test
        self.joint_timestamps = [1, 2, 5]
        self.action_timestamps = [1, 2, 10]

        # Get frequency
        joint_freq = 1 / dt_helper(self.joint_timestamps)
        action_freq = 1 / dt_helper(self.action_timestamps)
        print(f'{joint_freq=:.2f}\n{action_freq=:.2f}\n')


def open_serial_port(port, baudrate):
    ser = None
    try:
        # Open the serial port
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baudrate.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    return ser


def close_serial_port(ser):
        if ser is not None:
            ser.close()
            print("Serial port closed.")


def read_robot_joints(ser):
    # Give it up to one second to read one line
    for i in range(1000):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            numbers = line.split(', ')
            
            # Check if the line contains numbers
            if len(numbers) == 6:
                try:
                    num1 = int(numbers[0])
                    num2 = int(numbers[1])
                    num3 = int(numbers[2])
                    num4 = int(numbers[3])
                    num5 = int(numbers[4])
                    num6 = int(numbers[5])
                    #print("Nums: ", num1, num2, num3, num4, num5, num6)
                    return [num1, num4, 0]
                except ValueError:
                    print(f"Invalid numbers: {line}")
            else:
                print(f"Invalid line format: {line}")

        # Sleep 1ms
        time.sleep(0.001)

    # Error
    print("No serial data")
    return None


def write_robot_joints(ser, values):
    [num1, num2, num3] = values
    print(f"{int(num1)}, {int(num2)}")
    byte_array = bytearray([0x61, 0x1, 0x1, int(num1)//10, int(num2)//10, 0x0, 0x0, 0x0])
    ser.write(byte_array)


def read_from_serial_port(ser, output_file):
    try:
        # Open the output file
        with open(output_file, 'w') as file:
            while True:
                [num1, num2, num3] = read_robot_joints(ser)
                file.write(f"{num1}, {num2}\n")
                print(f"Read: {num1}, {num2}")

                # Sleep 5ms
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("Done reading.")


def write_to_serial_port(ser, input_file, plot):
    # Initialize lists for plotting
    timestamps = []
    num1_data = []
    num2_data = []

    # Open file
    try:
        # Read the input file
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip()
                numbers = line.split(',')
                
                # Check if the line contains exactly two numbers
                if len(numbers) == 2:
                    try:
                        # Get numbers
                        num1 = int(numbers[0])
                        num2 = int(numbers[1])
                        
                        # Write packet
                        write_robot_joints(ser, [num1, num2, 0])
                        print(f"Wrote: {num1}, {num2}")

                        # Append data for plotting
                        timestamp = datetime.datetime.now()
                        timestamps.append(timestamp)
                        num1_data.append(num1)
                        num2_data.append(700 - num2)

                    except ValueError:
                        print(f"Invalid numbers: {line}")
                else:
                    print(f"Invalid line format: {line}")
                
                # Sleep 5ms
                time.sleep(0.005)
    except IOError as e:
        print(f"I/O error: {e}")
    except KeyboardInterrupt:
        print("Done writing.")

    # Plot if requested
    if plot:
        plt.figure()
        plt.plot(timestamps, num1_data, label='Gripper', marker='X')
        plt.plot(timestamps, num2_data, label='Elbow', marker='.')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Plot')
        plt.legend()
        plt.grid(True)
        plt.show()


class ImageRecorder:
    def __init__(self, camera_names):
        self.camera_names = camera_names
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)

        # Open USB camera
        self.open_camera()

    def get_image(self, cam_name):
        # Get image from camera
        image = self.get_camera_frame()
        #print(image.shape)

        # Test blank image
        if False:
            width, height = 640, 480
            b, g, r = 0x3E, 0x88, 0xE5  # Orange
            image = np.zeros((height, width, 3), np.uint8)
            image[:, :, 0] = b
            image[:, :, 1] = g
            image[:, :, 2] = r
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

        # Save image
        setattr(self, f'{cam_name}_image', image)

    def get_images(self):
        image_dict = dict()
        cam_name = "cam_1"
        #for cam_name in self.camera_names: 
        image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def open_camera(self):
        # Open the USB camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open camera")
            return None

    def get_camera_frame(self):
        # Get camera frame
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Failed to capture camera image")
            return None

        # Display the frame
        #cv2.imshow('Frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'): return frame

        # Return frame
        return frame

    def close_camera(self):
        # Release the camera and close windows
        self.camera.release()
        cv2.destroyAllWindows()


def find_serial_port(pattern):
    ports = glob.glob(pattern)
    if ports: return ports[0]
    else:     raise IOError(f"No serial port found for {pattern}")


def record_video_to_numpy_array(duration=10, fps=30):
    # Open the USB camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Get the resolution
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = duration * fps

    # Create the matrix
    video_array = np.empty((total_frames, frame_height, frame_width, 3), dtype=np.uint8)

    # Get a frame
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Store the frame in the numpy array
        video_array[frame_count] = frame

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press 'q' to exit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_count += 1

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    return video_array


# Test
if False:
    video_array = record_video_to_numpy_array(duration=10, fps=30)
    print(f"Recorded video array shape: {video_array.shape}")

# Test
if False:
    image_recorder = ImageRecorder("cam_1")
    image_recorder.get_image("cam_1")
    images = image_recorder.get_images()
    print(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial port read/write utility")
    parser.add_argument('--play', action='store_true', help="Just play back input file")
    parser.add_argument('--file', type=str,            help="Input file to write to the serial port")
    parser.add_argument('--plot', action='store_true', help="Plot Num1 and Num2 over time")
    args = parser.parse_args()
    main(args.play, args.file, args.plot)
