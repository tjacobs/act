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
            numbers = line.split(',')
            
            # Check if the line contains exactly two numbers
            if len(numbers) == 2:
                try:
                    num1 = int(numbers[0])
                    num2 = int(numbers[1])
                    return [num1, num2, 0]
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
    print(f"{num1}, {num2}")
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


def find_serial_port(pattern):
    ports = glob.glob(pattern)
    if ports: return ports[0]
    else:     raise IOError(f"No serial port found for {pattern}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial port read/write utility")
    parser.add_argument('--play', action='store_true', help="Just play back input file")
    parser.add_argument('--file', type=str,            help="Input file to write to the serial port")
    parser.add_argument('--plot', action='store_true', help="Plot Num1 and Num2 over time")
    args = parser.parse_args()
    main(args.play, args.file, args.plot)
