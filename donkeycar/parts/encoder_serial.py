"""
Serial Encoder
"""

from datetime import datetime
from donkeycar.parts.teensy import TeensyRCin
import time
import serial

class SerialEncoder():
    __ser = False
    __debug = False
    __on = False
    __rpm = 0

    def __init__(self, ser_port='/dev/ttyACM0', baud=9600, debug=False):
        self.__ser = serial.Serial(port=ser_port, baudrate=baud, timeout=1)
        self.__debug = debug
        self.__on = True

    def update(self):
        # keep looping infinitely until the thread is stopped
        while(self.__on):

            # read the current RPM
            if self.__ser.in_waiting > 0:
                self.__rpm = int(self.__ser.readline().decode('utf-8'))

            #console output for debugging
            if(self.__debug):
                print('RPM:', self.__rpm)

    def run_threaded(self):
        return self.__rpm

    def shutdown(self):
        # indicate that the thread should be stopped
        print('Stopping Serial Encoder')
        self.__on = False
        self.__ser.close()
        time.sleep(.5)