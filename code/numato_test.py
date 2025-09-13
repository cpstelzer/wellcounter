
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:37:51 2024

@author: Stelzer Lab
"""

import serial
import time

# Set up the port and relay number
portName = "COM4"  # Adjust if necessary
relayNum = "1"     # Relay number to control

# Open port for communication
numato = serial.Serial(portName, 19200, timeout=1)

# Turn the relay on
numato.write("relay on {}\n\r".format(relayNum).encode())
print("Relay {} is ON".format(relayNum))

time.sleep(1)  # Wait for 1 second

user_input = input("Press return to end this program")


# Turn the relay off
numato.write("relay off {}\n\r".format(relayNum).encode())
print("Relay {} is OFF".format(relayNum))

# Close the port
numato.close()
