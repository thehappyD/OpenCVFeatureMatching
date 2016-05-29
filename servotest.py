import time
import smbus
import time

bus = smbus.SMBus(1)

#The arduino address
address = 0x04;


def writeNumber(value):
    bus.write_byte(address, value)
    # bus.write_byte_data(address, 0, value)
    return -1

def readNumber():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)
    return number


#Pan-tilt servo ids for the Arduino;
tiltChannel = 0;
panChannel = 1;

#Variables to track servo position
tiltServoPosition = 90;
panServoPosition = 90;

#Degree of change
stepSize = 1;

#Send initial pan/tilt angles to the arduino to make it look forward
print "Setting the tiltChannel";
bus.write_byte(address, tiltChannel);

print "Setting the tiltServoPosition to ",(tiltServoPosition);
bus.write_byte(address, tiltServoPosition);

print "Setting the panChannel";
bus.write_byte(address, panChannel);

print "Setting the panServoPosition to ",(panServoPosition);
bus.write_byte(address, panServoPosition);

while(1):
    channel = input("Enter the channel");
    position = input("Enter angle between 1 and 180");

    if(channel == 0):
        print "Setting the tiltChannel";
        bus.write_byte(address, tiltChannel);
        tiltServoPosition=position;
        print "Setting the tiltServoPosition to ",(tiltServoPosition);
        bus.write_byte(address, tiltServoPosition);

    elif(channel == 1):
        print "Setting the panChannel";
        bus.write_byte(address, panChannel);
        panServoPosition=position;
        print "Setting the panServoPosition to ",(panServoPosition);
        bus.write_byte(address, panServoPosition);
