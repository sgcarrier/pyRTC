import serial
import time

baud_rate = 9600  # whatever baudrate you are listening to
com_port1 = 'COM2'  # replace with your first com port path
com_port2 = 'COM7'  # replace with your second com port path

ComRead_timeout = 0.1   # Read timeout to avoid waiting while there is no data on the buffer
ComWr_timeout = 0.1     # Write timeout to avoid waiting in case of write error on the serial port

log = open('log.txt', 'a+')     # Open our log file, to put read data

From_PC_To_Device = True    # this variable is used to specify which port we're gonna read from

listener = serial.Serial(port=com_port1, baudrate=baud_rate, timeout=ComRead_timeout,
                         write_timeout=ComWr_timeout)

forwarder = serial.Serial(port=com_port2, baudrate=baud_rate, timeout=ComRead_timeout,
                          write_timeout=ComWr_timeout)

while 1:
    while (listener.inWaiting()) and From_PC_To_Device:
        serial_out = listener.readline()
        localtime = time.asctime(time.localtime(time.time()))
        Msg = "PC " + localtime + " " + serial_out
        Msg += "\n"
        log.write(Msg)
        print(serial_out)  # or write it to a file
        forwarder.write(serial_out)
    else:
        From_PC_To_Device = False

    while (forwarder.inWaiting()) and not From_PC_To_Device:
        serial_out = forwarder.readline()
        localtime = time.asctime(time.localtime(time.time()))
        Msg = "DEVICE " + localtime + " " + serial_out + "\n"
        log.write(Msg)
        print(serial_out)  # or write it to a file
        listener.write(serial_out)
    else:
        From_PC_To_Device = True
