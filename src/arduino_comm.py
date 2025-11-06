"""
Arduino communication module
Placeholder for serial communication with Arduino
"""


try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Note: pyserial not installed. Arduino communication disabled.")
    print("  Install with: pip install pyserial")

import time


def send_to_arduino(command, port='COM3', baudrate=9600):
    """
    Send command to Arduino via serial port
    Command format: "B1=0.20;B2=0.30;B3=0.10;B4=0.10;B5=0.10;B6=0.10;B7=0.10\n"
    
    Args:
        command: String command to send
        port: Serial port (default COM3 for Windows, adjust as needed)
        baudrate: Baud rate (default 9600)
    """
    if not SERIAL_AVAILABLE:
        print("Note: Arduino communication not available (pyserial not installed)")
        print(f"  Would send: {command.strip()}")
        return False
    
    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        
        # Send command
        ser.write(command.encode())
        print(f"OK: Command sent to Arduino: {command.strip()}")
        
        # Optional: Read response
        # response = ser.readline().decode().strip()
        # print(f"Arduino response: {response}")
        
        ser.close()
        return True
    
    except serial.SerialException as e:
        print(f"Warning: Arduino communication error: {e}")
        print("   Make sure Arduino is connected and port is correct")
        return False
    
    except Exception as e:
        print(f"Warning: Error: {e}")
        return False


def test_arduino_connection(port='COM3', baudrate=9600):
    """Test Arduino connection"""
    if not SERIAL_AVAILABLE:
        print("Note: pyserial not installed. Cannot test Arduino connection.")
        return False
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        ser.close()
        print(f"OK: Arduino connection successful on {port}")
        return True
    except Exception as e:
        print(f"Error: Arduino connection failed: {e}")
        return False


if __name__ == '__main__':
    # Test connection
    test_arduino_connection()

