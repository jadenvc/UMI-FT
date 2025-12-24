import serial.tools.list_ports
from umift.utils.print_utils import debug_print, info_print

def find_usb_device(path=''):
    """
    Find and return USB device path, preferring the one with shorter ID
    like /dev/tty.usbmodem1102
    """
    if path is not '':
        return path
    ports = serial.tools.list_ports.comports()
    usb_devices = [port.device for port in ports if 'usb' in port.device.lower()]
    
    if not usb_devices:
        debug_print("No USB devices found")
        return None
        
    # Sort by length of device name, shortest first
    usb_devices.sort(key=len)
    selected_device = usb_devices[0]
    info_print(f"Found USB device: {selected_device}")
    return selected_device
