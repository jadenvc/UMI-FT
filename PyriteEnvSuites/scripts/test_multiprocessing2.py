import os
import threading
import time
from array import array


def get_memory_address(arr):
    # Get the address of the underlying data buffer
    address = arr.buffer_info()[0]
    return address


class PIDTimePrinter:
    def __init__(self):
        self._value = array("i", [42])  # 'i' is for signed integer
        self._stop = False
        self._thread = threading.Thread(target=self._print_pid_and_time)
        self._thread.start()

    def stop(self):
        print(
            f"{os.getpid()} m adress: {get_memory_address(self._value)} Stopping PIDTimePrinter"
        )
        self._stop = True
        self._thread.join()

    def _print_pid_and_time(self):
        while not self._stop:
            print(
                f"{os.getpid()} m adress: {get_memory_address(self._value)}, self._stop: {self._stop}"
            )
            time.sleep(1)
        print(
            f"{os.getpid()} m adress: {get_memory_address(self._value)} PIDTimePrinter stopped"
        )


def worker(data, printer):
    i = 0
    while i < 5:
        print(i, ", ", data)
        time.sleep(1)
        i += 1
    printer.stop()


if __name__ == "__main__":
    # printer = PIDTimePrinter()
    # # Only passing necessary data
    # p = Process(target=worker, args=("worker", printer))
    # p.start()
    # p.join()

    data = array("i", [42])  # 'i' is for signed integer
    print(
        f"Parent PID: {os.getpid()}, Data Address: {get_memory_address(data)}, Value: {data.tolist()}"
    )
    data.append(99)
    print(
        f"Parent PID: {os.getpid()}, Data Address: {get_memory_address(data)}, Value: {data.tolist()}"
    )
