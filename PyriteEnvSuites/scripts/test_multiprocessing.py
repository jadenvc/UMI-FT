# define a class that initializes a timed loop which prints the process id and the time every 1 second
import multiprocessing
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from hardware_interfaces.workcell.test.python import (
    test_server_pybind as test_server,
)
from PyriteUtility.data_pipeline.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from PyriteUtility.data_pipeline.shared_memory.shared_memory_util import (
    ArraySpec,
)

# class TestServer:
#     def __init__(self):
#         self.process = multiprocessing.Process(target=self.run)
#         self.process.start()
#         self.data = np.random.rand(7)
#         self.timestamp = time.time()

#     def run(self):
#         while True:
#             print(f"TestServer Process ID: {os.getpid()} Time: {time.time()}")
#             self.data = np.random.rand(7)
#             self.timestamp = time.time()
#             time.sleep(1)

#     def get_data(self):
#         return {
#             "ts_pose_fb": self.data,
#             "timestamp_s": self.timestamp,
#         }


class Env:
    def __init__(self, shm_manager: SharedMemoryManager, time0: float):
        self.server = test_server.TestServer()
        self.shm_manager = shm_manager
        self.time0 = time0

        array_spec_ts_pose_fb = ArraySpec(
            name="ts_pose_fb",
            shape=(7,),
            dtype=np.float64,
        )
        array_spec_timestamp = ArraySpec(
            name="timestamp_s",
            shape=(),
            dtype=np.float64,
        )
        self.ring_buffer = SharedMemoryRingBuffer(
            shm_manager=shm_manager,
            array_specs=[array_spec_ts_pose_fb, array_spec_timestamp],
            get_max_k=10,
            get_time_budget=0.1,
            put_desired_frequency=5,
        )

        # self.process_ts_pose_fb = multiprocessing.Process(
        #     target=self.loop_generate_data
        # )
        # self.process_ts_pose_fb.start()

    # def loop_generate_data(self):
    #     while True:
    #         print(
    #             f"[loop_generate_data] Env Process ID: {os.getpid()} Time: {time.time()}"
    #         )

    #         new_frame = {
    #             "ts_pose_fb": np.random.rand(7),
    #             "timestamp_s": self.server.get_test(),
    #         }
    #         self.ring_buffer.put(new_frame)
    #         time.sleep(0.1)

    def get_data(self, seconds: int):
        # # deep copy data from the ring buffers
        # data = self.ring_buffer.get_last_k(sample_sizes)
        # time.sleep(0.5)
        # return data
        return self.server.get_test(seconds)


def test_func(env):
    while True:
        data = env.get_data(3)
        print("Test_func: ", data)


if __name__ == "__main__":
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    time0 = time.time()
    env = Env(shm_manager=shm_manager, time0=time0)
    time.sleep(2)
    print("---------- Starting test_func ----------------")

    process = multiprocessing.Process(target=test_func, args=(env,))
    process.start()
    for i in range(100):
        data = env.get_data(0.2)
        print("Main thread: ", data)
    print("---------- Ending test_func ----------------")
    process.join()
