import pickle
import time

import robotmq as rmq

client = rmq.RMQClient(
    client_name="asynchronous_client", server_endpoint="ipc:///tmp/feeds/0"
)

# client = rmq.RMQClient("asynchronous_client", "tcp://localhost:5555")
print("Client created")

while True:
    retrieve_start_time = time.time()
    retrieved_data, timestamp = client.pop_data(topic="test_checkpoints", n=-1)
    retrieve_end_time = time.time()

    if retrieved_data:
        received_data = pickle.loads(retrieved_data[0])

        print(
            f"Data size: {len(retrieved_data[0]) / 1024**2:.3f}MB. retrieve: {retrieve_end_time - retrieve_start_time:.4f}s, load: {time.time() - retrieve_end_time:.4f}s)"
        )

        statedict = received_data.state_dict()
        for key, value in statedict.items():
            print(key, ": ", value.shape)
        # # use the received payload
        # cfg = received_data['cfg']
        # print("dataset_path:", cfg.task.dataset.dataset_path)

        # cls = hydra.utils.get_class(cfg._target_)
        # workspace = cls(cfg)
        # workspace: BaseWorkspace
        # workspace.load_payload(received_data, exclude_keys=None, include_keys=None)

        # policy = workspace.model
        # if cfg.training.use_ema:
        #     policy = workspace.ema_model
        # policy.num_inference_steps = cfg.policy.num_inference_steps # DDIM inference iterations

        # device = torch.device('cpu')
        # policy.eval().to(device)
        # policy.reset()

        # # use normalizer saved in the policy
        # sparse_normalizer, dense_normalizer = policy.get_normalizer()

        # shape_meta = cfg.task.shape_meta
        # print("shape_meta:", shape_meta)
        break

    print("No data retrieved ...")
    time.sleep(0.2)
