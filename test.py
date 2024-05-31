import os, torch; print(os.environ["LOCAL_RANK"]); torch.distributed.init_process_group("nccl")
