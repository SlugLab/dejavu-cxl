# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_device_count():
    return torch.cuda.device_count()


def get_device():
    return torch.cuda.current_device()


class ModelParallelGroup:

    def __init__(self, tensor_para_size: int, pipeline_para_size: int, prompt_world_size=0):
        #self.check_parallel_size_validity(tensor_para_size, pipeline_para_size)

        rank = get_rank()
        device = rank % get_device_count()
        torch.cuda.set_device(device)

        # tp: tensor parallel, pp: pipeline parallel.
        self.tp_size = tensor_para_size
        self.tp_rank = rank % self.tp_size
        self.pp_size = pipeline_para_size

        if rank >= prompt_world_size:
            rank -= prompt_world_size

        self.pp_rank = rank // self.tp_size

    @staticmethod
    def check_parallel_size_validity(tensor_para_size, pipeline_para_size):
        world_size = tensor_para_size * pipeline_para_size #get_world_size()
        if world_size != tensor_para_size * pipeline_para_size:
            raise ValueError(
                f'[ERROR] Invalid tensor/pipeline parallel configuration. '
                f'world_size({world_size}) != tensor_para_size({tensor_para_size})'
                f' * pipeline_para_size({pipeline_para_size})')

    @property
    def is_pipeline_first(self):
        return self.pp_rank == 0

    @property
    def is_pipeline_last(self):
        return self.pp_rank == self.pp_size - 1


_model_para_group = None


def is_model_parallel_initailized():
    return _model_para_group is not None


def initialize_model_parallel(world_size: int, rank: int, backend="mpi"):
    import os
    print(f"Inside _model_para_group, model_para_group is {_model_para_group}")
    assert torch.cuda.is_available()
    assert not is_model_parallel_initailized(), \
        f'parallel group has been already initialized.'

    # Detect MPI environment and override rank/world_size if running under MPI
    # OpenMPI sets OMPI_COMM_WORLD_RANK and OMPI_COMM_WORLD_SIZE
    # MPICH sets PMI_RANK and PMI_SIZE
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        print(f'Detected OpenMPI: rank={rank}, world_size={world_size}')
    elif 'PMI_RANK' in os.environ:
        rank = int(os.environ['PMI_RANK'])
        world_size = int(os.environ['PMI_SIZE'])
        print(f'Detected MPICH: rank={rank}, world_size={world_size}')

    print(f'Initializing tensor and pipeline parallel, world size is {world_size}, rank is {rank}')

    # Set CUDA device based on rank before PyTorch distributed initialization
    # This prevents "Duplicate GPU detected" errors
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f'Set CUDA device to {device} for rank {rank}')

    # Set environment variables for PyTorch distributed initialization
    # This avoids race conditions when using MPI
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_ADDR'] = master_addr
    # Set PROMPT_MASTER_ADDR and TOKEN_MASTER_ADDR for NCCL initialization
    os.environ['PROMPT_MASTER_ADDR'] = os.environ.get('PROMPT_MASTER_ADDR', master_addr)
    os.environ['TOKEN_MASTER_ADDR'] = os.environ.get('TOKEN_MASTER_ADDR', master_addr)
    # Use a unique port to avoid conflicts (based on user ID + 20000)
    default_port = str(20000 + (os.getuid() % 10000))
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', default_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    print(f'Using MASTER_PORT={os.environ["MASTER_PORT"]}')

    # Use env:// init method which is more reliable with MPI
    dist.init_process_group(
        backend=dist.Backend.NCCL,
        init_method='env://'
    )
    print("After init!")

def init_group(tensor_para_size: int,
                       pipeline_para_size: int, prompt_world_size=0):

    _model_para_group = ModelParallelGroup(tensor_para_size, pipeline_para_size, prompt_world_size)


def get_tensor_para_rank():
    if _model_para_group is None:
        return 0
    return _model_para_group.tp_rank


def get_tensor_para_size():
    if _model_para_group is None:
        return 1
    return _model_para_group.tp_size


def get_pipeline_para_rank():
    if _model_para_group is None:
        return 0
    return _model_para_group.pp_rank


def get_pipeline_para_size():
    if _model_para_group is None:
        return 1
    return _model_para_group.pp_size


def is_pipeline_group_first():
    return _model_para_group is None or _model_para_group.is_pipeline_first


def is_pipeline_group_last():
    return _model_para_group is None or _model_para_group.is_pipeline_last


def destroy():
    dist.destroy_process_group()
    _model_para_group = None
