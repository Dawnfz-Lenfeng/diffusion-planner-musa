import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group
import subprocess

from diffusion_planner.utils import accelerator


def ddp_setup_universal(verbose=False, args=None):
       if args.ddp == False:
              print(f"do not use ddp, train on GPU 0")
              return 0, 0, 1

       # cuda | musa | cpu (musa = Moore Threads S5000, torch_musa + mccl)
       accel = accelerator.detect_backend(getattr(args, 'device', 'cuda'))

       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
              rank = int(os.environ["RANK"])
              world_size = int(os.environ['WORLD_SIZE'])
              gpu = int(os.environ['LOCAL_RANK'])
              # torchrun (incl. --standalone) already provides MASTER_ADDR/MASTER_PORT;
              # only fall back to the fixed port when they are absent, otherwise we can
              # collide with a leftover store and hang the rendezvous forever.
              if 'MASTER_ADDR' not in os.environ:
                     os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
                     os.environ["MASTER_ADDR"] = "localhost"
       elif 'SLURM_PROCID' in os.environ:
              rank = int(os.environ['SLURM_PROCID'])
              gpu = rank % accelerator.device_count(accel)
              world_size = int(os.environ['SLURM_NTASKS'])
              node_list = os.environ['SLURM_NODELIST']
              num_gpus = accelerator.device_count(accel)
              addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
              os.environ['MASTER_PORT'] = str(args.port)
              os.environ['MASTER_ADDR'] = addr
       else:
              print("Not using DDP mode")
              return 0, 0, 1

       os.environ['WORLD_SIZE'] = str(world_size)
       os.environ['LOCAL_RANK'] = str(gpu)
       os.environ['RANK'] = str(rank)

       accelerator.set_device(accel, gpu)
       dist_backend = accelerator.dist_backend(accel)
       dist_url = "env://"
       print('| distributed init (rank {}): {}, {} {}'.format(rank, dist_url, accel, gpu), flush=True)
       init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       return rank, gpu, world_size


def setup_for_distributed(is_master):
       """
       This function disables printing when not in master process
       """
       import builtins as __builtin__
       builtin_print = __builtin__.print

       def print(*args, **kwargs):
              force = kwargs.pop('force', False)
              if is_master or force:
                     builtin_print(*args, **kwargs)

       __builtin__.print = print


def get_world_size():
       if not is_dist_avail_and_initialized():
              return 1
       return dist.get_world_size()


def get_rank():
       if not is_dist_avail_and_initialized():
              return 0
       return dist.get_rank()


def get_model(model, use_ddp):
       if use_ddp:
              return model.module
       else:
              return model


def is_dist_avail_and_initialized():
       if not dist.is_available():
              return False
       if not dist.is_initialized():
              return False
       return True



def reduce_and_average_losses(loss_dict, device):
       torch.distributed.barrier()
       world_size = dist.get_world_size()
       for key in loss_dict.keys():
              loss_tensor = torch.tensor([loss_dict[key].item()]).to(device)
              dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
              loss_dict[key] = loss_tensor.item() / world_size
       return loss_dict