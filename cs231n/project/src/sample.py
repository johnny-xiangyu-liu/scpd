# (c) Meta Platforms, Inc. and affiliates. 
import argparse
import asyncio
import gc
import logging
import socket
from datetime import datetime, timedelta

import torch

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

# This function will leak tensors due to the reference cycles.
def simple_leak(tensor_size, gc_interval=None, num_iter=30000, device="cpu"):
    class Node:
        def __init__(self, T):
            self.tensor = T
            self.link = None

    for i in range(num_iter):
        A = torch.zeros(tensor_size, device=device)
        B = torch.zeros(tensor_size, device=device)
        a, b = Node(A), Node(B)
        # A reference cycle will force refcounts to be non-zero, when
        # a and b go out of scope.
        a.link, b.link = b, a
        # Python will eventually gc a and b, but may OOM on the CUDA
        # device before that happens (since python runtime doesn't
        # know about CUDA memory usage).

        # Since implicit gc is not called frequently enough due to
        # generational gc, adding an explicit gc is necessary as Python
        # runtime does not know about CUDA memory pressure.
        # https://en.wikipedia.org/wiki/Tracing_garbage_collection#Generational_GC_(ephemeral_GC)
        if gc_interval and i % int(gc_interval) == 0:
            gc.collect()

async def awaitable_leak(
    tensor_size, gc_interval=None, num_iter=100000, device="cpu"
):
    class AwaitableTensor:
        def __init__(self, tensor_size, device) -> None:
            self._tensor_size = tensor_size
            self._device = device
            self._tensor = None

        def wait(self) -> torch.Tensor:
            self._tensor = torch.zeros(self._tensor_size, device=self._device)
            return self._tensor

    class AwaitableTensorWithViewCallBack:
        def __init__(
            self,
            tensor_awaitable: AwaitableTensor,
            view_dim: int,
        ) -> None:
            self._tensor_awaitable = tensor_awaitable
            self._view_dim = view_dim
            # Add a view filter callback to the tensor.
            self._callback = lambda ret: ret.view(-1, self._view_dim)

        def wait(self) -> torch.Tensor:
            return self._callback(self._tensor_awaitable.wait())

    for i in range(num_iter):
        # Create an awaitable tensor
        a_tensor = AwaitableTensor(tensor_size, device)

        # Apply a view filter callback on the awaitable tensor.
        AwaitableTensorWithViewCallBack(a_tensor, 4).wait()

        # a_tensor will go out of scope.

        if gc_interval and i % int(gc_interval) == 0:
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A memory_leak binary instance")
    parser.add_argument(
        "--gc_collect_interval",
        default=None,
        help="Explicitly call GC every given interval. Default is off.",
    )
    parser.add_argument(
        "--workload",
        default="simple",
        help="Toggle which memory leak workload to run. Options are simple, awaitable.",
    )
    parser.add_argument(
        "--warn_tensor_cycles",
        action="store_true",
        default=False,
        help="Toggle whether to enable reference cycle detector.",
    )
    args = parser.parse_args()

    if args.warn_tensor_cycles:
        from tempfile import NamedTemporaryFile

        from torch.utils.viz._cycles import observe_tensor_cycles

        logger.info("Enabling warning for Python reference cycles for CUDA Tensors.")

        def write_and_log(html):
            with NamedTemporaryFile("w", suffix=".html", delete=False) as f:
                f.write(html)
                logger.warning(
                    "Reference cycle includes a CUDA Tensor see visualization of cycle %s",
                    f.name,
                )

        observe_tensor_cycles(write_and_log)
    else:
        # Start recording memory snapshot history
        start_record_memory_history()

    # Run the workload with a larger tensor size.
    # For smaller sizes, we will not CUDA OOM as gc will kick in often enough
    # to reclaim reference cycles before an OOM occurs.
    size = 2**26  # 256 MB
    try:
        if args.workload == "awaitable":
            size *= 2
            logger.info(f"Running tensor_size: {size*4/1024/1024} MB")
            asyncio.run(
                awaitable_leak(tensor_size=size, gc_interval=args.gc_collect_interval)
            )
        elif args.workload == "simple":
            logger.info(f"Running tensor_size: {size*4/1024/1024} MB")
            simple_leak(tensor_size=size, gc_interval=args.gc_collect_interval)
        else:
            raise Exception("Unknown workload.")
    except Exception:
        logger.exception(f"Failed to allocate {size*4/1024/1024} MB")

    # Create the memory snapshot file
    export_memory_snapshot()

    # Stop recording memory snapshot history
    stop_record_memory_history()