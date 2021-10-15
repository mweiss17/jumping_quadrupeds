import os
import sys
import submitit
from submitit.core.utils import CommandFunction

# create the submitit executor for creating and managing jobs
ex = submitit.AutoExecutor(folder=os.path.join(sys.argv[1], "Logs"))

# setup the executor parameters based on the cluster location
if ex.cluster == "slurm":
    ex.update_parameters(
        mem_gb=16,
        cpus_per_task=12,
        timeout_min=1000,
        tasks_per_node=1,
        nodes=1,
        slurm_partition="main",
        gres="gpu:rtx8000:1",
    )
command = ["python3"] + sys.argv[1:]
print(f"running: {command}")

ex.submit(CommandFunction(command))
