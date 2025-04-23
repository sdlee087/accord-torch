# accord-torch

Row-separable version of ACCORD graphical selection with CUDA availability. This code can be run in single-process mode or multi-process mode.

You can start the program as:

```
python main_sp.py --config cfg/config.yaml --log main.log
```

- main_sp.py (recommended for high-dimensional data): iterate omega is in sparse matrix format.
- main.py: iterate omega is in dense matrix format.
- main_sp_block.py: iterate omega is in sparse matrix format, assigning different lambda penalty for different column is possible.

### configuration
Configuration with **bold faces** does not have a default value, and thus has to be specified.

- **data_file**: input Numpy file to apply graphical selection. The matrix has to be in n-by-p format.
- **out_file**: name of the ouput omega. In multi-process mode, label number is automatically attached.
<br>

- l1: Lambda penalty to be appiled. Multiple values are possible.
- tau_init: Initial step size to start.
- eps: Convergence tolerance.
- beta: Rate of step size decrease applied for line search.
<br>

- max_outer: Maximum iteration number for outer loop.
- max_inner: Maximum iteration number for inner loop.
<br>

- resume: npz(npy) file for the iterate omega to resume. If not specified, the iterate will start from an identity matrix.
- CUDA: Option to use CUDA device.
- float64: Option to use float64. If False, the iterate will be computed in float32.
<br>

- log_interval: The outer iteration log (including the value of the object function) will appear for each times of number specified in this option, and the iteration will stop if the object function is not progressing. Default is 1.

#### configurations for multi-process mode
If the following configurations are not specified, the program will run in single-process mode.

- total_process: Number of total device (process) to be simultaneously run.
- row_divide: In multiprocess-mode, the partial omega will be divided by number of this option.
- row_min: Start row index for the range of partial omega.
- row_max: End row index for the range of partial omega.
<br>

- resume_from_whole: Whole npz omega matrix file to start the iteration.
- label_start: The label for each partial omega will start from this option. default is 0.
- log_file: log file name to record log on each device.
