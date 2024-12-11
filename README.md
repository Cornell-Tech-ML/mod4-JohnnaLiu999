# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


# Task 4.4b

Script for cuda_conv.py (sampling code included): https://github.com/JohnnaLiu999/cuda_conv.py/blob/1bd2ce6e3f86992581494990e98dcd7323b85c86/cuda_conv.py
Colab link: https://colab.research.google.com/drive/1g89yjTwtsuMzHtp5vslfTU4oDnzCF_3G?usp=sharing

Output of Colab for easier reference:
```
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
Conv1D Output Shape: (2, 4, 10)
Conv1D Output Data: [[[-1.95832259e+00  8.69499873e-01 -2.82527834e+00  2.59737191e+00
    8.69481384e-01 -3.16258412e+00 -9.25927071e-01  1.23626150e+00
   -3.92534794e-01 -4.65152891e+00]
  [-3.54050079e+00 -3.53693113e+00 -3.29232483e+00  6.86463987e+00
    5.05873184e-01  1.76110309e+00 -2.31340220e-03  6.39891119e+00
    2.81204666e+00 -7.77783400e-01]
  [-1.37987125e+00  2.49026389e+00  2.44859083e+00  1.64576300e+00
   -2.08667933e-01  1.64528141e+00 -1.64129656e+00 -3.97463929e-01
   -1.09290098e+00 -3.11180444e-01]
  [ 5.10370650e+00  2.55656139e+00 -3.39905351e+00  1.75700457e+00
    3.51100682e+00  1.01134414e+00 -2.60056783e+00  3.69186753e+00
    1.68113682e+00 -3.71822144e+00]]

 [[-7.09846914e-02 -1.24538079e+00  1.62330649e+00 -2.23901144e+00
    4.73917435e-01 -4.41846703e-01 -3.75998809e+00  4.23162596e+00
   -4.76367432e-02  1.19113577e+00]
  [ 1.41979053e+00  4.66736947e-02 -7.73154294e-02  1.71341907e+00
    4.25303131e-01 -2.83572628e+00 -3.02001705e+00 -4.18617691e+00
    1.09404402e+00  1.18234327e-01]
  [ 2.25206693e+00 -9.17076553e-01  4.91015469e-01  1.62622019e+00
    1.10828566e+00  3.08013144e+00  6.59048605e-01 -3.70529783e-01
    5.50863983e+00  2.52434183e-01]
  [ 1.14968477e-01 -1.81273814e+00 -8.15287868e-01 -3.85132664e+00
   -1.28978592e+00 -6.00243742e-01 -2.36703033e+00 -2.38486253e+00
    1.01486786e+00  5.91835853e-01]]]
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
Conv2D Output Shape: (2, 4, 5, 5)
Conv2D Output Data: [[[[-3.21796177e+00 -4.72159061e+00 -1.92152667e+00  3.62169971e+00
    -3.14740628e+00]
   [-3.05263867e+00 -9.67663561e+00  3.10644920e-01 -2.91461401e+00
    -3.31106456e+00]
   [-4.28886141e-01 -4.73995000e+00  2.07801717e+00  5.65781415e+00
    -3.14767950e-01]
   [-2.18909651e+00  2.42879446e+00 -5.39064244e-01  5.15322739e+00
    -1.38312593e-01]
   [ 3.75935815e-01 -2.16756165e-01 -5.82850534e-01  4.04571580e+00
     8.47913010e-01]]

  [[-8.73017300e+00 -3.67043532e+00 -2.36981497e+00  5.08906699e-01
    -5.33867779e-01]
   [ 2.49364882e+00  5.67833728e+00 -3.79299004e+00  6.73646664e+00
    -3.95748085e+00]
   [ 1.78421572e+00  5.65895086e+00 -2.80825552e+00  1.17559063e+01
    -1.08821461e+00]
   [-2.23660648e+00 -2.64678560e-01  2.30982249e+00  5.64419076e+00
     2.48829562e+00]
   [ 1.29903327e+00  1.65989801e-01  2.03231705e+00  7.78320103e-02
     3.87509765e+00]]

  [[-2.54754839e+00 -8.42964021e+00 -1.09889811e+01 -1.90838836e+00
    -1.38788641e+00]
   [-7.67619634e+00 -1.10178996e+01  9.25131594e-01  4.38963508e-01
     5.87283510e-01]
   [-6.93349205e-01  3.34247092e+00  5.35826776e+00 -6.11481373e-01
    -2.76577371e+00]
   [-4.23662221e+00 -2.92984761e+00 -4.21789993e+00 -6.87480737e-01
     2.28230356e+00]
   [-2.12135523e+00  6.46790215e-02 -7.65519421e-01  2.89402641e+00
     2.54050836e+00]]

  [[ 2.06733585e+00 -3.99654455e+00  4.64505224e+00 -1.66234013e+00
     2.16673812e+00]
   [ 1.54402710e+00  1.42543313e+01  2.20054976e+00  9.55388611e+00
     1.36351503e+00]
   [-2.55868517e+00  9.12995102e+00 -6.64457555e+00  3.38882568e-01
     2.13775537e+00]
   [ 3.85229422e+00 -1.01202076e+00 -3.99229660e+00  4.62821827e+00
     4.01556686e+00]
   [ 4.42166714e-02  2.82280259e+00  5.03215430e+00 -2.91574513e+00
    -1.09891203e+00]]]


 [[[ 5.46931891e-01  2.67188975e+00  3.28286231e+00  3.94092494e+00
    -5.12685945e+00]
   [-4.38936026e+00  1.17627036e+00  2.09871677e-01 -3.76180270e+00
    -8.50821557e-01]
   [-3.15307824e+00 -6.76063845e+00 -5.55339886e-01  8.13768470e+00
     5.44505708e+00]
   [-1.20244770e+00 -2.80665300e+00  1.16777714e+00  4.28571300e+00
    -2.05644438e+00]
   [ 3.91035216e+00  4.61411502e-01  4.43385239e-01 -1.14459520e+00
    -1.79272773e+00]]

  [[ 1.55379380e+00  3.24184578e+00  8.02268329e+00  2.21834690e+00
     2.67791373e+00]
   [ 1.10325528e+00 -5.11041325e+00  1.20049047e+00 -3.87874891e+00
    -1.38238701e+00]
   [-1.06330482e+00 -2.54540374e+00 -5.11005139e+00 -2.99081103e+00
    -8.98187342e-01]
   [ 3.02494738e+00  3.86349933e-01 -2.22886207e+00  3.82801736e+00
    -5.54559573e-01]
   [ 1.63000124e+00  5.01382742e+00  1.54916517e+00  3.73979197e+00
     5.64729445e-02]]

  [[-1.47785661e+01  5.98001960e+00  6.26011866e-01  2.71940212e+00
     2.90710485e-03]
   [-5.72869865e+00 -6.91761492e-02  2.98823270e+00 -1.13866398e+00
     3.80506578e+00]
   [-2.62112338e+00 -6.85083968e-01  3.66032299e+00 -1.90978556e+00
    -8.00099068e-01]
   [-2.48339488e+00  4.20068435e+00  1.65120780e-01 -2.80135235e+00
    -3.63734876e+00]
   [ 4.12089555e+00  1.20543892e+00 -2.52238269e-02  2.23301530e-01
     1.76813549e+00]]

  [[ 2.26395902e+00  1.05659629e+00 -1.76197745e+00 -3.46036951e+00
     9.84532813e-01]
   [ 4.80852235e+00  4.05791518e-01 -9.35791889e+00  5.10817540e+00
     8.31513210e-02]
   [ 5.22573120e+00  2.15996264e-01  1.01424213e+01 -8.54832212e+00
    -4.08616005e+00]
   [ 1.06849448e+01  1.90824798e+00 -5.37302950e+00 -2.75834403e+00
     2.30441335e+00]
   [-4.88272892e+00  2.64164776e+00 -2.94047892e+00  5.67586723e+00
     1.43982204e+00]]]]
```

# Task 4.5
Kindly see my training log at file sentiment.txt and mnist.txt in this repo. Both satisfy the accuracy requirements. 
