import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # ** INSTALL **
# # 1. conda create -n py310 python=3.10
# # 2. conda activate py310
# # 3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# # 4. python -m pip install "tensorflow=2.10"


# Setting up mixed precision
