# ok
# 数据集和数据加载的参数
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# 源数据集的参数
# 源数据集为：MNIST数据集
src_dataset = "MNIST"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# 目标数据集的参数
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# 设置模型的参数
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# 训练网络的参数
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 1
save_step_pre = 100
num_epochs = 2000
log_step = 100
save_step = 100
manual_seed = None

# 优化模型的参数
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
