import tensorflow as tf
from bert import modeling
import os

# 这里是下载下来的bert配置文件
bert_config = modeling.BertConfig.from_json_file("F:/Model/BERT/chinese_L-12_H-768_A-12/bert_config.json")
#  创建bert的输入
#  tf.placeholder()占位符号，相当于先声明一下，在解释性python中不用一上来就运算，避免消耗资源
input_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="segment_ids")

# 创建bert模型
model = modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
)

#bert模型参数初始化的地方
init_checkpoint = "F:/Model/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt"
use_tpu = False
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型

(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())