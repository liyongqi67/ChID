# -*- coding: utf-8 -*-
import tensorflow as tf

def get_flags():
    flags = tf.app.flags
    flags.DEFINE_float("lr", 0.001, "learning rate")
    flags.DEFINE_integer("batch_size", 32, "batch size")
    flags.DEFINE_boolean("is_train", False, "training or testing a model")
    flags.DEFINE_string("train_dir", "train", "path to save model")
    flags.DEFINE_string("model", "ar", "choose baseline")
    flags.DEFINE_string("gpu", "0", "choose gpu")
    flags.DEFINE_string('test_file', "/home/share/liyongqi/ChID/raw_data/dev.txt", '')
    flags.DEFINE_string('pred_file', 'pred_result.txt', '')

    FLAGS = flags.FLAGS

    return FLAGS