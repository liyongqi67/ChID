import numpy as np
import pickle

class Vocabulary:
    def __init__(self):
        with open("/home/share/liyongqi/ChID/2id.pkl", 'rb') as f:
            idiom2id = pickle.load(f)
            word2id = pickle.load(f)

        self.idiom2id = idiom2id

        self.word2id  = word2id


    def tran2id(self, token, is_idiom=False):
        if is_idiom:
            return self.idiom2id[token]
        else:
            if token in self.word2id:
                return self.word2id[token]
            else:
                return self.word2id["<UNK>"]



def caculate_acc(original_labels, pred_labels):
    """
    :param original_labels: look like [[list1], [list2], ...], num of list == batch size
        length of each list is not determined, for example, it may be 3, 4, 6
    :param pred_labels: [[pred_list1], [pred_list2], ...]
        length of each pred_list is padding to 10, we just care the first several ones
    :return: an array, looks like
    """

    acc_blank = np.zeros((2, 2), dtype=np.float32)
    acc_array = np.zeros((2), dtype=np.float32)

    for id in range(len(original_labels)): # batch_size
        ori_label = original_labels[id]
        pre_label = list(pred_labels[id])


        x_index = 0 if len(ori_label) == 1 else 1

        for real, pred in zip(ori_label, pre_label):
            acc_array[1] += 1
            acc_blank[x_index, 1] += 1

            if real == pred:
                acc_array[0] += 1
                acc_blank[x_index, 0] += 1

    return acc_array, acc_blank

