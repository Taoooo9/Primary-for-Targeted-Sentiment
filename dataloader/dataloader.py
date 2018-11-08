import re
import torch
from torch.autograd import Variable

def readdata(file):
    src_word = []
    src = []
    label_word = []
    label = []
    count = 0
    start = -1
    end = -1
    tag = ''
    first = True
    text = open(file, encoding='utf-8').readlines()
    temp = text[0]
   # with open(file, encoding="utf-8") as f:
    for line in text[1:]:
        if temp is not '\n':
            count += 1
            temp = temp.strip()
            word = re.findall(r'^(\S+)', temp)
            target = re.findall(r'(\S+)$', temp)
            if target[-1] != 'o' and first is True:
                tag = re.split(r'\S-', target[-1])
                tag = tag[-1]
                start = count
                first = False
            if (target[-1] == 'o' and first is False) or (line == '\n' and first is False):
                if line == '\n':
                    end = count
                    first = True
                else:
                    end = count - 1
                    first = True
            src.append(word[-1])
            label.append(target[-1])
            temp = line
        elif temp == '\n':
            temp = line
            src_word.append([src, tag, (start-1, end-1)])
            label_word.append([label, tag])
            count = 0
            src = []
            label = []
    return src_word, label_word

def pair_data_variable(batch, src_vocab, tar_vocab, config):
    batch_size = len(batch[0])
    start = batch[2][0]
    end = batch[2][1]
    src_martix = Variable(torch.LongTensor(batch_size, 1).zero_(), requires_grad=False)
    tar_martix = Variable(torch.LongTensor(1).zero_(), requires_grad=False)
    sentence = src_vocab.w2i(batch[0])
    for idx, word in enumerate(sentence):
        src_martix.data[idx][0] = word
    tar_martix.data[0] = tar_vocab.w2i(batch[1])

    return src_martix, tar_martix, start, end, batch_size

if __name__ == '__main__':
    file = '../data/Z_data/all.conll.train'
    a, b = readdata(file)

