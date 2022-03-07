import torch.optim as optim
import time
from pathlib import Path
import torch
import Dataset.alphabets as alphabets

def get_optimizer(model):

    optimimzer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = 0.0001
    )

    return optimimzer

def create_log_folder():
    root_out_dir = Path('output')
    #set up logger
    if not root_out_dir.exists():
        print('=> creating {}'.format(root_out_dir))
        root_out_dir.mkdir()

    dataset = '360CC'
    model = 'crnn'
    time_str = time.struct_time('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_out_dir / dataset / model / time_str / 'checkpoints'

    print('=> creating{}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents = True, exist_ok = True)

    tensorboard_log_dir = root_out_dir / dataset / model / time_str / 'log'
    print('=> creating{}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents = True, exist_ok = True)

    return {'chs_dir':str (checkpoints_output_dir), 'tb_dir':str(checkpoints_output_dir)}

def get_batch_label(dataset, index):
    label = []
    for idx in index:
        label.append(list(dataset.labels[idx].values())[0])
    return label

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case = False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' #for '-1' index

        self.dict = {}
        for i, char in enumerate(alphabet):
            #Note: index[0] is reserved for 'blank' required by wrap_etc so that begin from index[1]--(i + 1)
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw = False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i -1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            #batch mode
            assert t.numel() == length.sum(),"texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index - l], torch.IntTensor([l]), raw = raw))
                index += l
            return texts

def model_info(model):
    n_p = sum(x.numel() for x in model.parameters()) #number of parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad) #number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer','name', 'gradient', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.','')
        print('%5s %50s %9s %12g %20g %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()
        ))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

if __name__ == '__main__':
    pass
    # print(list(alphabets))
    # dict = {}
    # for i, char in enumerate(alphabets.alphabet):
    #     dict[char] = i + 1
    #     print(dict[char])
    #     print(dict)