import tokenization
#create_example


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


def create_examples(lines, set_type=None):
    """Creates examples for the training and dev sets."""
    examples = []
    labels = []
    labels_test = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        # print("guid:", guid) #guid: train-1
        text_a = tokenization.convert_to_unicode(line[1])
        # print("text_a", text_a)
        if set_type == "test":
            label = "台湾"  # 这里要设置成数据集中一个真实的类别
        else:
            label = tokenization.convert_to_unicode(line[0])
            # print("label:", label)
        labels.append(label)
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    # print("example:", examples)
    # print("label:", labels) #every sentence - labels
    # print("label_test:", labels_test) #[]
    return examples, labels, labels_test