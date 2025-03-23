import numpy as np
import json

sub = 1
path = 'dataset/NSD/processed_data/subj{:02d}/'.format(sub)
data_text = []
data_text.extend(np.load(path + 'train_cap.npy'))
data_text.extend(np.load(path + 'test_cap.npy'))

# -----------------------------------------------------------
vocabs_cap  = {}
""" Get: Special Characters """
com_chars = [x for x in " abcdefghijklmnopqrstuvwxyz"]
spe_chars = []
for text in data_text:
    text = text.replace("\n", ' ')
    text = ' '.join(text.split())
    text = text.lower()
    spe_chars.extend([x for x in text if x not in com_chars])

temp = []
[temp.append(x) for x in spe_chars if x not in temp]
spe_chars = temp

spe_chars.sort()
vocabs_cap['com_chars'] = com_chars
vocabs_cap['spe_chars'] = spe_chars

""" Read：Standardize data_text """
data_text_norm = []
for text in data_text:
    for sc in spe_chars:
        if sc in text:
            text = text.replace(sc, f' {sc} ')
    text = ' '.join(text.split())
    data_text_norm.append(text)

""" Count：Words """
counter = {}
for text in data_text_norm:
    text = ' '.join(text.split())
    for word in text.split():
        if word in counter.keys():
            counter[word.lower()] += 1
        else:
            counter[word.lower()] = 1
counter_spe_chars = [(k, v) for k, v in counter.items() if k in spe_chars]
counter_spe_chars = sorted(counter_spe_chars, key=lambda a: a[0])
counter_spe_chars = dict(counter_spe_chars)

counter_words = [(k, v) for k, v in counter.items() if k not in spe_chars]
counter_words = sorted(counter_words, key=lambda a: a[1], reverse=True)
counter_words = dict(counter_words)

vocabs_cap['counter_spe_chars'] = counter_spe_chars
vocabs_cap['counter_words'] = counter_words
vocabs_cap['counter_tokens'] = dict(**{'<unk>': 0, '<sos>': 0, '<eos>': 0,
                                            '<ls>': 0, '<ms>': 0, '<ss>': 0},
                                            **counter_spe_chars,
                                            **counter_words)

id2tokens = dict([(i, k) for i, k in enumerate(vocabs_cap['counter_tokens'].keys())])
tokens2id = dict([(v, k) for k, v in id2tokens.items()])
vocabs_cap['id2tokens'] = id2tokens
vocabs_cap['tokens2id'] = tokens2id
vocabs_cap['tokens_size'] = len(tokens2id)

""" Save：dict """
json.dump(vocabs_cap, open("dataset/NSD/processed_data/subj{:02d}/vocabs_cap.json".format(sub), 'w', encoding='utf-8'))

print("done")
