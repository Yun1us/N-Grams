import datasets
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm

#daten laden und Split 80/20
treebank = datasets.load_dataset("ptb_text_only", split="train")
splits   = treebank.train_test_split(test_size=0.2, seed=42)
train_data, test_data = splits["train"], splits["test"]
print(f"Train/​Test size: {len(train_data)}/{len(test_data)}")

def preprocess(example):
    words = example["sentence"].split()
    words.append("<STOP>")
    example["sentence"] = " ".join(words)
    return example

def apply_filter(example):
    return len(example["sentence"].split()) >= 3

train_data = train_data.map(preprocess).filter(apply_filter)
test_data  = test_data.map(preprocess).filter(apply_filter)

from collections import defaultdict

def estimate_unigram(dataset):
    counts = defaultdict(int)
    for example in dataset:
        for w in example["sentence"].split():
            counts[w] += 1
    return counts

import math 
import numpy as np
def unigram_sentence_logp(model, sentence):
    ttc = sum(model.values())
    sum_of_log2 = 0 
    for word in sentence:
        if word not in model: #words that are not in the model will be skipped later in unigram_perplexity 
           return -np.inf      # will be unecessary when smoothing is implemented 

        p_of_w = model[word] / ttc
        log2_of_w = math.log2(p_of_w)
        sum_of_log2 += log2_of_w
    return sum_of_log2

import numpy as np
def unigram_perplexity(model, test_data):
    total_log = 0
    total_words = 0
    for example in test_data:
        words = example["sentence"].split()
        logp = unigram_sentence_logp(model, words)
        if logp == np.inf or logp == -np.inf:
            continue

        total_log += logp
        total_words += len(words)

    avg_logp = total_log / total_words
    perplexity = 2**(-avg_logp)
    return perplexity   

from collections import defaultdict 
def remove_rares(train_data, test_data, threshold):

    def count_words(train_data):
        word_counts = defaultdict(int)
        for example in train_data:
            for word in example["sentence"].split():
                word_counts[word] += 1 
        return word_counts
    
    word_counts = count_words(train_data)
    
    rare_words = set()
    for word, count in word_counts.items():
        if count < threshold:
            rare_words.add(word)

    def replace_rares(example):
        words = example["sentence"].split()
        replaced_words = [word if word not in rare_words else "<unk>" for word in words]
        example["sentence"] = " ".join(replaced_words)
        return example

    cleaned_train = train_data.map(replace_rares)
    cleaned_test = test_data.map(replace_rares)

    return cleaned_train, cleaned_test



from collections import defaultdict

def estimate_bigram(dataset):
    u_counts = defaultdict(int)
    bigram_pairs = defaultdict(int)
    for example in dataset:
        words = example["sentence"].split()
        for word in words: 
            u_counts[word] += 1
        
        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]
            bigram_pairs[(w1, w2)] += 1 

    return u_counts, bigram_pairs

import math
import numpy as np
import numpy as np
unigrams, bigrams = estimate_bigram(train_data)

def bigram_sentence_logp(unigram_counts, bigram_counts, words):

    logp = 0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if bigram_counts[(w1, w2)] == 0:
            return -np.inf
        prob = bigram_counts[(w1, w2)] / unigram_counts[w1]
        logp += math.log2(prob)
    return logp

def count_zero(u_counts, b_pairs, dataset):
    zero_count = 0
    for example in dataset:
        words = example["sentence"].split()
        if bigram_sentence_logp(u_counts, b_pairs, words) == -np.inf:
            zero_count += 1
    return zero_count

from collections import defaultdict

def estimate_bigram_smoothed(dataset, alpha):
    u_counts = defaultdict(int)
    bigram_pairs = defaultdict(int)
    for example in dataset:
        words = example["sentence"].split()
        for word in words: 
            u_counts[word] += 1
        
        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]
            bigram_pairs[(w1, w2)] += 1 
    vocab_size = len(u_counts)
    return u_counts, bigram_pairs, vocab_size

import math, numpy as np
def bigram_sentence_logp_smoothed(u_counts, b_pairs, vocab_size, alpha, words):
    logp = 0
    for i in range(len(words) -1):
        w1, w2 = words[i], words[i+1]
        zähler = b_pairs[(w1,w2)] + alpha
        nenner = u_counts[w1] + alpha * vocab_size
        prob = zähler / nenner
        logp += math.log2(prob)

    return logp

s1 = ["the","the","the","<STOP>"]
s2 = ["i","love","computer","science","<STOP>"]

alpha = 1.0
u_counts, b_pairs, vocab = estimate_bigram_smoothed(train_data, alpha)
logp1 = bigram_sentence_logp_smoothed(u_counts, b_pairs, vocab, alpha, s1)
logp2 = bigram_sentence_logp_smoothed(u_counts, b_pairs, vocab, alpha, s2)


def bigram_perplexity_smoothed(u_counts, b_pairs, vocab_size, alpha, test_data):
    total_logp = 0.0
    total_words = 0

    for example in test_data:
        words = example["sentence"].split()
        logp = bigram_sentence_logp_smoothed(u_counts, b_pairs, vocab_size, alpha, words)
        if logp == -np.inf:
            continue


        total_logp += logp
        total_words += len(words)

    avg_logp = total_logp / total_words
    perplexity = 2 ** (-avg_logp)
    return perplexity







uni_model = estimate_unigram(train_data)
print("Amount of Words in Model:", len(uni_model))

#Unigram
uni = estimate_unigram(train_data)
s1 = ["the", "the", "the", "<STOP>"]
s2 = ["i", "love", "computer", "science", "<STOP>"]
print("Log-Probability without Smoothing s1:", unigram_sentence_logp(uni, s1))
print("Log-Probability without Smoothing s2:", unigram_sentence_logp(uni, s2))
print("Unigram-Perplexity:", unigram_perplexity(uni, test_data))

#Smoothed LOG
print("Smoothed log-prob s1:", logp1)
print("Smoothed log-prob s2:", logp2)

#Rare-Words 
new_train, new_test = remove_rares(train_data, test_data, threshold=3)
uni2 = estimate_unigram(new_train)
print("Unigram-Perplexity after <unk>:", unigram_perplexity(uni2, new_test))

#Bigram
u_counts, b_counts = estimate_bigram(train_data)
print("LogP s1:", bigram_sentence_logp(u_counts, b_counts, s1))
print("LogP s2:", bigram_sentence_logp(u_counts, b_counts, s2))
print("Zero-Count:", count_zero(u_counts, b_counts, test_data))

#Smoothed Bigram
u2, b2, V = estimate_bigram_smoothed(train_data, alpha=1.0)
print("Smoothed Bigram-Perplexity:", bigram_perplexity_smoothed(u2, b2, V, 1.0, test_data))



