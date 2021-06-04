import glob
import numpy as np
import random
import math as mt


def load_data(name_files):
    labels = []
    words_info = []
    for name in name_files:
        if 'spmsg' in name:
            labels.append(0)
        elif 'legit' in name:
            labels.append(1)
        with open(name) as f:
            words = f.read()
            words = words.replace("\n", " ")
            words = words.replace("Subject:", "")
            words = words.split(" ")
        #print(words)
        count = []
        for i in range(len(words)):
            if words[i] == "":
                pass
            else:
                count.append(int(words[i]))
        #print(count)
        words_info.append(count)
    words_info = np.expand_dims(np.array(words_info), axis= 1)
    #print(words_info.shape)
    labels = np.expand_dims(np.array(labels), axis=1)
    #print(labels.shape)
    dataset = np.concatenate((words_info, labels), axis=1)
    return dataset


def split_dataset(dataset, splitratio):
    trainsize = int(len(dataset) * splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset, copy]


def seperatebyclass(train_dataset):
    spam = []
    ham = []
    for a in train_dataset:
        if a[1] == 1:
            spam.append(a)
        elif a[1] == 0:
            ham.append(a)
    return np.array(spam), np.array(ham)


def summarizebyclass(spam, ham):
    dic_spam = {}
    dic_ham = {}
    spam_count = 0
    ham_count = 0
    spam_p_w = 0
    ham_p_w = 0
    for p in range(len(spam)):
        for q in range(len(spam[p][0])):
            spam_count += 1
            if spam[p][0][q] in dic_spam:
                dic_spam[spam[p][0][q]] += 1
            else:
                spam_p_w+= 1
                dic_spam[spam[p][0][q]] = 1
    for p in range(len(ham)):
        for q in range(len(ham[p][0])):
            ham_count += 1
            if ham[p][0][q] in dic_ham:
                dic_ham[ham[p][0][q]] += 1
            else:
                ham_p_w += 1
                dic_ham[ham[p][0][q]] = 1

    tot_ham = ham_count+ham_p_w
    tot_spam = spam_count+spam_p_w

    return dic_spam, dic_ham, tot_spam, tot_ham


def calculate_probabilities(spam, ham, train_dataset):
    prob_ham = mt.log(len(ham)/len(train_dataset))
    prob_spam = mt.log(len(spam)/len(train_dataset))
    return prob_spam, prob_ham


def prediction(test_dataset,spamdict,hamdict, prob_ham, prob_spam,totham,totspam):
    is_spam = []
    is_ham = []
    for q in test_dataset:
        prob_spam_ = prob_spam
        for w in q[0]:
            if w in spamdict:
                prob_spam_ += mt.log((spamdict[w])/totspam)
        is_spam.append(prob_spam_)
    for e in test_dataset:
        prob_ham_ = prob_ham
        for r in e[0]:
            if r in hamdict:
                prob_ham_ += mt.log((hamdict[r])/totham)
        is_ham.append(prob_ham_)
    return is_spam, is_ham


def get_prediction(inspam, inham):
    predictions = []
    for g in range(len(inspam)):
        if inspam[g] >= inham[g]:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions


def accuracy(testdataset, predicteddata):
    accuracy = 0
    for o in range(len(predicteddata)):
        if predicteddata[o] == testdataset[o][1]:
            accuracy += 1
    return (accuracy/len(predicteddata))*100


if __name__ == "__main__":
    path = 'F:\SecondSem\ML019\Bayesian\part2./*txt'
    files = glob.glob(path)
    data = load_data(files)
    test_, train_ = split_dataset(data, 0.3)
    spam, ham = seperatebyclass(train_)
    prob_spam, prob_ham = calculate_probabilities(spam, ham, train_)
    dic_spam, dic_ham, tot_spam, tot_ham = summarizebyclass(spam, ham)
    s, h = prediction(test_, dic_spam, dic_ham, prob_ham, prob_spam, tot_ham, tot_spam)
    pre = get_prediction(s, h)
    accu = accuracy(test_, pre)
    print('The accuracy is : ', accu)



















