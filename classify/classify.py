import numpy as np
import pandas as pd
def get_vocab():
    f = open("train.txt", 'r', encoding = 'utf-8')
    train = f.readlines()
    f.close()
    f = open("test.txt", 'r', encoding = 'utf-8')
    test = f.readlines()
    f.close()
    f = open("valid.txt", 'r', encoding = 'utf-8')
    valid = f.readlines()
    f.close()
    out = open("vocab.txt", 'w', encoding = 'utf-8')
    vocab = []
    for sentence in train:
        sentence = sentence.split(' ')
        for word in sentence:
            if not word in vocab:
                vocab.append(word)
                out.write(word + ' ')
    for sentence in test:
        sentence = sentence.split(' ')
        for word in sentence:
            if not word in vocab:
                vocab.append(word)
                out.write(word + ' ')
    for sentence in valid:
        sentence = sentence.split(' ')
        for word in sentence:
            if not word in vocab:
                vocab.append(word)
                out.write(word + ' ')

def get_features():
    with open("src-train.txt", 'r', encoding = 'utf-8') as f:
        train_en = f.readlines()
    with open("src-test.txt", 'r', encoding = 'utf-8') as f:
        test_en = f.readlines()
    with open("src-train-bpe.txt", 'r', encoding = 'utf-8') as f:
        train_en_bpe = f.readlines()
    with open("src-test-bpe.txt", 'r', encoding = 'utf-8') as f:
        test_en_bpe = f.readlines()
    with open("train.txt", 'r', encoding = 'utf-8') as f:
        train_enzh = f.readlines()
    with open("test.txt", 'r', encoding = 'utf-8') as f:
        test_enzh = f.readlines()
    with open("train-bpe.txt", 'r', encoding = 'utf-8') as f:
        train_enzh_bpe = f.readlines()
    with open("test-bpe.txt", 'r', encoding = 'utf-8') as f:
        test_enzh_bpe = f.readlines()
    with open("zhen/train.txt", 'r', encoding = 'utf-8') as f:
        train_zhen = f.readlines()
    with open("zhen/test.txt", 'r', encoding = 'utf-8') as f:
        test_zhen = f.readlines()
    with open("tgt-train.txt", 'r', encoding = 'utf-8') as f:
        train_zh = f.readlines()
    with open("tgt-test.txt", 'r', encoding = 'utf-8') as f:
        test_zh = f.readlines()
    with open("ppls/ppl_src_train.txt", 'r', encoding = 'utf-8') as f:
        ppl_train_en = f.readlines()
    with open("ppls/ppl_src_test.txt", 'r', encoding = 'utf-8') as f:
        ppl_test_en = f.readlines()
    with open("ppls/ppl_train.txt", 'r', encoding = 'utf-8') as f:
        ppl_train_enzh = f.readlines()
    with open("ppls/ppl_test.txt", 'r', encoding = 'utf-8') as f:
        ppl_test_enzh = f.readlines()
    with open("ppls/ppl_train_bpe.txt", 'r', encoding = 'utf-8') as f:
        ppl_train_enzh_bpe = f.readlines()
    with open("ppls/ppl_test_bpe.txt", 'r', encoding = 'utf-8') as f:
        ppl_test_enzh_bpe = f.readlines()
    with open("ppls/zhen_train.txt", 'r', encoding = 'utf-8') as f:
        ppl_train_zhen = f.readlines()
    with open("ppls/zhen_test.txt", 'r', encoding = 'utf-8') as f:
        ppl_test_zhen = f.readlines()
    with open("ppls/ppl_tgt_train.txt", 'r', encoding = 'utf-8') as f:
        ppl_train_zh = f.readlines()
    with open("ppls/ppl_tgt_test.txt", 'r', encoding = 'utf-8') as f:
        ppl_test_zh = f.readlines()
    en = corpus('news-commentary-v14.en')
    en_bpe = corpus('src-train-bpe-all.txt')
    zh = corpus('news-commentary-v14-1.zh')

    direction = "enzh"
    # direction = "zhen"
    direction = "enzh_bpe"
    if direction == "enzh":
        src_train = train_en
        src_test = test_en
        train = train_enzh
        test = test_enzh
        ref_train = train_zh
        ref_test = test_zh
        ppl_src_train = ppl_train_en
        ppl_src_test = ppl_test_en
        ppl_train = ppl_train_enzh
        ppl_test = ppl_test_enzh
        ppl_ref_train = ppl_train_zh
        ppl_ref_test = ppl_test_zh
        prob_train_path = "prob_train.txt"
        prob_test_path = "prob_test.txt"
        src = en
        ref = zh
    elif direction == "zhen":
        src_train = train_zh
        src_test = test_zh
        train = train_zhen
        test = test_zhen
        ref_train = train_en
        ref_test = test_en
        ppl_src_train = ppl_train_zh
        ppl_src_test = ppl_test_zh
        ppl_train = ppl_train_zhen
        ppl_test = ppl_test_zhen
        ppl_ref_train = ppl_train_en
        ppl_ref_test = ppl_test_en
        src = zh
        ref = en
    else:
        src_train = train_en_bpe
        src_test = test_en_bpe
        train = train_enzh_bpe
        test = test_enzh_bpe
        ref_train = train_zh
        ref_test = test_zh
        ppl_src_train = ppl_train_en
        ppl_src_test = ppl_test_en
        ppl_train = ppl_train_enzh_bpe
        ppl_test = ppl_test_enzh_bpe
        ppl_ref_train = ppl_train_zh
        ppl_ref_test = ppl_test_zh
        prob_train_path = "prob_train_bpe2.txt"
        prob_test_path = "prob_test_bpe2.txt"
        gold_train_path = "gold_train_bpe.txt"
        gold_test_path = "gold_test_bpe.txt"
        src = en_bpe
        ref = zh
    # out = open("features.txt", 'w', encoding = 'utf-8')
    X = []
    Y = []
    prob_test = get_prob(prob_test_path)
    prob_train = get_prob(prob_train_path)
    gold_test = get_prob(gold_test_path)
    gold_train = get_prob(gold_train_path)
    for i,sentence in enumerate(train[:10391]):
    # for i,sentence in enumerate(train[:5000]):
        sentence = sentence.replace('\n','').split()
        if len(sentence)==0:
            continue
        source = src_train[i].replace('\n','').split()
        reference = ref_train[i].replace('\n','').split()
        length_src = len(source)
        length = len(sentence)
        length_ref = len(reference)
        types_src = len(set(source))
        types = len(set(sentence))
        types_ref = len(set(sentence))
        freq_src = get_freq(source, src)
        freq = get_freq(sentence, ref)
        freq_ref = get_freq(reference, ref)
        ppl_src = float(ppl_src_train[i])
        ppl = float(ppl_train[i])
        ppl_ref = float(ppl_ref_train[i])
        prob = prob_train[i]
        gold = gold_train[i]
        # poss = pos([sentence])[0]
        # posr = pos([reference])[0]
        # pos1 = modified_p(poss, posr)
        p1 = modified_p(sentence, reference)
        # rec = recall(sentence, reference)
        try:
            p2 = modified_p(n_gram(sentence, 2), n_gram(reference, 2))
            # pos2 = modified_p(n_gram(poss, 2), n_gram(posr, 2))
        except:
            p2 = 0
            # pos2 = 0
        try:
            p3 = modified_p(n_gram(sentence, 3), n_gram(reference, 3))
            # pos3 = modified_p(n_gram(poss, 3), n_gram(posr, 3))
        except:
            p3 = 0
            # pos3 = 0
        # features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3, freq, freq_src, freq_ref, ppl, ppl_src, ppl_ref]
        features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3, freq, freq_src, freq_ref, prob, gold]
        # features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3]
        # features = [length,ppl]
        # out.write('0')
        # for feature in features:
        #     out.write(' ')
        #     out.write(str(feature))
        # out.write('\n')
        X.append(features)
        Y.append(0)

    for i,sentence in enumerate(test):
    # for i,sentence in enumerate(test[:5000]):
        sentence = sentence.replace('\n','').split()
        if len(sentence)==0:
            continue
        source = src_test[i].replace('\n','').split()
        reference = ref_test[i].replace('\n','').split()
        length_src = len(source)
        length = len(sentence)
        length_ref = len(reference)
        types_src = len(set(source))
        types = len(set(sentence))
        types_ref = len(set(sentence))
        freq_src = get_freq(source, src)
        freq = get_freq(sentence, ref)
        freq_ref = get_freq(reference, ref)
        ppl_src = float(ppl_src_test[i])
        ppl = float(ppl_test[i])
        ppl_ref = float(ppl_ref_test[i])
        prob = prob_test[i]
        gold = gold_test[i]
        # poss = pos([sentence])[0]
        # posr = pos([reference])[0]
        # pos1 = modified_p(poss, posr)
        p1 = modified_p(sentence, reference)
        # rec = recall(sentence, reference)
        try:
            p2 = modified_p(n_gram(sentence, 2), n_gram(reference, 2))
            # pos2 = modified_p(n_gram(poss, 2), n_gram(posr, 2))
        except:
            p2 = 0
            # pos2 = 0
        try:
            p3 = modified_p(n_gram(sentence, 3), n_gram(reference, 3))
            # pos3 = modified_p(n_gram(poss, 3), n_gram(posr, 3))
        except:
            p3 = 0
            # pos3 = 0
        # features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3, freq, freq_src, freq_ref, ppl, ppl_src, ppl_ref]
        features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3, freq, freq_src, freq_ref, prob, gold]
        # features = [length_src, length, length_ref, types_src, types, types_ref, p1, p2, p3]
        # features = [length,ppl]
        # out.write('1')
        # for feature in features:
        #     out.write(' ')
        #     out.write(str(feature))
        # out.write('\n')
        X.append(features)
        Y.append(1)

    return (np.array(X),np.array(Y))
  
def recall(trans, ref):
    match = 0
    for word in ref:
        if word in trans:
            match += 1
    return match/len(ref)
  
def n_gram(sentence, n):
    n_gram = []
    # sentence = sentence.replace('\n','')
    for i in range(len(sentence) - n + 1):
        n_gram.append(sentence[i: i + n])
    return n_gram

def modified_p(trans, ref):
    match = 0
    import copy
    temp = copy.deepcopy(ref)
    for word in trans:
        if word in temp:
            match += 1
            temp.remove(word)
    return match/len(trans)
  
def seen_percentage(sentence, corpus):
  seen = 0
  for n_gram in sentence:
    if n_gram in corpus:
      seen += 1
  return seen/len(sentence)

def corpus(path):
  dict = {}
  f = open(path, 'r', encoding='utf-8')
  lines = f.readlines()
  for line in lines:
    line = line.replace('\n','').split(' ')
    for word in line:
      if word not in dict.keys():
        dict[word] = 1
      else:
        dict[word] += 1
  return dict

def get_freq(sentence, corpus):
  freqs = 0
  for word in sentence:
    if word in corpus.keys():
      freqs += corpus[word]
  return freqs/len(sentence)
      
def pos():
    f = open("train.txt", 'r', encoding = 'utf-8')
    train = f.readlines()
    f.close()
    f = open("test.txt", 'r', encoding = 'utf-8')
    test = f.readlines()
    f.close()
    from ckiptagger import WS, POS, NER
    # ws = WS("./data")
    pos = POS("./data")
    # ner = NER("./data")
    out = []
    for line in train:
        line = line.replace('\n', '').split(' ')
        out.append(pos([line])[0])
    return out

def remove_space():
    t = open("src-train.txt", 'r', encoding='utf-8')
    out = open("src-train1.txt", 'w', encoding='utf-8')
    lines = t.readlines()
    for line in lines:
        new = line.replace(' ','')
        out.write(new)
    out.close() 

def cut():
    t = open("test1.txt", 'r', encoding='utf-8')
    out1 = open("test11.txt", 'w', encoding='utf-8')
    out2 = open("test12.txt", 'w', encoding='utf-8')
    lines = t.readlines()
    for i,line in enumerate(lines):
        if i < 1000:
            out1.write(line)
        else:
            out2.write(line)
    out1.close()
    out2.close()

def get_prob(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    probs = []
    for prob in lines:
        probs.append(float(prob.replace('\n', '')))
    return probs

def classify():
    X,Y = get_features()
    from sklearn.preprocessing import StandardScaler
    # print(X[-10:])
    # Test = np.array(X[:10391])
    # print(np.mean(Test[:,6]))
    # Test = np.array(X[10391:])
    # print(np.mean(Test[:,6]))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # print(X[:10])
    from sklearn.model_selection import train_test_split
    

    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    stacking = False

    if not stacking:
        accs = np.zeros(len(classifiers))

        for k in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size = 0.1)
            for i,clf in enumerate(classifiers):
                clf.fit(X_train, Y_train)
                X_pred = clf.predict(X_test)
                correct = 0
                # if i == 0:
                #     print(clf.get_params())
                for j,pred in enumerate(X_pred):
                    if pred == Y_test[j]:
                        correct += 1
                acc = correct/len(X_pred)
                print(k, acc)
                accs[i] += acc
        for acc in accs:
            print(acc/10)
    else:
        accs = 0
        for k in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size = 0.1)
            all_preds = []
            for clf in classifiers:
                clf.fit(X_train, Y_train)
                all_preds.append(clf.predict(X_test))
            correct = 0
            for i in range(len(X_test)):
                ones = 0
                zeros = 0
                for preds in all_preds:
                    if preds[i] == 1:
                        ones += 1
                    else: zeros += 1
                if ones > zeros:
                    pred = 1
                else: pred = 0
                if pred == Y_test[i]:
                    correct +=1
            acc = correct/len(X_test)
            print(k, acc)
            accs += acc
        print(accs/10)
    

def savefeatures():
    features,Y = np.array(get_features())
    out = open('lengths.csv','w')
    for i in range(len(features)):
        for fea in features[i]:
            out.write(str(fea)+',')
        out.write(str(Y[i]) + '\n')
    out.close()    

def wcsv(features, c, Y, outpath):
    df = pd.DataFrame(features,columns = c)
    df.insert(2,'label',Y)
    df.to_csv(outpath)

def plotppls():
    features,Y = get_features()
    wcsv(features, ['length','ppl'], Y, 'len_ppl_bpe.csv')
    # df = pd.read_csv('len_ppl.csv')
    # features = df.to_numpy()[:,1:-1]
    # features,Y = get_features()
    import matplotlib.pyplot as plt
    delta = 0.001
    # print(Y)
    f1, dot = plt.subplots()
    dot.plot(features[:10391][:,0] - delta, features[:10391][:,1], 'ro', markersize=3)
    dot.plot(features[10391:][:,0] + delta, features[10391:][:,1], 'bo', markersize=3)
    # plt.axis([3.5, 6.5, -25, -100])
    # f1.xlabel("sentence length")
    # f1.ylabel("perplexity")
    f2, box = plt.subplots()
    box.boxplot([features[:10391][:,1], features[10391:][:,1]], showfliers=True)
    plt.show()

def plot():
    df = pd.read_csv('lengths.csv')
    features = df.to_numpy()[:,:-1]
    import matplotlib.pyplot as plt
    # print(Y)
    plt.plot(features[:10391][:,0] - delta, features[:10391][:,1], 'ro', markersize=3)
    plt.plot(features[10391:][:,0] + delta, features[10391:][:,1], 'bo', markersize=3)

# pos()[0]
classify()
# plot()
# plotppls()
# savefeatures()


# f = open("train.txt", 'r', encoding = 'utf-8')
# train = f.readlines()
# f.close()

# f = open("ppls/ppl_train.txt", 'r', encoding = 'utf-8')
# ppl_train = f.readlines()
# f.close()
# f = open("ppls/ppl_test.txt", 'r', encoding = 'utf-8')
# ppl_test = f.readlines()
# f.close() 
# for i in range(len(test)):
#     print(float(ppl_test[i])/len(test[i])) 
# remove_space()
# import jieba as jb
# f = open('news-commentary-v14.zh', 'r', encoding = 'utf-8')
# out = open('news-commentary-v14-1.zh', 'w', encoding = 'utf-8' )
# lines = f.readlines()
# for line in lines:
#     line = ' '.join(jb.cut(line))
#     out.write(line)


# f = open("zhen/test.txt", 'r', encoding = 'utf-8')
# test = f.readlines()
# f.close()
# f = open("zhen/valid.txt", 'r', encoding = 'utf-8')
# valid = f.readlines()
# f.close()
# out = open("zhen/test1.txt",'w',encoding='utf-8')
# for line in test:
#     out.write(line)
# for line in valid:
#     out.write(line)
# out.close()