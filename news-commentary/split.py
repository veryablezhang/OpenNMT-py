import jieba as jb
import nltk

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def split():    
    f = open("news-commentary-v14.en-zh.tsv", 'r', encoding='utf-8')
    # f2 = open("wikititles-v1.zh-en.tsv", 'r', encoding='utf-8')
    # f = open("src-val.txt", 'r', encoding='utf-8')
    trainsrc = open("src-train.txt", 'w', encoding='utf-8')
    traintgt = open("tgt-train.txt", 'w', encoding='utf-8')
    valsrc = open("src-val.txt", 'w', encoding='utf-8')
    valtgt = open("tgt-val.txt", 'w', encoding='utf-8')
    testsrc = open("src-test.txt", 'w', encoding='utf-8')
    testtgt = open("tgt-test.txt", 'w', encoding='utf-8')
    # valsrc = open("tests.txt", 'w', encoding='utf-8')
    # valtgt = open("testt.txt", 'w', encoding='utf-8')
    lines = 0
    paras = 0
    pair = f.readline()
    while pair:
        pair = pair.replace('\n','')
        pair = pair.split('\t')
        if not (pair[0] == '' or pair[1] == ''):
            tokens = nltk.word_tokenize(pair[0])
            source = []
            for token in tokens:
                source.append(token.lower())
            target = jb.cut(pair[1])
            if (lines%60)==10:
                valsrc.write(' '.join(source)+'\n')
                valtgt.write(' '.join(target)+'\n')
            elif (lines%60)!=40:
                trainsrc.write(' '.join(source)+'\n')
                traintgt.write(' '.join(target)+'\n')
            else:
                testsrc.write(' '.join(source)+'\n')
                testtgt.write(' '.join(target)+'\n')
            lines += 1
        else:
            paras += 1
        if lines%5000 == 0:
            print(lines)
        pair = f.readline()
    print(lines, paras)
    f.close()
    # lines = 0
    # pair = f2.readline()
    # while pair:
    #     pair = pair.replace('\n','')
    #     pair = pair.split('\t')
    #     if not (pair[0] == '' or pair[1] == ''):
    #         tokens = nltk.word_tokenize(pair[1])
    #         source = []
    #         for token in tokens:
    #             source.append(token.lower())
    #         target = jb.cut(pair[0])
    #         if (lines%6)==0:
    #             valsrc.write(' '.join(source)+'\n')
    #             valtgt.write(' '.join(target)+'\n')
    #         elif (lines%6)!=3:
    #             trainsrc.write(' '.join(source)+'\n')
    #             traintgt.write(' '.join(target)+'\n')
    #         else:
    #             testsrc.write(' '.join(source)+'\n')
    #             testtgt.write(' '.join(target)+'\n')
    #         lines += 1
    #     pair = f2.readline()
    #     if lines%5000 == 0:
    #         print(lines)
    # f2.close()
    trainsrc.close()
    traintgt.close()
    valsrc.close()
    valtgt.close()
    testsrc.close()
    testtgt.close()

def test():
    testsrc = open("src-test.txt", 'r', encoding='utf-8')
    testtgt = open("tgt-test.txt", 'r', encoding='utf-8')
    o1 = open("src-test-2.txt", 'w', encoding='utf-8')
    o2 = open("tgt-test-2.txt", 'w', encoding='utf-8')
    f = testsrc.readlines()
    o1.writelines(f[:2000])
    f = testtgt.readlines()
    o2.writelines(f[:2000])

def bleu(pred,target):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import corpus_bleu
    pred = open(pred, 'r', encoding='utf-8')
    tgt = open(target, 'r', encoding='utf-8')
    scores = []
    references = []
    predictions = []
    prediction = pred.readline()
    target = tgt.readline()
    while prediction:
        prediction = prediction.split(' ')
        target = target.split(' ')
        references.append([target])
        predictions.append(prediction)
        score = sentence_bleu([target],prediction)
        scores.append(score)
        prediction = pred.readline()
        target = tgt.readline()
    # print(corpus_bleu(references, predictions))
    return scores

def remove0():
    valsrc = open("src-train.txt", 'r', encoding='utf-8')
    valtgt = open("tgt-train.txt", 'r', encoding='utf-8')
    outsrc = open("src-train1.txt", 'w', encoding='utf-8')
    outtgt = open("tgt-train1.txt", 'w', encoding='utf-8')
    src = valsrc.readlines()
    tgt = valtgt.readlines()
    for i,a in enumerate(tgt):
        if not len(a) == 1:
            if not len(src[i]) == 1:
                outsrc.write(src[i])
                outtgt.write(a)

# out = open("../classify/a.txt", 'w', encoding = 'utf-8')
# with open("../classify/src-train-bpe.txt", 'r', encoding='utf-8') as f:
#     for i in range(50):
#         out.write(f.readline())
# out.close()

# src = open("src-train-bpe.txt", 'r', encoding='utf-8')
# # tgt = open("tgt-train.txt", 'r', encoding='utf-8')
# outsrc = open("../classify/src-train_bpe.txt", 'w', encoding='utf-8')
# # outtgt = open("tgt-train1.txt", 'w', encoding='utf-8')
# r = src.readlines()
# for line in r[:20782]:
#     outsrc.write(line)
# src.close()
# outsrc.close()
# r = tgt.readlines()
# for line in r[:20782]:
#     outtgt.write(line)
# split()
import numpy as np
pred = "../classify/train-bpe2.txt"
target = "../classify/tgt-train.txt"
scores = bleu(pred,target)
print(np.mean(scores))
target = "../classify/tgt-test.txt"
pred = "../classify/test-bpe2.txt"
scores = bleu(pred,target)
print(np.mean(scores))
