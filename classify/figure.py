import matplotlib.pyplot as plt
import numpy as np
with open("log1.txt",'r',encoding='utf-8') as f:
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    step = 0
    log=f.readline()
    while(log):
        log = log.split()
        if "Step" in log:
            index = log.index("Step")
            step = int(log[index + 1].split('/')[0])
            if step>950:
                acc = float(log[index + 3][:-1])
                ppl = float(log[index + 5][:-1])
                train_x.append(step)
                train_y.append([acc,ppl])
        if "perplexity:" in log:
            dev_x.append(step)
            ppl = float(log[-1])
            log = f.readline().split()
            acc = float(log[-1])
            dev_y.append([acc,ppl])
        log = f.readline()
y = 'acc'
if y == 'acc':
    train_y = np.array(train_y)[:,0]
    dev_y = np.array(dev_y)[:,0]
else:
    train_y = np.array(train_y)[:,1]
    dev_y = np.array(dev_y)[:,1]
    y = 'ppl'
plt.plot(train_x, train_y, label = "train")
plt.plot(dev_x, dev_y, label = "test")
plt.xlabel("steps")
plt.ylabel(y) 
plt.legend()
plt.show()