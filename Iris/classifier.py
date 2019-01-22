import neural as nn
import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':

    f = open('Iris.csv',mode = 'r')
    
    feature = []
    label = []
    for line in f:
        data = line.split(',')
        try:
            d = [float(i) for i in data[1:5]]
            feature.append(d)
            if data[5]=='Iris-setosa\n':
                l = [1,0,0]
            elif data[5] == 'Iris-versicolor\n':
                l = [0,1,0]
            elif data[5] == 'Iris-virginica\n':
                l = [0,0,1]
            label.append(l)
        except:
            pass
    feature = np.array(feature)
    label = np.array(label)
    #let's devide the set into train and test set
    train_feature = np.concatenate((feature[0:15][:],feature[51:51+15][:],feature[101:101+15][:],))
    train_target = np.concatenate((label[0:15][:],label[51:51+15][:],label[101:101+15][:]))

    test_feature = np.concatenate((feature[15:51][:],feature[66:101][:],feature[116:151][:]))
    test_target = np.concatenate((label[15:51][:],label[66:101][:],label[116:151][:]))

    #print(test_feature.shape,test_target.shape)
    l1 = nn.denselayer(ip_size = 4,op_size = 20)
    l2 = nn.denselayer(ip_size = 20,op_size = 10)
    l3 = nn.denselayer(ip_size = 10,op_size = 3)
    l4 = nn.softmax_layer()
    net = nn.dense_net()

    epoch = 25000
    #training phase
    loss = []
    alpha = 3e-3
    for e in range(epoch):
        r = net.forward_pass(train_feature,l1,l2,l3,l4)
        if e%50 == 0:
            loss.append(net.avg_loss(target = train_target , prediction = r , batch_size = len(train_feature)))
            print(net.avg_loss(target = train_target , prediction = r , batch_size = len(train_feature)))  
        net.backward_pass(l1,l2,l3,learning_rate = alpha ,target = train_target)

    r = net.forward_pass(test_feature,l1,l2,l3,l4)
    right_guess = 0
    total_test_samples = len(r)
    for i in range(total_test_samples):
        if np.argmax(r[i])==np.argmax(test_target[i]):
            right_guess += 1

    accuracy = right_guess/total_test_samples
    print("Accuracy :{} % ".format(accuracy*100))
    plt.plot(loss)
    plt.xlabel("epoch----->")
    plt.ylabel("cross entropy loss----->")
    plt.show()




    

    


