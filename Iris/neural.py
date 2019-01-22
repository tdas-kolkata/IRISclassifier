import numpy as np

class softmax_layer:
    def cal_op(self,ip):
        self.ip = ip
        self.op = np.exp(ip)
        for r in range(len(self.op)):
            self.op[r] = self.op[r]/sum(self.op[r])

    # def cal_diff(self,target):
    #     return self.prev_layer.op - target


class denselayer:
    def __init__(self,ip_size,op_size):
        self.ip_size = ip_size
        self.op_size = op_size
        self.weight = np.random.randn(ip_size,op_size)
        self.bias = np.random.randn(1,op_size)

    def cal_op(self,ip):
        self.op = np.dot(ip,self.weight)
        self.op = self.op + self.bias
        self.op = self.__nonlin(self.op,deriv=False)

    @staticmethod
    def __nonlin(ip,deriv=False):
        if deriv==False:
            a = np.exp(ip)
            return (a/(a+1))
        if deriv==True:
            return ip*(1-ip)


    def cal_diff(self,layer_input,error):
        error = error * self.__nonlin(self.op,deriv = True)
        self.bias_diff = error.sum(axis=0, dtype='float', keepdims=True)
        self.weight_diff = np.dot(layer_input.T,error)
        propagated_error = np.dot(error,self.weight.T)
        return propagated_error

    def update(self,alpha):
        self.bias -= alpha*self.bias_diff
        self.weight -= alpha*self.weight_diff


class dense_net:
    def avg_loss(self,target,prediction,batch_size): 
         #here we are using cross entropy loss
        loss = np.multiply(target,np.log(prediction))
        loss = -loss
        #print(loss)
        return sum(sum(loss))/batch_size

    def forward_pass(self,ip,*layers):
        '''input: test set, target for test case,batchsize ,layers'''
        self.test = ip
        layers[0].cal_op(self.test)
        depth = len(layers)
        for i in range(1,depth):
            layers[i].cal_op(layers[i-1].op)
        return layers[-1].op

    def backward_pass(self,*layers,**parameters):
        alpha = parameters['learning_rate']
        error = layers[-1].op - parameters['target']
        for i in range(len(layers)-1,0,-1):
            error = layers[i].cal_diff(layers[i-1].op,error) #derivative is calculated layerinput and error is passed as parameters
            layers[i].update(alpha)                          #weigths and bias are updated
        layers[0].cal_diff(self.test,error)
        layers[0].update(alpha)

'''if __name__=='__main__':
    l0 = denselayer(4,3)
    l1 = denselayer(3,3)
    l2 = denselayer(3,3)
    softmax = softmax_layer()
    test = np.array([[5.1,3.5,1.4,0.2],[7.0,3.2,4.7,1.4],[5.4,4.4,1.5,0.4]])
    target = np.array([[1,0,0],[0,1,0],[1,0,0]])
    network = dense_net()
    for i in range(100000):
        a = network.forward_pass(test, l0,l1,l2,softmax)
        network.backward_pass(l0,l1,l2,learning_rate = 0.01,target = target)
        if i %200 == 0 :
            l = network.avg_loss(target,a,batch_size = 3)
            print(l)
    print(a,'\n')'''



