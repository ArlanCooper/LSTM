import numpy as np
class SigmoidActivator(object):
    def forward(self,weighted_input):#函数方程
        return 1.0/(1.0+np.exp(-weighted_input))
    def backward(self,output):#函数导数
        return output*(1-output)
class TanhActivator(object):
    def forward(self,weighted_input):#函数方程
        return 2.0/(1.0+np.exp(-2*weighted_input))-1.0
    def backward(self,output):#函数导数
        return 1-output*output
class LSTMLayer(object):
    def __init__(self,input_width,state_width,learning_rate):
        self.input_width=input_width
        self.state_width=state_width
        self.learning_rate=learning_rate
        #门的激活函数
        self.gate_activator=SigmoidActivator()
        #输出的激活函数
        self.output_activator=TanhActivator()
        #单圈时刻初始化为t0
        self.times=0
        #各个时刻的单元状态向量c
        self.c_list=self.init_state_vec()
        #各个时刻的输出向量h
        self.h_list=self.init_state_vec()
        #各个时刻的遗忘门f
        self.f_list=self.init_state_vec()
        #各个时刻的输入门i
        self.i_list=self.init_state_vec()
        #各个时刻的输出门o
        self.o_list=self.init_state_vec()
        #各个时刻的即时状态c~
        self.ct_list=self.init_state_vec()
        #遗忘门权重矩阵wfh,wfx,偏置项bf
        self.Wfh,self.Wfx,self.bf=(self.init_weight_mat())
        #输入门权重wih,wix,偏置项bi
        self.Wih,self.Wix,self.bi=(self.init_weight_mat())
        #输出门权重矩阵woh,wox,bo
        self.Woh,self.Wox,self.bo=(self.init_weight_mat())
        #单元状态权重矩阵
        self.Wch,self.Wcx,self.bc=(self.init_weight_mat())
    def init_state_vec(self):
        '''
        初始化保存状态向量
        '''
        state_vec_list=[]
        state_vec_list.append(np.zeros((self.state_width,1)))
        return state_vec_list
    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        #numpy.random.uniform(low,high,size):左闭右开
        Wh=np.random.uniform(-1e-4,1e-4,(self.state_width,self.state_width))
        Wx=np.random.uniform(-1e-4,1e-4,(self.state_width,self.input_width))
        b=np.zeros((self.state_width,1))
        return Wh,Wx,b
    #前向传播计算
    def forward(self,x):
        '''
        根据式1-式6进行前向计算
        '''
        self.times+=1
        #遗忘门
        fg=self.calc_gate(x,self.Wfx,self.Wfh,self.bf,self.gate_activator)
        self.f_list.append(fg)
        #输入门
        ig=self.calc_gate(x,self.Wix,self.Wih,self.bi,self.gate_activator)
        self.i_list.append(ig)
        #输出门
        og=self.calc_gate(x,self.Wox,self.Woh,self.bo,self.gate_activator)
        self.o_list.append(og)
        #即时状态
        ct=self.calc_gate(x,self.Wcx,self.Wch,self.bc,self.gate_activator)
        self.ct_list.append(ct)
        #单元状态
        c=fg*self.c_list[self.times-1]+ig*ct
        self.c_list.append(c)
        #输出
        h=og*self.output_activator.forward(c)
        self.h_list.append(h)
    def calc_gate(self,x,Wx,Wh,b,activator):
        '''
        计算门
        '''
        h=self.h_list[self.times-1] #上次的LSTMs输出
        net=np.dot(Wh,h)+np.dot(Wx,x)+b
        gate=activator.forward(net)
        return gate
    #反向传播计算
    def backward(self,x,delta_h):
        '''
        
        实现LSTM训练算法
        '''
        self.calc_delta(delta_h)
        self.calc_gradient(x)
    def calc_delta(self,delta_h):
        '''
        初始化各个时刻的误差项
        '''
        self.delta_h_list=self.init_delta()#输出误差项
        self.delta_o_list=self.init_delta()#输出门误差项
        self.delta_i_list=self.init_delta() #输入门误差项
        self.delta_f_list=self.init_delta()#遗忘门误差项
        self.delta_ct_list=self.init_delta()#即时输出误差项
        #保存从上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1]=delta_h
        #迭代计算每个时刻的误差项
        for k in range(self.times,0,-1):
            self.calc_delta_k(k)
    def init_delta(self):
        '''
        初始化误差项
        '''
        delta_list=[]
        for i in range(self.times+1):
            delta_list.append(np.zeros((self.state_width,1)))
        return delta_list
    def calc_delta_k(self,k):
        '''
        根据k时刻的delta_h,计算k时刻的delta_f,delta_i,delta_o,delta_ct,以及k-1时刻的delta_h
        '''
        #获得k时刻前向计算的值
        ig=self.i_list[k]
        og=self.o_list[k]
        fg=self.f_list[k]
        ct=self.c_list[k]
        c_prev=self.c_list[k]
        tanh_c=self.output_activator.forward(ct)
        delta_k=self.delta_h_list[k]
        #根据式9计算dlta_o
        delta_o=(delta_k*tanh_c*self.gate_activator.backward(og))
        delta_f=(delta_k*og*(1-tanh_c*tanh_c)*c_prev*self.gate_activator.backward(fg))
        delta_i=(delta_k*og*(1-tanh_c*tanh_c)*ct*self.gate_activator.backward(ig))
        delta_ct=(delta_k*og*(1-tanh_c*tanh_c)*ig*self.output_activator.backward(ct))
        delta_h_prev=(np.dot(delta_o.transpose(),self.Woh)
                     +np.dot(delta_i.transpose(),self.Wih)
                     +np.dot(delta_f.transpose(),self.Wfh)
                     +np.dot(delta_ct.transpose(),self.Wch)).transpose()
        #保存全部的delta值
        self.delta_h_list[k-1]=delta_h_prev
        self.delta_f_list[k]=delta_f
        self.delta_i_list[k]=delta_i
        self.delta_o_list[k]=delta_o
        self.delta_ct_list[k]=delta_ct
     #计算梯度
    def calc_gradient(self,x):
        #初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad,self.Wfx_grad,self.bf_grad=(self.init_weight_gradient_mat())
        #初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad,self.Wix_grad,self.bi_grad=(self.init_weight_gradient_mat())
        #初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad,self.Wox_grad,self.bo_grad=(self.init_weight_gradient_mat())
        #初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad,self.Wcx_grad,self.bc_grad=(self.init_weight_gradient_mat())
        #计算对上一次输出h的权重梯度
        for t in range(self.times,0,-1):
            #计算各个时刻的梯度
            (Wfh_grad,bf_grad,Wih_grad,bi_grad,Woh_grad,bo_grad,Wch_grad,bc_grad)=(self.calc_gradient_t(t))
            #实际梯度是各个时刻梯度之和
            self.Wfh_grad+=Wfh_grad
            self.bf_grad+=bf_grad
            self.Wih_grad+=Wih_grad
            self.bi_grad+=bi_grad
            self.Woh_grad+=Woh_grad
            self.bo_grad+=bo_grad
            self.Wch_grad+=Wch_grad
            self.bc_grad+=bc_grad
            print('-------%d------' %t)
            print('wfh_grad:',Wfh_grad)
            print('self.wfh_grad:',self.Wfh_grad)
        #计算对本次输入x的权重梯度
        xt=x.transpose()
        self.Wfx_grad=np.dot(self.delta_f_list[-1],xt)
        self.Wix_grad=np.dot(self.delta_i_list[-1],xt)
        self.Wox_grad=np.dot(self.delta_o_list[-1],xt)
        self.Wcx_grad=np.dot(self.delta_ct_list[-1],xt)
    def init_weight_gradient_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh_grad=np.zeros((self.state_width,self.state_width))
        Wx_grad=np.zeros((self.state_width,self.input_width))
        b_grad=np.zeros((self.state_width,1))
        return Wh_grad,Wx_grad,b_grad
    def calc_gradient_t(self,t):
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev=self.h_list[t-1].transpose()
        Wfh_grad=np.dot(self.delta_f_list[t],h_prev)
        bf_grad=self.delta_f_list[t]
        Wih_grad=np.dot(self.delta_i_list[t],h_prev)
        bi_grad=self.delta_i_list[t]
        Woh_grad=np.dot(self.delta_o_list[t],h_prev)
        bo_grad=self.delta_o_list[t]
        Wch_grad=np.dot(self.delta_ct_list[t],h_prev)
        bc_grad=self.delta_ct_list[t]
        return Wfh_grad,bf_grad,Wih_grad,bi_grad,Woh_grad,bo_grad,Wch_grad,bc_grad
    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh-=self.learning_rate*self.Whf_grad
        self.Wfx-=self.learning_rate*self.Wfx_grad
        self_bf-=self.learning_rate*self.bf_grad
        self_Wih-=self.learning_rate*self.Wih_grad
        self_Wix-=self.learning_rate*self.Wix_grad
        self_bi-=self.learning_rate*self.bi_grad
        self_Woh-=self.learning_rate*self.Woh_grad
        self_Wox-=self.learning_rate*self.Wox_grad
        self.bo-=self.learning_rate*self.bo_grad
        self.Wch-=self.learning_rate*self.Wch_grad
        self.Wcx-=self.learning_rate*self.Wox_grad
        self.bc-=self.learning_rate*self.bc_grad
    #梯度检查的实现
    def reset_state(self):
        #当前状态
        self.times=0
        #各个时刻的单元状态向量
        self.c_list=self.init_state_vec()
        #各个时刻的输出向量h
        self.h_list=self.init_state_vec()
        #各个时刻的遗忘门f
        self.f_list=self.init_state_vec()
        #各个时刻的输出门o
        self.o_list=self.init_state_vec()
        #各个时刻的输入门i
        self.ct_list=self.init_state_vec()
#剃度检查代码
def data_set():
    x=[np.array([[1],[2],[3]]),np.array([[2],[3],[4]])]
    d=np.array([[1],[2]])
    return x,d
def gradient_check():
    '''
    梯度检查
    '''
    error_function=lambda o:o.sum()
    lstm=LSTMLayer(3,2,1e-3)
    #计算forward值
    x,d=data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])
    #求取sensitivity_map
    sensitivity_aray=np.ones(lstm.h_list[-1].shape,dtype=np.float64)
    #计算梯度
    lstm.backward(x[1],sensitivity_aray)
    #检查梯度
    epsilon=10e-4
    print('wfh:',lstm.Wfh,lstm.Wfh.shape)
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i,j]+=epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1=error_function(lstm.h_list[-1])
            print('1:h_list[-1]:',lstm.h_list[-1],err1)
            lstm.Wfh[i,j]-=2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2=error_function(lstm.h_list[-1])
            print('2:h_list[-1]:',lstm.h_list[-1],err2)
            expect_grad=(err1-err2)/(2*epsilon) #导数公式
            lstm.Wfh[i,j]+=epsilon
            print('weight(%d,%d):expected-actural %.4e -%.4e ' %(i,j,expect_grad,lstm.Wfh_grad[i,j]))
    return lstm
    
        
