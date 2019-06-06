import numpy as np
import tensorflow as tf
from Graph import *

class NNetLayers(object):
    tfConvs=[tf.layers.conv1d,tf.layers.conv2d,tf.layers.conv3d]
    tfConvsT=[tf.layers.conv1d,tf.layers.conv2d_transpose,tf.layers.conv3d_transpose]
    tfMPools=[tf.layers.max_pooling1d,tf.layers.max_pooling2d,tf.layers.max_pooling3d]
    tfAPools=[tf.layers.average_pooling1d,tf.layers.average_pooling2d,tf.layers.average_pooling3d]
    tflayer_map={"dense":tf.layers.dense,"conv":tfConvs,"conv1d":tf.layers.conv1d,"conv2d":tf.layers.conv2d,
                 "conv3d":tf.layers.conv3d,"conv_t":tfConvsT,"conv1d_t":tf.layers.conv1d,
                 "conv2d_t":tf.layers.conv2d_transpose,"conv3d_t":tf.layers.conv3d_transpose,"dropout":tf.layers.dropout,
                 "Maxpool":tfMPools,"Averagepool":tfAPools,"Apool":tfAPools,"Avpool":tfAPools,"flatten":tf.layers.Flatten,
                 "LSTM":tf.contrib.rnn.LayerNormBasicLSTMCell,"lstm":tf.contrib.rnn.LayerNormBasicLSTMCell,
                 "+":tf.math.add,"/":tf.math.divide,"-":tf.math.subtract,"*":tf.math.multiply,"Input":tf.placeholder,
                 "input":tf.placeholder,"Inputs":tf.placeholder,"inputs":tf.placeholder,
                 "mean_squared_error":tf.losses.mean_squared_error,"MSE":tf.losses.mean_squared_error,
                 "softmax_cross_entropy":tf.losses.softmax_cross_entropy,"SoftCE":tf.losses.softmax_cross_entropy,
                 "sigmoid_cross_entropy":tf.losses.sigmoid_cross_entropy,"SigtCE":tf.losses.sigmoid_cross_entropy
                }
    tfDenseArg=["inputs","units","activation","use_bias","kernel_initializer","bias_initializer","kernel_regularizer",
               "bias_regularizer","activity_regularizer","kernel_constraint","bias_constraint","trainable","name","reuse"]
    tfConvArg=["inputs","filters","kernel_size","strides","padding","data_format","dilation_rate","activation","use_bias",
                 "kernel_initializer","bias_initializer","kernel_regularizer","bias_regularizer","activity_regularizer",
                 "kernel_constraint","bias_constraint","trainable","name","reuse"]
    tfConvTArg=["inputs","filters","kernel_size","strides","padding","data_format","activation","use_bias",
               "kernel_initializer","bias_initializer","kernel_regularizer","bias_regularizer","activity_regularizer",
               "kernel_constraint","bias_constraint","trainable","name","reuse"]
    tfDropoutArg=["inputs","rate","noise_shape","seed","training","name"]
    tfFlattenArg=["inputs","name"]
    tfPoolArg=["inputs","pool_size","strides","padding","data_format","name"]
    tfoperatorargs=["x","y","name"]
    tfoperatorinputargs=["dtype","shape","name"]
    tfMSE=["labels","predictions","weights","scope","loss_collection","reduction"]
    tfSCE=["labels","logits","weights","label_smoothing","scope","loss_collection","reduction"]
    tfArgList={tf.layers.dense:tfDenseArg,tf.layers.conv1d:tfConvArg,tf.layers.conv2d:tfConvArg,tf.layers.conv3d:tfConvArg,
               tf.layers.conv2d_transpose:tfConvTArg,tf.layers.conv3d_transpose:tfConvTArg,tf.layers.dropout:tfDropoutArg,
               tf.layers.Flatten:tfFlattenArg,tf.layers.max_pooling1d:tfPoolArg,tf.layers.max_pooling2d:tfPoolArg,
               tf.layers.max_pooling3d:tfPoolArg,tf.layers.average_pooling1d:tfPoolArg,tf.layers.average_pooling2d:tfPoolArg,
               tf.layers.average_pooling3d:tfPoolArg,tf.math.add:tfoperatorargs,tf.math.divide:tfoperatorargs,
               tf.math.multiply:tfoperatorargs,tf.math.subtract:tfoperatorargs,tf.placeholder:tfoperatorinputargs,
               tf.losses.mean_squared_error:tfMSE,tf.losses.softmax_cross_entropy:tfSCE,
               tf.losses.sigmoid_cross_entropy:tfSCE
              }
    inputx2=["+","-","*","/","mean_squared_error","softmax_cross_entropy","sigmoid_cross_entropy","MSE","SoftCE","SigtCE"]
    Input_variation=["Input","Inputs","input","inputs"]

    def __init__(self,ntype,dim=None,net_specification={},custom_func=None,name=None):
        self.type=ntype
        self.dim=dim
        self.net_specification=net_specification
        self.custom_func=custom_func
        self.values=None
        self.name=name
        self.trainning_var=None
        self.W=None
        self.b=None

    def build_layer(self,inp,pack):
        self.pack=pack
        if pack is "tf":
            return self.build_tf(inp)
        elif pack is "keras":
            return self.build_keras(inp)
        elif pack is "pytorch":
            return self.build_pytorch(inp)

    def build_keras(self,inp):
        pass
    def build_pytorch(self,inp):
        pass
    def build_tf(self,inp):
        layer_func=self._get_func("tf",self.type,self.dim)
        args=self._fill_func(inp,layer_func)
        if type(args) is list:
            self.layer=layer_func(*args)
        elif type(args is dict):
            self.layer=layer_func(**args)
        return self.layer
    def _get_func(self,pack,ntype,dim=None):
        if pack is "tf":
            if self.custom_func is not None:
                return self.custom_func
            lfunc=self.tflayer_map[ntype]
            if type(lfunc) is list:
                lfunc=lfunc[dim-1]
        elif pack is "keras":
            pass
        else:
            pass
        return lfunc
    def get_input_length(self):
        if self.type in self.Input_variation:
            return 0
        elif self.type in self.inputx2:
            return 2
        else:
            return 1
    def _fill_func(self,Input,layer_func):
        input_len=self.get_input_length()
        inp=[i.layer for i in Input]
        assert(len(inp)==input_len)
        if type(self.net_specification) is list:
            args=inp+self.net_specification
        elif type(self.net_specification) is dict:
            args={}
            if "inputs" in self.tfArgList[layer_func]:
                args["inputs"]=inp[0]
            if "x" in  self.tfArgList[layer_func]:
                args["x"]=inp[0]
            if "labels" in  self.tfArgList[layer_func]:
                args["labels"]=inp[0]
            if "y" in  self.tfArgList[layer_func]:
                args["y"]=inp[1]
            if "predictions" in  self.tfArgList[layer_func]:
                args["predictions"]=inp[1]
            if "logits" in  self.tfArgList[layer_func]:
                args["logits"]=inp[1]
            if self.custom_func is not None:
                return args
            if self.custom_func is not None:
                return args
            #check everything is alright and only feed required args
            for arg in self.tfArgList[layer_func]: 
                if arg in self.net_specification:
                    args[arg]=self.net_specification[arg]
        if "training" in self.tfArgList[layer_func] and self.trainning_var is not None:
            args["training"]=self.trainning_var
        return args
    def set_train_var(self,var):
        self.trainning_var=var
    def helpfunc(self,pack=None,ntype=None):
        if ntype is None:
            ntype=self.type
        if pack is None or pack is "tf":
            print("Tensorflow Parameters:")
            for i in self.tfArgList[self._get_func("tf",ntype)]:
                print(" "+i)
        elif pack is None or pack is "keras":
            pass
    def save_weights(self,args={}):
        if self.pack is "tf":
            assert("sess" in args)
            self._save_weights_tf(args["sess"])
        elif self.pack is "keras":
            self._save_weights_keras()
        elif self.pack is "pytorch":
            self._save_weights_pytorch()
    def _save_weights_tf(self,sess):
        Name=self.layer.name.partition(":")[0].partition("/")[0]
        W=[v for v in tf.trainable_variables() if v.name == Name+"/kernel:0"]
        b=[v for v in tf.trainable_variables() if v.name == Name+"/bias:0"]
        if len(W)>0:
            self.W=W[0].eval(sess)
            self.b=b[0].eval(sess)
    def assign_weights(self,args={}):
        if self.pack is "tf":
            assert("sess" in args)
            self._assign_weights_tf(args["sess"])
        elif self.pack is "keras":
            self._assign_weights_keras()
        elif self.pack is "pytorch":
            self._assign_weights_pytorch()
    def _assign_weights_tf(self,sess):
        Name=self.layer.name.partition(":")[0].partition("/")[0]
        W=[v for v in tf.trainable_variables() if v.name == Name+"/kernel:0"]
        b=[v for v in tf.trainable_variables() if v.name == Name+"/bias:0"]
        if self.W is not None:
            W[0].load(self.W,sess)
        if self.b is not None:
            b[0].load(self.b,sess)
    def _assign_weights_keras(self):
        pass
    def _assign_weights_pytorch(self):
        pass
    
class Optimizer(object):
    Optimizers_tf={"Adam":tf.train.AdamOptimizer,"AdamOptimizer":tf.train.AdamOptimizer,
                   "Adagrad":tf.train.AdagradOptimizer,"AdagradOptimizer":tf.train.AdagradOptimizer,
                   "Adadelta":tf.train.AdadeltaOptimizer,"AdadeltaOptimizer":tf.train.AdadeltaOptimizer}
    Optimizers_keras={}
    Optimizers_pythorch={}
    Optimizers={"tf":Optimizers_tf,"keras":Optimizers_keras,"pythorch":Optimizers_pythorch}
    

    def __init__(self,opttype,inputs,args,ntype=None):
        self.ntype=ntype
        self.opttype=opttype
        self.inputs=inputs
        self.args=args
        self.built=0
        if ntype is not None:
            self.build(ntype)
    def build(self,ntype):
        if not self.built:
            if type(self.inputs) is list:
                Input=[Inp.layer for Inp in self.inputs]
            else:
                Input=self.inputs.layer
            self.ntype=ntype
            if type(self.args) is list:
                self.optimizer=self.Optimizers[self.ntype][self.opttype](*self.args).minimize(Input)
            elif type(self.args) is dict:
                self.optimizer=self.Optimizers[self.ntype][self.opttype](**self.args).minimize(Input)
            self.built=1

        

class NNet(object):
    def __init__(self,inputs,outputs,Net,name=None):
        self.name=name
        self.inputs=inputs # dict or list of inputs
        self.outputs=outputs
        self.Net=Net
        self.optimizers={}
        self.loss={}     #loss functions
        
        self.layers={}
        for v in self.Net.V:
            self.layers[v.name]=v
        
        self.ntype=None
        self.net_built=0
    def build_net(self,pack):
        self.ntype=pack
        if pack is "tf":
            self._build_tf_net()
        elif pack is "keras":
            self._build_keras_net()
        elif pack is "pytorch":
            self._build_pytorch_net()
        for optimizer in self.optimizers:
            self.optimizers[optimizer].build(self.ntype)
    def create_optimizer(self,name,optimizer_type,inputs,args={"learning_rate":0.01}):
        if self.optimizers is None:
            self.optimizers={}
        if type(inputs) is list:
            Input=[]
            for inp in inputs:
                Input+=[self.layers[inp]]
        else:
            Input=self.layers[inputs]
        self.optimizers[name]=Optimizer(optimizer_type,Input,args,self.ntype)
    def display(self):
        #to change away from nx library
        G=nx.DiGraph()
        for e in self.Net.E:
            G.add_edge(e[0].name,e[1].name)
        nx.draw(G, with_labels = True)
        plt.show()
    def _build_tf_net(self):
        self._is_training=tf.placeholder_with_default(True,shape=())
        built_layer_list=[]
        for inp in self.inputs:
            self.inputs[inp].build_layer([],"tf")
            built_layer_list+=[self.inputs[inp]]
        forward_net=self.Net.get_forward_graph()
        for layers in forward_net:
            for layer in layers:
                layer.set_train_var(self._is_training)
                if layer not in built_layer_list:
                    InputFrom=self.Net.E_in[layer]
                    layer_input=InputFrom
                    layer.build_layer(layer_input,"tf")
        self.net_built=1
    def _build_keras_net(self):
        raise Exception("not implemented for keras yet")
    def _build_pytorch_net(self):
        raise Exception("not implemented for pytorch yet")
    def save_net(self):
        pass
    def load_net(self):
        pass
    def init_weights(self,args={}):
        for name,layer in self.layers.items():
            layer.assign_weights(args)
    def run_net(self,inputs,outputs=None):
        if not self.net_built:
            raise Exception("Net not built, run build_net(tool_type)")
        if self.ntype is "tf":
            out=self._run_net_tf(inputs)
        elif self.ntype is "keras":
            pass
        elif self.ntype is "pytorch":
            pass
        return out
    def _run_net_tf(self,inputs,outputs=None):
        #implement selecting outputs
        
        #to change maybe to remove sess elsewhere
                
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        
        #reassign all weights
        self.init_weights({"sess":sess})

        feed_dict={}
        feed_dict[self._is_training]=False
        for inp in inputs:
            if inp in self.inputs:
                feed_dict[self.inputs[inp].layer]=inputs[inp]
                
        if type(self.outputs) is list:
            tf_out=[out.layer for out in self.outputs]
        elif type(self.outputs) is dict:
            tf_out=[out.layer for out in self.outputs.values]
        out=sess.run(tf_out,feed_dict=feed_dict)
        sess.close()
        return out
    def train_net(self,inputs,optimizer_name,outputs=None,batch_size=1,test_perc=0,epochs=1):
        if not self.net_built:
            raise Exception("Net not built, run build_net(tool_type)")
        if self.ntype is "tf":
            out=self._train_net_tf(inputs,optimizer_name,outputs,batch_size,test_perc,epochs)
        elif self.ntype is "keras":
            pass
        elif self.ntype is "pytorch":
            pass
        return out
    def _train_net_tf(self,inputs,optimizer_name,outputs=None,batch_size=1,test_perc=0,epochs=1):
        """
        inputs: a matrix of inputs, the 1st dimension is assumed to be the batch size
        optimizer_name: name of the optimizer function used
        outputs: actual outputs for training purpose, None if not used
        batch_size: batch_size of every train step
        test_perc: percentage of test set, test ratio+train ratio=1
        epochs: number of trainning epochs
        
        """
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        exp_set_size=None
        for inp in inputs:
            if exp_set_size is not None:
                assert(inputs[inp].shape[0]==exp_set_size)
            else:
                exp_set_size=inputs[inp].shape[0]
        
        Opt=self.optimizers[optimizer_name].optimizer
        run_vals=[Opt]
        
        #reassign all weights
        self.init_weights({"sess":sess})
        
        if outputs is None:
            if type(self.outputs) is list:
                run_vals+=[out.layer for out in self.outputs]
            elif type(self.outputs) is dict:
                run_vals+=[out.layer for out in self.outputs.values]
        else:
            if type(outputs) is list:
                run_vals+=[self.layers[out].layer for out in outputs]
            elif type(outputs) is dict:
                run_vals+=[self.layers[out].layer for out in outputs.values]

        if test_perc==0:
            Test_Set=[]
            train_range=exp_set_size-1
        else:
            Test_Set=[]
            train_range=exp_set_size-1
            #to change and implement test_perc
            
        last_outs=[]
        for i in range(epochs):
            inp_set=np.random.choice(train_range, batch_size, replace=False)
            feed_dict={}
            feed_dict[self._is_training]=True
            for inp in inputs:
                if inp in self.inputs:
                    feed_dict[self.inputs[inp].layer]=inputs[inp][inp_set]

            out=sess.run(run_vals, feed_dict=feed_dict)
            out=out[1:len(out)]
            last_outs+=[out]
            if len(last_outs)>10:
                last_outs=last_outs[1:len(last_outs)]
                
        for name,layer in self.layers.items():
            layer.save_weights({"sess":sess})
        sess.close()
        return last_outs
            