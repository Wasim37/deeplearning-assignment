from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


''' 构建一个LSTM模型，用于音乐生成
本练习结束后，你可以用深度学习生成自己的爵士音乐
模型实现使用的框架是keras

思考：
实际场景中，音乐的生成难度应该比较大，特征工程与音乐生成后的后期处理都会涉及一些音乐领域方面的知识
这些知识本练习中有涉及到，但是都大大的简化了
'''

# 音乐数据的预处理很重要，本练习你可以把 "value" 看作一个音符，它包含一个音高和一个持续时间。
# 例如，如果您按下特定钢琴键0.5秒，那么您刚刚弹奏了一个音符。
# 实际在音乐理论中，"value" 实际上比这更复杂 - 具体来说，它还捕获了同时播放多个音符所需的信息。
# 例如，在播放音乐作品时，可以同时按下两个钢琴键，但是本任务中我们不需要关心这些音乐理论的细节
# 你只需要知道你即将获取一个"value"数据集，然后通过RNN模型训练并生成新的"value"序列
# 本系统"value"数据集一共有78个独特的值，每个时间序列都是78中的某一个，用one-hot表示

# 打印训练数据相关信息
# shape of X:', (样本数量, Tx, 78)
X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)



#  We will use an LSTM with 64 dimensional hidden states. Lets set n_a = 64.
n_a = 64

# 现在需要创建一个具有多个输入和输出的Keras模型。
# 如果你正在构建一个RNN，在测试阶段，整个输入序列都是预先给定的，比如你的输入是一个词语，输出是一个标签，都是已知的。Keras具有非常简单的内置函数来构建整个模型。
# 但是，对于序列生成，测试阶段我们并不知道整个输入序列的值，我们使用x(t)=y(t-1)，一次生成一个。所以代码会更复杂一点，你需要实现自己的for循环来遍历不同的时间序列。

# 函数djmodel()将使用for循环调用LSTM层Tx次，每次调用权重都是共享的。也就是说，它不应该每次重新初始化权重
# 在Keras中实现权重共享的关键步骤是：
#   1、定义图层对象（我们将为此使用全局变量）。
#   2、当inputs前向传播时调用这些对象。
reshapor = Reshape((1, 78))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
# Dense 全连接层 https://blog.csdn.net/u012969412/article/details/70882296
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D



# 模型结构
# GRADED FUNCTION: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model with the 
    """
    
    # Define the input of your model with a shape 
    X = Input(shape=(Tx, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda X: X[:,t,:])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    ### END CODE HERE ###
    
    return model



# 传入参数定义模型
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

# 使用adam, 并编译模型
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# 初始化参数，并开始模型的训练
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
model.fit([X, a0, c0], list(Y), epochs=100)

assert 1==2

print("---------------------------------------------------------------------------------------------------------------------------")


# 现在模型已经训练完成，还需要实现一个函数，用这个模型来生成音乐

# 音乐推理(inference)模型
# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model

# 传入参数定义音乐推理模型
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)



# GRADED FUNCTION: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    使用推理模型预测values的下一个值

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(np.array(pred), axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=x_initializer.shape[-1])

    ### END CODE HERE ###

    return results, indices

x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))



# 生成音乐
# generate_music内部会调用predict_and_sample生成系列value值，这些值接着会进行后期处理，变为音乐和弦
# 大多数音乐生成的算法都会使用一些后期处理，因为如果没有这样的后期处理，很难生成听起来很好的音乐，这些处理需要对音乐相关知识有一定的了解
# 很多时候音乐的输出质量不仅仅取决于RNN的质量, 也取决于后期处理的质量。后期的处理能造成很大的影响，我们的实现也使用了后期处理
out_stream = generate_music(inference_model)