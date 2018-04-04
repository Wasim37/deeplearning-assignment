from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

''' 将人类可读日期翻译成机器可读日期
构建一个神经机器翻译（NMT）模型，将人类可读日期（“2009年6月25日”）转换为机器可读日期（“2009-06-25”）。
您将使用注意模型来完成此操作，这是模型序列中最复杂的序列之一

本次任务创建的模型可用于语言翻译，将一种语言翻译为另一种语言，如英语翻译为印地语。但是，语言翻译需要大量的数据集，并且通常需要跑几天的GPU。
为了让你在不使用海量数据集的前提下仍有机会使用这些模型，我们用更简单的“日期转换”任务替代语言翻译任务。

网络将输入以各种可能格式（例如“1958年8月29日”，“03/30/1968”，“1987年6月24日”）写成的日期，并将它们转换成标准化的，机器可读的日期（例如“1958 -08-29“，”1968-03-30“，”1987-06-24“）
'''

print("--------------------------------------了解数据集并对数据进行预处理----------------------------------------------------------")


# 我们将在一个包含10000个人类可读日期的数据集及其等效的标准化机器可读日期上训练模型
# 加载并打印前10个数据集
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print(dataset[:10])


# 我们对数据进行预处理，将原始文本数据映射到相应的索引值。
# 并设置Tx = 30（我们假设它是人类可读日期的最大长度;如果我们得到更长的输入，我们将不得不截断它），Ty = 10（因为“YYYY-MM-DD”是10长字符）。
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)


# 随意更改index, 你可以查看预处理后的例子
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])


print("--------------------------------------------带注意力机制的机器翻译----------------------------------------------------")

# 如果你需要把一个书的段落从法文翻译成英文，那么你不会阅读完整段文字，关闭这本书后再去翻译它
# 一般你会在翻译过程中，同时阅读/重读，并专注于你正写下的英语相对应的法语部分。

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1, name='dot')


# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###
    
    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)


# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True), name='bidirectional_1')(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model



model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))


model.summary()


### START CODE HERE ### (≈2 lines)
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
### END CODE HERE ###


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


#model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)


#model.load_weights('models/model.h5')


#EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
#for example in EXAMPLES:
    
    #source = string_to_int(example, Tx, human_vocab)
    #source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    #prediction = model.predict([source, s0, c0])
    #prediction = np.argmax(prediction, axis = -1)
    #output = [inv_machine_vocab[int(i)] for i in prediction]
    
    #print("source:", example)
    #print("output:", ''.join(output))
    
    
#model.summary()


#attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)


print("----------------------------------------------------总结---------------------------------------------------------------")

#下面是你应该从本次任务中记住的：

# 1、机器翻译模型可用于从一个序列映射到另一个序列。它们不仅用于翻译人类语言（如法语 - >英语），还用于日期格式翻译等任务。
# 2、注意力机制允许网络在产生输出的特定部分时专注于输入的最相关部分。
# 3、使用注意力机制的网络可以把长度为Tx的输入转化为长度为Ty的输出 ，Tx和Ty的长度可以不一样
# 4、你可以可视化注意力的权重，去观察当生成每一个输出的时候，网络把注意力放在哪儿