import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt


''' 用词向量表示来构建一个Emojifier

你有没有想过让你的短信更具表现力？你的emojifier应用程序将帮助你做到这一点。
比如有这么一句话，"恭喜！让我们喝咖啡聊天，爱你！" emojifier可以自动将它变成"恭喜！让我们喝咖啡和谈话☕️。爱你！❤️"

你会实现一个模型，它接受一个句子的输入（比如“我们今晚去看看棒球比赛！”）并找到最适合这个句子的表情符号。
在许多表情符号界面中，您需要记住❤️是“心脏”符号而不是“爱”符号。
但是使用词向量，即使你的训练集只将几个词明确地与特定的表情符号相关联，你的算法也能够将测试集中的单词进行概括并关联到相应的表情符号，即使这些词并没有在你的训练集中出现。
这使您即使使用小型训练集，也可以建立从句子到表情符号的精确映射。

在本练习中，您将从使用词嵌入的基线模型（Emojifier-V1）开始，然后构建更复杂的模型（Emojifier-V2），最后进一步整合LSTM。

--------------------------

Emojifier-V1 是自己定义了一个基础的模型结构，包括损失函数等, 它没有使用Basic RNN
Emojifier-V2 整合的是LSTM，通过lstm的长短期记忆，解决了Emojifier-V1无法解决的语句排序问题（V2的模型定义使用的是Keras，细节尚未了解）
Emojifier 两个模型都使用了预先训练好的 Glove 词向量

'''

# 加载数据集
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# max(X_train, key=len) 按长度选取X_train中最长的句子
maxLen = len(max(X_train, key=len).split())
print(maxLen)

index = 1
print(X_train[index], label_to_emoji(Y_train[index]))


# 模型的输入是一个对应于一个句子的字符串（例如“我爱你”）。
# 在代码中，输出将是一个形状（1,5）的概率向量，然后你通过一个argmax层来提取最有可能的表情符号的索引。
# 下面通过convert_to_one_hot，将Y标签转换为softmax分类器可接受的shape
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# 随便更改index值，验证转换后的 one-hot 向量值
index = 50
print(Y_train[index], "is converted into one hot", Y_oh_train[index])


print("----------------------------------------------------Emojifier-V1---------------------------------------------------------------")


# 将输入句子转换为单词向量表示，然后将其平均到一起
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


word = "cucumber"
index = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])


# 将每个句子转换为小写，然后将句子拆分为单词列表。X.lower()并X.split()可能有用。
# 对于句子中的每个单词，访问其GloVe表示。然后，平均所有这些值
# GRADED FUNCTION: sentence_to_avg

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    将一个字符串的句子转换为单词列表，提取每个单词的Glove表示，并将其值平均到编码句子含义的单个向量中
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(word_to_vec_map[words[0]].shape)
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    
    ### END CODE HERE ###
    
    return avg


# 验证 sentence_to_avg
avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = ", avg)


# GRADED FUNCTION: model

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    n_y = 5                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    
    # Optimization loop
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples
            
            ### START CODE HERE ### (≈ 4 lines of code)
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.matmul(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(Y_oh[i] * np.log(a))
            ### END CODE HERE ###
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)

X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
 'Lets go party and drinks','Congrats on the new job','Congratulations',
 'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
 'You totally deserve this prize', 'Let us go play football',
 'Are you down for football this afternoon', 'Work hard play harder',
 'It is suprising how people can be dumb sometimes',
 'I am very disappointed','It is the best day in my life',
 'I think I will end up alone','My life is so boring','Good job',
 'Great so awesome'])

print(X.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(type(X_train))


pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)


# 检测训练集和测试集的训练精度，你会发现效果都不错
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

# 在训练集中，算法看到了"i love you"对应标签❤️ 。但是训练集中没有出现“adore”(崇拜)一词。但是运行下面的代码，你会发现句子"i adore you"预测的标签也是❤，是不是很神奇
# 但是算法没有得到“不快乐”的正确答案。因为这种算法忽略了词序
X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)


# 打印混淆矩阵能帮助您了解您的模型对哪些类别的识别是有困难的
print(Y_test.shape)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)


# 你应该记住这部分内容：
# 1、即使只有127个训练例子，你也可以得到一个相当好的Emojifying模型。这是泛化能力强的词向量赋予的。
# 2、Emojify-V1在诸如“这部电影不好，不愉快”这样的句子上表现不佳，因为它不理解单词的组合 - 它只是将所有单词的嵌入向量集中在一起，而没有关注单词的排序。您将在下一部分中构建一个更好的算法。


print("----------------------------------------------------Emojifier-V2---------------------------------------------------------------")


# 让我们建立一个LSTM模型，它将句子序列作为输入。这个模型将能够考虑文字排序。
# Emojifier-V2将继续使用预先训练的词嵌入来表示单词，但会将它们输入到LSTM中，其工作是预测最合适的表情符号。

import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


# 本次练习中，每次训练Keras使用的是小批量数据。然而，大多数深度学习框架要求同一个小批量中的所有序列具有相同的长度。
# 因为这是允许矢量化工作的原因：如果你有一个3字的句子和一个4字的句子，那么他们所需要的计算是不同的（一个需要3个步骤的LSTM，一个需要4个步骤）

# 常见的解决方法是使用填充。具体而言，设置最大序列长度，并将所有序列填充到相同长度
# 例如，最大序列长度为20，我们可以用“0”填充每个句子，以便每个输入句子的长度为20
# 在这个例子中，任何超过20个单词的句子都必须被截断。选择最大序列长度的一个简单方法是只选择训练集中最长句子的长度

# GRADED FUNCTION: sentences_to_indices

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    将一组句子(字符串) 转换为与句子中的单词对应的索引数组。
    输出的 shape 需要能够赋予 Embedding() 函数（就像图四描述的一样）
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
            
    ### END CODE HERE ###
    
    return X_indices


# 验证 sentences_to_indices，检测结果
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1, word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)


# GRADED FUNCTION: pretrained_embedding_layer

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    创建一个Keras嵌入图层并且加载预先训练好的Glove50维度的词向量
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    # 一个Embedding()图层可以使用预训练好的值进行初始化。训练过程中这些值可以是固定的，也可以是在数据集上进一步训练的。
    # 但是，如果您标记的数据集很小，则通常不值得尝试训练大量的预先训练的嵌入集。所以此处 trainable=False
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])




# 构建Emojifier-V2模型。您将使用您已构建的嵌入图层来完成此操作，并将其输出提供给LSTM网络
#
# GRADED FUNCTION: Emojify_V2

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings) # return_sequences 是否要返回每个隐藏的状态或仅返回最后一个状态。
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, output=X)
    
    ### END CODE HERE ###
    
    return model


# 运行以下单元格以创建模型并检查其摘要
# 因为数据集中的所有句子都少于10个单词，所以我们选择了max_len = 10
# 你应该看到你的架构，它使用了“20,223,927”参数，其中20,000,050（词嵌入）是无需训练的，其余的223,877需要训练
# 因为我们的词汇量有400,001字（有效索引从0到400,000），所以不需训练的参数有 400,001 * 50 = 20,000,050个
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()

# 输出：
#_________________________________________________________________
#图层（类型）输出形状参数＃   
#================================================== ===============
#input_4（InputLayer）（None，10）0         
#_________________________________________________________________
#embedding_5（嵌入）（无，10,50）20000050  
#_________________________________________________________________
#lstm_5（LSTM）（无，10，128）91648     
#_________________________________________________________________
#dropout_5（Dropout）（None，10，128）0         
#_________________________________________________________________
#lstm_6（LSTM）（无，128）131584    
#_________________________________________________________________
#dropout_6（Dropout）（None，128）0         
#_________________________________________________________________
#dense_3（Dense）（None，5）645       
#_________________________________________________________________
#activation_3（激活）（无，5）0         
#================================================== ===============
#总参数：20,223,927
#可训练参数：223,877
#不可训练的参数：20,000,050


# 像往常一样，在Keras中创建模型后，您需要编译它并定义要使用的损失，优化程序和指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型，训练到最后，模型在训练集上应该拥有接近100％的精确度
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

# 评估测试集的精确度
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# 您大致会获得80％至95％的测试精度。运行下面的代码查看错误标记的示例
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())
        
        
# 更改下面的句子去查看你的预测结果，记得确保所有的词都在Glove词嵌入中
x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


# 以前，Emojify-V1模型没有正确标注“不快乐”，但我们的Emojiy-V2的实现是正确的（Keras的输出每次都是随机的，所以你可能没有得到相同的结果）
# 目前的模型在理解否定（如“不高兴”）方面仍然不是很强大，因为训练集很小，有很多否定的例子
# 但是如果训练集较大，在理解这样复杂的句子时，LSTM模型会比Emojify-V1模型好得多。


print("----------------------------------------------------总结---------------------------------------------------------------")

# 你应该记住的是：

# 1、如果您的NLP任务的训练集较小，则使用词嵌入可以显着帮助您的算法。词嵌入能让你的模型在测试集上的词工作良好，即使那些词并没有出现在您的训练集中。
# 2、用Keras训练序列模型（以及大多数其他深度学习框架）需要了解一些重要细节：
#  （1）要使用小批量，序列需要填充，以便小批量中的所有示例具有相同的长度。
#  （2）一个Embedding()图层可以使用预训练好的值进行初始化。训练过程中这些值可以是固定的，也可以是在数据集上进一步训练的。但是，如果您标记的数据集很小，则通常不值得尝试训练大量的预先训练的嵌入集。
#  （3）LSTM()有一个标志return_sequences来决定是否要返回每个隐藏的状态或仅返回最后一个状态。
#  （4）您可以在使用LSTM()后调用Dropout()以调整您的网络。


