clear
clc
%强行搞7张图过来,4猫3狗
pic = imread('1.jpg');  pic1 = reshape(pic,64*64*3,1);
pic = imread('2.jpg');  pic2 = reshape(pic,64*64*3,1);
pic = imread('3.jpg');  pic3 = reshape(pic,64*64*3,1);
pic = imread('4.jpg');  pic4 = reshape(pic,64*64*3,1);
pic = imread('5.jpg');  pic5 = reshape(pic,64*64*3,1);
pic = imread('6.jpg');  pic6 = reshape(pic,64*64*3,1);
pic = imread('7.jpg');  pic7 = reshape(pic,64*64*3,1);

X = [pic1 pic2 pic3 pic4 pic5 pic6 pic7];
X = double(X);
clear pic1 pic2 pic3 pic4 pic5 pic6 pic7 pic
m = 7;
Y = [1 1 1 1 0 0 0];
[w b] = initialize_with_zeros(64*64*3);

[w,b,cost_i] = optimize(w,b,X,Y,60000,0.000000001,m);


%检验已完成参数
pic = imread('test1.jpg');
pic = reshape(pic,64*64*3,1);
pic = double(pic);
predict(w,b,pic)



function [cost dw db] = propagate(m,w,b,X,Y)
    % m --- 训练样本数
    % w --- 权重,(nx,1)维
    % b --- 偏置值,实数R
    % X --- 样本集 (nx,m)维
    % Y --- 样本标签 (1,m)维
    % cost --- 回归函数的代价函数
    % dw --- 对 w 的导数,(nx,1)维
    % db --- 对 b 的导数,实数R
    K = w.'*X+b;
    A = sigmoid(K);                       % (1,m)维
    cost = -1/m*sum(Y.*log(A)+(1-Y).*log(1-A));  % (1,1)维  
    dz = A-Y;                                   % (1,m)维
    dw = 1/m*X*dz.';                            % (nx,1)维
    db = 1/m*sum(dz);                           % 实数R
end

function [para_w para_b cost_i] = optimize(w,b,X,Y,num_iter,rate,m)
    % w --- 权重,(nx,1)维
    % b --- 偏置值,实数R
    % X --- 样本集 (nx,m)维
    % Y --- 样本标签 (1,m)维
    % num_iter --- 迭代次数
    % rate --- 学习率
    % params 学习中的各个w,b值
    % dw,db  学习中的各梯度
    % costs  学习中的各个代价值,可用来观察代价曲线变化
    for i=1:num_iter
       [cost dw db] = propagate(m,w,b,X,Y);
       w = w-rate*dw;
       b = b-rate*db;
       if i==1
           cost_i = cost;
       else
           cost_i = [cost_i; cost];
       end
       para_w = w;
       para_b = b;       
    end
end

function cat = predict(w,b,pic)
    m = 1;
    A = sigmoid(w.'*pic+b);
    if A<0.5 
        cat = 0;
    else
        cat = 1;
    end
end

function s = sigmoid(x)
    s = 1./(1+exp(-x));
end
function [w b] = initialize_with_zeros(dim)
    w = zeros(dim,1);
    b = 0;
end