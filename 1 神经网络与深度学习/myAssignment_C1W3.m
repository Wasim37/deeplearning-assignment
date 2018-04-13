clear

%形状1
theta = linspace(1,360,400);    k = 4;
r = 4*sind(theta*k);
x1 = r.*cosd(theta)+0.4*rand(1,400);
x2 = r.*sind(theta)+0.4*rand(1,400);
Y(1:50) = 1;    Y(51:100) = 0;
Y(101:150) = 1; Y(151:200) = 0;
Y(201:250) = 1; Y(251:300) = 0;
Y(301:350) = 1; Y(351:400) = 0;
X = [x1;x2];

% % 形状2
% x11 = [-2;3]+rand(2,100);   Y(1:100)=1;
% x12 = [1;1.5]+rand(2,100);    Y(101:200)=0;
% x13 = [3;0]+rand(2,100);   Y(201:300)=1;
% x14 = [-2;-2]+rand(2,100);  Y(301:400)=0;
% X = [x11 x12 x13 x14];

for i=1:length(X)
    if Y(i)==1
        plot(X(1,i),X(2,i),'g*'); hold on
    else 
        plot(X(1,i),X(2,i),'black*'); hold on
    end
end



% 定义神经网络结构
nx = 2;         % 输入层
nh = 20;         % 中间隐藏层
ny = 1;         % 输出
m = size(X,2);  % 样本数量

% 参数初始化
W1 = 0.2*randn(nh,nx);
b1 = zeros(nh,1);
W2 = 0.2*randn(ny,nh);
b2 = zeros(ny,1);
rate = 1.2;
iter = 1000;

% 循环
for i=1:2000
    rate = rate-exp(-i);
    % 前向传播
    Z1 = W1*X+b1;
    A1 = tanh(Z1);
    Z2 = W2*A1+b2;
    A2 = sigmoid(Z2);

    % 计算cost
    logprobs = Y.*log(A2)+(1-Y).*log(1-A2);
    cost = squeeze(-1/m*sum(logprobs))

    % 计算梯度
    dZ2 = A2-Y;
    dW2 = 1/m*dZ2*A1.';
    db2 = 1/m*sum(dZ2,2);
    
    dZ1 = (W2.')*dZ2.*(1-A1.^2);
    dW1 = 1/m*dZ1*X.';
    db1 = 1/m*sum(dZ1,2);

    %参数更新
    W1 = W1-rate*dW1;
    b1 = b1-rate*db1;
    W2 = W2-rate*dW2;
    b2 = b2-rate*db2;
end


% 预测
X = [4*rand(1,400) -4*rand(1,400) -4*rand(1,400) 4*rand(1,400);...
    4*rand(1,400) -4*rand(1,400) 4*rand(1,400) -4*rand(1,400)];
Z1 = W1*X+b1;
A1 = tanh(Z1);
Z2 = W2*A1+b2;
A2 = sigmoid(Z2);
for i=1:length(X)
    if A2(i)>0.5
        plot(X(1,i),X(2,i),'ro');   hold on
    else
        plot(X(1,i),X(2,i),'bo');   hold on
    end
end

function s = sigmoid(x)
    s = 1./(1+exp(-x));
end
