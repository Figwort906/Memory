function [p_out] = UWBlocation(p,err)
% p = [5,5;10,9;15,12;20,22;25,3]; %real location
% 定义函数 UWBlocation，输入参数为 p（真实位置矩阵）和 err（误差系数），函数返回值为 p_out（计算得到的位置）。
% 这里给出了真实位置的示例，是一个二维坐标矩阵。

p_out = p./p;
% 初始化 p_out，这里目前的操作结果是将每个坐标变成了(1,1)，可能后续会被重新赋值。

X_ = p(:,1);
Y_ = p(:,2);
% 将输入的真实位置矩阵 p 的第一列（x 坐标）赋值给 X_，第二列（y 坐标）赋值给 Y_。

baseP = [5,10;10,5;30,25;25,30]; %Location of signal transmission
% 定义信号发射源的位置矩阵 baseP。

baseX_ = baseP(:,1);
baseY_ = baseP(:,2);
% 将信号发射源位置矩阵的第一列（x 坐标）赋值给 baseX_，第二列（y 坐标）赋值给 baseY_。

R = zeros(length(X_),length(baseX_)); %radius
% 初始化距离矩阵 R，用于存储每个真实位置点到每个信号发射源的距离。

for i=1:length(X_)
    R(i,:) = ((X_(i)-baseX_).^2+(Y_(i)-baseY_).^2).^0.5;
end
% 计算每个真实位置点到每个信号发射源的距离，使用两点间距离公式。

time = R/(3e8);
% 将距离转换为时间，假设信号传播速度为 3e8 m/s。

time_actually = time + err*randn(length(X_),length(baseX_)).*time;
% 引入误差，通过在实际时间上加上误差系数 err 乘以随机数生成的误差项。

R_calcu = time_actually*3e8; %radius calculated
% 根据带有误差的时间重新计算距离。

H = [
    baseX_(2)-baseX_(1),baseY_(2)-baseY_(1);
    baseX_(3)-baseX_(1),baseY_(3)-baseY_(1);
    baseX_(4)-baseX_(1),baseY_(4)-baseY_(1)];
% 定义矩阵 H，用于后续计算。

for i=1:length(X_)
    % HX=a
    a = 0.5*[
        baseX_(2).^2+baseY_(2).^2-R_calcu(i,2).^2-baseX_(1).^2-baseY_(1).^2+R_calcu(i,1).^2;
        baseX_(3).^2+baseY_(3).^2-R_calcu(i,3).^2-baseX_(1).^2-baseY_(1).^2+R_calcu(i,1).^2;
        baseX_(4).^2+baseY_(4).^2-R_calcu(i,4).^2-baseX_(1).^2-baseY_(1).^2+R_calcu(i,1).^2];
    p_out(i,:) = (pinv(H)*a)';
end
% 通过循环计算每个真实位置点对应的计算位置 p_out，使用最小二乘法求解。

%% plot
fig(p,p_out);
% 调用绘图函数 fig。

%% function
    function fig(p,p_out)
        figure;
        scatter(p(:,1),p(:,2),'g');
        hold on
        scatter(p_out(:,1),p_out(:,2),'r');
        legend('ideal','calcu');
    end
end