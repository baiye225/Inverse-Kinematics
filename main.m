 clc, clear, close all, close all force

%% setup all initial parameters
%"Robot3DoF" "Robot5DoF" "SCARA"

MyRobot       = "Robot5DoF";
MyRobotFKData = "DataRobot5DoF4.mat";
MyMLResult    = "MLResultRobot4DoF2.mat";


% initialize classes
obj = IKMLSIM;
obj = obj.RobotInit(MyRobot);

%% generate data set
% Data = obj.GetDataSet();

%% Machine learning analysis
load(fullfile(".", "FK Data", MyRobotFKData))
MLResult = obj.MLAnalysis(Data);
   
load(fullfile(".", "ML Result", MyMLResult))
obj.PlotResult(MLResult);

%% simulate trajectory
% traj = obj.GetTraj("Traj1");
% net = MLResult.net;
% [AllTraj, Result] = obj.TrajSIM(net, traj);
% obj.PlotTraj(AllTraj)






