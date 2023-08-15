% backup previous or unused methods(do not use!!!)
classdef IKOthers
    properties
        % FABRIK vs Normal ANN
        % test data
        % [0, 0, 90] -> [0, 30, 120]
        % TargetAng = [0, 30, 120];
        % MyPos = [0, 0, 10;
        %          0, 0, 16;
        %          6, 0, 16];
        % TargetPos = [6, 0, 10];
        
        % [0, 70, 10] -> [0, 75, 20]
        % TargetAng = [0, 75, 20];
        % MyPos = [0, 0, 10;
        %          5.6382, 0, 12.0521;
        %          11.5470, 0, 13.0940];
        % TargetPos = [11.7727, 0, 11.0300];
        
        % [0, 80, -10] -> [0, 75, -15] (multiple solution)
        % TargetAng = [0, 75, -15];
        % MyPos = [0, 0, 10;
        %          5.9088, 0, 11.0419;
        %          11.5470, 0, 13.0940];
        % TargetPos = [10.9917, 0, 14.5529];
        
        % [20, 70, 10] -> [30, 75, 20] (spatial)
        % TargetAng = [30, 75, 20];
        % MyPos = [0         0   10.0000;
        %         5.2981    1.9284   12.0521;
        %         10.8506    3.9493   13.0940];
        % TargetPos = [10.1955, 5.8864, 11.0300];
    end

    methods
        %% function: prepare the current input based on the method
        function Input = GetInput(obj, MethodIndex, targetPos, MyLastData)
            % <input>
            % MethodIndex - different trainning method(1,2,3)
            % targetPos   - the current desired position
            % MyLastData  - the last status based on MyData

            % normalize position and angle separately(optional)
            targetPos = obj.NormalizeData(targetPos, "Position", "forward");

            % single method or multiple method
            if obj.TrainingType == "All"
                MyInputType = obj.TypeSet(MethodIndex);
            else
                MyInputType = obj.TrainingType;
            end

            % get the curent input
            switch MyInputType
                case "Normal" % normal
                    Input = targetPos;

                case "Type1" % type1
                    LastAng = MyLastData(:, end-2:end);

                    % normalize position and angle separately(optional)
                    [~, LastAng] = obj.NormalizeData(LastAng,...
                                                       "Now", "Angle", "forward");
                    Input   = [targetPos LastAng];

                case "Type2" % type2
                    % update the information at the previous status
                    LastAng       = MyLastData(:, end-2:end);
                    LastAllPos    = obj.Theta2Pos(LastAng);
                    LastJointBPos = LastAllPos(:, 4:6);
                    LastPos       = MyLastData(:, 1:3);

                    % normalize position and angle separately(optional)
                    LastJointBPos = obj.NormalizeData(LastJointBPos,...
                                                      "Now", "Position", "forward");
                    LastPos  = obj.NormalizeData(LastPos, "Now", "Position", "forward");
                    Input         = [targetPos LastPos LastJointBPos];

                case "Type3" % type3
                    % to be done
            end

        end
        
        %% function: Machine Learning feedback simulation
        function [AllTraj, Result] = TrajSIM(obj, net, traj)
            % initialize data
            n       = length(traj(:, 1)); % total iteration
            MyAng   = zeros(n, 3);        % predicted joint angle
            MyPos   = zeros(n, 9);        % predicted end-effector and joint position
            load("NormalPara.mat")        % manually get normalization parameters
            obj.NormalPara = NormalPara;

            % initialize 1st position and joint angle
            MyAng(1, :) = [0 0 90];         
            MyPos(1, :) = obj.Theta2Pos(MyAng(1, :));

            % integrate position and angle as the same version of all
            % methods
            MyData = cell(1, 3);
            for i = 1: 1: length(net)
                MyData{i} = [MyPos(:, 1:3) MyAng];
            end
 
            % initialize waitbar paramters
            f = waitbar(0, "Predict Position and Joint Angle...");

            % start to simulate since the 2nd point, assume 1st point is known     
            for i = 2: 1: n
                % start to predict at the current desired position
                for j = 1: 1: length(net)   
                    % prepare input
                    Input     = obj.GetInput(j, traj(i, :), ...
                                             MyData{j}(i-1, :));

                    % use trained network to predict joint angle
                    OutputAng = net{j}(Input');
                    
                    % recover the data from the normalization
                    OutputAng = obj.NormalizeData(OutputAng',...
                                                      "Now", "Angle", "reverse");

                    % use predicted joint angle to calculate predicted position
                    OutputPos = obj.Theta2Pos(OutputAng);

                    % integrate the output
                    MyData{j}(i, :) = [OutputPos(:, 1:3) OutputAng];
                end
                % update waitbar
                MsgLine1 = sprintf("Predicting...");
                MsgLine2 = sprintf("%d/%d (%0.2f%%)", i-1, n-1, (i/n)*100);
                msg      = {MsgLine1, MsgLine2};
                waitbar((i-1)/(n-1), f, msg);
            end
            % close wait bar
            close(f);

            % integrate all trajectory
            AllTraj = cell(1, length(net) + 1);            
            AllTraj{1} = traj;                  % 1st. desired trajectory  
            for i = 1: 1: length(net)
                AllTraj{i + 1} = MyData{i}; % 2-4. trajectories of three methods
            end
            
            % calculate traj MSE
            Result = zeros(1, length(net));    
            for i = 1: 1: length(net)
                Result(i) = immse(AllTraj{1}, AllTraj{i + 1}(:, 1:3));
            end
            
        end

        %% function: Machine Learning Test for Trajectory Simulation
        function net = MLTrajSIM(obj, Data)
        
            % train all original data
            % initialize normalization data
            Data = obj.SetupNormal(Data);
            
            % prepare training input/output
            TrainData = obj.GetInputOutput(Data);

            % process input data with related types
            TrainData = obj.ProcessInput(TrainData);

            % train data (all types may include)
            for i = 1: 1: length(TrainData)
                net{i} = obj.ANNTrain(TrainData(i));

            end
     
        end

        %% function: ANN test
        function Result = ANNTest(~, net, TrainData, TestData)
            TrainPredict  = net(TrainData.Input')';
            TestPredict   = net(TestData.Input')';

            % integrate results
            Result.TrainData = TrainData;
            Result.TestData  = TestData;
            Result.TrainPredict = TrainPredict;
            Result.TestPredict = TestPredict;
            Result.ModelType = "ANN";
        end
        
        %% function: 3 DoF -> convert joint angle(degree 1x3) into position
        function position = Theta2Pos(obj, thetaDegree)

%                 % Euler transformation
%                 % position end-effector A
%                 positionA = MRTheta1 * (MRTheta2 * (MRTheta3 * [0; 0; obj.L] +...
%                            [0; 0; obj.L]) + [0; 0; obj.H]);
%     
%                 % position joint B
%                 positionB = MRTheta1 * MRTheta2 *[0; 0; obj.L] + [0; 0; obj.H];
%     
%                 % position joint C
%                 % no rotation affect position C ([0; 0; H])
%                 positionC = MRTheta1 * [0; 0; obj.H];


            end

        %% function: 4 DoF -> convert joint angle(degree 1x3) into position
        function position = Theta2Pos2(obj, thetaDegree)
            % degree to rad
            thetaRad = deg2rad(thetaDegree);

            for i = 1: 1: length(thetaRad(:, 1))
                % get the current matrix of rotation(MR)
                MRTheta1 = obj.GetMatrixRotation(thetaRad(i, 1), 'z');
                MRTheta2 = obj.GetMatrixRotation(thetaRad(i, 2), 'y');
                MRTheta3 = obj.GetMatrixRotation(thetaRad(i, 3), 'y');
                MRTheta4 = obj.GetMatrixRotation(thetaRad(i, 4), 'y');
    
                % Euler transformation
                % position joint D(fixed joint)
                positionD = MRTheta1 * [0; 0; obj.RobotGeo(1)];

                % position joint C
                positionC = MRTheta1 * MRTheta2 * [0; 0; obj.RobotGeo(2)]...
                            + positionD;

                % position joint B
                positionB = MRTheta1 * MRTheta2...
                            * MRTheta3 * [0; 0; obj.RobotGeo(3)]...
                            + positionC;

                % position end-effector A
                positionA = MRTheta1 * MRTheta2...
                            * MRTheta3 * MRTheta4 * [0; 0; obj.RobotGeo(4)]...
                            + positionB;
    
                % intergrate position
                position(i, :) = [positionA' positionB' positionC' positionD'];
            end            
        end
        %% function: generate angle & position data(old version)
        function Data = GetIKData2(obj)
            % initialize waitbar paramters
            f = waitbar(0, "Generate data set...");

            % initialize other paramters
            total = 1;
            n     = 1;

            % generate joint angle set
            obj.RobotJoint
            for i = 1: 1: length(obj.RobotJoint(:, 1))
                JointSet{i} = obj.RobotJoint(i, 1) :...
                              obj.RobotJointInterval(i) :...
                              obj.RobotJoint(i, 2);
                total = total * length(JointSet{i});
            end
            
            % start to generate position by forward kinematic
            for i = 1 : 1 : length(JointSet{1})
                for j = 1 : 1 : length(JointSet{2})
                    for k = 1 : 1 : length(JointSet{3})
                        % update waitbar
                        msg = sprintf("Process data set with %d/%d (%0.2f%%)",...
                                            n, total, n/total * 100);
                        waitbar(n/total, f, msg);
            
                        % get the current joint angle
                        theta1       = JointSet{1}(i);
                        theta2       = JointSet{2}(j);
                        theta3       = JointSet{3}(k);        
                        thetaDegree  = [theta1 theta2 theta3]; 

                        % convert the current joint angle into position
                        position = obj.Theta2Pos(thetaDegree);

                        % append it to Data
                        if obj.ValiPos(position(1, 1:3)) == "True"
                            Data(n, :) = [position thetaDegree];
                            n = n + 1;
                        end
                    end
                end
            end
            
            % close waitbar
            close(f)
        end

        %% function: generate random angle & position data
        function Data = GetIKDataRnd(obj)
            % initialize waitbar paramters
            f = waitbar(0, "Generate data set...");

            % initialize other paramters
            total = 10000;
           
            % start to generate position by forward kinematic
            for i = 1 : 1 : total   
                % get the current joint angle
                theta1 = round(rand()* (obj.ThetaRange(1, 2) -obj.ThetaRange(1, 1))...
                               + obj.ThetaRange(1, 1));
                theta2 = round(rand()* (obj.ThetaRange(2, 2) -obj.ThetaRange(2, 1))...
                               + obj.ThetaRange(2, 1));
                theta3 = round(rand()* (obj.ThetaRange(3, 2) -obj.ThetaRange(3, 1))...
                               + obj.ThetaRange(3, 1));       

                thetaDegree = [theta1 theta2 theta3];

                % convert the current joint angle into position
                position = obj.Theta2Pos(thetaDegree);

                % append it to Data
                Data(i, :) = [position thetaDegree];

                % update waitbar
                msg = sprintf("Process data set with: %d/%d (%0.2f%%)",...
                               i, total, i/total * 100);   
                waitbar(i/total, f, msg);
            end
            
            % close waitbar
            close(f)
        end
        
        %% function: plot fitness result
        function PlotResult(obj, Result)
%             MyPara.ModelType = ModelType;
%             MyPara.SampleType = "Training";
%             MyPara.MySubTitle = ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3"];
%             MyPara.ErrorUnit = "Degree";
%             MyPara.FigurePos = [100, 300, 800, 600];
%             obj.PlotSingle(Result, MyPara)
% 
%             MyPara.ModelType = ModelType;
%             MyPara.SampleType = "Testing";
%             MyPara.MySubTitle = ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3"];
%             MyPara.ErrorUnit = "Degree";
%             MyPara.FigurePos = [1000, 300, 800, 600];
%             obj.PlotSingle(Result, MyPara)
% 
%             MyPara.ModelType = ModelType;
%             MyPara.SampleType = "Position";
%             MyPara.MySubTitle = ["Trainning Position", "Testing Position"];
%             MyPara.ErrorUnit = "mm";
%             MyPara.FigurePos = [500, 300, 800, 600];
%             obj.PlotSingle(Result, MyPara)
        end
        
        %% function: plot single group comparison(train or test)(invalid)
        function PlotSingle2(obj, MyResult, MyPara)
            % split data
            if  MyPara.SampleType == "Training"
                type = "Train";
            elseif MyPara.SampleType == "Testing"
                type = "Test";
            elseif MyPara.SampleType == "Position"
                type = "Pos";
            end
                 
            % plot
            figure
            set(gcf, 'Position',  MyPara.FigurePos)
            MySubTitle = MyPara.MySubTitle;
            MyTitle = tiledlayout(length(MySubTitle), 1, 'TileSpacing','Compact');
            
            
            % start to plot each joint angle eror
            for i = 1: 1: length(MySubTitle)
                ax(i) = nexttile;

                % start to plot each type error
                for j = 1: 1: length(MyResult)
                    if type == "Train" || type == "Test"
                        CurrentData = MyResult(j).(type)(:, i);
                    elseif type == "Pos"
                        CurrentData = MyResult(j).(type){i};
                    end
                    plot(CurrentData, 'DisplayName', 'CurrentData')
                    hold on
                end
                hold off
%                 yMean = mean(CurrentData, 'all');
%                 axis([-inf inf -inf yMean * 10])
                title(MySubTitle(i));
                axis padded;
                grid minor;
            end
            
            % set parameters of the figure
            if obj.TrainingType == "All"
                legend(ax(1), obj.TypeSet)
            end
            xlabel(MyTitle, "Date Sample Index");
            ylabel(MyTitle, compose("Numerical Error(%s)", MyPara.ErrorUnit));    
            title(MyTitle, compose("%s result vs Joint angle",...
                  MyPara.ModelType), compose("<%s Samples>",MyPara.SampleType))
        end
    end

end