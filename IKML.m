  % Machine Learning For IK
classdef IKML < IKRobot & IKProperty
    properties
        % the number of set of train-test data by random division
        NumSet      = 100;
        NumKFoldSet = 10;
        NumFold     = 10;

        % paramter for data set waitbar text
        Para = {["ANN", "Random Division"];
                ["SVM", "Random Division"]; 
                ["ANN", "K-Fold"];
                ["SVM", "K-Fold"];
                ["ANN", "All"]};
        
        % switch of normalization("on" or "off")
        NormalSwitch = "on";

    end

    methods
        %% function: Machine Learning Analysis
        function Result = MLAnalysis(obj, Data)
            % normalize data and setup normalized threhold
            [obj, Data] = obj.SetupNormal(Data);
    
%             % get train and test data
%             [KTrainSet, KTestSet] =...
%                       obj.GetTrainTestDataKFoldSet(Data, obj.NumKFoldSet);
%     
%             %run machine learning
%             Result.ANNKFoldSet = obj.MLTrainTestSet(@obj.MLTrainTestKFold,...
%                                                     @obj.ANNTrainTest,....
%                                                     KTrainSet, KTestSet, ...
%                                                     obj.Para{3});
    
    %         Result.SVMKFoldSet = obj.MLTrainTestSet(@obj.MLTrainTestKFold,...
    %                                                 @obj.SVMTrainTest,....
    %                                                 KTrainSet, KTestSet, ...
    %                                                 obj.Para{4});
            
            % train all data
            Result = obj.MLFullTrain(@obj.ANNTrainTest, Data, obj.Para{5});

        end

        %% function: pre-process original data
        function [obj, Data] = SetupNormal(obj, Data)
            % find max and min in the normalization   
            DataPos = Data(:, obj.MLInputIndex);
            obj.RobotNormalPara.Position.max = max(DataPos, [], 'all');
            obj.RobotNormalPara.Position.min = min(DataPos, [], 'all');

            DataAng = Data(:, obj.MLOutputIndex);
            obj.RobotNormalPara.Angle.max = max(DataAng, [], 'all');
            obj.RobotNormalPara.Angle.min = min(DataAng, [], 'all');

            % normalize position and angle separately(optional)
            if obj.NormalSwitch == "on"
                fprintf("Start to normalize data...\r");
                Data(:, obj.MLInputIndex) = obj.NormalizeData(Data(:, obj.MLInputIndex),...
                                                              "Position", "forward");
                Data(:, obj.MLOutputIndex) = obj.NormalizeData(Data(:, obj.MLOutputIndex),...
                                                               "Angle", "forward");
            end
             
        end

        %% function: normalization and reverse normalization
        function Data = NormalizeData(obj, Data, type, command)
            % load max and min in the normalization
            DataMax = obj.RobotNormalPara.(type).max;
            DataMin = obj.RobotNormalPara.(type).min;

            % normalize or recover
            if command == "forward" && obj.NormalSwitch == "on"
                % normalize the data
                Data = (Data - DataMin) / (DataMax - DataMin);

            elseif command == "reverse" && obj.NormalSwitch == "on"
                % recover the data from the normalization
                Data = Data * (DataMax - DataMin) + DataMin;
            end
        end

        %% function: ANN all data analysis(100% training data)
        function Result = MLFullTrain(obj, MLFunc, Data, Para)
            % initialize parameters
            ModelTyple    = Para(1);
            DataSelection = Para(2);
            
            % split data into input and output
            [TrainData.Input, TrainData.Output] = obj.GetMLInputOutput(Data);

            % process input data
            TrainData = obj.ProcessInput(TrainData);

            % machine learning analysis
            fprintf("Start to use <%s> analysis with <%s> data...\r",...
                    ModelTyple, DataSelection);
            Result = MLFunc(TrainData);
            
            % Save normalization paramters
            Result.RobotNormalPara = obj.RobotNormalPara;

            % process training result
            Result.TrainData = obj.ProcessJoint(Result.TrainData); 
            Result.TrainData = obj.ProcessPos(Result.TrainData); 

            % plot result
            obj.PlotResult(Result);

        end
        %% function: randomly distribute sets of train-test data
        function [TrainSet, TestSet] = GetTrainTestDataSet(obj, Data, NumSet)
           % initialize set of train and test data
           TrainSet = cell(NumSet, 1);
           TestSet = cell(NumSet, 1);

           % start to generate set data
            for i = 1: 1: NumSet
                [Train, Test] = obj.GetTrainTestData(Data);
                TrainSet{i} = Train;
                TestSet{i} = Test;
            end
        end

        %% function: randomly distribute training data and testing data
        function [Train, Test] = GetTrainTestData(~, Data)
            % randomly divide train and test data
            [TrainData, ~, TestData] = dividerand(Data', 9/10, 0, 1/10);
            
            % transpose train, test data
            TrainData = TrainData';
            TestData = TestData';

            % get training and testing input/output
            Train = obj.GetInputOutput(TrainData);
            Test = obj.GetInputOutput(TestData);
        end
        
        %% function: split data into input and output
        function Result = GetInputOutput(~, Data)
            % the last 3 columns are outputs, others are inputs
            Result.Input  = Data(:, 1:end-3);
            Result.Output = Data(:, end-2:end);        
        end

        %% function: Get tran and test day set by k-fold
        function [KTrainSet, KTestSet] =...
                           GetTrainTestDataKFoldSet(obj, Data, NumKFoldSet)
           % initialize set of train and test data
           KTrainSet = cell(NumKFoldSet, 1);
           KTestSet = cell(NumKFoldSet, 1);

           % start to generate set data
            for i = 1: 1: NumKFoldSet
                [KTrain, KTest] = obj.GetTrainTestDataKFold(Data);
                KTrainSet{i} = KTrain;
                KTestSet{i} = KTest;
            end

        end

        %% function: Get train and test data by k-fold
        function [Train, Test] = GetTrainTestDataKFold(obj, Data)            
            % get the number of samples
            Num = length(Data(:, 1));         
            
            % generate two k-fold model
            cvData = cvpartition(Num, 'kfold', obj.NumFold);
            
            % start prepare all data
            for i = 1 : obj.NumFold
                % get success/failure training/test index
                Index.Train{i} = find(training(cvData,i));
                Index.Test{i}  = find(test(cvData,i));   
   
                % prepare training/test data at each fold
                TrainData = Data(Index.Train{i}, :);
                TestData = Data(Index.Test{i}, :);

                % split data into input and output
                [Train.Input{i}, Train.Output{i}] =...
                                                obj.GetMLInputOutput(TrainData);
                [Test.Input{i}, Test.Output{i}] =...
                                                obj.GetMLInputOutput(TestData);
            end
        end
        
        %% function: pick up input(position) and output(joint angle)
        function [Input, Output] = GetMLInputOutput(obj, Data)
            % the number of joint angle is Dof
            % data = [1, 2, 3, ..., end - n, end - n + 1, .., end]
            %         |                    |                     |
            %               Input                 Output
            Input  = Data(:, obj.MLInputIndex);
            Output = Data(:, obj.MLOutputIndex);
        end

        %% function Train, Test, Analyze, Plot Data set
        function ResultSetAll = MLTrainTestSet(~, MLTrainTestFunc,...
                                   MLFunc, TrainDataSet, TestDataSet, Para)
            % initialize parameters
            n             = length(TrainDataSet);
            ResultSet     = cell(1, n);
            ModelTyple    = Para(1);
            DataSelection = Para(2);

            % initialize waitbar paramters
            f = waitbar(0, "Train and Test Data Set...");

            % start get result set
            for i = 1: 1: n
                % train and test data
                ResultSet{i} = MLTrainTestFunc(MLFunc,...
                                          TrainDataSet{i}, TestDataSet{i});

                ResultSetAll.ElapsedTime(i) = ResultSet{i}.ElapsedTime;
                ResultSetAll.TrainAccuracy(i) = ResultSet{i}.TrainAccuracy;
                ResultSetAll.TestAccuracy(i) = ResultSet{i}.TestAccuracy;

                % update waitbar
                MsgLine1 = sprintf("%s Process data set with %s",...
                                    ModelTyple, DataSelection);
                MsgLine2 = sprintf("%d/%d (%0.2f%%)", i, n, (i/n)*100);
                msg      = {MsgLine1, MsgLine2};
                waitbar(i/n, f, msg);
            end

            % close waitbar
            close(f)
            
            % integrate other result
            ResultSetAll.ModelType = ModelTyple;
            ResultSetAll.DataSelection = DataSelection;
            ResultSetAll.Num = n;
            ResultSetAll.DataPointNum = ResultSet{1}.DataPointNumNow;
            
            % display numerical result
%             obj.DisplayNumericalResultsSet(ResultSetAll)
        end
    
        %% function: MLTrainTest <Random Division>
        function Result = MLTrainTest(obj, MLFunc, TrainData, TestData)
            ResultData = MLFunc(TrainData, TestData);

            % Get numerical results
            Result = obj.GetTrainTestAccuracy(ResultData);
            Result.ModelType    = OtherPara.ModelType;
            Result.ElapsedTime  = OtherPara.ElapsedTime;
            Result.TrainData    = TrainData;
            Result.TestData     = TestData;
            Result.TrainPredict = ResultData.TrainPredict;
            Result.TestPredict  = ResultData.TestPredict;
            Result.DataPointNumNow = length(TrainData.Input(1, :)) / 3;

        end

        %% function: MLTrainTest <K-fold>
        function Result = MLTrainTestKFold(obj, MLFunc, KTrainData, KTestData)
            % initialize outputs
            n             = length(KTrainData.Input);
            TrainAccuracy = zeros(1, n);
            TestAccuracy  = zeros(1, n);
            ElapsedTime   = zeros(1, n);

            % manually get normalization parameters(testing)
%             load("NormalPara.mat")        
%             obj.NormalPara = NormalPara;

            % start train and test data at each fold
            for i = 1: 1: n                 
                % train and test data
                TrainData.Input = KTrainData.Input{i}; % get the current fold
                TrainData.Output = KTrainData.Output{i};
                TestData.Input = KTestData.Input{i};
                TestData.Output = KTestData.Output{i};

                % process input data
                [TrainData, TestData] = obj.ProcessTrainTestInput(...
                                            TrainData, TestData);

                % train-test data (may have different kinds of input)
                for j = 1: 1: length(TrainData)
                    ResultData(j) = MLFunc(TrainData(j), TestData(j));                   
                end

                % plot result
                obj.PlotTrainTestResult(ResultData)
                
                % manually end(testing)
                error("stop running!!!")

                % train and test analysis
                Result = obj.GetTrainTestAccuracy(ResultData);

                % accumulate partial results
                TrainAccuracy(i) = Result.TrainAccuracy;
                TestAccuracy(i)  = Result.TestAccuracy;
                ElapsedTime(i) = OtherPara.ElapsedTime;
            end


            % integrate outputs
            Result.MeanTrainAccuracy = mean(Result.TrainAccuracy);
            Result.MeanTestAccuracy = mean(Result.TestAccuracy);
            Result.DataPointNumNow = length(TrainData.Input(1, :)) / 3;
            Result.ElapsedTime = sum(ElapsedTime);
            Result.ModelType = OtherPara.ModelType;
           
            % display result
%             obj.DisplayKFoldResults(Result)

        end
    
        %% function: process input data as the related type
        function [TrainData, TestData] = ProcessTrainTestInput(obj, TrainData, TestData)
            % process training input
            TrainData = obj.ProcessInput(TrainData);

            % process test input
            TestData  = obj.ProcessInput(TestData);
        end
        %% function: process input data as the related type
        function Data = ProcessInput(obj, Data)
           % generate all kinds of input data

           % normal
           DataSetInput{1} = Data.Input(:, 1:3);
           
           % type1
           DataSetInput{2} = [];
%            DataSetInput{2} = [Data.Input(:, 1:3),...
%                               [zeros(1,3); Data.Output(1:end-1, :)]];
           % type2
           DataSetInput{3} = [];
%            DataSetInput{3} = [Data.Input(:, 1:3),...
%                               [zeros(1,3); Data.Input(1:end-1, 1:3)],...
%                               [zeros(1,3); Data.Input(1:end-1, 4:6)]];
           
           % type3
           DataSetInput{4} = Data.Input;
               
           % pick up the choice based on the training type
           if obj.TrainingType == "Normal"
               Data.Input = DataSetInput{1};
           elseif obj.TrainingType == "Type1" 
               Data.Input = DataSetInput{2};
           elseif obj.TrainingType ==  "Type2"
               Data.Input = DataSetInput{3};
          elseif obj.TrainingType ==   "Type3"
               Data.Input = DataSetInput{4};
           elseif obj.TrainingType ==  "All"
               for i = 1: 1: length(obj.TypeSet)
                    DataSet(i) = struct('Input', DataSetInput{i},...
                                         'Output', Data.Output);
               end
               Data = DataSet;
           end
        end

        %% ANN train and Test
        function Result = ANNTrainTest(obj, TrainData, TestData)
           % initialize current timestamp
           tStart = cputime;

           % train net
           Result.net = obj.ANNTrain(TrainData);

           % use trained net to predict train data
           Result.TrainData = obj.ANNTest(Result.net, TrainData);

           % use trained net to predict train data
           if exist('TestData','var')
               Result.TestData = obj.ANNTest(Result.net, TestData);
           end

           % add elapsed time
           Result.ElapsedTime = cputime - tStart;

           % add model type
           Result.ModelType = "ANN";

        end

        %% function: ANN train
        function net = ANNTrain(~, TrainData)
            % initialize parameters
            AllTrainFcn = ['trainlm', 'trainbr', 'trainbfg', 'trainrp',...
                           'trainscg', 'traincgb', 'traincgf', 'traincgp',...
                           'trainoss', 'traingdx', 'traingdm', 'traingd'];

            MyTrainFcn      = 'trainlm';  % Levenberg-Marquardt backpropagation.
            hiddenLayerSize =  [20, 20, 20];     
            net = feedforwardnet(hiddenLayerSize, MyTrainFcn);

            % make division
            net.divideParam.trainRatio  = 100/100;
            net.divideParam.valRatio    = 0/100;
            net.divideParam.testRatio   = 0/100;

            % setup parameters
            net.trainParam.epochs       = 1E5;
            net.trainParam.showWindow   = true;

            % train and test data
            [net, ~] = train(net, TrainData.Input', TrainData.Output', ...
                             'useParallel','no',...
                             'useGPU', 'no',...
                             'showResources', 'yes');            
        end

        %% function: ANN predict train/test data
        function Data = ANNTest(~, net, Data)
            Data.Predict  = net(Data.Input')';
        end

        %% SVM train and test
        function Result = SVMTrainTest(~, TrainData, TestData)
            % initialize parameters
            tStart      = cputime;
            ModelType   = "SVM";           
            % train and test data
%             Model           = fitrsvm(TrainData.Input, TrainData.Output,...
%                                     'Standardize',true);
            Model         = fitrsvm(TrainData.Input, TrainData.Output,...
                            'KernelFunction','linear','Standardize',true);
            TrainPredict  = predict(Model, TrainData.Input);                
            TestPredict   = predict(Model, TestData.Input);  

            % display elapsed time
            ElapsedTime = cputime - tStart;
            
            % integrate results
            Result.TrainData = TrainData;
            Result.TestData  = TestData;
            Result.TrainPredict = TrainPredict;
            Result.TestPredict = TestPredict;
            Result.ModelType = ModelType;
            Result.ElapsedTime = ElapsedTime;
        end

        %% function Get Training and Testing accuracy
        function Result = GetTrainTestAccuracy(~, ResultData)
            % export all data
            TrainData    = ResultData.TrainData;
            TestData     = ResultData.TestData;
            TrainPredict = ResultData.TrainPredict;
            TestPredict  = ResultData.TestPredict;

            % train analysis
            Result.Trainloss     = immse(TrainPredict, TrainData.Output);
            Result.RSquareTrain  = 1 - sum((TrainData.Output - TrainPredict).^2)/...
                            sum((TrainData.Output - mean(TrainData.Output)).^2);
            TrainError           = TrainData.Output - TrainPredict;
            Result.TrainAccuracy = mean(1 - abs(TrainError ./ TrainData.Output));
            
            % test analysis
            Result.TestMSE        = immse(TestPredict,TestData.Output);
            Result.RsquareTest   = 1 - sum((TestData.Output - TestPredict).^2)/...
                            sum((TestData.Output - mean(TestData.Output)).^2);
            TestError            = TestData.Output - TestPredict;
            Result.TestAccuracy  = mean(1 - abs(TestError ./ TestData.Output));

        end

        %% function: plot fitness result(train and test)
        function PlotTrainTestResult(obj, ResultAll)
            for i = 1: 1: length(ResultAll)
                % process training result
                ResultAll(i).TrainData = obj.ProcessJoint(ResultAll(i).TrainData); 
                ResultAll(i).TrainData = obj.ProcessPos(ResultAll(i).TrainData);  
                
                % process testing result
                ResultAll(i).TestData = obj.ProcessJoint(ResultAll(i).TestData); 
                ResultAll(i).TestData = obj.ProcessPos(ResultAll(i).TestData);  

            end

                    
            % get other paramters
            Result = ResultAll(1);
            MyPara.ModelType = Result.ModelType; 

            % plot result of joint error
            % setup joint angle general paramters
            MyPara.DataType = "Joint";
            MyPara.MySubTitle = ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3"];
            MyPara.ErrorUnit = "Degree";

            % plot training joint angle result
            MyPara.SampleType = "Training";
            MyPara.FigurePos = [100, 300, 800, 600];
            obj.PlotSingle(Result.TrainData.OutputError, MyPara)

            % plot testing joint angle result
            MyPara.SampleType = "Testing";
            MyPara.FigurePos = [1000, 300, 800, 600];
            obj.PlotSingle(Result.TestData.OutputError, MyPara)

            % plot trainning and testing position result
            % setup position general paramters
            MyPara.DataType = "Position";
            MyPara.SampleType = "Training and Testing";
            MyPara.MySubTitle = ["Trainning Position", "Testing Position"];
            MyPara.ErrorUnit = "mm";
            MyPara.FigurePos = [500, 300, 800, 600];
            
            obj.PlotSingle(Result.PosError, MyPara)
        end

        %% function: plot fitness result (train only)
        function PlotResult(obj, Result)

            % get other paramters
            MyPara.ModelType = Result.ModelType; 

            % plot result of joint error
            % setup joint angle general paramters
            MyPara.DataType  = "Joint";
            MyPara.MySubTitle = obj.RobotPlotSubTitle;
            MyPara.ErrorUnit  = "Degree";

            % plot training joint angle result
            MyPara.SampleType = "Training";
            MyPara.FigurePos = [100, 300, 800, 600];
            obj.PlotSingle(Result.TrainData.OutputError, MyPara)

            % plot trainning and testing position result
            % setup position general paramters
            MyPara.DataType = "Position";
            MyPara.SampleType = "Training";
            MyPara.MySubTitle = "Trainning Position";
            MyPara.ErrorUnit = "mm";
            MyPara.FigurePos = [500, 300, 800, 600];
            
            obj.PlotSingle(Result.TrainData.PosError, MyPara)
        end

        %% function: process net result related to joint data
        function MyResult = ProcessJoint(obj, MyResult)    
            % reverse normalization(optional)
            MyResult = obj.ReverseNor(MyResult);
                    
            % comparison of joint angle
            MyResult.OutputError = abs(MyResult.Output - MyResult.Predict);
      
        end

        %% function: reverse normalization
        function data = ReverseNor(obj, data)
            data.Input   = obj.NormalizeData(data.Input, "Position", "reverse");
            data.Output  = obj.NormalizeData(data.Output, "Angle", "reverse");
            data.Predict = obj.NormalizeData(data.Predict, "Angle", "reverse");     
        end

        %% function: process net result related to position data
        function MyResult = ProcessPos(obj, MyResult)  
            % use Forward kinematics(FK) to calculate predicted postion
            MyResult.PredictPos = obj.RobotJoint2Pos(MyResult.Predict);

            % comparison of end effector position to get position error
            MyResult.PosError{1} = obj.GetEuDis(MyResult.PredictPos(:, 1:3),...
                                                MyResult.Input(:, 1:3));
        end

        %% function: calculate euclidean distance
        function Result = GetEuDis(~, Pos1, Pos2)
            PosDiff = abs(Pos1 - Pos2); 
            Result = sqrt(PosDiff(:, 1).^2 +...
                          PosDiff(:, 2).^2 +...
                          PosDiff(:, 3).^2);

        end
        
        %% function: plot single group comparison(train or test)
        function PlotSingle(obj, MyResult, MyPara)               
            % Initialize figure layout
            figure
            set(gcf, 'Position',  MyPara.FigurePos) % figure location
            SubPlotNum = length(MyPara.MySubTitle); % the number of subplot
            MyLayout = tiledlayout(SubPlotNum, 1, 'TileSpacing','Compact');
            
            
            % start to plot each joint angle error
            for i = 1: 1: SubPlotNum
                ax(i) = nexttile;
                SingleData = obj.GetSingleData(MyResult, MyPara.DataType, i);
                plot(SingleData)
                title(MyPara.MySubTitle(i));
                axis padded;
                grid minor;
            end
            
            xlabel(MyLayout, "Date Sample Index");
            ylabel(MyLayout, compose("Numerical Error(%s)", MyPara.ErrorUnit));    
            title(MyLayout, compose("Difference between %s result vs %s",...
                                    MyPara.ModelType, MyPara.DataType),...
                            compose("<%s Samples>",MyPara.SampleType))
        end

        %% function Pick up joint angle or position data to plot
        function SingleData = GetSingleData(~, Data, DataType, Index)
            if DataType == "Joint"
                % get joint angle error in the whole column
                SingleData = Data(:, Index);
            elseif DataType == "Position"
                % get positio error in the single cell
                SingleData = Data{Index};
            else
                error("Cannot recognize the data type!!!");
            end

        end
       
    end

end

















