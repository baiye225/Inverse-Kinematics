% the general class to keep all dynamic properties of all classes

classdef IKProperty

    properties
        % parameter for different kinds of training samples
        % ("Normal", "Type1", "Type2", "All")
        TrainingType = "Normal";
        TypeSet = ["Normal", "Type1", "Type2", "Type3"];
        
        % the current robot parameters
        RobotIndex;         % robot type index
        RobotGeo;           % length of each arm
        RobotJoint;         % joint range of each joint
        RobotJointInterval; % angle increment of each joint
        RobotInitialJoint;  % initial joint status
        RobotDoF;           % the number of joint angle(Degree of Freedom)
        RobotJointNum       % the number of joints
        RobotJoint2Pos;     % Euler function (rotation/translation -> position)
        RobotNormalPara;    % normalization range(angle and position
        RobotValiPos;       % work envelope checking system
        RobotFABRIKArm;     % robot arm for FABRIK algorithm
        RobotFABRIKSolver;  % inverse kinematic solver based on FABRIK
        RobotPlotSubTitle;  % subtitle of training fitness result
        
        % All robot Euler transformation
        AllJoint2Pos;
        
        % All robot work envelope checking system
        AllValiPos;

        % All robot geometry
        AllRobotGeo = {[10, 6, 6];        % 3DoF Joint Arm
                       [10, 6, 6, 1];     % 5DoF Joint Arm
                       [20, 10, 10, 20];  % 3DoF SCARA
                       [10, 6, 6, 1]};    % 4DoF Joint Arm

        % All robot joint angle range
        AllRobotJoint = {[-90, 90; 0, 120; -20, 170];                   % 3DoF Joint Arm
                         [-80, 80; 0, 120; -20, 170; -80, 80; -80, 80]; % 5DoF Joint Arm
                         [-90, 90; -160, 160; 0, 1];                    % 3DoF SCARA
                         [-80, 80; 0, 120; -20, 170; -90, 90]};         % 4DoF Joint Arm 

        % All joint angle interval
        AllRobotJointInterval = {[5, 5, 5];             % 3DoF Joint Arm
                                 [10, 10, 10, 10, 10];  % 5DoF Joint Arm
                                 [5, 5, 0.01]           % 3DoF SCARA
                                 [5, 5, 5, 10]};        % 4DoF Joint Arm 

        % All robot initial joint
        AllRobotInitialJoint = {[0, 0, 90];       % 3DoF Joint Arm
                                [0, 0, 90, 0, 0]; % 5DoF Joint Arm
                                [0, 0, 1];        % 3DoF SCARA
                                [0, 0, 90, 0]};   % 4DoF Joint Arm   

        % All the number of joints(all joints + end-effector)
        AllRobotJointNum = [3;  % 3DoF Joint Arm
                            4;  % 5DoF Joint Arm
                            4;  % 3DoF SCARA
                            4]; % 4DoF Joint Arm 
        
        % the current machine learning data parameters
        MLInputIndex; 
        MLOutputIndex;
        ANNMLResult      = ["MLResultRobot3DoF1.mat",... % 3DoF Joint Arm
                            "MLResultRobot5DoF1.mat",... % 5DoF Joint Arm
                            "MLResultSCARA1.mat",...     % 3DoF SCARA
                            "MLResultRobot4DoF1.mat"];   % 4DoF Joint Arm 
        
        NovelANNMLResult = ["MLResultRobot3DoF3.mat",... % 3DoF Joint Arm
                            "MLResultRobot5DoF2.mat",... % 5DoF Joint Arm
                            "MLResultSCARA3.mat",...     % 3DoF SCARA
                            "MLResultRobot4DoF2.mat"];   % 4DoF Joint Arm    
        
        % forward kinematic data real-time index
        FKDataIndex = 0;

        % All FABRIK arm matrix
        FARBRIKArm = {[6, 6];
                      [6, 6, 1];
                      [10, 10, 0];
                      [6, 6, 1]};


        % All FABRIK inverse kinematic solver
        AllFABRIKSolver;

        % Training fitness figure parameters
        AllSubTitle = {["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3"];
                       ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3",...
                        "Roll \theta_4","Pitch \theta_5"];
                       ["\Joint_1 \theta_1", "\Joint_2 \theta_2", "Prismatic Joint \theta_3"];
                       ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3", "Pitch \theta_5"]};
       
    end

    methods

        %% initialize the robot parameters
        function obj = RobotInit(obj, RobotType)
            % confirm the robot type index
            switch RobotType
                case "Robot3DoF"
                    obj.RobotIndex = 1;
                case "Robot5DoF"
                    obj.RobotIndex = 2;
                case "SCARA"
                    obj.RobotIndex = 3;
                case "Robot4DoF"
                    obj.RobotIndex = 4;
            end

            % initialize robot geometry
            obj.RobotGeo = obj.AllRobotGeo{obj.RobotIndex};

            % initialize the range of joint angle, angle interval, DoF,
            % initiali joint, joint mumber
            obj.RobotJoint         = obj.AllRobotJoint{obj.RobotIndex};
            obj.RobotJointInterval = obj.AllRobotJointInterval{obj.RobotIndex};
            obj.RobotDoF           = length(obj.RobotJoint(:, 1));
            obj.RobotInitialJoint  = obj.AllRobotInitialJoint{obj.RobotIndex};
            obj.RobotJointNum      = obj.AllRobotJointNum(obj.RobotIndex);

            % initialize theta/joint to position function
            obj.AllJoint2Pos   = {@obj.Theta2Pos, @obj.Theta2Pos2, @obj.Joint2Pos3, @obj.Joint2Pos4};
            obj.RobotJoint2Pos = obj.AllJoint2Pos{obj.RobotIndex};

            % initialize work envelope checking system
            obj.AllValiPos   = {@obj.ValiPos, @obj.ValiPos2, @obj.ValiPos3, @obj.ValiPos};
            obj.RobotValiPos = obj.AllValiPos{obj.RobotIndex};

            % initialize machine learning input and output index
            obj.MLInputIndex = 1: obj.RobotJointNum * 3;
            obj.MLOutputIndex = obj.RobotJointNum * 3 + 1: obj.RobotJointNum * 3 + obj.RobotDoF;

            % initialize robot FARBIK parameter
            obj.RobotFABRIKArm = obj.FARBRIKArm{obj.RobotIndex};
            obj.AllFABRIKSolver = {@obj.FABRIKSolver1, @obj.FABRIKSolver2,...
                                   @obj.FABRIKSolver3, @obj.FABRIKSolver2};
            obj.RobotFABRIKSolver =  obj.AllFABRIKSolver{obj.RobotIndex};

            % initialize robot training fitness plot subtitle
            obj.RobotPlotSubTitle = obj.AllSubTitle{obj.RobotIndex};
        end

    end
end
















