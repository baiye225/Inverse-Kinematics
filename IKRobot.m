% Inverse Kinematic For Robot Arm

classdef IKRobot < IKProperty
    properties
        % empty
    end

    methods
        
        %% function: generate angle & position data
        function Data = GetDataSet(obj)

            % generate joint angle set          
            [JointSet, total] = obj.GetJointAngleSet();

            % initialize waitbar paramters
            WBPara.f = waitbar(0, "Generate data set...");
            WBPara.total = total;

            % prepare forward kinematic parameters
            MyDoFIndex        = 1;                       % initialize index of the current degree
            FKPara.MyJoint    = zeros(1, obj.RobotDoF);  % initialize current joint set
            FKPara.JointSet   = JointSet;                % joint angle set

            % initialize FK data set
            Data = zeros(total,  obj.RobotJointNum * 3 + obj.RobotDoF);  



            % start to generate position by forward kinematic
            [~, Data] = CalFKData(obj, Data, MyDoFIndex, FKPara, WBPara);  
            Data      = Data(any(Data, 2), :); % remove zero row from the Data

            % close waitbar
            close(WBPara.f)
        end      

        %% generate joint angle set
        function [JointSet, total] = GetJointAngleSet(obj)
            % initialize joint angle set cell
            JointSet = cell(obj.RobotDoF);
            total = 1; 

            % generate joint angle set
            for i = 1: 1: obj.RobotDoF
                JointSet{i} = obj.RobotJoint(i, 1) :...
                              obj.RobotJointInterval(i) :...
                              obj.RobotJoint(i, 2);
                total = total * length(JointSet{i});
            end            

        end
        %% use recursion to generated nested for loop of FK
        function [obj, Data] = CalFKData(obj, Data, MyDoFIndex, FKPara, WBPara)    
                for i = 1: 1: length(FKPara.JointSet{MyDoFIndex})
                    % get the current joint angle 
                    FKPara.MyJoint(1, MyDoFIndex) = FKPara.JointSet{MyDoFIndex}(i);

                    % recursion
                    if MyDoFIndex < obj.RobotDoF
                        % recurse to the next joint
                        [obj, Data]= obj.CalFKData(Data, MyDoFIndex + 1, FKPara, WBPara);
                    else
                        % use complete joint angle set to generate position
                        position = obj.RobotJoint2Pos(FKPara.MyJoint);

                        % check if the position is in the threshold
                        if obj.RobotValiPos(position, FKPara.MyJoint) == "True"
                            % collect data set(position + joint angle) with current joint
                            obj.FKDataIndex = obj.FKDataIndex + 1;
                            Data(obj.FKDataIndex, :) = [position, FKPara.MyJoint];
                            
                            % update and display waitbar status
                            currentDataNum = obj.FKDataIndex;
                            msg = sprintf("Process data set with %d/%d (%0.2f%%)",...
                                           currentDataNum, WBPara.total,...
                                           (currentDataNum / WBPara.total) * 100);
                            waitbar(currentDataNum/WBPara.total, WBPara.f, msg);
                        end
                    end
                    
                end
        end

        %% function: 3 DoF -> convert joint angle(degree 1x3) into position
        function position = Theta2Pos(obj, thetaDegree)
            % degree to rad
            thetaRad = deg2rad(thetaDegree);

            % start to transform joint to position  
            for i = 1: 1: length(thetaRad(:, 1))
                % get the current matrix of rotation(MR)
                MRTheta1 = obj.GetMatrixRotation(thetaRad(i, 1), 'z');
                MRTheta2 = obj.GetMatrixRotation(thetaRad(i, 2), 'y');
                MRTheta3 = obj.GetMatrixRotation(thetaRad(i, 3), 'y');
    
                % position joint C (fixed joint)
                positionC = MRTheta1 * [0; 0; obj.RobotGeo(1)];

                % position joint B
                positionB = MRTheta1 * MRTheta2 * [0; 0; obj.RobotGeo(2)]...
                            + positionC;
    
                % position end-effector A
                positionA = MRTheta1 * MRTheta2...
                            * MRTheta3 * [0; 0; obj.RobotGeo(3)]...
                            + positionB;
    
                % intergrate position
                position(i, :) = [positionA' positionB' positionC'];
            end
        end

        %% function: 5 DoF -> convert joint angle(degree 1x5) into position
        function position = Theta2Pos2(obj, thetaDegree)
            % degree to rad
            thetaRad = deg2rad(thetaDegree);
            
            % the number of theta set
            n = length(thetaRad(:, 1));

            % initialize waitbar if process nuerous theta set(optional)
            if n > 1
                f = waitbar(0, "Generate data set...");
                f.Name = "5DoF Joint to Poisition";
            end

            for i = 1: 1: length(thetaRad(:, 1))
                % update waitbar(optional)
                if n > 1
                    msg = sprintf("Convert joint to position with %d/%d (%0.2f%%)",...
                                  i, n, (i / n) * 100);
                    waitbar((i / n), f, msg);
                end
                
                % get the current matrix of rotation(MR)
                MRTheta1 = obj.GetMatrixRotation(thetaRad(i, 1), 'z');
                MRTheta2 = obj.GetMatrixRotation(thetaRad(i, 2), 'y');
                MRTheta3 = obj.GetMatrixRotation(thetaRad(i, 3), 'y');
                MRTheta4 = obj.GetMatrixRotation(thetaRad(i, 4), 'z');
                MRTheta5 = obj.GetMatrixRotation(thetaRad(i, 5), 'y');
    
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
                            * MRTheta3 * MRTheta4...
                            * MRTheta5 * [0; 0; obj.RobotGeo(4)]...
                            + positionB;
    
                % intergrate position
                position(i, :) = [positionA' positionB' positionC' positionD'];

            end 
            
            % close waitbar
            if n > 1
                close(f)
            end
        end

        %% function: SCARA -> convert joint angle and movement (degree 1x3) into position
        function position = Joint2Pos3(obj, JointInput)
            % degree to rad
            thetaRad = deg2rad(JointInput(:, 1:2));

            for i = 1: 1: length(thetaRad(:, 1))
                % get the current matrix of rotation(MR)
                MRTheta1 = obj.GetMatrixRotation(thetaRad(i, 1), 'z');
                MRTheta2 = obj.GetMatrixRotation(thetaRad(i, 2), 'z');
    
                % Euler transformation

                % position joint D(fixed joint)
                positionD = [0; 0; obj.RobotGeo(1)];

                % position joint C
                positionC = MRTheta1 * [obj.RobotGeo(2); 0; 0] + positionD;
    
                % position joint B
                positionB = MRTheta1 * MRTheta2 * [obj.RobotGeo(3); 0; 0]...   
                            + positionC;

                % position end-effector A
                positionA = [0; 0; obj.RobotGeo(4)] * JointInput(i, 3)...
                            + positionB - [0; 0; obj.RobotGeo(1)];
    
                % intergrate position
                position(i, :) = [positionA' positionB' positionC' positionD'];
            end   
        end

        %% function: 4 DoF -> convert joint angle(degree 1x3) into position
        function position = Joint2Pos4(obj, thetaDegree)
            % degree to rad
            thetaRad = deg2rad(thetaDegree);
            
            % the number of theta set
            n = length(thetaRad(:, 1));

            % initialize waitbar if process nuerous theta set(optional)
            if n > 1
                f = waitbar(0, "Generate data set...");
            end

            for i = 1: 1: length(thetaRad(:, 1))
                % update waitbar(optional)
                if n > 1
                    msg = sprintf("Convert joint to position with %d/%d (%0.2f%%)",...
                                  i, n, (i / n) * 100);
                    waitbar((i / n), f, msg);
                end
                
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
                            * MRTheta3 * MRTheta4...
                            * [0; 0; obj.RobotGeo(4)]...
                            + positionB;
    
                % intergrate position
                position(i, :) = [positionA' positionB' positionC' positionD'];
            end

            % close waitbar
            if n > 1
                close(f)
            end
        end
        %% function: generate matrix of rotation
        function result = GetMatrixRotation(~, theta, type)
            MR1 = [1, 0,          0;
                   0, cos(theta), -sin(theta);
                   0, sin(theta), cos(theta)];
        
            MR2 = [cos(theta),  0, sin(theta);
                    0,          1, 0;
                   -sin(theta), 0, cos(theta)];
        
            MR3 = [cos(theta), -sin(theta), 0;
                   sin(theta), cos(theta),  0;
                    0,         0,           1];
        
            switch type
                case 'x'
                    result = MR1;
                case 'y'
                    result = MR2;
                case 'z'
                    result = MR3;
                otherwise
                    error("unidentified matrix of rotation")
            end
        end

        %% function: convert joint angle(rad 1x3) into position
        function position = Theta2PosDH(obj, Theta)
            % prepare paramter  
            ThetaRad = deg2rad(Theta); % degree to rad
            alpha = [pi/2, 0, 0];
            r     = [0, 6, 6];
            d     = [10, 0, 0];
            T     = eye(4);

            % D-H FK solver
            % end effector A
            for i = 1: 1: length(Theta)
                T = T * obj.DHMatrix(ThetaRad(i), alpha(i), r(i), d(i)); 
            end
            
            % joint B
            for i = 1: 1: length(Theta) - 1
                TJB = T * obj.DHMatrix(ThetaRad(i), alpha(i), r(i), d(i)); 
            end

            % intergrate position
            positionA = T(1:3, 4);
            positionB = TJB(1:3, 4);
            position = [positionA; positionB];
        end

        %% function: single link homogenous transformation matrix
        function T = DHMatrix(theta, alpha, r, d)    
            T = [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha),  r*cos(theta);
                 sin(theta), cos(theta)*cos(alpha),  -cos(theta)*sin(alpha), r*sin(theta);
                 0,          sin(alpha),             cos(alpha),             d;
                 0,          0,                      0,                      1];

        end

        %% function: check position within work envelope(3DoF)
        function Result = ValiPos(~, Pos, ~)
            % get position
            Pos = Pos(1, 1:3);
            x   = Pos(1);
            y   = Pos(2);
            z   = Pos(3);
            
            % get pitch and roll joint of the gripper
            if x >=0 && sqrt(x^2+y^2) >= 1 && z > 0
                Result = "True";
            else
                Result = "False";
            end
        end

        %% function: check position within work envelope(5DoF)
        function Result = ValiPos2(~, Pos, Joint)
            % get EE position
            EEPos = Pos(1, 1:3);
            x     = EEPos(1);
            y     = EEPos(2);
            z     = EEPos(3);
            
            % get Joint1 position
            J1Pos   = Pos(1, 4:6);
            J1x     = J1Pos(1);
            J1y     = J1Pos(2);
            J1z     = J1Pos(3);
            
            % get pitch and roll joint of the gripper
            roll     = Joint(4);
            pitch    = Joint(5);
            
            if x >=0 && sqrt(x^2+y^2) >= 1 && z > 0 &&...
               not(pitch == 0 && roll ~= 0) && J1x >=0 &&...
               sqrt(J1x^2+J1y^2) >= 1 && J1z > 0
                Result = "True";
            else
                Result = "False";
            end
        end
        %% function: check position within work envelope(SCARA)
        function Result = ValiPos3(~, Pos, ~)
            Pos = Pos(1, 1:3);
            x   = Pos(1);
            y   = Pos(2);
            z   = Pos(3);     

            if x >=0 && sqrt(x^2+y^2) >= 1 && z > 0
                 Result = "True";
            else
                Result = "False";
            end               
        end

        %% function: scatter plot all position data points
        function PlotAllPos(~, Data)
            Position = Data(:, 1:3);
            for i = 1: 1: length(Position(:, 1))
                theta3 = Data(i, 6);
                if theta3 == 0 || Position(i, 3) == 0
                    plot3(Position(i, 1), Position(i, 2), Position(i, 3), 'b.')
                    hold on
                end
            end
            hold off
            xlabel('x')
            ylabel('y')
            zlabel('z')
            axis equal
            grid on
        end
        
        %% function: generate random trajectory
        function traj = GetTraj1(obj)
            % initialize paramters
            n     = 200;            % the number of data points
            traj  = zeros(200, 6);  % trajectory data set
            
            % generate initial joint angle randomly
            thetaFactor = rand(3, 1); 
            ThetaDeg = zeros(3, 1);
            for i = 1: 1: length(ThetaDeg)
                ThetaDeg(i) = thetaFactor(i) * (obj.ThetaRange(i, 2) -...
                           obj.ThetaRange(i, 1)) + obj.ThetaRange(i, 1);
            end
            
            % set initial joint angle manually
            ThetaDeg = [0; 0; 90];

            % initialize the position
            position = obj.Theta2Pos(obj, ThetaDeg);
            traj(1, :) = [position', ThetaDeg'];

            % start to generate trajectory
            for i = 1: 1: n

                % update angle
                for j = 1: 1: length(ThetaDeg)

                    % generate the 1st New angle
                    ThetaDegChange = obj.GetUpdatedAngle(obj);
                    NewThetaDeg = ThetaDeg(j) + ThetaDegChange;

                    % re-generate until it is in the angle range
                    while NewThetaDeg < obj.ThetaRange(j, 1) ||...
                            NewThetaDeg > obj.ThetaRange(j, 2)
                        ThetaDegChange = obj.GetUpdatedAngle(obj);
                        NewThetaDeg = ThetaDeg(j) + ThetaDegChange;                        
                    end

                    % update the joint angle
                    ThetaDeg(j) = NewThetaDeg;
                end

                % convert angle into position
                position = obj.Theta2Pos(ThetaDeg);

                % append it into traj data set
                traj(i, :) = [position, ThetaDeg'];
            end
        end

        %% function: get updated angle based on the range
        function result = GetUpdatedAngle(obj)
            result = rand(1) *...
                     (obj.ThetaDegChangeRange(2) - obj.ThetaDegChangeRange(1)) +...
                      obj.ThetaDegChangeRange(1);
        end

        %% function: generate custom trajectory
        function traj = GetTraj(obj, type)
            % get raw trajectory path
            switch type
                case "Traj1"
                    traj = obj.Traj1();
                case "Traj2"
                    traj = obj.Traj2();
                case "Traj3"
                    traj = obj.Traj3();                    
                otherwise
                    error("unidentified trajectory type! ")
            end
            
            % initialize joint angle
            ThetaDeg0 = [0, 0, 90];

            % initialize position
            Pos0 = obj.Theta2Pos(ThetaDeg0);
            
            % shift trajectory and setup 1st at the initial position of the
            % end effector
            traj = traj - (traj(1, :) - Pos0(1, 1:3));
        end
        
        %% function: trajectory1: circle top to bottom
        function traj = Traj1(~)
            % trajectory parameters
            c1  = 2;    % coefficient1
            c2  = 2;    % coefficient2
            c3  = -0.2; % coefficient3

            h    = 6;           % trajectory's height
            nInt = 0.1;         % interval of the for loop
            num  = h / abs(c3); % the number of iteration based on the height
            k    = 1;           % the current iteration

            % start to generate trajectory
            for i = 0: nInt: num
                x = c1 * sin(i);
                y = c2 * cos(i);
                z = c3 * i;
                traj(k, :) = [x y z];
                k = k + 1;
            end
        end

        %% function: trajectory2: direct line top to bottom
        function traj = Traj2(~)
            % trajectory parameters
            c    = -0.2;        % coefficient1
            h    = 6;           % trajectory's height
            nInt = 0.1;         % interval of the for loop
            num  = h / abs(c); % the number of iteration based on the height
            k    = 1;           % the current iteration

            % start to generate trajectory
            for i = 0: nInt: num
                x = 0;
                y = 0;
                z = c * i;
                traj(k, :) = [x y z];
                k = k + 1;
            end
        end 

        %% function: trajectory3: 1/4 cicle
        function traj = Traj3(~)
            % trajectory parameters
            h    = 6;           % trajectory's height
            nInt = 0.1;         % interval of the for loop
            k    = 1;           % the current iteration
    
            % start to generate trajectory
            for i = 0: nInt: h
                y = i;
                x = sqrt(h^2 - y^2);
                z = 0;
                traj(k, :) = [x y z];
                k = k + 1;
            end
        end  
        %% function: plot trajectory
        function PlotTraj(obj, traj)        
            % plot traj
            figure    
            
            % start point
            plot3(traj{1}(1, 1), traj{1}(1, 2), traj{1}(1, 3),...
                  'ro', 'MarkerSize', 10)   
            hold on

            % end point
            plot3(traj{1}(end, 1), traj{1}(end, 2), traj{1}(end, 3), ...
                  'g*', 'MarkerSize', 10)

            % trajectory
            for i = 1: 1: length(traj)
                h(i) = plot3(traj{i}(:, 1), traj{i}(:, 2), traj{i}(:, 3));
            end 
            hold off
            
            % set up properties
            if length(traj) == 2
                legendName = ["Original", obj.TypeSet];
            else
                legendName = ["Original", obj.TrainingType];
            end

            legend(h, legendName)
            grid minor
            axis([-12 12 -12 12 0 22])
            axis normal;
            xlabel("x (mm)")
            ylabel("y (mm)")
            zlabel("z (mm)")
            set(gcf, 'Position',  [100, 300, 800, 600])
            
            % plot joint angle
            obj.PlotJointAngle(traj{2}(:, end-2:end))
        end
        
        %% function: plot all joint angles
        function PlotJointAngle(~, Data)
            figure
            
            % setup title and subtile
            MySubTitle = ["Base \theta_1", "Shoulder \theta_2", "Elbow \theta_3"];
            MyTitle = tiledlayout(length(MySubTitle), 1, 'TileSpacing','Compact');

             % start to plot each joint angle eror
            for i = 1: 1: length(MySubTitle)
                nexttile;         
                plot(Data(:, i), 'DisplayName', 'Data(:, i)')
                title(MySubTitle(i));
                axis padded;
                grid minor;
            end

            % set parameters of the figure
            set(gcf, 'Position',  [1000, 300, 800, 600])
            xlabel(MyTitle, "Date Sample Index");
            ylabel(MyTitle, "Theta (degree)");    
            title(MyTitle, "Displacement Joint Angle in the Trajectory")
        end

        %% function: animate robot movement
        function RobotAnimate(obj, Data)
            % prepare position data
            total  = length(Data(:,1));
            EndA   = Data(:, 1:3);
            JointB = Data(:, 4:6);
            JointC = [zeros(total, 2), ones(total, 1) * obj.H];
            BaseO  =  zeros(total, 3);  

            figure
            % start to animate
            for i = 1: 1: total
                % update joints and end-effector
                
                hA = plot3(EndA(i, 1), EndA(i, 2), EndA(i, 3)); 
                hold on
                hB = plot3(JointB(i, 1), JointB(i, 2), JointB(i, 3));
                hC = plot3(JointC(i, 1), JointC(i, 2), JointC(i, 3));
                hO = plot3(BaseO(i, 1), BaseO(i, 2), BaseO(i, 3));

                % update links between joints and end-effectors
                l1 = plot3([EndA(i, 1), JointB(i, 1)],...
                           [EndA(i, 2), JointB(i, 2)],...
                           [EndA(i, 3), JointB(i, 3)]);
                l2 = plot3([JointC(i, 1), JointB(i, 1)],...
                           [JointC(i, 2), JointB(i, 2)],...
                           [JointC(i, 3), JointB(i, 3)]);
                l3 = plot3([JointC(i, 1), BaseO(i, 1)],...
                           [JointC(i, 2), BaseO(i, 2)],...
                           [JointC(i, 3), BaseO(i, 3)]);

                % plot trajectory
                plot3(EndA(1:i, 1), EndA(1:i, 2), EndA(1:i, 3))
                hold off
                
                % setup parameters
                set(hA, 'Color', 'green', 'Marker', 'o')
                set(hB, 'Color', 'red', 'Marker', 'o')
                set(hC, 'Color', 'red', 'Marker', 'o')
                set(hO, 'Color', 'red', 'Marker', 'o')
                set(l1, 'Color', 'blue', 'LineWidth', 3)
                set(l2, 'Color', 'blue', 'LineWidth', 3)
                set(l3, 'Color', 'blue', 'LineWidth', 3)
                grid minor
                axis([-max(EndA(:, 1)) max(EndA(:, 1))...
                     -max(EndA(:, 2)) max(EndA(:, 2))...
                     -max(EndA(:, 3)) max(EndA(:, 3))]);
                xlabel("x (mm)")
                ylabel("y (mm)")
                zlabel("z (mm)")
                pause(0.1);
%                 set(gcf,)
            end
        end

    end
end





















