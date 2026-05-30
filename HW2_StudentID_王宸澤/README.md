# Execution Environment
OS: Ubuntu 22.04.5 LTS
Docker version: 29.5.0
python version: 3.10.20
GPU: RTX 5060 desktop
GPU driver: nvidia-driver-595-open
CPU: 13th Gen Intel(R) Core(TM) i5-13450HX
ROS2 version: iron

# Installation commands
pip install gymnasium
pip install scipy
pip install stable-baseline3
pip install pyyaml
pip install tensorboard

# How to reproduce your training run:

> cd HW2_StudentID_王宸澤

> python3 main_step1.py --mode train --timesteps 70000

> python3 main_step2.py --mode train --resume-from models/ppo_drone_step1.zip --timesteps 100000

> python3 main_step3.py --mode train --resume-from models/ppo_drone_step2.zip --timesteps 150000

> python3 main_step4.py --mode train --resume-from models/ppo_drone_step3.zip --timesteps 350000


# How to load and test the trained model
Before running, open the Gazebo, and then, open the teleop and press S to make sure the drone is not moving. 
> cd HW2_StudentID_王宸澤

(for stage1:)
> python3 main_step1.py --mode test --resume-from models/ppo_drone_step1.zip

(for stage2:)
> python3 main_step2.py --mode test --resume-from models/ppo_drone_step2.zip

(for stage3 and randomly choose one of two trajectory:)
> python3 main_step3.py --mode test --resume-from models/ppo_drone_step3.zip

(for stage3 and choose traj1 as trajectory for this test:)
> python3 main_step3.py --mode test --resume-from models/ppo_drone_step3.zip --traj-num 1

(for stage3 and choose traj2 as trajectory for this test:)
> python3 main_step3.py --mode test --resume-from models/ppo_drone_step3.zip --traj-num 2

(for stage4 and randomly choose one of two trajectory:)
> python3 main_step4.py --mode test --resume-from models/ppo_drone_step4.zip

(for stage4 and choose traj1 as trajectory for this test:)
> python3 main_step4.py --mode test --resume-from models/ppo_drone_step4.zip --traj-num 1

(for stage4 and choose traj2 as trajectory for this test:)
> python3 main_step4.py --mode test --resume-from models/ppo_drone_step4.zip --traj-num 2


# Other things you can know
1. Trajectories what I have given are long. 
2. Tracking traj1 for main_step4.py for once would cost about 1 and half minute
3. Tracking traj2 for main_step4.py for once would cost about 4 minute
