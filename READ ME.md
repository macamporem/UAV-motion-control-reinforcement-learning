Attitude Control: Implement quadrotor attitude control via Reinforcement Learning

The Policy Gradient algorithm in uav-rl-policy-gradients-discrete learns the optimal policy to maintain the quadrotor attitude stabilized.  The quadrotor action space has eight possible actions: port roll, startboard roll, fore pitch, aft pitch, yaw (cc and anti cc), no thrust, balance thrust.

- The quadrotor C++ motion control outputs the state via log files
- The  policy gradient algorithm communicates the commanded quadrotor action via log file