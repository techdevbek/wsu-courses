
%
% Script to demonstrate how to solve small mixed integer programs
% using the Matlab function intlinprog.
%

% The problem we will solve is:
%
% max z = 526 x1 + 168 x2 +  74 x3 + 102 x4
% s.t.    172 x1 +  60 x2 +  96 x3 +  48 x4 <= 2950
%         144 x1 +  36 x2 +  54 x3 +  30 x4 >= 1000
%         4.4 x1 - 5.4 x2 +11.9 x3 - 4.5 x4 <= 0
%         2.4 x1 + 9.6 x2 -21.6 x3 - 2.0 x4 >= 0
%             x1 +     x2 +     x3 +     x4  = 25
%         x >= 0
%         x1,x3,x4 in R
%         x2 in Z

% First build the objective vector.  This is a column vector
% as the entries are delineated by a semicolon.
c = [ 526 ; 168 ;  74 ; 102 ];

% Next, the constraint matrix for the inequality constraints.
% Note that the inequalities must be Ax <= b, so some sign 
% changes result in converting >= into <=.
A = [  172   60   96      48 
      -144  -36  -54     -30
       4.4  -5.4  11.9  -4.5
      -2.4  -9.6  21.6     2 ];

% Next the right-hand-side vector for the inequalities
% Sign changes can occur here too.  
b = [2950 ; -1000 ; 0 ; 0];

% The coefficient matrix for the equality constraints and
% the right hand side vector.
Ae=ones(1,4);     % Ae = [1 1 1 1];  
be = [25];

% Next, we provide any lower and upper bound vectors, one
% value for each decision variable.  In this example all
% lower bound are zero and there are no upper bounds.
lb=zeros(4,1);    % lb = [0;0;0;0];
ub=inf(4,1);      % ub = [inf;inf;inf;inf];

% Lastly, we can specify which variables are required to be integer.
% If no variables are integer then iv=[];  In our example, only x4
% is integer.
iv=[2];    

% Now, call the solver.  The input variables c,iv,...,ub must be
% written in the exact order given.  In this example we have a 
% maximization problem - so, as the solver expects a minimization
% problem we instead pass "-c" as the objective vector.
[xstar,zstar]=intlinprog(-c,iv,A,b,Ae,be,lb,ub);

% We can view the final result by viewing the values in your
% workspace or by typing the variable name at the command
% window prompt, or by including the following:
zstar
xstar

% *** CAUTION *** The solver may also display additional information
% to the command window.  You should ignore this extra information 
% because it can be misleading.  The actual solution is given in 
% your variables zstar and xstar.






