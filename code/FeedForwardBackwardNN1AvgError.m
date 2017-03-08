%------------------ Initialization of Neural Network ----------------- 
NN=[256 100 60 10];               % assign number of layers and number of nodes in each layer
Num_of_runs = 12;				  % assign number of runs to calculate average minimum test_error
min_test_errors = [];			  % store minimum test error for each run
min_avg_test_error = 0;			  % store average minimum test_error

for i=1:Num_of_runs				  % loop over for number_of_runs assigned above

alpha = 0.2;                        % learning rate used in gradient discent to update betas
target_mse = 0.05;                  % target mean square error
Max_Iteration = 2000;                 % number of iteration in Neural Network
Min_Error = Inf;                    % Stores minimum test error
Min_Error_Iteration = -1;           % Stores iteration in which minimum error is obtained
Iteration = 0;                      % stores current iteration
train_mse = Inf;                    % initialization of training error to Infinity
test_mse = Inf;                     % initialization of test error to Infinity
TrainErr = [];                      % array to store training error
TestErr = [];                       % array to store test error
Iter = [];                          % store iteration number

%------------------ Load training dataset ----------------- 

load train.txt                              
[train_row, train_col] = size(train);
ip_matrix = train(:,2:train_col);               % remove 1st column and store rest of columns for training (first column is the output digit)
[ip_row, ip_col] = size(ip_matrix);             % get rows and columns in training matrix
temp_op_matrix = train(:,1);                    % parse only first column from input file
op_matrix = zeros(ip_row, 10);                  % represent the digit by 10 bit binary with 1 representing digit e.g. class0 = 1 means the digit is zero

for i = 1:ip_row                                % create an op_matrix with column 1 as class0 and represents digit 0, column 2 as class1 represents digit 1 and so on
    for j = 0:9
        if temp_op_matrix(i) == j
            op_matrix(i,j+1) = 1;
        else
            op_matrix(i,j+1) = 0;
        end
    end
end

[op_row, op_col] = size(op_matrix);

%------------ Load testing dataset --------------------------

load test.txt
[test_row, test_col] = size(test);               
test_ip_matrix = test(:,2:test_col);            % create a test input matrix by removing 1st column (1st column is output in test.txt file)
[test_R, test_C] = size(test_ip_matrix);
temp_test_op_matrix = test(:,1);                % save the 1st column of test.txt in a temporary matrix
test_op_matrix = zeros(test_R, 10);             % save the 10 bit representation (obtained using indicator function) of digit

for i = 1:test_R                                % create a test_op_matrix with column 1 as class0, column 2 as class1 and so on (column1 = 1 => digit 0,column2 = 1 => digit 1)
    for j = 0:9
        if temp_test_op_matrix(i) == j
            test_op_matrix(i,j+1) = 1;
        else
            test_op_matrix(i,j+1) = 0;
        end
    end
end

[test_op_row, test_op_col] = size(test_op_matrix);


%------------ throw error if output row is not equal to input row ---------
if ip_row ~= op_row
    error('some values in first column of train.txt are missing');
end

%------------ throw error if number of nodes in input layer is not equal to number of features ---------
if NN(1) ~= ip_col
    error('number of nodes in input layer is not equal to number of features in train dataset');
end


% -------------- Initialize Weight or Beta Matrix of the neural network
B=cell(length(NN)-1,1);     % numbe of beta matrix required is = (no of hidden layer + 1)

for i=1:length(NN)-1        % Assign uniform random values in [-0.7, 0.7] 
      B{i} =[1.4.*rand(NN(i)+1,NN(i+1))-0.7];	
end

% --------------- store Best Beta (BB) in a matrix and then write in a file at
% the end of program ----------------------------

BB=cell(length(NN)-1,1);     % numbe of beta matrix required is = (no of hidden layer + 1)

for i=1:length(NN)-1        % assign random values to the elements of cell
      BB{i} =[rand(NN(i)+1,NN(i+1))];	
end


% --------------- Initialize matrix for term T -------------
T=cell(length(NN),1);
for i=1:length(NN)
	T{i} =ones (NN(i),1);
end


% --------------- Initialize matrix for term Z (activation function) ---
Z=cell(length(NN),1);
for i=1:length(NN)-1
	Z{i} =zeros (NN(i)+1,1); % initialize Z with zeros
end
Z{end} =zeros (NN(end),1);  % no Bias term at the output layer


% --------------- Initialize matrix for term T for test (here T is replaced by U) -------------
U=cell(length(NN),1);
for i=1:length(NN)
	U{i} =ones (NN(i),1);
end


% --------------- Initialize matrix for term X (activation function) for test ---
X=cell(length(NN),1);
for i=1:length(NN)-1
	X{i} =zeros (NN(i)+1,1); % initialize Z with zeros
end
X{end} =zeros (NN(end),1);  % no Bias term at the output layer

% --------------- Initialize matrix for term delta (d - error term)----
d=cell(length(NN),1);
for i=1:length(NN)
	d{i} =zeros (NN(i),1);
end

avg_error=cell(length(NN)-1,1);                 % initialize a matrix to store (Z*d) and (d)
for i = 1:length(NN)-1
    avg_error{i} = cell(2,1);
end

for i = 1:length(NN)-1
    avg_error{i}{1} = zeros(NN(i+1),NN(i));     % store addition of term (Z*d) for each iteration (will be used at the end of training by all samples to calculate Betas)
    avg_error{i}{2} = zeros(NN(i+1),1);         % store addition of term (d) for each iteration (will be used at the end of training by all samples to calculate Beta0)
end

while (test_mse > target_mse) && (Iteration < Max_Iteration)    % outer loop with exit conditions
 CSqErr=0;                                                      % Cumulative Sq Err of each Sample; we will take the average after computing error for each sample (=> mse)
  for j=1:ip_row                                                % // loop through all the samples in the training dataset	
     Z{1} = [ip_matrix(j,:) 1]';                                % // Load sample one by one with bias=1
      Yk   = op_matrix(j,:)';                                   % // Load Corresponding Desired or Target output
  
      % ---------- start of forward propagation for training ----
  
      for i=1:length(NN)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(NN)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % // end of forward propagation 
       
      CSqErr= CSqErr+sum((Yk-Z{end}).^2);           % collect sample wise Cumulative Sq Err for whole training samples
   
    
     % --------- Compute error term delta 'd' for each of the node except the input unit
    d{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end});     % // delta error term for the output layer
    
     for i=length(NN)-1:-1:2 
         d{i}=Z{i}(1:end-1).*(1-Z{i}(1:end-1)).*sum(d{i+1}'*B{i}(1:end-1,:)'); % //compute the error term for all the hidden layer (and skip the input layer).
     end
     
    % ----- Store the term (d*Z) and (d) required to update Beta at the end
    % of training
    for i = 1: length(NN)-1
       avg_error{i}{1} = plus(avg_error{i}{1},d{i+1}*Z{i}(1:end-1)');
       avg_error{i}{2} = plus(avg_error{i}{2}, d{i+1});
    end
    
  end
  
  % ----------------------- Forward propagation for test ----------------
  CSqErr_test=0;
   for j=1:test_R                       % // loop through all the samples in the training dataset		
     X{1} = [test_ip_matrix(j,:) 1]';   % // Load sample one by one with bias=1
      Y   = test_op_matrix(j,:)'; 	    % // Load Corresponding Desired or Target output
  
      % ------ start of forward propagation for testing ----
      for i=1:length(NN)-1
       	     U{i+1} = B{i}' * X{i};
            
             if (i+1)<length(NN)
               X{i+1}=[(1./(1+exp(-U{i+1}))) ;1];
             else  
               X{i+1}=(1./(1+exp(-U{i+1}))); 
             end
      end  % // end of forward propagation
    
    CSqErr_test = CSqErr_test+sum((Y-X{end}).^2);
   end
    
  % ---- update Betas after training and testing is finished (because we are doing batch learning neural network)  
  for i=1:length(NN)-1 
          B{i}(1:end-1,:)=B{i}(1:end-1,:)-alpha.*(1/ip_row .*avg_error{i}{1}'); 
          B{i}(end,:)=B{i}(end,:)-alpha.*(1/ip_row.*avg_error{i}{2}');  			% // update weight connected to the bias unit(or, intercept)	
  end
  
  CSqErr= (CSqErr) /(ip_row);                  % //Average training error of ip_row sample after training 
    train_mse=CSqErr
    
  CSqErr_test= (CSqErr_test) /(test_R);        % //Average test error of test_R sample after testing 
  test_mse = CSqErr_test
  
  Iteration  = Iteration+1                     % increase iteration counter by 1

  TrainErr = [TrainErr train_mse];             % store training error of each iteration
  TestErr = [TestErr test_mse];                % store test error of each iteration
  Iter = [Iter Iteration];                     % store iteration count
    
    if test_mse < Min_Error                    % if current test error is less than minimum error then store current test error as minimum error
        Min_Error= test_mse;
        Min_Error_Iteration=Iteration;
        
        % store BB (Best Beta's) which provided minimum test error
        for i=1:length(NN)-1
            BB{i} = B{i};        
        end
        
    end
    
end    % end of while loop                                     

Min_Error               %print minimum error on console
Min_Error_Iteration     %print minimum error iteration on console


min_test_errors = [min_test_errors Min_Error];			 	% save minimum test error every time in a vector
end

min_test_errors												% print min_test_errors vector
avg_min_test_errors = sum(min_test_errors)/Num_of_runs		% take average of min_test_errors and print on console


    