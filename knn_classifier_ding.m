 function accuracy = knn_classifier_ding(data,fea_sel,k)
d_length = length(data);

%% divide data to train and test set

train_data = data(1:d_length/2,:);
test_data = data(d_length/2+1:end,:);

%% operate the active feature user selected
train_fea = train_data(:,fea_sel);
test_fea  = test_data(:,fea_sel);

correct = 0 ;
wrong = 0;

for i = 1: length(test_fea) 
    class0 = 0;
    class1 = 0;
    % search the k samples that around test data
    neibor_idx = knnsearch(train_fea,test_fea(i,:),'K',k);
    % compare sample label and classify to the two categories
    if sum(train_data(neibor_idx,9)) > k/2
        class1 = 1;
    else 
        class0 = 1;
    end
    % compute the corrected classified samples
    if class0 == 1 && test_data(i,9) == 0
        correct = correct + 1;
    elseif class1 == 1 && test_data(i,9) == 1
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
accuracy = 1 - wrong/length(test_data);

