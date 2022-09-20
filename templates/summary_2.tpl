Train Summary Report
$datetime

=================================
Global Configuration:
---------------------

Architecture: $model
Normalized: $normalized
Save model: $save_model
Plot Loss: $plot_loss
Device: $device
Require Grad: $require_grad
Weights Init.: $weight_init
Hyperparams:
	Epochs: $epochs
	Batch size: $batch_size
	Learning rate: $learning_rate
	Weight Decay: $weight_decay
	Criterion: $criterion
	Optimizer: $optimizer


=================================
Fold results:
---------------------

Fold [$fold_id_1_1]: train ($n_train_1_1) / test ($n_test_1_1)
	Normalization: (mean=$mean_1_1, std=$std_1_1)
	Elapsed time: train=$fold_train_time_1_1, test=$fold_test_time_1_1
	Accuracy: $accuracy_1_1
	Precision: $precision_1_1
	Recall: $recall_1_1
	F1: $f1_1_1
	Confusion Matrix: $tn_1_1 , $fp_1_1
	                  $fn_1_1 , $tp_1_1

Fold [$fold_id_1_2]: train ($n_train_1_2) / test ($n_test_1_2)
	Normalization: (mean=$mean_1_2, std=$std_1_2)
	Elapsed time: train=$fold_train_time_1_2, test=$fold_test_time_1_2
	Accuracy: $accuracy_1_2
	Precision: $precision_1_2
	Recall: $recall_1_2
	F1: $f1_1_2
	Confusion Matrix: $tn_1_2 , $fp_1_2
	                  $fn_1_2 , $tp_1_2

Fold [$fold_id_1_3]: train ($n_train_1_3) / test ($n_test_1_3)
	Normalization: (mean=$mean_1_3, std=$std_1_3)
	Elapsed time: train=$fold_train_time_1_3, test=$fold_test_time_1_3
	Accuracy: $accuracy_1_3
	Precision: $precision_1_3
	Recall: $recall_1_3
	F1: $f1_1_3
	Confusion Matrix: $tn_1_3 , $fp_1_3
	                  $fn_1_3 , $tp_1_3

Fold [$fold_id_1_4]: train ($n_train_1_4) / test ($n_test_1_4)
	Normalization: (mean=$mean_1_4, std=$std_1_4)
	Elapsed time: train=$fold_train_time_1_4, test=$fold_test_time_1_4
	Accuracy: $accuracy_1_4
	Precision: $precision_1_4
	Recall: $recall_1_4
	F1: $f1_1_4
	Confusion Matrix: $tn_1_4 , $fp_1_4
	                  $fn_1_4 , $tp_1_4

Fold [$fold_id_1_5]: train ($n_train_1_5) / test ($n_test_1_5)
	Normalization: (mean=$mean_1_5, std=$std_1_5)
	Elapsed time: train=$fold_train_time_1_5, test=$fold_test_time_1_5
	Accuracy: $accuracy_1_5
	Precision: $precision_1_5
	Recall: $recall_1_5
	F1: $f1_1_5
	Confusion Matrix: $tn_1_5 , $fp_1_5
	                  $fn_1_5 , $tp_1_5


===================================================================================


Fold [$fold_id_2_1]: train ($n_train_2_1) / test ($n_test_2_1)
	Normalization: (mean=$mean_2_1, std=$std_2_1)
	Elapsed time: train=$fold_train_time_2_1, test=$fold_test_time_2_1
	Accuracy: $accuracy_2_1
	Precision: $precision_2_1
	Recall: $recall_2_1
	F1: $f1_2_1
	Confusion Matrix: $tn_2_1 , $fp_2_1
	                  $fn_2_1 , $tp_2_1

Fold [$fold_id_2_2]: train ($n_train_2_2) / test ($n_test_2_2)
	Normalization: (mean=$mean_2_2, std=$std_2_2)
	Elapsed time: train=$fold_train_time_2_2, test=$fold_test_time_2_2
	Accuracy: $accuracy_2_2
	Precision: $precision_2_2
	Recall: $recall_2_2
	F1: $f1_2_2
	Confusion Matrix: $tn_2_2 , $fp_2_2
	                  $fn_2_2 , $tp_2_2

Fold [$fold_id_2_3]: train ($n_train_2_3) / test ($n_test_2_3)
	Normalization: (mean=$mean_2_3, std=$std_2_3)
	Elapsed time: train=$fold_train_time_2_3, test=$fold_test_time_2_3
	Accuracy: $accuracy_2_3
	Precision: $precision_2_3
	Recall: $recall_2_3
	F1: $f1_2_3
	Confusion Matrix: $tn_2_3 , $fp_2_3
	                  $fn_2_3 , $tp_2_3

Fold [$fold_id_2_4]: train ($n_train_2_4) / test ($n_test_2_4)
	Normalization: (mean=$mean_2_4, std=$std_2_4)
	Elapsed time: train=$fold_train_time_2_4, test=$fold_test_time_2_4
	Accuracy: $accuracy_2_4
	Precision: $precision_2_4
	Recall: $recall_2_4
	F1: $f1_2_4
	Confusion Matrix: $tn_2_4 , $fp_2_4
	                  $fn_2_4 , $tp_2_4

Fold [$fold_id_2_5]: train ($n_train_2_5) / test ($n_test_2_5)
	Normalization: (mean=$mean_2_5, std=$std_2_5)
	Elapsed time: train=$fold_train_time_2_5, test=$fold_test_time_2_5
	Accuracy: $accuracy_2_5
	Precision: $precision_2_5
	Recall: $recall_2_5
	F1: $f1_2_5
	Confusion Matrix: $tn_2_5 , $fp_2_5
	                  $fn_2_5 , $tp_2_5
	                  

=================================
Global Performance:
---------------------

Elapsed total time: $execution_time.
Folds Acc.: $folds_accuracy
Mean: $cross_v_mean
StdDev: $cross_v_stddev
CI:(95%) : $cross_v_interval






