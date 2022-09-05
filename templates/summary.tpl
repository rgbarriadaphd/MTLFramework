Train Summary Report
$datetime

=================================
Global Configuration:
---------------------

Architecture: $model
Images: $image_type
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

Fold [$fold_id_1]: train ($n_train_1) / test ($n_test_1)
	Normalization: (mean=$mean_1, std=$std_1)
	Elapsed time: train=$fold_train_time_1, test=$fold_test_time_1
	Accuracy: $accuracy_1
	Precision: $precision_1
	Recall: $recall_1
	F1: $f1_1
	Confusion Matrix: $tn_1 , $fp_1
	                  $fn_1 , $tp_1

Fold [$fold_id_2]: train ($n_train_2) / test ($n_test_2)
	Normalization: (mean=$mean_2, std=$std_2)
	Elapsed time: train=$fold_train_time_2, test=$fold_test_time_2
	Accuracy: $accuracy_2
	Precision: $precision_2
	Recall: $recall_2
	F1: $f1_2
	Confusion Matrix: $tn_2 , $fp_2
	                  $fn_2 , $tp_2

Fold [$fold_id_3]: train ($n_train_3) / test ($n_test_3)
	Normalization: (mean=$mean_3, std=$std_3)
	Elapsed time: train=$fold_train_time_3, test=$fold_test_time_3
	Accuracy: $accuracy_3
	Precision: $precision_3
	Recall: $recall_3
	F1: $f1_3
	Confusion Matrix: $tn_3 , $fp_3
	                  $fn_3 , $tp_3

Fold [$fold_id_4]: train ($n_train_4) / test ($n_test_4)
	Normalization: (mean=$mean_4, std=$std_4)
	Elapsed time: train=$fold_train_time_4, test=$fold_test_time_4
	Accuracy: $accuracy_4
	Precision: $precision_4
	Recall: $recall_4
	F1: $f1_4
	Confusion Matrix: $tn_4 , $fp_4
	                  $fn_4 , $tp_4

Fold [$fold_id_5]: train ($n_train_5) / test ($n_test_5)
	Normalization: (mean=$mean_5, std=$std_5)
	Elapsed time: train=$fold_train_time_5, test=$fold_test_time_5
	Accuracy: $accuracy_5
	Precision: $precision_5
	Recall: $recall_5
	F1: $f1_5
	Confusion Matrix: $tn_5 , $fp_5
	                  $fn_5 , $tp_5


=================================
Global Performance:
---------------------

Elapsed total time: $execution_time.
Folds Acc.: $folds_accuracy
Mean: $cross_v_mean
StdDev: $cross_v_stddev
CI:(95%) : $cross_v_interval






