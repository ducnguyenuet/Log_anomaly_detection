�	'����ER@'����ER@!'����ER@	����&�?����&�?!����&�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$'����ER@�HP�Q@A���e@Y�:pΈ�?*	�����)|@2F
Iterator::Modelm���{��?!e���p�U@)n���?1�4���T@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[2]::ConcatenatejM�S�?!<����!@)j�q���?1v�A�` @:Preprocessing2S
Iterator::Model::ParallelMap�?�߾�?!�DM�R@)�?�߾�?1�DM�R@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatF%u��?!��wo@)Ǻ����?1Q��u�@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipŏ1w-!�?!�
Cz�*@)�+e�Xw?1��G�[=�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��g��s�?!B�@��"@)/n��b?1�d�2J?�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����Mb`?!,�@�h�?)����Mb`?1,�@�h�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor_�Q�[?!>B��D%�?)_�Q�[?1>B��D%�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice/n��R?!�d�2J?�?)/n��R?1�d�2J?�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B93.5 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�HP�Q@�HP�Q@!�HP�Q@      ��!       "      ��!       *      ��!       2	���e@���e@!���e@:      ��!       B      ��!       J	�:pΈ�?�:pΈ�?!�:pΈ�?R      ��!       Z	�:pΈ�?�:pΈ�?!�:pΈ�?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationN
nohigh"B93.5 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 