�	�y�):L@�y�):L@!�y�):L@	����tH�?����tH�?!����tH�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$�y�):L@;�O���H@A9��v��@Y��z6��?*	43333W@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[2]::ConcatenateK�=�U�?!�/����@@)B>�٬��?1���/�>@:Preprocessing2F
Iterator::ModelX�5�;N�?!����ROB@)�
F%u�?1����ڄ;@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��ׁsF�?!��Z�s5@)L7�A`�?1o�$�a�1@:Preprocessing2S
Iterator::Model::ParallelMap�J�4�?!K�Ú�3"@)�J�4�?1K�Ú�3"@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�ʡE��?!6��O@)��_vOv?1��4�f@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorF%u�k?!�k��5�@)F%u�k?1�k��5�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��镲�?!�Y��	B@)��_vOf?1��4�f@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicea2U0*�S?!27�g>��?)a2U0*�S?127�g>��?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor/n��R?!C�k4y�?)/n��R?1C�k4y�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B87.9 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;�O���H@;�O���H@!;�O���H@      ��!       "      ��!       *      ��!       2	9��v��@9��v��@!9��v��@:      ��!       B      ��!       J	��z6��?��z6��?!��z6��?R      ��!       Z	��z6��?��z6��?!��z6��?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationN
nohigh"B87.9 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 