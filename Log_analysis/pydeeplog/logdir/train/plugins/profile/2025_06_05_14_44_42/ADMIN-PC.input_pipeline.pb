	������R@������R@!������R@	�g����?�g����?!�g����?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$������R@�a��4�O@A��?��&@Yŏ1w-�?*	�����ق@2F
Iterator::Model#��~j��?!�Қ��R@)��q���?1�&��Q@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip���ZӼ�?!���L�9@)�:pΈ�?1d̫[(@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenatez�):�˯?!z��h�$@)�(��?1�@iM~v#@:Preprocessing2S
Iterator::Model::ParallelMapΈ����?!�v���@)Έ����?1�v���@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatK�=�U�?!n���J@)9��v���?1����=@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�/�$�?!��"��&@)/n��r?1*R�ȘW�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�խ 7g�?)HP�s�b?1�խ 7g�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensorŏ1w-!_?!���!�(�?)ŏ1w-!_?1���!�(�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�~j�t�X?!
ps����?)�~j�t�X?1
ps����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B84.2 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�a��4�O@�a��4�O@!�a��4�O@      ��!       "      ��!       *      ��!       2	��?��&@��?��&@!��?��&@:      ��!       B      ��!       J	ŏ1w-�?ŏ1w-�?!ŏ1w-�?R      ��!       Z	ŏ1w-�?ŏ1w-�?!ŏ1w-�?JCPU_ONLY