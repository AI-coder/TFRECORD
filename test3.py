import tensorflow as tf


# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# num_shards定义了写入多少个文本，instances_per_shard定义了每个文件中有多少个数据
num_shards = 2
instances_per_shard = 4
for i in range(num_shards):
    # 数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分
    # m为文件总数，n为当前文件编号
    filename = ('/tensorflow_google/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    # 将数据封装成Example写入TFRecord文件中
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
files = tf.train.match_filenames_once("/tensorflow_google/data.tfrecords-*")
# 通过tf.train.string_input_producer创建输入队列，文件列表为files
# shuffle为False避免随机打乱读文件的顺序，只会打乱文件列表的顺序，文件内部样例输出顺序不变
# 一般来说，shuffle设置为True
# num_epochs=1时，在读完数据一轮后会自动停止
filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=1)
# 读取并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })

with tf.Session() as sess:
    # tf.train.match_filenames_once函数需要初始化
    tf.local_variables_initializer().run()
    print(sess.run(files))
    # 声明tf.train.Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 多次执行获取数据的操作
    for i in range(10):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
