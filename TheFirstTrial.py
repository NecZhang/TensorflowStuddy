import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)

# The first testing program
mA = tf.constant([[1, 2], [3, 4]])
mB = tf.constant([[5, 6], [7, 8]])
mC = tf.matmul(mA, mB)
mD = tf.add(mA, mB)
print(mC)
print(mD)

# The basic conception
random_float = tf.random.uniform(shape=())      # 定义一个随机数（标量）
print(random_float)

zero_vector = tf.zeros(shape=(2))               # 定义一个有两个元素的零向量
print(zero_vector)

A = tf.constant([[1., 2.], [3., 4.]])             # 定义两个2*2的常量矩阵
B = tf.constant([[5., 6.], [7., 8.]])             #

# 查看矩阵A的形状、类型、值
print(A.shape)          # 输出（2，2），即矩阵的长和宽均为2
print(A.dtype)
print(A.numpy())

# Operations
C = tf.add(A, B)        # 计算矩阵和
D = tf.matmul(A, B)     # 计算矩阵积
print(C)
print(D)
