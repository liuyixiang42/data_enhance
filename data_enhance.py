import tensorflow as tf
import numpy as np

seq_length = 100 # 时序数据的长度
batch_size = 128 # 批次大小
latent_dim = 32 # 隐变量向量的大小
epochs = 10000 # 训练迭代次数

def load_data():
    data = np.load('data.npy')
    return data

# 定义生成器和判别器模型
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(seq_length * num_features))
    model.add(tf.keras.layers.Reshape((seq_length, num_features)))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_features, return_sequences=True, activation='sigmoid'))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(seq_length, num_features)))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# 定义生成器和判别器的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 定义损失函数和评估指标
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# 定义训练步骤
@tf.function
def train_step(real_seq):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成随机噪声
        noise = tf.random.normal([batch_size, latent_dim])
        # 通过生成器生成伪造的时序数据
        fake_seq = generator(noise, training=True)

        # 计算判别器对真实和伪造数据的输出
        real_output = discriminator(real_seq, training=True)
        fake_output = discriminator(fake_seq, training=True)

        # 计算损失
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # 计算生成器和判别器的梯度
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # 应用梯度更新模型
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

# 定义数据增强函数
def generate_samples(model, input_seq, num_samples=1):
    generated_seq = []
    for _ in range(num_samples):
        # 生成随机噪声
        noise = tf.random.normal([1, latent_dim])
        # 通过生成器生成伪造的时序数据
        fake_seq = model(noise, training=False)

        # 将伪造的数据与输入数据拼接在一起
        generated_seq.append(np.concatenate([input_seq, fake_seq.numpy()], axis=0))

    return np.array(generated_seq)

# 加载数据集
data = load_data()

# 获取数据集中时序数据的特征维度
num_features = data.shape[-1]

# 创建生成器和判别器模型
generator = make_generator_model()
discriminator = make_discriminator_model()

# 定义训练循环
for epoch in range(epochs):
    # 随机选择一批真实的时序数据
    idx = np.random.randint(0, data.shape[0], batch_size)
    real_seq = data[idx]

    # 训练一次生成器和判别器
    gen_loss, disc_loss = train_step(real_seq)

    # 每隔一定步数，输出一下损失和生成的数据
    # 输出损失
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

    # 输出生成的数据
    if (epoch + 1) % 500 == 0:
        # 随机选择一条输入数据
        input_seq = data[np.random.randint(0, data.shape[0], 1)]
        # 生成多条增强数据
        generated_seq = generate_samples(generator, input_seq, num_samples=5)
        # 输出增强数据
        print(f"Generated Samples at epoch {epoch + 1}:")
        for i in range(generated_seq.shape[0]):
            print(generated_seq[i])





