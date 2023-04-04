import tensorflow as tf
import numpy as np
import utils.functions as func


# updater 函数的作用是根据训练类型(train_type)和输入的图像(in_im)，对输入的噪声图像(noiser)进行更新，并执行反向传播算法。具体含义如下：

# 根据训练类型(train_type)的不同，updater 函数会有不同的实现逻辑：
# 如果 train_type 为 'no_data'，表示不需要使用数据集进行训练，此时 updater 函数只需要执行一次反向传播算法即可。

# 如果 train_type 为 'with_range'，表示需要使用数据集中的一部分样本进行训练。updater 函数会从数据集中随机选择一张图像，将其转换为噪声图像，并将其复制到输入的噪声图像(noiser)中，然后执行反向传播算法。

# 如果 train_type 为 'with_data'，表示需要使用完整的数据集进行训练。updater 函数会从数据集中随机选择一批(batch_size)图像，将其转换为噪声图像，并将其复制到输入的噪声图像(noiser)中，然后执行反向传播算法。

# updater 函数的参数包括噪声图像(noiser)、会话对象(sess)、更新操作(update)、输入的图像(in_im)、批大小(batch_size)、图像大小(size)和图像列表(img_list)。

# 在 updater 函数中，根据不同的训练类型(train_type)，使用 func.img_preprocess 函数将数据集中的图像转换为噪声图像，并将其复制到输入的噪声图像(noiser)中。

# 最后，使用会话对象(sess)和更新操作(update)，执行反向传播算法，更新模型的参数。

# 因此，这段代码的主要作用是定义一个 updater 函数，用于根据不同的训练类型，将噪声图像更新为更接近于原图像的形式，并执行反向传播算法，以训练模型。
def get_update_operation_func(train_type, in_im, sess, update, batch_size, size, img_list):
    if train_type == 'no_data':
        def updater(noiser, sess=sess, update=update):
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_range':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size):
            image_i = 'data/gaussian_noise.png'
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(image_i,
                                                            size=size, augment=True))
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_data':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size, img_list=img_list):
            rander = np.random.randint(low=0, high=(len(img_list)-batch_size))
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(
                    img_list[rander+j].strip(), size=size, augment=True))
            sess.run(update, feed_dict={in_im: noiser})
    return updater