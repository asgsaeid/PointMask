import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data_loader import *
from model_cls import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from schedules import onetenth_50_75
import matplotlib.pyplot as plt


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_model.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def main():
    nb_classes = 40
    train_file = 'PATH/ModelNet40/ply_data_train.h5'
    val_file = 'PATH/ModelNet40/ply_data_test.h5'

    num_samples_train = len(h5py.File(train_file, mode='r')['data'])
    num_samples_val = len(h5py.File(val_file, mode='r')['data'])
    epochs = 500
    batch_size = 32

    train = DataGenerator(train_file, batch_size, nb_classes, train=True)
    val = DataGenerator(val_file, batch_size, nb_classes, train=False)

    model = point_mask(nb_classes)
    model.summary()
    lr = 0.001
    adam = Adam(lr=lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if not os.path.exists('PATH/results/'):
        os.mkdir('PATH/results/')
    checkpoint = ModelCheckpoint('PATH/results/pointmask.h5',
                                 monitor='val_acc', mode='max', save_weights_only=True, save_best_only=True, verbose=1)
    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=num_samples_train // batch_size,
                                  validation_data=val.generator(),
                                  validation_steps=num_samples_val // batch_size,
                                  epochs=epochs,
                                  callbacks=[checkpoint, onetenth_50_75(lr)],
                                  verbose=1)

    plot_history(history, 'PATH/results/')
    save_history(history, 'PATH/results/')


if __name__ == '__main__':
    main()
