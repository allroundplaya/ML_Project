from CNN import CNN
from preprocess import preprocess_all_data
from settings import *

batch_size = 100
num_classes = 29
epochs = 12
x_train_path = DATA_DIR + "/train_X.npz"
y_train_path = DATA_DIR + "/train_y.npz"
x_test_path = DATA_DIR + "/test_X.npz"
y_test_path = DATA_DIR + "/test_y.npz"


preprocess_all_data()

cnn = CNN()
cnn.load_dataset(x_train_path=x_train_path, y_train_path=y_train_path,
                 x_test_path=x_test_path, y_test_path=y_test_path)
cnn.make_cnn_model()
cnn.train_model(batch_size=batch_size, epochs=epochs)
cnn.print_score()

cnn.save_model()



