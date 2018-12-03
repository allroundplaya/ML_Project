from CNN import CNN

batch_size = 14
num_classes = 29
epochs = 6

cnn = CNN()
cnn.load_dataset()
cnn.make_cnn_model()
cnn.train_model(batch_size=batch_size, epochs=12)
cnn.print_score()

cnn.save_model()



