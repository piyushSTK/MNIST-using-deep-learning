import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()


images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:7]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
#plt.show()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
dummy =np.zeros((1797,10))
print dummy.shape
for i in range(0,1797):
	dummy[i][digits.target[i]]=1
print dummy[0]
print digits.target.shape
X_train,X_test,Y_train,Y_test=train_test_split(data,dummy,test_size=0.33,random_state=2)

################################################the real model
model=Sequential()
model.add(Dense(100,input_dim=64,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(10,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=15,batch_size=100)
_,accuracy=model.evaluate(X_test,Y_test)
print accuracy