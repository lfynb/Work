import pandas as pd

#用pandas读取数据
iris_data = pd.read_csv('iris.data')
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
print(iris_data.head(5))
iris_data.describe()
#数据可视化
import matplotlib.pyplot as plt
import seaborn as sb
sb.pairplot(iris_data.dropna(), hue='class')

#用决策树划分
from sklearn.model_selection import train_test_split
all_inputs = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
all_classes = iris_data['class'].values
from sklearn.tree import DecisionTreeClassifier
    
    
decision_tree_classifier = DecisionTreeClassifier()
#    划分训练系与测试集
(training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_inputs,all_classes,test_size=0.25,random_state=1)
# 在训练集上训练分类器  
decision_tree_classifier.fit(training_inputs,training_classes) 
# 使用分类准确性在测试集上验证分类器
decision_tree_classifier.score(testing_inputs,testing_classes)

from sklearn.model_selection import cross_val_score
import numpy as np
cv_scores = cross_val_score(decision_tree_classifier,all_inputs,all_classes,cv=10)
#cross_val_score返回我们可以想象的分数列表
#对分类器的性能进行合理估
print(cv_scores)
sb.distplot(cv_scores)
plt.title('Average score:{}'.format(np.mean(cv_scores)))

#SVM划分
def show_accuracy(y_hat,y_train,str):
    pass

def SVM():
    from sklearn import svm

    # kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
    classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)

    #调用ravel()函数将矩阵转变成一维数组
    classifier.fit(training_inputs,training_classes.ravel())
   
    # （4）计算svm分类器的准确率
    print("SVM-输出训练集的准确率为：", classifier.score(training_inputs, training_classes))
    y_hat = classifier.predict(training_inputs)
    show_accuracy(y_hat, training_classes, '训练集')
    print("SVM-输出测试集的准确率为：", classifier.score(testing_inputs, testing_classes))
    y_hat = classifier.predict(testing_inputs)
    show_accuracy(y_hat, testing_classes, '测试集')

SVM()

#knn划分
def knn():
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(training_inputs, training_classes)
    y_pred = knn.predict(testing_inputs)
    print(np.mean(y_pred == testing_classes))
knn()



