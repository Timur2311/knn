import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_excel('tulipa.xlsx')
data.head()
# calculate the correlation
corr = data.corr()
# plot the heatmap
sns.heatmap(corr, cmap="Blues", annot=True)

x = data[['stamen_length','ovary_color','bulb_shell_pubescence', 'coloring_of_integumentary_scales', 'shell_consistency','bulb_shell_duration']]
y = data['type_class']

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.20, random_state=3)
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train,y_train)
scr=knn.score(X_test,y_test)
print("Score for your model is ",scr)

#1: '1-tip', 2: '2-tip', 3: '3-tip', 4: '4-tip'
res=knn.predict([[100,100,3000,100,100,100000]])
if res==1:
    print("1-tip")
elif res==2:
    print("2-tip")
elif res==3:
    print("3-tip")
else:
    print("4-tip")
		


#Find best K value
k_range = range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
