import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



df = pd.read_csv('C:/Users/Asus/Pokemon.csv')
newdf = df.drop(['Name', '#', 'Type 1', 'Type 2'], axis='columns')
print(newdf)
print(df.columns)
sns.histplot(data=newdf, x='Attack', y='Generation')
plt.title('Amount of attack by Pokemons throughout the years')
plt.show()
sns.scatterplot(data=newdf, x='Attack', y='Defense')
plt.title('Attack and defense of Pokemons')
plt.show()
sns.boxplot(data=newdf, x='Generation', y='Total')
plt.show()
sns.heatmap(newdf.corr(), annot=True, cmap='coolwarm')
plt.show()
train_data, test_data, train_labels, test_labels = train_test_split(newdf, newdf['Legendary'], test_size=0.2, random_state=42)


nb_clf = GaussianNB()
lda_clf = LinearDiscriminantAnalysis()
qda_clf = QuadraticDiscriminantAnalysis()
nb_clf.fit(train_data, train_labels)
lda_clf.fit(train_data, train_labels)
qda_clf.fit(train_data, train_labels)


nb_preds = nb_clf.predict(test_data)
lda_preds = lda_clf.predict(test_data)
qda_preds = qda_clf.predict(test_data)
nb_accuracy = (nb_preds == test_labels).mean()
lda_accuracy = (lda_preds == test_labels).mean()
qda_accuracy = (qda_preds == test_labels).mean()
print("Naïve Bayes accuracy for the Pokemon dataset:", nb_accuracy)
print("LDA accuracy for the Pokemon dataset:", lda_accuracy)
print("QDA accuracy for the Pokemon dataset:", qda_accuracy)  