import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import GC
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from Bio.Alphabet import IUPAC

# Load the genomic sequence data
def load_data(file_path):
	sequences = []
	labels = []
	with open(file_path, "r") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			sequences.append(str(record.seq))
			labels.append(record.id.split('|')[1])
	return sequences, labels

if __name__ == "__main__":
	# Data Analysis Procedures:
	# 1. **Data Loading and Preprocessing:**
	file_path = 'genomic_sequences.fasta'
	sequences, labels = load_data(file_path)

	#2. **Sequence Feature Extraction:**
	# Calculate GC content
	gc_content = [GC(seq) for seq in sequences]

	# Count nucleotide frequencies
	nucleotide_counts = [Counter(seq) for seq in sequences]
	a_count = [count['A'] for count in nucleotide_counts]
	c_count = [count['C'] for count in nucleotide_counts]
	g_count = [count['G'] for count in nucleotide_counts]
	t_count = [count['T'] for count in nucleotide_counts]

	#3. **Create a Data Frame to store features:**
	feature_data = pd.DataFrame({
		'GC_content': gc_content,
		'A_count': a_count,
		'C_count': c_count,
		'G_count': g_count,
		'T_count': t_count,
		'Label': labels
	})

	#4. **Data Exploration and Visualization:**
	print(feature_data.describe())
	feature_data.hist(bins=20, figsize=(10, 8))
	plt.show()

	#5. **Split data into training and testing sets:**
	X = feature_data.drop(columns=['Label'])
	y = feature_data['Label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#6. **Model Training using Random Forest:**
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=100, random_state=42)
	clf.fit(X_train, y_train)

	#7. **Model Evaluation:**
	y_pred = clf.predict(X_test)
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("Classification Report:")
	print(classification_report(y_test, y_pred))

	#8. **Importance of features:**
	importances = clf.feature_importances_
	feature_names = X.columns
	indices = np.argsort(importances)[::-1]
	plt.figure(figsize=(10, 6))
	plt.title("Feature Importances")
	plt.bar(range(X.shape[1]), importances[indices], align="center")
	plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
	plt.show()
