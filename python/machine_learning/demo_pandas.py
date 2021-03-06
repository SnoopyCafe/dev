import pandas as pd

names = ['student1','student2','student3','student4','student5']
grades_pct = [80,75,65,87,92]
studentDataset = list(zip(names,grades_pct))

print (studentDataset, "\n")

# Data Frame
df = pd.DataFrame(data=studentDataset, columns=['Names','grades_pct'])
print (df, "\n")

# Export to CSV
# path = 'student_data.csv'
# df.to_csv(path)

# Sorting
print (df.sort_values(['grades_pct'],ascending=True), "\n")

# Dataset stats
print (df.describe())

