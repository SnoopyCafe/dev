import pandas as pd

names = ['student1','student2','student3','student4','student5']
grades_pct = [80,75,65,87,92]
studentDataset = list(zip(names,grades_pct))
print (studentDataset)

# Data Frame
df = pd.DataFrame(data=studentDataset, columns=['Names','grades_pct'])

path = 'student_data.csv'
df.to_csv(path)
print (df)