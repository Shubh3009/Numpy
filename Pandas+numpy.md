# NumPy & Pandas Documentation Notes (Session-wise)

## Session 13: NumPy Fundamentals

### 1. What is NumPy?
- **Definition**: NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object (`ndarray`) and tools for fast array operations.
- **Core Object**: `ndarray` – n-dimensional array of homogeneous data types.
- **Features**: Mathematical, logical, shape manipulation, sorting, linear algebra, statistics, random simulation, etc.

### 2. NumPy Arrays vs Python Lists
| Feature | NumPy Arrays | Python Lists |
|---------|-------------|--------------|
| Size | Fixed at creation | Dynamic |
| Data Type | Homogeneous | Heterogeneous |
| Performance | Faster, memory efficient | Slower |
| Operations | Vectorized operations | Loop-based |

### 3. Creating NumPy Arrays
```python
import numpy as np

# From list
a = np.array([1, 2, 3])
# 2D array
b = np.array([[1, 2, 3], [4, 5, 6]])
# With dtype
c = np.array([1, 2, 3], dtype=float)
# Using arange
d = np.arange(1, 11, 2)
# With reshape
e = np.arange(16).reshape(2, 2, 2, 2)
# Ones and zeros
ones = np.ones((3, 4))
zeros = np.zeros((3, 4))
# Random
rand = np.random.random((3, 4))
# Linspace
lin = np.linspace(-10, 10, 10, dtype=int)
# Identity matrix
iden = np.identity(3)
```

### 4. Array Attributes
```python
a = np.arange(8).reshape(2, 2, 2)
print(a.ndim)    # 3
print(a.shape)   # (2, 2, 2)
print(a.size)    # 8
print(a.dtype)   # int64
print(a.itemsize) # 8 (bytes)
```

### 5. Changing Data Type
```python
a = a.astype(np.int32)
```

### 6. Array Operations
```python
a1 = np.arange(12).reshape(3, 4)
a2 = np.arange(12, 24).reshape(3, 4)

# Scalar operations
print(a1 * 2)
print(a2 == 15)

# Vector operations
print(a1 * a2)
```

### 7. Array Functions
```python
# Math functions
print(np.sin(a1))
print(np.exp(a1))
print(np.log(a1))

# Statistical functions
print(np.mean(a1, axis=0))
print(np.median(a1, axis=1))
print(np.std(a1))
print(np.var(a1))

# Rounding
print(np.round(a1))
print(np.ceil(a1))
print(np.floor(a1))

# Dot product
a3 = np.arange(12).reshape(4, 3)
print(np.dot(a2, a3))
```

### 8. Indexing and Slicing
```python
a1 = np.arange(10)
a2 = np.arange(12).reshape(3, 4)
a3 = np.arange(8).reshape(2, 2, 2)

# Basic indexing
print(a1[2])
print(a2[1, 0])
print(a3[1, 0, 1])

# Slicing
print(a1[2:5:2])
print(a2[0:2, 1::2])
print(a2[::2, 1::2])
```

### 9. Iterating
```python
for i in np.nditer(a3):
    print(i)
```

### 10. Reshaping
```python
print(a2.reshape(2, 6))
print(a2.T)  # Transpose
print(a2.ravel())  # Flatten
```

### 11. Stacking
```python
# Horizontal stacking
print(np.hstack((a1, a2)))
# Vertical stacking
print(np.vstack((a1, a2)))
```

### 12. Splitting
```python
# Horizontal split
print(np.hsplit(a2, 2))
# Vertical split
print(np.vsplit(a2, 3))
```

---

## Session 14: NumPy Advanced

### 1. NumPy Array vs Python Lists
**Speed Comparison**:
```python
import time

# Python list
start = time.time()
c = [a[i] + b[i] for i in range(len(a))]
print(time.time() - start)  # ~3.26 seconds

# NumPy array
start = time.time()
c = a_np + b_np
print(time.time() - start)  # ~0.06 seconds
```

**Memory Comparison**:
```python
import sys
a_list = [i for i in range(1000000)]
a_np = np.arange(1000000, dtype=np.int8)
print(sys.getsizeof(a_list))  # ~8MB
print(sys.getsizeof(a_np))    # ~1MB
```

### 2. Advanced Indexing
```python
a = np.arange(24).reshape(6, 4)

# Normal indexing
print(a[1, 2])          # 5
print(a[1:3, 1:3])      # [[4, 5], [7, 8]]

# Fancy indexing
print(a[:, [0, 2, 3]])  # Select specific columns

# Boolean indexing
mask = a > 50
print(a[mask])
print(a[a % 2 == 0])
print(a[(a > 50) & (a % 2 == 0)])
```

### 3. Broadcasting
Rules for broadcasting:
1. Make arrays have same number of dimensions
2. Each dimension should be same size or size 1

```python
# Same shape
a = np.arange(6).reshape(2, 3)
b = np.arange(6, 12).reshape(2, 3)
print(a + b)

# Different shape (broadcasting)
a = np.arange(12).reshape(4, 3)
b = np.arange(3)
print(a + b)  # b broadcast to shape (4, 3)
```

### 4. Working with Mathematical Formulas
```python
# Sigmoid function
def sigmoid(array):
    return 1 / (1 + np.exp(-array))

# Mean Squared Error
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)
```

### 5. Working with Missing Values
```python
a = np.array([1, 2, 3, 4, np.nan, 6])
print(a[~np.isnan(a)])  # Remove NaN values
```

### 6. Plotting Graphs
```python
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = x ** 2
plt.plot(x, y)
plt.show()
```

---

## Session 15: NumPy Tricks

### 1. `np.concatenate()`
```python
c = np.array([[0, 1, 2], [3, 4, 5]])
d = np.array([[6, 7, 8], [9, 10, 11]])

print(np.concatenate((c, d), axis=0))  # Vertical
print(np.concatenate((c, d), axis=1))  # Horizontal
```

### 2. `np.unique()`
```python
e = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(np.unique(e))  # [1, 2, 3, 4, 5, 6]
```

### 3. `np.expand_dims()`
```python
a = np.array([11, 53, 28, 50, 38, 37])
print(np.expand_dims(a, axis=0).shape)  # (1, 6)
print(np.expand_dims(a, axis=1).shape)  # (6, 1)
```

### 4. `np.where()`
```python
print(np.where(a > 50))  # Indices where condition is True
print(np.where(a > 50, 0, a))  # Replace values >50 with 0
print(np.where(a % 2 == 0, 0, a))  # Replace even values with 0
```

### 5. `np.argmax()` / `np.argmin()`
```python
print(np.argmax(a))  # Index of max value
print(np.argmax(b, axis=0))  # Max along columns
print(np.argmax(b, axis=1))  # Max along rows
print(np.argmin(a))  # Index of min value
```

### 6. `np.cumsum()` / `np.cumprod()`
```python
print(np.cumsum(a))  # Cumulative sum
print(np.cumsum(b, axis=1))  # Cumulative sum along columns
print(np.cumprod(a))  # Cumulative product
```

### 7. `np.percentile()`
```python
print(np.percentile(a, 50))  # 50th percentile (median)
print(np.median(a))  # Median
```

### 8. `np.histogram()`
```python
hist, bins = np.histogram(a, bins=[0, 50, 100])
print(hist)  # Frequency counts
print(bins)  # Bin edges
```

### 9. `np.corrcoef()`
```python
salary = np.array([20000, 40000, 25000, 35000, 60000])
experience = np.array([1, 3, 2, 4, 2])
print(np.corrcoef(salary, experience))
```

### 10. `np.isin()`
```python
items = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(a[np.isin(a, items)])  # Elements in 'a' that are in 'items'
```

### 11. `np.flip()`
```python
print(np.flip(a))  # Reverse array
print(np.flip(b, axis=1))  # Reverse along columns
```

### 12. `np.put()`
```python
np.put(a, [0, 1], [110, 530])  # Replace elements at indices 0,1
```

### 13. `np.delete()`
```python
print(np.delete(a, [0, 2, 4]))  # Delete elements at indices
```

### 14. Set Functions
```python
m = np.array([1, 2, 3, 4, 5])
n = np.array([3, 4, 5, 6, 7])

print(np.union1d(m, n))  # [1, 2, 3, 4, 5, 6, 7]
print(np.intersect1d(m, n))  # [3, 4, 5]
print(np.setdiff1d(m, n))  # [1, 2]
print(np.setxor1d(m, n))  # [1, 2, 6, 7]
print(np.in1d(m, 1))  # [True, False, False, False, False]
```

### 15. `np.clip()`
```python
print(np.clip(a, a_min=25, a_max=75))  # Limit values between 25 and 75
```

---

## Session 16: Pandas Series

### 1. What is Pandas?
- **Definition**: Fast, powerful, flexible open-source data analysis/manipulation tool.
- **Series**: 1D array-like object holding data of any type.

### 2. Creating Series
```python
import pandas as pd

# From list
country = pd.Series(['India', 'Pakistan', 'USA', 'Nepal', 'Srilanka'])
runs = pd.Series([13, 24, 56, 78, 100])

# With custom index
marks = pd.Series([67, 57, 89, 100], index=['maths', 'english', 'science', 'hindi'])

# From dictionary
marks_dict = {'maths': 67, 'english': 57, 'science': 89, 'hindi': 100}
marks_series = pd.Series(marks_dict)

# From CSV
subs = pd.read_csv('subs.csv', squeeze=True)
vk = pd.read_csv('kohli_ipl.csv', index_col='match_no', squeeze=True)
```

### 3. Series Attributes
```python
print(marks_series.size)
print(marks_series.dtype)
print(marks_series.name)
print(marks_series.is_unique)
print(marks_series.index)
print(marks_series.values)
```

### 4. Series Methods
```python
# Head/Tail
print(vk.head(3))
print(vk.tail(10))

# Sample
print(movies.sample(5))

# Value counts
print(movies.value_counts())

# Sorting
print(vk.sort_values(ascending=False))
print(movies.sort_index(ascending=False, inplace=True))

# Math methods
print(vk.count())
print(subs.sum())
print(subs.mean())
print(subs.median())
print(subs.mode())
print(subs.std())
print(subs.var())
print(subs.min())
print(subs.max())
print(subs.describe())
```

### 5. Series Indexing
```python
# Integer indexing
print(x[2])

# Slicing
print(vk[5:16])
print(vk[-5:])

# Fancy indexing
print(vk[[1, 3, 4, 5]])

# Label indexing
print(movies['2 States (2014 film)'])
```

### 6. Editing Series
```python
marks_series[1] = 100
marks_series['evs'] = 100  # Add new index
runs_ser[2:4] = [100, 100]
runs_ser[[0, 3, 4]] = [0, 0, 0]
movies['2 States (2014 film)'] = 'Alia Bhatt'
```

### 7. Type Conversion & Membership
```python
print(list(marks_series))
print(dict(marks_series))
print('2 States (2014 film)' in movies)
print('Alia Bhatt' in movies.values)
```

### 8. Boolean Indexing on Series
```python
# Find 50s and 100s
print(vk[vk == 50].size)
print(vk[vk == 100].size)

# Find ducks
print(vk[vk == 0].size)

# Find good days (above mean)
print(subs[subs > subs.mean()].size)

# Actors with >20 movies
num_movies = movies.value_counts()
print(num_movies[num_movies > 20])
```

### 9. Important Series Methods
```python
# astype
print(vk.astype('int16'))

# between
print(vk[vk.between(51, 99)].size)

# clip
print(subs.clip(100, 200))

# drop_duplicates
temp = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
print(temp.drop_duplicates(keep='last'))

# isnull / dropna / fillna
temp = pd.Series([1, 2, 3, np.nan, 5, 6, np.nan, 8, np.nan, 10])
print(temp.isnull().sum())
print(temp.dropna())
print(temp.fillna(temp.mean()))

# isin
print(vk[vk.isin([49, 99])])

# apply
print(movies.apply(lambda x: x.split()[0].upper()))

# copy
new = vk.head().copy()
new[1] = 100
```

---

## Session 17: Pandas DataFrames

### 1. Creating DataFrames
```python
# From lists
student_data = [[100, 80, 10], [90, 70, 7], [120, 100, 14], [80, 50, 2]]
df1 = pd.DataFrame(student_data, columns=['iq', 'marks', 'package'])

# From dictionary
student_dict = {
    'name': ['nitish', 'ankit', 'rupesh', 'rishabh', 'amit', 'ankita'],
    'iq': [100, 90, 120, 80, 0, 0],
    'marks': [80, 70, 100, 50, 0, 0],
    'package': [10, 7, 14, 2, 0, 0]
}
students = pd.DataFrame(student_dict)
students.set_index('name', inplace=True)

# From CSV
movies = pd.read_csv('movies.csv')
ipl = pd.read_csv('ipl-matches.csv')
```

### 2. DataFrame Attributes
```python
print(movies.shape)
print(ipl.dtypes)
print(movies.index)
print(movies.columns)
print(movies.values)

# Head/Tail/Sample
print(movies.head(2))
print(ipl.tail(5))
print(ipl.sample(5))

# Info and describe
movies.info()
print(movies.describe())

# Null values and duplicates
print(movies.isnull().sum())
print(movies.duplicated().sum())
```

### 3. Renaming Columns
```python
students.rename(columns={'marks': 'percent', 'package': 'lpa'}, inplace=True)
```

### 4. Math Methods
```python
print(students.sum(axis=0))  # Column-wise sum
print(students.mean(axis=1))  # Row-wise mean
print(students.var())  # Variance
```

### 5. Selecting Columns
```python
# Single column
print(movies['title_x'])

# Multiple columns
print(movies[['year_of_release', 'actors', 'title_x']])
```

### 6. Selecting Rows
```python
# Using iloc (position-based)
print(movies.iloc[5])  # Single row
print(movies.iloc[0:3])  # Multiple rows
print(movies.iloc[[0, 4, 5]])  # Fancy indexing

# Using loc (label-based)
print(students.loc['nitish'])
print(students.loc['nitish':'rishabh':2])
print(students.loc[['nitish', 'ankita', 'rupesh']])
```

### 7. Selecting Both Rows and Columns
```python
print(movies.iloc[0:3, 0:3])
print(movies.iloc[0:2, ['title_x', 'poster_path']])
```

### 8. Boolean Indexing
```python
# Find all final winners
final_winners = ipl[ipl['MatchNumber'] == 'Final'][['Season', 'WinningTeam']]

# Super over finishes
super_overs = ipl[ipl['SuperOver'] == 'Y'].shape[0]

# Matches CSK won in Kolkata
csk_kolkata = ipl[(ipl['City'] == 'Kolkata') & (ipl['WinningTeam'] == 'Chennai Super Kings')].shape[0]

# Toss winner is match winner percentage
toss_match_winner = (ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0] / ipl.shape[0]) * 100

# Action movies with rating > 7.5
mask1 = movies['genres'].str.contains('Action')
mask2 = movies['imdb_rating'] > 7.5
action_high_rated = movies[mask1 & mask2]
```

### 9. Adding New Columns
```python
movies['Country'] = 'India'
```

### 10. DataFrame Functions
```python
# astype
ipl['ID'] = ipl['ID'].astype('int32')
ipl.info()
```

---

## Session 18: DataFrame Methods

### 1. `value_counts()`
```python
# Series
a = pd.Series([1, 1, 1, 2, 2, 3])
print(a.value_counts())

# DataFrame
marks = pd.DataFrame([[100, 80, 10], [90, 70, 7], [120, 100, 14], [80, 70, 14], [80, 70, 14]],
                     columns=['iq', 'marks', 'package'])
print(marks.value_counts())

# Real-world example
player_counts = ipl['Player_of_Match'].value_counts()
```

### 2. `sort_values()`
```python
# Series
x = pd.Series([12, 14, 1, 56, 89])
print(x.sort_values(ascending=False))

# DataFrame with NaN handling
students = pd.DataFrame({
    'name': ['nitish', 'ankit', 'rupesh', np.nan, 'mrityunjay', np.nan, 'rishabh', np.nan, 'aditya', np.nan],
    'college': ['bit', 'iit', 'vit', np.nan, np.nan, 'vlsi', 'ssit', np.nan, np.nan, 'git'],
    'branch': ['eee', 'it', 'cse', np.nan, 'me', 'ce', 'civ', 'cse', 'bio', np.nan],
    'cgpa': [6.66, 8.25, 6.41, np.nan, 5.6, 9.0, 7.4, 10, 7.4, np.nan],
    'package': [4, 5, 6, np.nan, 6, 7, 8, 9, np.nan, np.nan]
})

students.sort_values('name', na_position='first', ascending=False, inplace=True)

# Multiple columns
movies.sort_values(['year_of_release', 'title_x'], ascending=[True, False])
```

### 3. `rank()`
```python
batsman['batting_rank'] = batsman['batsman_run'].rank(ascending=False)
batsman.sort_values('batting_rank')
```

### 4. `sort_index()`
```python
# Series
marks_series.sort_index(ascending=False)

# DataFrame
movies.sort_index(ascending=False)
```

### 5. `set_index()` / `reset_index()`
```python
# Set index
batsman.set_index('batter', inplace=True)

# Reset index
batsman.reset_index(inplace=True)
marks_series.reset_index()

# Series to DataFrame
marks_series.reset_index()
```

### 6. `rename()`
```python
movies.set_index('title_x', inplace=True)
movies.rename(columns={'imdb_id': 'imdb', 'poster_path': 'link'}, inplace=True)
movies.rename(index={'Uri: The Surgical Strike': 'Uri', 'Battalion 609': 'Battalion'})
```

### 7. `unique()` / `nunique()`
```python
print(ipl['Season'].unique())
print(ipl['Season'].nunique())
```

### 8. `isnull()` / `notnull()` / `hasnans`
```python
print(students['name'].isnull())
print(students['name'].notnull())
print(students['name'].hasnans)
print(students.isnull())
print(students.notnull())
```

### 9. `dropna()`
```python
# Series
print(students['name'].dropna())

# DataFrame
print(students.dropna(how='any'))
print(students.dropna(how='all'))
print(students.dropna(subset=['name']))
print(students.dropna(subset=['name', 'college']))
students.dropna(inplace=True)
```

### 10. `fillna()`
```python
# Series
print(students['name'].fillna('unknown'))
print(students['package'].fillna(students['package'].mean()))
print(students['name'].fillna(method='bfill'))
```

### 11. `drop_duplicates()`
```python
# Series
temp = pd.Series([1, 1, 1, 2, 3, 3, 4, 4])
print(temp.drop_duplicates())

# DataFrame
print(marks.drop_duplicates(keep='last'))

# Real-world example
ipl['all_players'] = ipl['Team1Players'] + ipl['Team2Players']
def did_kohli_play(players_list):
    return 'V Kohli' in players_list

ipl['did_kohli_play'] = ipl['all_players'].apply(did_kohli_play)
last_match_delhi = ipl[(ipl['City'] == 'Delhi') & (ipl['did_kohli_play'] == True)].drop_duplicates(subset=['City', 'did_kohli_play'])
```

### 12. `drop()`
```python
temp = pd.Series([10, 2, 3, 16, 45, 78, 10])
print(temp.drop(index=[0, 6]))
```

---

## Session 19: GroupBy Objects

### 1. Creating GroupBy Objects
```python
movies = pd.read_csv('imdb-top-1000.csv')
genres = movies.groupby('Genre')
```

### 2. Aggregation Functions
```python
# Basic statistics
print(genres.std())
print(genres.mean())

# Top 3 genres by total earnings
top_genres = movies.groupby('Genre').sum()['Gross'].sort_values(ascending=False).head(3)

# Genre with highest avg IMDb rating
highest_rated_genre = movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False).head(1)

# Director with most popularity
popular_director = movies.groupby('Director')['No_of_Votes'].sum().sort_values(ascending=False).head(1)

# Highest rated movie of each genre
highest_per_genre = movies.groupby('Genre')['IMDB_Rating'].max()

# Number of movies by each actor
actor_counts = movies.groupby('Star1')['Series_Title'].count().sort_values(ascending=False)
```

### 3. GroupBy Attributes and Methods
```python
# Number of groups
print(len(genres))
print(movies['Genre'].nunique())

# Size of each group
print(genres.size())

# First/last/nth item
print(genres.first())
print(genres.last())
print(genres.nth(6))

# Get specific group
print(genres.get_group('Fantasy'))

# Groups dictionary
print(genres.groups)

# Describe
print(genres.describe())

# Sample
print(genres.sample(2))

# Unique values per group
print(genres['Director'].unique())
```

### 4. Multiple Aggregations
```python
genres.agg({
    'Runtime': ['min', 'mean'],
    'IMDB_Rating': 'mean',
    'No_of_Votes': ['sum', 'max'],
    'Gross': 'sum',
    'Metascore': 'min'
})
```

### 5. Custom Apply Functions
```python
# Number of movies starting with 'A' for each group
def foo(group):
    return group['Series_Title'].str.startswith('A').sum()

print(genres.apply(foo))

# Ranking within each group
def rank_movie(group):
    group['genre_rank'] = group['IMDB_Rating'].rank(ascending=False)
    return group

print(genres.apply(rank_movie))

# Normalized rating within each group
def normal(group):
    group['norm_rating'] = (group['IMDB_Rating'] - group['IMDB_Rating'].min()) / (group['IMDB_Rating'].max() - group['IMDB_Rating'].min())
    return group

print(genres.apply(normal))
```

### 6. GroupBy on Multiple Columns
```python
# Group by director and actor
duo = movies.groupby(['Director', 'Star1'])
print(duo.size())

# Get specific group
print(duo.get_group(['Aamir Khan', 'Amole Gupte']))

# Most earning actor-director combo
top_combo = duo['Gross'].sum().sort_values(ascending=False).head(1)

# Best actor-genre combo by Metascore
best_combo = movies.groupby(['Star1', 'Genre'])['Metascore'].mean().reset_index().sort_values('Metascore', ascending=False).head(1)

# Multiple aggregations on multiple groupby
print(duo.agg(['min', 'max', 'mean']))
```

---

## Session 20: Merging

### 1. `pd.concat()`
```python
import pandas as pd

# Read data
courses = pd.read_csv('courses.csv')
students = pd.read_csv('students.csv')
nov = pd.read_csv('reg-month1.csv')
dec = pd.read_csv('reg-month2.csv')

# Concatenate vertically
regs = pd.concat([nov, dec], ignore_index=True)
print(regs)

# Append method (deprecated but still works)
regs_append = nov.append(dec, ignore_index=True)

# Concatenate with keys (MultiIndex)
multi = pd.concat([nov, dec], keys=['Nov', 'Dec'])
print(multi.loc['Dec'])

# Concatenate horizontally
horizontal = pd.concat([nov, dec], axis=1)
```

### 2. `merge()` - Inner Join
```python
# Inner join on student_id
inner_join = students.merge(regs, how='inner', on='student_id')
print(inner_join)
```

### 3. `merge()` - Left Join
```python
# Left join on course_id
left_join = courses.merge(regs, how='left', on='course_id')
print(left_join)
```

### 4. `merge()` - Right Join
```python
# Add new students
temp_df = pd.DataFrame({
    'student_id': [26, 27, 28],
    'name': ['Nitish', 'Ankit', 'Rahul'],
    'partner': [28, 26, 17]
})
students = pd.concat([students, temp_df], ignore_index=True)

# Right join
right_join = students.merge(regs, how='right', on='student_id')
print(right_join)
```

### 5. `merge()` - Outer Join
```python
outer_join = students.merge(regs, how='outer', on='student_id')
print(outer_join.tail(10))
```

### 6. Practical Problems

**1. Total Revenue Generated**
```python
total_revenue = regs.merge(courses, how='inner', on='course_id')['price'].sum()
print(total_revenue)
```

**2. Month by Month Revenue**
```python
temp_df = pd.concat([nov, dec], keys=['Nov', 'Dec']).reset_index()
monthly_revenue = temp_df.merge(courses, on='course_id').groupby('level_0')['price'].sum()
print(monthly_revenue)
```

**3. Registration Table**
```python
registration_table = regs.merge(students, on='student_id').merge(courses, on='course_id')[['name', 'course_name', 'price']]
print(registration_table)
```

**4. Revenue per Course (Bar Chart)**
```python
revenue_per_course = regs.merge(courses, on='course_id').groupby('course_name')['price'].sum()
revenue_per_course.plot(kind='bar')
```

**5. Students Enrolled in Both Months**
```python
common_student_id = np.intersect1d(nov['student_id'], dec['student_id'])
common_students = students[students['student_id'].isin(common_student_id)]
print(common_students)
```

**6. Courses with No Enrollment**
```python
course_id_list = np.setdiff1d(courses['course_id'], regs['course_id'])
no_enrollment = courses[courses['course_id'].isin(course_id_list)]
print(no_enrollment)
```

**7. Students Not Enrolled in Any Course**
```python
student_id_list = np.setdiff1d(students['student_id'], regs['student_id'])
not_enrolled = students[students['student_id'].isin(student_id_list)]
percentage = (not_enrolled.shape[0] / students.shape[0]) * 100
print(f"Percentage: {percentage}%")
```

**8. Student Name -> Partner Name (Self Join)**
```python
partner_names = students.merge(students, how='inner', left_on='partner', right_on='student_id')[['name_x', 'name_y']]
print(partner_names)
```

**9. Top 3 Students by Number of Enrollments**
```python
top_enrollments = regs.merge(students, on='student_id').groupby(['student_id', 'name'])['name'].count().sort_values(ascending=False).head(3)
print(top_enrollments)
```

**10. Top 3 Students by Total Spending**
```python
top_spenders = regs.merge(students, on='student_id').merge(courses, on='course_id').groupby(['student_id', 'name'])['price'].sum().sort_values(ascending=False).head(3)
print(top_spenders)
```

**11. Alternative Merge Syntax**
```python
# Using pd.merge directly
merged = pd.merge(students, regs, how='inner', on='student_id')
```

**12. IPL Problems (Examples)**
```python
# Find top 3 stadiums with highest sixes/match ratio
# Find orange cap holder of all seasons
# These would require additional data and calculations
```

---

## Summary of Key Concepts

### NumPy:
- **Arrays**: Efficient numerical computations with `ndarray`
- **Operations**: Vectorized operations, broadcasting
- **Functions**: Mathematical, statistical, linear algebra
- **Indexing**: Basic, fancy, boolean indexing
- **Manipulation**: Reshaping, stacking, splitting

### Pandas:
- **Series**: 1D labeled array
- **DataFrame**: 2D labeled data structure
- **Data Manipulation**: Indexing, slicing, filtering
- **Data Cleaning**: Handling missing values, duplicates
- **Grouping**: Split-apply-combine operations
- **Merging**: Combining datasets with different join types
- **Time Series**: Date/time functionality (not covered in these sessions)

### Best Practices:
1. Use vectorized operations instead of loops
2. Leverage boolean indexing for filtering
3. Use `inplace=True` to modify data structures directly
4. Chain operations for cleaner code
5. Always check data types and null values
6. Use appropriate join types when merging
7. Leverage groupby for aggregate analysis

This comprehensive documentation covers all major concepts from the provided sessions with practical examples for each command and function.
