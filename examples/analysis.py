from progpy.analysis import show_heatmap
from progpy.datasets import nasa_cmapss

(training, testing, eol) = nasa_cmapss.load_data(1)

show_heatmap(training)

# Notice that some values have no color- this is because they are constant. Let's drop these
for feature in ['setting3', 'sensor1', 'sensor5', 'sensor10', 'sensor16', 'sensor18', 'sensor19']:
    training.drop(feature, axis=1)
show_heatmap(training)

# Here you can see high correlations between sensor 14 and 9
