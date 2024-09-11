# Package emodel

A package for epidemiological modeling based on a compartmental approach.

## Import package

```python
import epidemmo
```

## Сreating a simple SIR model

```python
from epidemmo import ModelBuilder
from matplotlib import pyplot as plt

builder = ModelBuilder()
builder.add_stage('S', 100).add_stage('I', 1).add_stage('R')
builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

model = builder.build()
result_df = model.start(70)

result_df.plot(title='SIR', ylabel='population', xlabel='time')
plt.show()
```

`start(70)` - the start method, which takes the simulation duration, returns a pandas.DataFrame with the simulation results.

### Simulation result

![sir example](https://raw.githubusercontent.com/Paul-NP/EpidemicModel/master/documentation/images/sir_example.png)

## Use standard models

The package contains several standard epidemiological models.

```python
from epidemmo import Standard

model = Standard.SIR_builder().build()
result = model.start(40)
```

You can change start num for every stage.
Also you can change Factor's values.

```python
from epidemmo import Standard

model = Standard.SIR_builder().build()
model.set_start_stages(S=1000, I=10, R=0)
model.set_factors(beta=0.5)
```

## Print and write results table

After using the model you can print result table in console.

```python
from epidemmo import Standard

model = Standard.SIR_builder().build()
model.start(60)
model.print_result_table()
```
or writing result in csv files

```python
from epidemmo import Standard

model = Standard.SIR_builder().build()
model.start(60)
model.write_results()
```
