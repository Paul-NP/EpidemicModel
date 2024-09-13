# Пакет epidemmo

Пакет для создания эпидемиологических моделей.
Структура создаваемых моделей основана на идеях компартментального моделирования.


## Импорт пакета

```python
import epidemmo
```

## Создание простой SIR модели

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

`start(70)` - метод, который принимает длительность моделирования, а возвращает pd.DataFrame с результатами моделирования.


### Результаты моделирования

![sir example](https://raw.githubusercontent.com/Paul-NP/EpidemicModel/master/documentation/images/sir_example.png)

## Использование стандартных моделей

Пакет содержит несколько стандартных эпидемиологических моделей.

```python
from epidemmo import Standard

model = Standard.get_SIR_builder().build()
result = model.start(40)
```
Вы можете изменить стартовую численность каждой стадии, а также изменить значение параметров модели.

```python
from epidemmo import Standard

model = Standard.get_SIR_builder().build()
model.set_start_stages(S=1000, I=10, R=0)
model.set_factors(beta=0.5)
```

## Вывод и запись табличных результатов

После запуска модели Вы можете вывести результаты в виде таблицы (PrettyTable) в консоль.

```python
from epidemmo import Standard

model = Standard.get_SIR_builder().build()
model.start(60)
model.print_result_table()
```
или записать результаты в csv файлы, включая
1. файл с изменением численности каждой стадии
2. файл с изменением значений всех параметров во времени
3. файл с изменением интенсивности потоков модели во времени

```python
from epidemmo import Standard

model = Standard.get_SIR_builder().build()
model.start(60)
model.write_results()
```