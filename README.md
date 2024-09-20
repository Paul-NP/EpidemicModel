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

## Получение системы LaTex уравнений 

Вы можете получить LaTex уравнения модели.
Для этого при добавлении факторов необходимо указать свойство `latex_repr`.
Вы можете использовать параметр `simplified` для упрощения уравнений (представления в классическом варианте).

```python
from epidemmo import ModelBuilder

builder = ModelBuilder()
builder.add_stage('S', 100).add_stage('I', 1).add_stage('R')
builder.add_factor('beta', 0.4, latex_repr=r'\beta').add_factor('gamma', 0.1, latex_repr=r'\gamma')
builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

model = builder.build()

latex_str = model.get_latex()
print(latex_str)

latex_str_simplified = model.get_latex(simplified=True)
print(latex_str_simplified)
```
Будут сформированы следующие системы дифференциальных уравнений:

```latex
\begin{equation}\label{eq:SIR_full}
    \begin{cases}
        \frac{dS}{dt} =  - S \cdot (1 - (1 - \frac{\beta}{N})^{I})\\
        \frac{dI}{dt} = S \cdot (1 - (1 - \frac{\beta}{N})^{I}) - I \cdot \gamma\\
        \frac{dR}{dt} = I \cdot \gamma\\
    \end{cases}
\end{equation}

\begin{equation}\label{eq:SIR_classic}
    \begin{cases}
        \frac{dS}{dt} =  - \frac{S \cdot \beta \cdot I}{N}\\
        \frac{dI}{dt} = \frac{S \cdot \beta \cdot I}{N} - I \cdot \gamma\\
        \frac{dR}{dt} = I \cdot \gamma\\
    \end{cases}
\end{equation}
``` 
Для компиляции необходимо подключить пакет `amsmath` 
Результат компиляции LaTex выглядит следующим образом:
![equation example](https://raw.githubusercontent.com/Paul-NP/EpidemicModel/master/documentation/images/equations_example.png)