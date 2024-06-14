Пакет для эпидемиологического моделирования на основе компартментального моделирования.
>>> from emodel import EpidemicModel

Создание экземпляра модели из содержимого json файла
>>> m = EpidemicModel.from_json(json_content, struct_version='kk_2024')
('kk_2024' протокол описания модели (Киреев-Кулдарев-2024)

Получение результатов. Метод start принимающий длительность моделирования возвращает pandas.DataFrame с результатами моделирования
>>> result = m.start(100)

Пример функции моделирования из файла с json структурой и сохранение результатов в csv таблицу
>>> def modelling_from_json(filename_json: str, filename_csv: str, time: int):
        with open(filename_json, encoding='utf8') as file:
            json_content = file.read()
            e_model = EpidemicModel.from_json(json_content, struct_version='kk_2024')
            result = e_model.start(time)
            result.to_csv(filename_csv, sep=',')