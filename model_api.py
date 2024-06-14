from model import EpidemicModel


# создание экземпляра модели из содержимого json файла
# m = EpidemicModel.from_json(json_content, struct_version='kk_2024')
# struct_version = 'kk_2024' протокол описания модели (Киреев-Кулдарев-2024)
# метод start принимающий длительность моделирования возвращает pandas.DataFrame с результатами моделирования

# пример функции моделирования из файла с json структурой и сохранение результатов в csv таблицу
def modelling_from_json(filename_json: str, filename_csv: str, time: int):
    with open(filename_json, encoding='utf8') as file:
        json_content = file.read()
        e_model = EpidemicModel.from_json(json_content, struct_version='kk_2024')
        result = e_model.start(time)
        result.to_csv(filename_csv, sep=',')


modelling_from_json('temp/test_model.json', 'temp/test_result.csv', 200)