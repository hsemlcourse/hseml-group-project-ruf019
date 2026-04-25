[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание отмены бронирования отеля

**Студент:** Фёдоров Руслан Олегович

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** бинарная классификация. По характеристикам бронирования нужно предсказать, будет ли бронь отменена.

**Датасет:** Hotel Booking Demand — https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

**Целевая переменная:** `is_canceled`, где `1` означает отмену бронирования, а `0` — бронирование без отмены.

**Целевая метрика:** F1-score. Дополнительно считаются Accuracy, Precision, Recall и ROC-AUC.

F1-score выбран как основная метрика, потому что в задаче важны оба типа ошибок: если модель не заметит вероятную отмену, отель может потерять деньги из-за простоя номера. Если модель ошибочно посчитает обычную бронь рискованной, то это тоже может привести к неверным решениям.


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные, метрики CP1
│   └── raw                     # Исходный файл hotel_bookings.csv
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и сравнение моделей
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Отчёт по проекту
├── src
│   ├── preprocessing.py        # Предобработка данных и sklearn-препроцессинг
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-ruf019.git
cd hseml-group-project-ruf019

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить baseline-модель
Откройте notebooks/02_baseline.ipynb и выполните ячейки сверху вниз

# 5. Запустить ноутбук с экспериментами
Откройте notebooks/03_experiments.ipynb и выполните ячейки сверху вниз
```

Тесты запускаются следующим образом:
```bash
python -m pytest tests/test.py -p no:cacheprovider
```

Линтер:
```bash
python -m ruff check src/ tests/ --line-length 120
```

## Данные
- `data/raw/` — исходный файл `hotel_bookings.csv`.
- `data/processed/` — предобработанные данные и таблица метрик.

Исходный датасет содержит 119390 строк и 32 колонки. В таргете `is_canceled` 75166 бронирований без отмены и 44224 отменённых бронирования.

На этапе CP1 выполнена базовая подготовка данных:
- заполнены пропуски в `children`, `country`, `agent`, `company`;
- удалены дубликаты;
- удалены строки с нулевым количеством гостей;
- удалены строки с нулевой длительностью проживания;
- удалены строки с отрицательным `adr`;
- исключены признаки `reservation_status` и `reservation_status_date`, так как они приводят к утечке целевой переменной;
- добавлены новые признаки `arrival_month_num`, `total_guests`, `total_nights`, `has_children`, `has_agent`, `has_company`, `adr_per_person`, `room_type_changed`;
- сделан стратифицированный split на train/validation/test с фиксированным `random_state=42`.


## Результаты
Первые результаты на validation-выборке для моделей с feature engineering:
| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Примечание |
|--------|----------|-----------|--------|----|---------|------------|
| Baseline: Logistic Regression | 0.7969 | 0.6841 | 0.4838 | 0.5668 | 0.8521 | Baseline-модель для сравнения с более сложными подходами |
| KNeighborsClassifier | 0.7990 | 0.6696 | 0.5296 | 0.5915 | 0.8392 | Простая нелинейная модель для первого сравнения |
| Decision Tree | 0.8154 | 0.6924 | 0.5901 | 0.6371 | 0.8710 | Дерево решений с ограничением глубины |
| Random Forest | 0.8161 | 0.8213 | 0.4223 | 0.5578 | 0.8894 | Ансамбль деревьев |
| Лучшая модель: Gradient Boosting | 0.8354 | 0.7576 | 0.5892 | 0.6629 | 0.8949 | Лучшая модель CP1 по F1-score |

В `02_baseline.ipynb` baseline-модель `LogisticRegression` обучается на очищенных данных без feature engineering. В `03_experiments.ipynb` один и тот же набор моделей сначала обучается на данных без feature engineering, а затем повторно на данных с новыми признаками. Видно, что feature engineering сильнее всего помог `GradientBoosting`: его F1-score вырос с `0.6464` до `0.6629`. Полная таблица метрик сохранена в `data/processed/metrics_cp1.csv`. Лучшая модель сохранена в `models/best_model_cp1.joblib`.


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
