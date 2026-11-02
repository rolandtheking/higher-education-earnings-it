import os
import pandas as pd

# Словарь волн, префиксов и годов проведения опроса
WAVES_INFO = {
        'r34': {'prefix': 'dd', 'year': 2025},
        'r33': {'prefix': 'cc', 'year': 2024},
        'r32': {'prefix': 'bb', 'year': 2023},
        'r31': {'prefix': 'aa', 'year': 2022},
        'r30': {'prefix': 'z', 'year': 2021},
        'r29': {'prefix': 'y', 'year': 2020},
        'r28': {'prefix': 'x', 'year': 2019},
        'r27': {'prefix': 'w', 'year': 2018},
        'r26': {'prefix': 'v', 'year': 2017},
        'r25': {'prefix': 'u', 'year': 2016},
}

# Шаблоны поиска переменных во всех волнах РМЭЗ
VAR_PATTERNS = {
        'idind': ['idind'],
        'age': ['_age', 'age'],
        'diplom': ['_diplom', 'diplom'],
        'educ': ['_educ', 'educ'],
        'marst': ['_marst', 'marst'],
        'gender': ['h5', '_h5'],
        'wage': ['j10', '_j10'],  # Доход за 30 дней
        'wage_avg': ['j13.2', 'j13_2', '_j13_2'],  # Среднемесячный за 12 мес.
        'isco_main': ['j2cod08', '_j2cod08'],  # ISCO-08 основная работа
        'isco_add': ['j33cod08', '_j33cod08'],  # ISCO-08 доп. работа
        'industry': ['j4.1', 'j4_1', '_j4_1'],  # Отрасль занятости
        'hours_week': ['j6.2', 'j6_2', '_j6_2'],  # Часов в неделю
        'job_start_yr': ['j5a', '_j5a'],  # Год начала работы
        'qual_mismatch': ['j1.1.8', 'j1_1_8', '_j1_1_8'],  # Несоответствие квалификации
        'job_changed': ['j29.1', 'j29_1', '_j29_1'],  # Смена профессии/работы
        'job_change_cnt': ['j536', '_j536'],  # Частота смен работы
        'status': ['status', '_status'],  # Тип поселения
        'psu': ['psu', '_psu'],  # Код ПЕО / региона
}


def _find_column(df_columns, prefix, patterns):
    """Устойчивый поиск названия колонки с учетом вариаций написания."""
    cols_set = set(df_columns)
    for pat in patterns:
        candidates = [f'{prefix}{pat}', f'{prefix}_{pat}', pat]
        for cand in candidates:
            if cand in cols_set:
                return cand
            cand_alt = cand.replace('.', '_')
            if cand_alt in cols_set:
                return cand_alt
    return None


def load_single_wave(file_path, prefix, year):
    """Загружает одну волну РМЭЗ и применит универсальное переименование."""
    temp_df = pd.read_stata(file_path, convert_categoricals=False)
    cols_in_file = set(temp_df.columns)

    rename_map = {}
    for clean_name, patterns in VAR_PATTERNS.items():
        found = _find_column(cols_in_file, prefix, patterns)
        if found:
            rename_map[found] = clean_name

    extracted = temp_df[list(rename_map.keys())].rename(columns=rename_map)
    extracted['year'] = year
    return extracted


def load_all_waves(data_dir='data', waves_info=None):
    """Объединяет все имеющиеся файлы Stata в единый пул наблюдений."""
    if waves_info is None:
        waves_info = WAVES_INFO

    dfs = []
    for wave, meta in waves_info.items():
        possible_paths = [
                os.path.join(data_dir, f'{wave}iall.dta'),
                os.path.join(data_dir, f'{wave}iall.DTA'),
                os.path.join(data_dir, 'STATA', f'{wave}iall.dta'),
        ]
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if file_path:
            try:
                wave_df = load_single_wave(file_path, meta['prefix'], meta['year'])
                dfs.append(wave_df)
                print(
                        f"[DataLoader] Волна {wave} ({meta['year']} г.): загружено"
                        f' {len(wave_df)} строк.'
                )
            except Exception as e:
                print(f'[DataLoader] Ошибка при загрузке {wave}: {e}')

    if not dfs:
        raise FileNotFoundError('Файлы волн не найдены в директории ' + data_dir)

    df_pooled = pd.concat(dfs, ignore_index=True)
    print(
            f'[DataLoader] Успешно объединено {len(df_pooled)} наблюдений за'
            f' {len(dfs)} волн.'
    )
    return df_pooled