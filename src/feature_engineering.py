import numpy as np
import pandas as pd

DEFAULT_CPI_FACTORS = {
        2016: 1.55,
        2017: 1.51,
        2018: 1.45,
        2019: 1.40,
        2020: 1.34,
        2021: 1.25,
        2022: 1.12,
        2023: 1.06,
        2024: 1.03,
        2025: 1.00,
}

ISCO_IT_CODES = [
        2511,
        2512,
        2513,
        2514,
        2519,  # Разработчики ПО и аналитики
        2521,
        2522,
        2523,
        2529,  # Специалисты по БД и сетям
        3511,
        3512,
        3513,
        3514,  # IT-техники и поддержка
        1330,  # Руководители в сфере ИКТ
]


def prepare_econometric_dataset(df_pooled, cpi_map=None):
    """Полная подготовка массива к эконометрическому оцениванию."""
    if cpi_map is None:
        cpi_map = DEFAULT_CPI_FACTORS

    df = df_pooled.copy()

    # 1. Удаление технических кодов ответов
    missing_codes = [99999996, 99999997, 99999998, 99999999]
    df.replace(missing_codes, np.nan, inplace=True)

    # 2. Фильтрация трудоспособных граждан (20-60 лет) с зарплатой и отработанными часами
    df_clean = df[
            (df['age'] >= 20)
            & (df['age'] <= 60)
            & (df['wage'] > 0)
            & (df['hours_week'] > 0)
    ].copy()

    # 3. Дефлирование и логарифмирование реальной почасовой оплаты
    df_clean['cpi'] = df_clean['year'].map(cpi_map)
    df_clean['real_wage'] = df_clean['wage'] * df_clean['cpi']
    df_clean['hourly_wage'] = df_clean['real_wage'] / (
            df_clean['hours_week'] * 4.33
    )
    df_clean['ln_earn'] = np.log(df_clean['hourly_wage'])

    # Триммирование 1% крайних выбросов
    p1, p99 = df_clean['ln_earn'].quantile([0.01, 0.99])
    df_clean = df_clean[
            (df_clean['ln_earn'] >= p1) & (df_clean['ln_earn'] <= p99)
    ]

    # 4. Переменные человеческого капитала
    df_clean['higher_educ'] = (df_clean['diplom'] == 6).astype(int)

    # Расчет стажа работы на текущем месте (Tenure)
    df_clean['tenure'] = df_clean['year'] - df_clean['job_start_yr']
    df_clean.loc[(df_clean['tenure'] < 0) | (df_clean['tenure'] > 50), 'tenure'] = (
            np.nan
    )

    # Флаг IT-профессионалов по ISCO-08
    df_clean['is_it_prof'] = df_clean['isco_main'].isin(ISCO_IT_CODES).astype(int)

    # 5. Возрастные когорты
    age_bins = [19, 24, 29, 34, 39, 44, 49, 60]
    age_labels = [
            'age_20_24',
            'age_25_29',
            'age_30_34',
            'age_35_39',
            'age_40_44',
            'age_45_49',
            'age_50_60',
    ]
    df_clean['age_group'] = pd.cut(
            df_clean['age'], bins=age_bins, labels=age_labels
    )

    # Бинарные дамми для когорт (референтная группа: 20-24 года)
    age_dummies = pd.get_dummies(
            df_clean['age_group'], drop_first=True, dtype=int
    )
    cohort_cols = list(age_dummies.columns)
    df_clean = pd.concat([df_clean, age_dummies], axis=1)

    # 6. Построение Z-индекса SkillChange по профессиям ISCO-08
    df_clean['skill_mismatch_flag'] = df_clean['qual_mismatch'].isin(
            [4, 5]
    ).astype(int)

    skill_change_by_isco = (
            df_clean.groupby('isco_main')['skill_mismatch_flag']
            .mean()
            .reset_index()
            .rename(columns={'skill_mismatch_flag': 'SkillChange'})
    )
    df_clean = df_clean.merge(skill_change_by_isco, on='isco_main', how='left')

    df_clean['SkillChange_norm'] = (
            df_clean['SkillChange'] - df_clean['SkillChange'].mean()
    ) / df_clean['SkillChange'].std()

    # 7. Перекрестные члены (Interaction terms)
    interaction_cols = []
    for col in cohort_cols:
        inter_name = f'{col}_x_SkillChange'
        df_clean[inter_name] = df_clean[col] * df_clean['SkillChange_norm']
        interaction_cols.append(inter_name)

    # Итоговая очистка от NaN перед запуском регрессии
    required_vars = (
            [
                    'ln_earn',
                    'SkillChange_norm',
                    'higher_educ',
                    'tenure',
                    'isco_main',
                    'gender',
                    'status',
                    'year',
                    'idind',
            ]
            + cohort_cols
            + interaction_cols
    )

    df_model = df_clean.dropna(subset=required_vars).copy()
    print(
            f'[FeatureEng] Выборка готова к оцениванию: {len(df_model)} наблюдений'
            f" ({df_model['idind'].nunique()} уникальных индивидов)."
    )

    return df_model, cohort_cols, interaction_cols