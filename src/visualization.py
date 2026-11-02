import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use(
        'seaborn-v0_8-whitegrid'
        if 'seaborn-v0_8-whitegrid' in plt.style.available
        else 'default'
)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


def plot_skill_change_coefficients(summary_table, save_path=None):
    """Строит график оценки коэффициентов взаимодействия gamma_a

    (Interaction Term Error-Bar Plot) с 95% доверительными интервалами для Pooled
    OLS и Panel FE.
    """
    df_inter = summary_table[
            summary_table['Variable'].str.contains('_x_SkillChange')
    ].copy()

    label_map = {
            'age_25_29_x_SkillChange': '25–29 лет',
            'age_30_34_x_SkillChange': '30–34 года',
            'age_35_39_x_SkillChange': '35–39 лет',
            'age_40_44_x_SkillChange': '40–44 года',
            'age_45_49_x_SkillChange': '45–49 лет',
            'age_50_60_x_SkillChange': '50–60 лет',
    }
    df_inter['Age_Cohort'] = df_inter['Variable'].map(label_map)

    # Расчет 95% доверительных интервалов (Z = 1.96)
    df_inter['OLS_CI'] = 1.96 * df_inter['OLS_SE']
    df_inter['FE_CI'] = 1.96 * df_inter['FE_SE']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    y_positions = np.arange(len(df_inter))
    offset = 0.15

    # Линия нулевого эффекта
    ax.axvline(
            0, color='#999999', linestyle='--', linewidth=1.2, label='Нулевой эффект'
    )

    # Оценка Pooled OLS (Темно-синий)
    ax.errorbar(
            df_inter['OLS_Coef'] * 100,
            y_positions + offset,
            xerr=df_inter['OLS_CI'] * 100,
            fmt='o',
            color='#1B365D',
            ecolor='#1B365D',
            capsize=4,
            capthick=1.2,
            markersize=7,
            label='Сквозная OLS (Pooled OLS, cluster CRV1)',
    )

    # Оценка Panel FE (Темно-красный)
    ax.errorbar(
            df_inter['FE_Coef'] * 100,
            y_positions - offset,
            xerr=df_inter['FE_CI'] * 100,
            fmt='s',
            color='#8C1D40',
            ecolor='#8C1D40',
            capsize=4,
            capthick=1.2,
            markersize=7,
            label='Панельная FE (Entity & Time Fixed Effects)',
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_inter['Age_Cohort'], fontsize=11, fontweight='medium')
    ax.invert_yaxis()

    ax.set_xlabel(
            'Изменение реальной почасовой заработной платы (%) на 1 std. dev. индекса'
            ' SkillChange',
            fontsize=11,
            fontweight='bold',
            labelpad=10,
    )
    ax.set_title(
            'Оценка штрафа за скорость деактуализации навыков (γ_a) по возрастным'
            ' когортам',
            fontsize=12,
            fontweight='bold',
            pad=15,
    )

    ax.legend(
            loc='lower left', frameon=True, facecolor='#ffffff', framealpha=0.95
    )

    # Аннотация экономического пика
    ax.annotate(
            'Максимальный чистый штраф FE:\n-3.7% (p < 0.001) в возрасте 30–34'
            ' года',
            xy=(-3.69, 1 - offset),
            xytext=(-7.5, 0.3),
            arrowprops=dict(
                    arrowstyle='->', color='#8C1D40', lw=1.2, connectionstyle='arc3,rad=.2'
            ),
            fontsize=9.5,
            color='#8C1D40',
            fontweight='bold',
            bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='#FFF5F5',
                    edgecolor='#8C1D40',
                    lw=0.8,
            ),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_age_wage_trajectories(df_model, save_path=None):
    """Строит профили «возраст-заработок» для ИКТ/STEM и прочих секторов

    с доверительными интервалами (95% Confidence Ribbons).
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    stats = (
            df_model.groupby(['age_group', 'is_it_prof'], observed=True)['hourly_wage']
            .agg(
                    mean='mean',
                    sem='sem',
                    count='count',
            )
            .reset_index()
    )

    stats['ci95'] = 1.96 * stats['sem']

    it_data = stats[stats['is_it_prof'] == 1].reset_index(drop=True)
    non_it_data = stats[stats['is_it_prof'] == 0].reset_index(drop=True)

    x_axis = np.arange(len(it_data))
    categories = [
            '20–24',
            '25–29',
            '30–34',
            '35–39',
            '40–44',
            '45–49',
            '50–60',
    ]

    ax.plot(
            x_axis,
            it_data['mean'],
            marker='o',
            linewidth=2.2,
            color='#006699',
            label='Специалисты ИКТ / STEM (ISCO-08: 251, 252, 351, 1330)',
    )
    ax.fill_between(
            x_axis,
            it_data['mean'] - it_data['ci95'],
            it_data['mean'] + it_data['ci95'],
            color='#006699',
            alpha=0.15,
    )

    ax.plot(
            x_axis,
            non_it_data['mean'],
            marker='s',
            linewidth=2.2,
            color='#4A5568',
            linestyle='--',
            label='Занятые в прочих отраслях экономики',
    )
    ax.fill_between(
            x_axis,
            non_it_data['mean'] - non_it_data['ci95'],
            non_it_data['mean'] + non_it_data['ci95'],
            color='#4A5568',
            alpha=0.12,
    )

    ax.set_xticks(x_axis)
    ax.set_xticklabels(categories, fontsize=10.5)

    ax.set_xlabel('Возрастная группа (лет)', fontsize=11, fontweight='bold')
    ax.set_ylabel(
            'Реальная почасовая заработная плата (руб., цены 2025 г.)',
            fontsize=11,
            fontweight='bold',
    )
    ax.set_title(
            'Эмпирические траектории накопления человеческого капитала: ИКТ vs Прочие'
            ' секторы',
            fontsize=12,
            fontweight='bold',
            pad=15,
    )

    ax.legend(
            loc='upper left', frameon=True, facecolor='#ffffff', framealpha=0.95
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_skill_change_distribution(df_model, save_path=None):
    """Строит распределение индекса SkillChange по укрупненным группам профессий ISCO-08."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    df_top_isco = df_model.copy()
    df_top_isco['isco_group'] = (
            df_top_isco['isco_main'].astype(str).str[0].map({
                    '1': 'Руководители (ISCO 1)',
                    '2': 'Специалисты высшего уровня (ISCO 2)',
                    '3': 'Специалисты среднего уровня (ISCO 3)',
                    '4': 'Офисные служащие (ISCO 4)',
                    '5': 'Работники сферы услуг (ISCO 5)',
                    '7': 'Квалифицированные рабочие (ISCO 7)',
                    '8': 'Операторы установок (ISCO 8)',
            })
    )

    df_plot = df_top_isco.dropna(subset=['isco_group'])

    sns.boxplot(
            data=df_plot,
            x='SkillChange_norm',
            y='isco_group',
            palette='Blues_r',
            fliersize=1,
            linewidth=0.9,
            ax=ax,
    )

    ax.axvline(
            0, color='#999999', linestyle='--', linewidth=1, label='Среднее по экономике'
    )
    ax.set_xlabel(
            'Нормированный индекс скорости деактуализации навыков'
            ' (SkillChange_norm, Z-score)',
            fontsize=10.5,
            fontweight='bold',
    )
    ax.set_ylabel('')
    ax.set_title(
            'Вариация показателя нестабильности навыков по укрупненным группам'
            ' ISCO-08',
            fontsize=12,
            fontweight='bold',
            pad=12,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax