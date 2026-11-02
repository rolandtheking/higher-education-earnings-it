from linearmodels.panel import PanelOLS
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_pooled_ols(
        df_model, cohort_cols, interaction_cols, cluster_col='isco_main'
):
    """Оценивает Pooled OLS с кластеризацией стандартных ошибок по ISCO-08."""
    formula = (
            'ln_earn ~ '
            + ' + '.join(cohort_cols)
            + ' + '
            + ' + '.join(interaction_cols)
            + ' + SkillChange_norm + higher_educ + tenure + C(gender) + C(status) +'
            ' C(year)'
    )

    model = smf.ols(formula, data=df_model).fit(
            cov_type='cluster', cov_kwds={'groups': df_model[cluster_col]}
    )
    return model


def run_panel_fe(
        df_model,
        cohort_cols,
        interaction_cols,
        entity_col='idind',
        time_col='year',
):
    """Оценивает панельную модель с индивидуальными и временными фиксированными эффектами."""
    panel_df = df_model.set_index([entity_col, time_col])

    exog_vars = cohort_cols + interaction_cols + ['tenure']
    exog = sm.add_constant(panel_df[exog_vars])

    model = PanelOLS(
            panel_df['ln_earn'], exog, entity_effects=True, time_effects=True
    ).fit(cov_type='clustered', cluster_entity=True)

    return model


def get_model_summary_table(ols_model, fe_model, cohort_cols, interaction_cols):
    """Сводит параметры Pooled OLS и Panel FE в единый сопоставимый DataFrame."""
    records = []

    ols_params, ols_pvalues, ols_bse = (
            ols_model.params,
            ols_model.pvalues,
            ols_model.bse,
    )
    fe_params, fe_pvalues, fe_bse = (
            fe_model.params,
            fe_model.pvalues,
            fe_model.std_errors,
    )

    all_vars = cohort_cols + interaction_cols + ['tenure']
    if 'higher_educ' in ols_params:
        all_vars.append('higher_educ')

    for v in all_vars:
        records.append({
                'Variable': v,
                'OLS_Coef': ols_params.get(v, None),
                'OLS_SE': ols_bse.get(v, None),
                'OLS_PValue': ols_pvalues.get(v, None),
                'FE_Coef': fe_params.get(v, None),
                'FE_SE': fe_bse.get(v, None),
                'FE_PValue': fe_pvalues.get(v, None),
        })

    return pd.DataFrame(records)