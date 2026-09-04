import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from cimfe.utils.metrics import metric


def evaluate_model_unseen(X, y, model='rf', seed=2025):
    """Subject-grouped 5-fold CV (column 0 of X is the subject id), so no subject's
    data appears in both train and test within a fold."""
    Model = RandomForestRegressor if model == 'rf' else GradientBoostingRegressor
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, groups=X[:, 0]):
        X_train, X_test = X[train_idx, 1:], X[test_idx, 1:]
        y_train, y_test = y[train_idx], y[test_idx]

        fitted = Model(n_estimators=200, random_state=seed)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_test)
        mse, mae, rmse, mape, mspe, r2 = metric(pred, y_test)
        scores.append([rmse, r2])

    avg_score = np.mean(scores, axis=0)
    scores.append(avg_score)
    print('RMSE: {:.4f}, R2: {:.4f}'.format(avg_score[0], avg_score[1]))
    return scores


def select_features_unseen(df, model='rf', seed=2025, step=2):
    """Greedily grow a feature set by Gini importance, picking the size that
    maximizes subject-grouped CV R2."""
    Model = RandomForestRegressor if model == 'rf' else GradientBoostingRegressor
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    fitted = Model(n_estimators=200, random_state=2025)
    fitted.fit(X, y)
    importance = fitted.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': df.columns[1:-1], 'Gini Importance': importance}).sort_values('Gini Importance', ascending=False)

    r2_scores = []
    rmse_scores = []
    for n in np.arange(2, len(feature_imp_df), step):
        rf_selected_features = feature_imp_df.loc[feature_imp_df['Gini Importance'].nlargest(n).index, 'Feature'].values
        accs = []
        for i, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(X, groups=df['subj'].values)):
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
            fold_model = Model(random_state=2025, n_estimators=200)
            fold_model.fit(df_train.loc[:, rf_selected_features], df_train['sCr change'])
            pred = fold_model.predict(df_test.loc[:, rf_selected_features])
            mse, mae, rmse, mape, mspe, r2 = metric(pred, df_test['sCr change'])
            accs.append([mse, mae, rmse, mape, mspe, r2])
        accs = np.array(accs).mean(axis=0)
        print(f'{n} selected feature - MSE: {accs[0]:.4f} MAE: {accs[1]:.4f} RMSE: {accs[2]:.4f} R2: {accs[5]:.4f}')
        r2_scores.append(accs[5])
        rmse_scores.append(accs[2])

    print('--' * 50)
    n = np.arange(2, len(feature_imp_df), step)[np.argmax(r2_scores)]
    selected_features = list(feature_imp_df.loc[feature_imp_df['Gini Importance'].nlargest(n).index, 'Feature'].values)
    print(selected_features)
    return selected_features
