# -*- coding=utf-8 -*-

def gen_model():
    import xgboost as xgb
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics

    TRAIN_FEATURE = ''
    TRAIN_LABEL = ''
    TEST_FEATURE = ''
    TEST_LABEL = ''
    MODEL = ''

    X_train = np.loadtxt(fname=TRAIN_FEATURE, delimiter=',')
    y_train = np.loadtxt(fname=TRAIN_LABEL, delimiter=',')
    X_test = np.loadtxt(fname=TEST_FEATURE, delimiter=',')
    y_test = np.loadtxt(fname=TEST_LABEL, delimiter=',')

    # for GCC-4.4.0.model
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'error',
              'n_estimators': 134,
              'max_depth': 10,
              'gamma': 0,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'min_child_weight': 9,
              'eta': 0.01,
              'nthread': 2,
              'silent': 0,
              'scale_pos_weight': 1.6,
              'seed': 14}

    xgb_model = xgb.XGBClassifier(**params).fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)
    actuals = y_test

    acc = metrics.accuracy_score(actuals, predictions)
    recall = metrics.recall_score(actuals, predictions)
    precision = metrics.precision_score(actuals, predictions)
    f1 = metrics.f1_score(actuals, predictions)
    auc = metrics.roc_auc_score(actuals, predictions)

    print(confusion_matrix(actuals, predictions))

    print('[' +'acc=' + str(acc) + ', ' +
          'recall=' + str(recall) + ', ' +
          'precession=' + str(precision) + ', ' +
          'f1=' + str(f1) + ', ' +
          'auc=' + str(auc) + '' +']')

    import tools.file_tools as ftools
    ftools.delete_if_exists(MODEL)
    xgb_model.save_model(MODEL)


if __name__ == '__main__':
    gen_model()
