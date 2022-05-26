import torch
from torch import nn
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import scipy.stats as sps
from Diploma.model import *
from Diploma.training import *


class MlpDiffsRTSPredictor:
  def __init__(self, device='cuda', path='Diploma/'):
    self.path = path
    self.data = self.load_data(self.path + 'data.csv')
    self.data_with_features = self.add_features_to_data(self.data)

    ts = self.data['TRADEDATE'].max()
    self.interval_max = ts.year * 365 + ts.day_of_year
    self.max_trained = self.interval_max

    self.model = RTSNet(in_dim=143, out_dim=1, n_layers=5, p=0.28).to(device)
    if (device == 'cuda'):
      self.model.load_state_dict(torch.load(self.path + 'mlp_diffs_cuda'))
    else:
      self.model.load_state_dict(torch.load(self.path + 'mlp_diffs_cpu'))


  def load_data(self, path):
    """
    Выгружает данные из path
    """
    data = pd.read_csv(path, sep=',')
    data = data[['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VALUE']]
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'])
    data.sort_values('TRADEDATE', inplace=True)
    data = data[data['TRADEDATE'] >= '2015-01-01'].reset_index(drop=True)

    return data


  def create_rolling_features(
      self, data, shifts=5, features=None
  ):
      """
          Создает обучающий датасет из признаков, полученных из дат и значений ряда ранее.
          При этом используются занчения ряда со сдвигами на неделю и год назад.
      """
      if features is None:
        features = ['CLOSE']

      df = data.copy()
      for feature in features:
        df[f'{feature}_rolling_{shifts}'] = (df['CLOSE'] - df['CLOSE'].shift(1)).rolling(shifts, min_periods=1).mean().shift(1, axis = 0)
      
      # удалим первые строчки с nan
      drop_indices = df.index[df.isna().sum(axis=1) > 0]
      df = df.dropna()#(index=drop_indices)

      df = df.copy()
      return df

    
  def create_date_features(self, date):
      """Создает фичи из даты"""
      
      row = {}
      row['dayofweek'] = date.dayofweek
      row['quarter'] = date.quarter
      row['month'] = date.month
      row['year'] = date.year
      row['dayofyear'] = date.dayofyear
      row['dayofmonth'] = date.day
      row['weekofyear'] = date.weekofyear
      return row

  def create_shifted_features(
      self, data, shifts=5, features=None
  ):
      """
          Создает обучающий датасет из признаков, полученных из дат и значений ряда ранее.
          При этом используются занчения ряда со сдвигами на неделю и год назад.
      """
      if features is None:
        features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VALUE']

      df = data.copy()
      for feature in features:
        for shift in range(1, shifts + 1):
          df[f'{feature}_{shift}'] =  df[f'{feature}'].shift(shift, axis = 0)
      
      # удалим первые строчки с nan
      drop_indices = df.index[df.isna().sum(axis=1) > 0]
      df = df.dropna()#(index=drop_indices)

      df = df.copy()
      return df


  def add_features_to_data(self, data):
    """
    Добавляет в датасет фичи, которые нужно для предсказания
    """
    shifts = 26
    shifts_rolling = 5

    data_with_features = self.create_shifted_features(
        data, shifts=shifts, features=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VALUE']
      )
    
    q = pd.DataFrame([self.create_date_features(d) for d in data_with_features['TRADEDATE']])
    data_with_features = pd.merge(data_with_features, q, left_index=True, right_index=True)
    data_with_features = self.create_rolling_features(data_with_features, shifts=shifts_rolling)

    # data_all_with_features['y'] = data_all_with_features['CLOSE'].shift(-1) - data_all_with_features['CLOSE']
    # data_all_with_features.sort_values('TRADEDATE').dropna(inplace=True)

    self.features = data_with_features.columns.values
    self.features = self.features[(self.features != 'TRADEDATE') & (self.features != 'y')]

    return data_with_features


  def predict(self):
    """Предсказывает, увеличится или уменьшится завтра"""

    data_test = self.data_with_features.iloc[-1]
    X_test = data_test[self.features].values    
    self.model.eval()
    x_test = torch.Tensor(X_test.astype(np.float64)).unsqueeze(0).to(device)
    # print(x_test.size())
    result = self.model(x_test).detach().cpu().item()

    return 'Вырастет' if result >= 0 else 'Упадет'


  def add_data(self, extra_data):
    """
    Добавляет новые данные. Формат -- pd.DataFrame
    с колонками ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VALUE']
    """
    self.data = self.data.append(extra_data).reset_index(drop=True)
    self.data_with_features = self.add_features_to_data(self.data)

    ts = self.data['TRADEDATE'].max()
    self.interval_max = ts.year * 365 + ts.day_of_year


  def re_train(self, device='cuda'):
    """
    Заново обучает модель, если есть новые данные
    """
    if (self.max_trained == self.interval_max):
      return

    data_with_features = self.data_with_features.copy()
    data_with_features['y'] = self.data_with_features['CLOSE'].shift(-1) - self.data_with_features['CLOSE']
    data_with_features.sort_values('TRADEDATE').dropna(inplace=True)

    batch_size = 108

    # data_train = data.iloc[:-30]

    X_train = data_with_features[self.features].values
    y_train = data_with_features['y'].values

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = RTSNet(in_dim=self.features.shape[0], out_dim=1, n_layers=5, p=0.28).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2944, betas=(0.974, 0.879))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5847)

    model, history = train(
        model, criterion, optimizer,
        train_batch_gen, val_batch_gen=None,
        num_epochs=50,
        visualize_freq=10000,
        scheduler=scheduler,
        interval_max=self.interval_max

    )

    self.max_trained = self.interval_max
    self.model = model
